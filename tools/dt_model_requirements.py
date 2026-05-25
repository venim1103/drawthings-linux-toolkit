#!/usr/bin/env python3
"""Audit and download Draw Things model dependencies from ModelZoo metadata.

This script reads draw-things-community/Libraries/ModelZoo/Sources/ModelZoo.swift
as the source of truth for model specs and SHA256 hashes.

Examples:
  python tools/dt_model_requirements.py --list --filter ltx-2.3
  python tools/dt_model_requirements.py --model ltx_2.3_22b_distilled_1.1_q6p.ckpt
  python tools/dt_model_requirements.py --model "LTX-2.3 22B [distilled] 1.1 (6-bit)" --download-missing
  python tools/dt_model_requirements.py --model ltx_2.3_22b_distilled_1.1_q6p.ckpt --verify-existing --repair-mismatched
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELZOO_SWIFT = WORKSPACE_ROOT / "draw-things-community/Libraries/ModelZoo/Sources/ModelZoo.swift"
DEFAULT_MODELS_DIR = WORKSPACE_ROOT / "dt-models"
DEFAULT_DOWNLOAD_BASE = "https://static.libnnc.org"
DEFAULT_CHUNK_SIZE = 1024 * 1024

OverwritePolicy = Literal["ask", "never", "always"]


@dataclass
class ModelSpec:
    name: str
    file: str
    version: str
    text_encoder: str | None
    autoencoder: str | None
    image_encoder: str | None
    clip_encoder: str | None
    t5_encoder: str | None
    diffusion_mapping: str | None
    additional_clip_encoders: List[str]
    stage_models: List[str]
    latents_upscalers: List[str]


def _read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path.read_text(encoding="utf-8")


def _extract_builtin_section(swift_text: str) -> str:
    anchor = "public static let builtinSpecifications: [Specification] = ["
    start = swift_text.find(anchor)
    if start < 0:
        raise RuntimeError("Could not locate builtinSpecifications in ModelZoo.swift")

    i = start + len(anchor)
    depth = 1
    while i < len(swift_text):
        ch = swift_text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return swift_text[start + len(anchor) : i]
        i += 1

    raise RuntimeError("Unterminated builtinSpecifications array in ModelZoo.swift")


def _extract_spec_blocks(section: str) -> List[str]:
    blocks: List[str] = []
    i = 0
    needle = "Specification("
    while True:
        start = section.find(needle, i)
        if start < 0:
            break
        j = start + len(needle)
        depth = 1
        while j < len(section) and depth > 0:
            ch = section[j]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            j += 1
        if depth != 0:
            raise RuntimeError("Unterminated Specification(...) block in builtinSpecifications")
        blocks.append(section[start:j])
        i = j
    return blocks


def _first_string(block: str, key: str) -> str | None:
    m = re.search(rf"\b{re.escape(key)}\s*:\s*\"([^\"]+)\"", block)
    return m.group(1) if m else None


def _first_enum(block: str, key: str) -> str | None:
    m = re.search(rf"\b{re.escape(key)}\s*:\s*\.([A-Za-z0-9_]+)", block)
    return m.group(1) if m else None


def _array_strings(block: str, key: str) -> List[str]:
    m = re.search(rf"\b{re.escape(key)}\s*:\s*\[(.*?)\]", block, re.S)
    if not m:
        return []
    return re.findall(r'"([^\"]+)"', m.group(1))


def _latents_upscalers(block: str) -> List[str]:
    m = re.search(r"\blatentsUpscalers\s*:\s*\[(.*?)\]", block, re.S)
    if not m:
        return []
    return re.findall(r'\.init\(\s*file:\s*"([^\"]+)"', m.group(1))


def parse_model_specs(swift_text: str) -> List[ModelSpec]:
    section = _extract_builtin_section(swift_text)
    blocks = _extract_spec_blocks(section)
    out: List[ModelSpec] = []
    for b in blocks:
        name = _first_string(b, "name")
        file = _first_string(b, "file")
        version = _first_enum(b, "version")
        if not name or not file or not version:
            continue
        out.append(
            ModelSpec(
                name=name,
                file=file,
                version=version,
                text_encoder=_first_string(b, "textEncoder"),
                autoencoder=_first_string(b, "autoencoder"),
                image_encoder=_first_string(b, "imageEncoder"),
                clip_encoder=_first_string(b, "clipEncoder"),
                t5_encoder=_first_string(b, "t5Encoder"),
                diffusion_mapping=_first_string(b, "diffusionMapping"),
                additional_clip_encoders=_array_strings(b, "additionalClipEncoders"),
                stage_models=_array_strings(b, "stageModels"),
                latents_upscalers=_latents_upscalers(b),
            )
        )
    return out


def parse_sha_mapping(swift_text: str) -> dict[str, str]:
    pairs = re.findall(r'"([^\"]+)"\s*:\s*"([0-9a-f]{64})"', swift_text, flags=re.I | re.S)
    return {k: v.lower() for (k, v) in pairs}


def resolve_spec(specs: Iterable[ModelSpec], model_ref: str) -> ModelSpec:
    ref = model_ref.strip().lower()
    candidates = list(specs)

    exact = [
        s
        for s in candidates
        if s.file.lower() == ref or s.name.lower() == ref
    ]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise RuntimeError(
            "Model reference is ambiguous (exact match on multiple entries):\n"
            + "\n".join(f"  - {s.file} ({s.name})" for s in exact)
        )

    partial = [
        s
        for s in candidates
        if ref in s.file.lower() or ref in s.name.lower()
    ]
    if len(partial) == 1:
        return partial[0]
    if len(partial) == 0:
        raise RuntimeError(
            f"No model match for: {model_ref}\n"
            "Tip: run with --list to see known model ids."
        )

    raise RuntimeError(
        "Model reference is ambiguous. Use a more specific file name:\n"
        + "\n".join(f"  - {s.file} ({s.name})" for s in partial[:20])
    )


def required_files_for_spec(
    spec: ModelSpec,
    *,
    include_upscalers: bool,
    include_dependencies: bool,
) -> List[str]:
    files: List[str] = []

    def add(name: str | None) -> None:
        if not name:
            return
        if name not in files:
            files.append(name)

    add(spec.file)
    if not include_dependencies:
        return files

    add(spec.text_encoder or ("clip_vit_l14_f16.ckpt" if spec.version == "v1" else "open_clip_vit_h14_f16.ckpt"))
    add(spec.autoencoder or "vae_ft_mse_840000_f16.ckpt")
    add(spec.image_encoder)
    add(spec.clip_encoder)
    for x in spec.additional_clip_encoders:
        add(x)
    add(spec.t5_encoder)
    add(spec.diffusion_mapping)
    for x in spec.stage_models:
        add(x)
    if include_upscalers:
        for x in spec.latents_upscalers:
            add(x)

    return files


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ordered_unique(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _format_bytes(value: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    n = float(value)
    for unit in units:
        if n < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(n)}{unit}"
            return f"{n:.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}TB"


def _format_duration(seconds: float) -> str:
    if not seconds or seconds == float("inf") or seconds < 0:
        return "?"
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _render_progress(label: str, done: int, total: int, started_at: float, width: int = 26) -> str:
    elapsed = max(time.time() - started_at, 1e-6)
    speed = done / elapsed
    if total > 0:
        pct = min(100.0, (done / total) * 100.0)
        filled = min(width, int((pct / 100.0) * width))
        bar = "#" * filled + "-" * (width - filled)
        remaining = max(0, total - done)
        eta = remaining / speed if speed > 0 else float("inf")
        return (
            f"{label} [{bar}] {pct:6.2f}% "
            f"{_format_bytes(done)}/{_format_bytes(total)} "
            f"{_format_bytes(speed)}/s ETA {_format_duration(eta)}"
        )
    return f"{label} {_format_bytes(done)} {_format_bytes(speed)}/s"


def _ask_overwrite(path: Path, policy: OverwritePolicy) -> tuple[bool, OverwritePolicy]:
    if policy == "always":
        return True, policy
    if policy == "never":
        return False, policy
    if not sys.stdin.isatty():
        print(
            f"warning: {path.name} exists; non-interactive session with overwrite-policy=ask, skipping overwrite"
        )
        return False, policy

    while True:
        answer = input(
            f"{path} exists. Overwrite? [y]es/[n]o/[a]ll/[s]kip-all: "
        ).strip().lower()
        if answer in {"y", "yes"}:
            return True, policy
        if answer in {"n", "no", ""}:
            return False, policy
        if answer in {"a", "all"}:
            return True, "always"
        if answer in {"s", "skip", "skip-all", "none"}:
            return False, "never"
        print("Please answer y, n, a, or s.")


def _response_status(resp) -> int:
    status = getattr(resp, "status", None)
    if isinstance(status, int):
        return status
    code = resp.getcode()
    return int(code) if code is not None else 0


def _response_total_bytes(resp, resume_from: int) -> int:
    content_range = resp.headers.get("Content-Range", "")
    m = re.match(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", content_range)
    if m and m.group(3).isdigit():
        return int(m.group(3))
    content_length = resp.headers.get("Content-Length", "")
    if content_length.isdigit():
        return resume_from + int(content_length)
    return 0


def _stream_response_to_tmp(
    resp,
    tmp_path: Path,
    *,
    resume_from: int,
    chunk_size: int,
    label: str,
) -> tuple[int, int]:
    total_bytes = _response_total_bytes(resp, resume_from)
    downloaded = resume_from
    started_at = time.time()
    last_draw = 0.0
    mode = "ab" if resume_from > 0 else "wb"

    with tmp_path.open(mode) as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_draw >= 0.2:
                print(
                    "\r" + _render_progress(label, downloaded, total_bytes, started_at),
                    end="",
                    flush=True,
                )
                last_draw = now

    print("\r" + _render_progress(label, downloaded, total_bytes, started_at) + " " * 8)
    return downloaded, total_bytes


def download_file(
    url: str,
    out_path: Path,
    *,
    index: int,
    total_files: int,
    overwrite_policy: OverwritePolicy,
    retries: int,
    timeout: int,
    chunk_size: int,
) -> tuple[str, OverwritePolicy]:
    existed_before = out_path.exists()
    if existed_before:
        overwrite, overwrite_policy = _ask_overwrite(out_path, overwrite_policy)
        if not overwrite:
            return "skipped", overwrite_policy

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")

    # Symlink aliases are not safe resume sources (they can point to another model file).
    if out_path.is_symlink():
        out_path.unlink(missing_ok=True)
        existed_before = False

    # A stale .part symlink can append bytes into an unrelated target; always drop it.
    if tmp.is_symlink():
        tmp.unlink(missing_ok=True)

    # If we are replacing an existing file, preserve current bytes in .part so resume can continue.
    if existed_before and out_path.exists() and not tmp.exists():
        os.replace(out_path, tmp)

    attempts = retries + 1
    action = "overwritten" if existed_before else "downloaded"
    label = f"[{index}/{total_files}] {out_path.name}"

    for attempt in range(1, attempts + 1):
        try:
            resume_from = tmp.stat().st_size if tmp.exists() else 0
            headers = {"User-Agent": "dt-model-requirements/1.1"}
            if resume_from > 0:
                headers["Range"] = f"bytes={resume_from}-"
                print(f"Resuming {out_path.name} at {_format_bytes(resume_from)}")

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = _response_status(resp)

                # Some proxies/CDNs ignore Range and return full content (200) even when Range was requested.
                if resume_from > 0 and status != 206:
                    tmp.unlink(missing_ok=True)
                    resume_from = 0
                    req = urllib.request.Request(url, headers={"User-Agent": "dt-model-requirements/1.1"})
                    with urllib.request.urlopen(req, timeout=timeout) as fresh_resp:
                        downloaded, total_bytes = _stream_response_to_tmp(
                            fresh_resp,
                            tmp,
                            resume_from=0,
                            chunk_size=chunk_size,
                            label=label,
                        )
                else:
                    downloaded, total_bytes = _stream_response_to_tmp(
                        resp,
                        tmp,
                        resume_from=resume_from,
                        chunk_size=chunk_size,
                        label=label,
                    )

            if total_bytes > 0 and downloaded < total_bytes:
                raise RuntimeError(
                    "incomplete download (connection ended early): "
                    f"received {downloaded} of {total_bytes} bytes"
                )

            os.replace(tmp, out_path)
            return action, overwrite_policy

        except urllib.error.HTTPError as exc:
            if exc.code == 416 and tmp.exists():
                # Range beyond EOF or stale .part state; reset and retry from zero.
                tmp.unlink(missing_ok=True)
                if attempt < attempts:
                    print(
                        f"download attempt {attempt}/{attempts} got HTTP 416 for {out_path.name}; "
                        "resetting partial data and retrying..."
                    )
                    continue
            if attempt >= attempts:
                raise RuntimeError(f"download failed for {out_path.name}: {exc}") from exc
            print(
                f"download attempt {attempt}/{attempts} failed for {out_path.name}: {exc}; retrying..."
            )

        except Exception as exc:
            # Keep partial data so next retry can resume instead of starting from zero.
            if attempt >= attempts:
                raise RuntimeError(f"download failed for {out_path.name}: {exc}") from exc
            print(
                f"download attempt {attempt}/{attempts} failed for {out_path.name}: {exc}; retrying..."
            )

    return action, overwrite_policy


def _validate_positive(value: int, name: str) -> None:
    if value <= 0:
        raise RuntimeError(f"{name} must be > 0")


def _validate_non_negative(value: int, name: str) -> None:
    if value < 0:
        raise RuntimeError(f"{name} must be >= 0")


def _print_urls(title: str, files: Iterable[str], base_url: str) -> None:
    files = list(files)
    if not files:
        return
    print("")
    print(title)
    for file_name in files:
        encoded = urllib.parse.quote(file_name)
        print(f"{base_url.rstrip('/')}/{encoded}")


def _verify_hash_or_raise(path: Path, expected_sha: str, file_name: str) -> None:
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"SHA256 mismatch for {file_name}: expected {expected_sha}, got {actual_sha}"
        )


def print_model_list(
    specs: List[ModelSpec],
    filter_text: str | None,
    *,
    models_dir: Path,
    downloaded_only: bool,
) -> None:
    rows = specs
    if filter_text:
        x = filter_text.lower()
        rows = [s for s in rows if x in s.file.lower() or x in s.name.lower() or x in s.version.lower()]
    if downloaded_only:
        rows = [s for s in rows if (models_dir / s.file).exists()]
    for s in rows:
        print(f"{s.file}\t{s.name}\t{s.version}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit and download Draw Things model dependencies")
    ap.add_argument("--model", help="Model file or display name (exact or partial)")
    ap.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR), help="Draw Things models directory")
    ap.add_argument("--modelzoo-swift", default=str(DEFAULT_MODELZOO_SWIFT), help="Path to ModelZoo.swift")
    ap.add_argument("--download-base", default=DEFAULT_DOWNLOAD_BASE, help="Download base URL")
    ap.add_argument("--download-missing", action="store_true", help="Download any missing required files")
    ap.add_argument(
        "--repair-mismatched",
        action="store_true",
        help="Re-download files with hash mismatches (requires --verify-existing)",
    )
    ap.add_argument(
        "--redownload-existing",
        action="store_true",
        help="Queue all required files for download even if they already exist",
    )
    ap.add_argument(
        "--overwrite-policy",
        choices=["ask", "never", "always"],
        default="ask",
        help="Behavior when a queued download target already exists",
    )
    ap.add_argument(
        "--yes",
        action="store_true",
        help="Shortcut for --overwrite-policy always",
    )
    ap.add_argument("--retries", type=int, default=2, help="Retries per file when download fails")
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds")
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Download chunk size in bytes")
    ap.add_argument(
        "--include-dependencies",
        action="store_true",
        default=True,
        help="Include dependent model files in requirements (default: true)",
    )
    ap.add_argument(
        "--no-include-dependencies",
        dest="include_dependencies",
        action="store_false",
        help="Only require the primary model file",
    )
    ap.add_argument("--include-upscalers", action="store_true", default=True, help="Include latent upscalers in requirements (default: true)")
    ap.add_argument("--no-include-upscalers", dest="include_upscalers", action="store_false", help="Exclude latent upscalers")
    ap.add_argument("--list", action="store_true", help="List known built-in model mappings")
    ap.add_argument("--filter", help="Filter text for --list")
    ap.add_argument(
        "--downloaded-only",
        action="store_true",
        help="With --list, show only models present in --models-dir",
    )
    ap.add_argument("--verify-existing", action="store_true", help="Verify sha256 of already-present files when hash metadata exists")

    args = ap.parse_args()

    modelzoo_path = Path(args.modelzoo_swift)
    models_dir = Path(args.models_dir)

    try:
        _validate_non_negative(args.retries, "--retries")
        _validate_positive(args.timeout, "--timeout")
        _validate_positive(args.chunk_size, "--chunk-size")
        if args.yes:
            args.overwrite_policy = "always"
        if args.repair_mismatched and not args.verify_existing:
            raise RuntimeError("--repair-mismatched requires --verify-existing")

        swift_text = _read_text(modelzoo_path)
        specs = parse_model_specs(swift_text)
        if not specs:
            raise RuntimeError("No model specifications parsed from ModelZoo.swift")
        sha_map = parse_sha_mapping(swift_text)

        if args.list:
            print_model_list(
                specs,
                args.filter,
                models_dir=models_dir,
                downloaded_only=args.downloaded_only,
            )
            return 0

        if not args.model:
            raise RuntimeError("--model is required unless using --list")

        spec = resolve_spec(specs, args.model)
        required = required_files_for_spec(
            spec,
            include_upscalers=args.include_upscalers,
            include_dependencies=args.include_dependencies,
        )

        print(f"Model: {spec.name}")
        print(f"File: {spec.file}")
        print(f"Version: {spec.version}")
        print(f"Models dir: {models_dir}")
        print("")

        missing: List[str] = []
        mismatched: List[str] = []
        for f in required:
            p = models_dir / f
            exists = p.exists()
            expected_sha = sha_map.get(f)
            status = "present" if exists else "missing"
            extra = ""
            if exists and args.verify_existing and expected_sha:
                actual_sha = sha256_file(p)
                if actual_sha != expected_sha:
                    status = "hash-mismatch"
                    extra = f" expected={expected_sha} actual={actual_sha}"
                    mismatched.append(f)
                else:
                    extra = f" sha256={actual_sha}"
            elif exists and expected_sha:
                extra = f" sha256={expected_sha}"
            elif expected_sha:
                extra = f" sha256={expected_sha}"

            print(f"[{status:13}] {f}{extra}")
            if not exists:
                missing.append(f)

        print("")
        print(f"Required files: {len(required)}")
        print(f"Missing files:  {len(missing)}")
        if args.verify_existing:
            print(f"Mismatched files: {len(mismatched)}")

        _print_urls("Download URLs for missing files:", missing, args.download_base)

        if mismatched and args.verify_existing and not args.repair_mismatched:
            print("")
            print("Tip: run with --repair-mismatched to re-download hash-mismatched files.")

        download_targets: List[str] = []
        if args.redownload_existing:
            download_targets.extend(required)
        if args.download_missing:
            download_targets.extend(missing)
        if args.repair_mismatched:
            download_targets.extend(mismatched)
        download_targets = _ordered_unique(download_targets)

        if download_targets:
            _print_urls("Download URLs for queued files:", download_targets, args.download_base)

            print("")
            print(f"Downloading queued files ({len(download_targets)})...")
            overwrite_policy: OverwritePolicy = args.overwrite_policy
            downloaded_count = 0
            overwritten_count = 0
            skipped_count = 0

            for i, f in enumerate(download_targets, start=1):
                encoded = urllib.parse.quote(f)
                url = f"{args.download_base.rstrip('/')}/{encoded}"
                out_path = models_dir / f
                action, overwrite_policy = download_file(
                    url,
                    out_path,
                    index=i,
                    total_files=len(download_targets),
                    overwrite_policy=overwrite_policy,
                    retries=args.retries,
                    timeout=args.timeout,
                    chunk_size=args.chunk_size,
                )

                if action == "skipped":
                    skipped_count += 1
                    continue
                if action == "overwritten":
                    overwritten_count += 1
                else:
                    downloaded_count += 1

                expected_sha = sha_map.get(f)
                if expected_sha:
                    _verify_hash_or_raise(out_path, expected_sha, f)

            print(
                "Download complete. "
                f"downloaded={downloaded_count}, overwritten={overwritten_count}, skipped={skipped_count}."
            )

        if mismatched and args.verify_existing and not args.repair_mismatched:
            return 3
        if missing and not (args.download_missing or args.redownload_existing):
            return 2
        return 0

    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
