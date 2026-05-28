#!/usr/bin/env python3
"""Validate converted Draw Things checkpoint structure.

This is intended as a fast post-conversion integrity check so the wrapper can
fail early when converter output is structurally incompatible with runtime.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import struct
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


LTX23_SOURCE_REQUIRED_KEYS: Sequence[str] = (
    "transformer_blocks.47.attn1.to_q.weight",
    "audio_patchify_proj.weight",
    "video_embeddings_connector.learnable_registers",
    "audio_embeddings_connector.learnable_registers",
    "video_embeddings_connector.transformer_1d_blocks.7.attn1.to_q.weight",
    "audio_embeddings_connector.transformer_1d_blocks.7.attn1.to_q.weight",
)

LTX23_REQUIRED_PREFIX_MIN_COUNTS: Dict[str, int] = {
    "__dit__[t-a_embedder-": 1,
    "__dit__[t-x_embedder-": 1,
    "__text_audio_connector__[t-down_proj-": 1,
    "__text_video_connector__[t-down_proj-": 1,
}

KNOWN_INVALID_PLACEHOLDER_KEYS = {
    "__dit__[]",
    "__text_audio_connector__[]",
    "__text_video_connector__[]",
}


def _parse_safetensors_header(path: Path) -> dict:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"cannot read safetensors header length: {path}")
        header_len = struct.unpack("<Q", raw)[0]
        header_bytes = f.read(header_len)
    payload = json.loads(header_bytes)
    if not isinstance(payload, dict):
        raise ValueError(f"safetensors header is not an object: {path}")
    return payload


def _header_has_key(header: dict, key: str) -> bool:
    return (
        key in header
        or f"model.diffusion_model.{key}" in header
        or f"model.{key}" in header
    )


def detect_profile_from_source(source_safetensors: Path) -> str:
    try:
        header = _parse_safetensors_header(source_safetensors)
    except Exception:
        return "none"

    if all(_header_has_key(header, key) for key in LTX23_SOURCE_REQUIRED_KEYS):
        return "ltx2_3"
    return "none"


def load_tensor_keys(ckpt_path: Path) -> List[str]:
    try:
        con = sqlite3.connect(str(ckpt_path))
        cur = con.cursor()
        cur.execute("SELECT name FROM tensors")
        rows = cur.fetchall()
        con.close()
    except sqlite3.Error as exc:
        raise RuntimeError(f"failed to read tensors table from {ckpt_path}: {exc}") from exc

    keys = [str(row[0]) for row in rows if row and row[0] is not None]
    return keys


def count_prefix(keys: Iterable[str], prefix: str) -> int:
    return sum(1 for key in keys if key.startswith(prefix))


def infer_profile_from_output_keys(keys: Sequence[str]) -> str:
    ltx_markers = (
        "__text_audio_connector__",
        "__text_video_connector__",
        "__dit__[t-a2v_",
        "__dit__[t-v2a_",
    )
    if any(any(marker in key for marker in ltx_markers) for key in keys):
        return "ltx2_3"
    return "none"


def validate_ckpt(
    ckpt_path: Path,
    profile: str,
    source_safetensors: Path | None,
) -> int:
    if not ckpt_path.exists():
        print(f"RESULT=FAIL missing checkpoint file: {ckpt_path}")
        return 2

    try:
        keys = load_tensor_keys(ckpt_path)
    except Exception as exc:
        print(f"RESULT=FAIL {exc}")
        return 2

    resolved_profile = profile
    if resolved_profile == "auto":
        resolved_profile = "none"
        if source_safetensors is not None and source_safetensors.exists():
            resolved_profile = detect_profile_from_source(source_safetensors)
        if resolved_profile == "none":
            resolved_profile = infer_profile_from_output_keys(keys)

    key_set = set(keys)

    print("== Converted Checkpoint Validation ==")
    print(f"file={ckpt_path}")
    if source_safetensors is not None:
        print(f"source_safetensors={source_safetensors}")
    print(f"tensor_count={len(keys)}")
    print(f"profile={resolved_profile}")

    failures: List[str] = []

    if not keys:
        failures.append("tensors table is empty")

    placeholder_hits = sorted(key for key in KNOWN_INVALID_PLACEHOLDER_KEYS if key in key_set)
    if placeholder_hits:
        failures.append(f"invalid placeholder keys found: {placeholder_hits}")

    if resolved_profile == "ltx2_3":
        print("== LTX2.3 Required Prefix Checks ==")
        for prefix, min_count in LTX23_REQUIRED_PREFIX_MIN_COUNTS.items():
            found = count_prefix(keys, prefix)
            print(f"{prefix} count={found} required>={min_count}")
            if found < min_count:
                failures.append(
                    f"missing required LTX2.3 tensor prefix: {prefix} (found {found}, expected >= {min_count})"
                )

    if failures:
        print("== Failures ==")
        for failure in failures:
            print(f"- {failure}")
        print("RESULT=FAIL")
        return 1

    print("RESULT=PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate converted Draw Things checkpoint")
    parser.add_argument("--file", required=True, help="Path to converted .ckpt")
    parser.add_argument(
        "--profile",
        choices=["auto", "none", "ltx2_3"],
        default="auto",
        help="Validation profile (default: auto)",
    )
    parser.add_argument(
        "--source-safetensors",
        default="",
        help="Optional source safetensors path used for profile auto-detection",
    )
    args = parser.parse_args()

    source = Path(args.source_safetensors).expanduser().resolve() if args.source_safetensors else None
    ckpt = Path(args.file).expanduser().resolve()

    return validate_ckpt(ckpt, args.profile, source)


if __name__ == "__main__":
    sys.exit(main())
