#!/usr/bin/env python3
"""Merge one or more converted Draw Things LoRA `.ckpt` files into an unquantized base `.ckpt`.

Operates directly in Draw Things `.ckpt` space (after `dt_lora_convert.sh` has already applied
rotary de-interleaving and `alpha / rank` + `--scale-factor` scaling to `__up__`/`__down__`),
writing an inline unquantized `_f16.ckpt` ready for `bash tools/dt_quantize.sh <name> --codec q6p`.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.dt_weellm.sqlite_store import CkptTensorRow, DrawThingsCkptStore
from tools.dt_weellm.transforms import (
    DATATYPE_CODE_TO_NAME,
    DT_BFLOAT16,
    DT_FLOAT16,
    DT_FLOAT32,
)

UP_SUFFIX = "__up__"
DOWN_SUFFIX = "__down__"
DIFF_SUFFIX = "__diff__"


@dataclass(frozen=True)
class LoraSpec:
    path: Path
    weight: float = 1.0


def parse_lora_spec(raw_spec: str) -> LoraSpec:
    """Parse `PATH[:WEIGHT]` into a validated `LoraSpec`."""
    text = raw_spec.strip()
    if not text:
        raise ValueError("LoRA specification must not be empty")

    path_str = text
    weight = 1.0
    if ":" in text:
        candidate_path, candidate_weight = text.rsplit(":", 1)
        if candidate_path:
            try:
                parsed_weight = float(candidate_weight)
            except ValueError:
                parsed_weight = math.nan
            if not math.isnan(parsed_weight):
                path_str = candidate_path
                weight = parsed_weight

    if not math.isfinite(weight):
        raise ValueError(f"LoRA weight must be finite: {raw_spec!r}")
    return LoraSpec(path=Path(path_str).expanduser().resolve(), weight=weight)


def _ensure_unquantized_meta(meta: CkptTensorRow, label: str) -> None:
    codec_id = meta.codec_identifier & ~0x1000_0000
    if codec_id != 0 or meta.reserved != 0:
        raise ValueError(
            f"{label} tensor {meta.name!r} is quantized/encoded (codec={meta.codec_identifier:#x}); "
            "merging requires unquantized _f16.ckpt stores"
        )
    if meta.datatype not in (DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16):
        raise ValueError(f"{label} tensor {meta.name!r} has unsupported datatype code {meta.datatype}")


def _collect_lora_targets(
    lora_store: DrawThingsCkptStore,
    base_keys: set[str],
    allow_unmatched: bool = False,
) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, str], List[str]]:
    """Group LoRA store keys into `(up_key, down_key)` pairs and direct delta keys per base tensor."""
    pairs_partial: Dict[str, Dict[str, str]] = {}
    direct_deltas: Dict[str, str] = {}
    unmatched: List[str] = []

    for key in sorted(lora_store.list_keys()):
        meta = lora_store.read_metadata(key)
        if meta is None:
            raise ValueError(f"Unreadable LoRA metadata for {key!r}")
        _ensure_unquantized_meta(meta, "LoRA")

        if key.endswith(UP_SUFFIX):
            base_key = key[: -len(UP_SUFFIX)]
            if not base_key:
                raise ValueError(f"Malformed LoRA up key: {key!r}")
            pairs_partial.setdefault(base_key, {})["up"] = key
        elif key.endswith(DOWN_SUFFIX):
            base_key = key[: -len(DOWN_SUFFIX)]
            if not base_key:
                raise ValueError(f"Malformed LoRA down key: {key!r}")
            pairs_partial.setdefault(base_key, {})["down"] = key
        elif key.endswith(DIFF_SUFFIX):
            base_key = key[: -len(DIFF_SUFFIX)]
            if not base_key or base_key in direct_deltas:
                raise ValueError(f"Invalid or duplicate LoRA diff key: {key!r}")
            direct_deltas[base_key] = key
        elif key in base_keys:
            if key in direct_deltas:
                raise ValueError(f"Duplicate direct LoRA delta key: {key!r}")
            direct_deltas[key] = key
        else:
            raise ValueError(
                f"Unsupported LoRA tensor key format (not __up__/__down__/__diff__ or base key): {key!r}"
            )

    complete_pairs: Dict[str, Tuple[str, str]] = {}
    for base_key, sides in sorted(pairs_partial.items()):
        if "up" not in sides or "down" not in sides:
            raise ValueError(f"Incomplete LoRA up/down pair for target {base_key!r}: found {sorted(sides)}")
        if base_key in direct_deltas:
            raise ValueError(f"Target {base_key!r} has both rank-pair and direct delta in the same LoRA")
        if base_key not in base_keys:
            unmatched.append(base_key)
            continue
        complete_pairs[base_key] = (sides["up"], sides["down"])

    filtered_direct: Dict[str, str] = {}
    for base_key, lora_key in sorted(direct_deltas.items()):
        if base_key not in base_keys:
            unmatched.append(base_key)
            continue
        filtered_direct[base_key] = lora_key

    if unmatched and not allow_unmatched:
        raise ValueError(
            f"LoRA contains {len(unmatched)} target(s) absent from base checkpoint: {unmatched[:10]}"
        )
    return complete_pairs, filtered_direct, unmatched


def _compute_lora_pair_delta(
    up_f32: np.ndarray,
    down_f32: np.ndarray,
    base_shape: Sequence[int],
    weight: float,
    target_name: str,
) -> np.ndarray:
    """Compute `weight * (up @ down)` in float32 matching `base_shape`."""
    if up_f32.ndim < 2 or down_f32.ndim < 2:
        raise ValueError(
            f"LoRA up/down tensors for {target_name!r} must be at least 2-D, got {up_f32.shape} and {down_f32.shape}"
        )
    if not np.isfinite(up_f32).all() or not np.isfinite(down_f32).all():
        raise ValueError(f"Non-finite LoRA weights detected for {target_name!r}")

    target_tuple = tuple(int(d) for d in base_shape)
    expected_elems = math.prod(target_tuple)

    # Support 2-D linear `[out, rank] x [rank, in]` as well as 4-D conv `[out, rank, 1, 1] x [rank, in, kh, kw]`
    # and unit-padded 3-D linear `[1, out, in]`.
    if up_f32.ndim == 4 and tuple(up_f32.shape[2:]) != (1, 1):
        raise ValueError(
            f"Unsupported 4-D LoRA up spatial shape for {target_name!r}: {up_f32.shape} (expected trailing (1, 1))"
        )

    up_2d = np.ascontiguousarray(up_f32.reshape(up_f32.shape[0], -1), dtype=np.float32)
    down_2d = np.ascontiguousarray(down_f32.reshape(down_f32.shape[0], -1), dtype=np.float32)

    if up_2d.shape[1] != down_2d.shape[0] or up_2d.shape[1] == 0:
        raise ValueError(
            f"Incompatible LoRA rank dimensions for {target_name!r}: up={up_f32.shape}, down={down_f32.shape}"
        )

    merged_2d_elems = up_2d.shape[0] * down_2d.shape[1]
    if merged_2d_elems != expected_elems:
        raise ValueError(
            f"LoRA delta element count {merged_2d_elems} (from up={up_f32.shape}, down={down_f32.shape}) "
            f"does not match base tensor {target_name!r} shape {target_tuple}"
        )

    if len(target_tuple) == 2 and (up_2d.shape[0], down_2d.shape[1]) != target_tuple:
        raise ValueError(
            f"LoRA matrix shape {(up_2d.shape[0], down_2d.shape[1])} mismatches 2-D base "
            f"{target_name!r} shape {target_tuple}"
        )
    if len(target_tuple) == 4 and (up_2d.shape[0] != target_tuple[0] or down_2d.shape[1] != math.prod(target_tuple[1:])):
        raise ValueError(
            f"Conv LoRA shape up={up_f32.shape}, down={down_f32.shape} mismatches 4-D base "
            f"{target_name!r} shape {target_tuple}"
        )

    delta_f32 = (up_2d @ down_2d).reshape(target_tuple)
    if weight != 1.0:
        delta_f32 = delta_f32 * np.float32(weight)
    if not np.isfinite(delta_f32).all():
        raise ValueError(f"Non-finite LoRA product for {target_name!r}")
    return delta_f32


def _write_raw_unmodified_tensor(
    out_store: DrawThingsCkptStore,
    meta: CkptTensorRow,
    payload: bytes,
) -> None:
    """Copy an unmodified tensor's exact bytes and metadata into `out_store`."""
    clean_type = int(meta.tensor_type) & 0xFFFFFFFF
    clean_format = int(meta.tensor_format) & 0xFFFFFFFF
    out_store.con.execute("DELETE FROM tensor_splits WHERE name=?", (meta.name,))
    split = len(payload) > out_store.chunk_bytes
    out_store.con.execute(
        "INSERT OR REPLACE INTO tensors (name, type, format, datatype, dim, data) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            meta.name,
            clean_type,
            clean_format | ((1 << 32) if split else 0),
            int(meta.datatype),
            sqlite3.Binary(meta.dim_blob),
            sqlite3.Binary(payload[: out_store.chunk_bytes]),
        ),
    )
    if split:
        out_store.con.executemany(
            "INSERT INTO tensor_splits (name, part, data) VALUES (?, ?, ?)",
            (
                (meta.name, part, sqlite3.Binary(payload[offset : offset + out_store.chunk_bytes]))
                for part, offset in enumerate(range(out_store.chunk_bytes, len(payload), out_store.chunk_bytes))
            ),
        )


def merge_loras_into_ckpt(
    base_ckpt: str | Path,
    loras: Sequence[LoraSpec],
    output_ckpt: str | Path,
    overwrite: bool = False,
    allow_unmatched: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Merge one or more converted Draw Things LoRAs into `base_ckpt` and write `output_ckpt`."""
    if not loras:
        raise ValueError("At least one LoRA specification is required")

    base_path = Path(base_ckpt).expanduser().resolve()
    out_path = Path(output_ckpt).expanduser().resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"Base checkpoint not found: {base_path}")
    if base_path == out_path:
        raise ValueError("Base and output checkpoints must be different files")
    for spec in loras:
        if not spec.path.is_file():
            raise FileNotFoundError(f"LoRA checkpoint not found: {spec.path}")
        if spec.path == out_path:
            raise ValueError(f"LoRA and output checkpoints must be different files: {spec.path}")
        if not math.isfinite(spec.weight):
            raise ValueError(f"LoRA weight must be finite: {spec.weight}")

    if not dry_run:
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Output exists; pass --force to overwrite: {out_path}")
        if any(Path(str(out_path) + sfx).exists() for sfx in ("-tensordata", "-wal", "-shm", "-journal")):
            raise ValueError(f"Refusing to overwrite checkpoint with existing sidecar/journal: {out_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

    staging_path: Optional[Path] = None
    try:
        with ExitStack() as stack:
            base_store = stack.enter_context(DrawThingsCkptStore(base_path, writable=False))
            base_keys_list = base_store.list_keys()
            if not base_keys_list:
                raise ValueError(f"Base checkpoint contains no tensors: {base_path}")
            base_keys = set(base_keys_list)

            base_metas: Dict[str, CkptTensorRow] = {}
            for key in base_keys_list:
                meta = base_store.read_metadata(key)
                if meta is None:
                    raise ValueError(f"Unreadable base tensor metadata: {key!r}")
                _ensure_unquantized_meta(meta, "Base")
                base_metas[key] = meta

            lora_plans: List[Tuple[LoraSpec, DrawThingsCkptStore, Dict[str, Tuple[str, str]], Dict[str, str]]] = []
            modified_keys: set[str] = set()
            total_pair_ops = 0
            total_direct_ops = 0
            total_unmatched = 0

            for spec in loras:
                l_store = stack.enter_context(DrawThingsCkptStore(spec.path, writable=False))
                pairs, directs, unmatched = _collect_lora_targets(
                    l_store, base_keys=base_keys, allow_unmatched=allow_unmatched
                )
                if not pairs and not directs:
                    raise ValueError(f"LoRA applies zero matching tensors to base checkpoint: {spec.path}")
                lora_plans.append((spec, l_store, pairs, directs))
                modified_keys.update(pairs.keys())
                modified_keys.update(directs.keys())
                total_pair_ops += len(pairs)
                total_direct_ops += len(directs)
                total_unmatched += len(unmatched)

            out_store: Optional[DrawThingsCkptStore] = None
            if not dry_run:
                fd, staging_name = tempfile.mkstemp(prefix=f".{out_path.name}.tmp.", dir=out_path.parent)
                os.close(fd)
                staging_path = Path(staging_name)
                out_store = stack.enter_context(DrawThingsCkptStore(staging_path, writable=True))
                out_store.con.execute("PRAGMA synchronous=OFF")

            max_abs_delta = 0.0
            dtype_counts: Dict[str, int] = {}

            for idx, key in enumerate(base_keys_list, start=1):
                meta = base_metas[key]
                dt_label = DATATYPE_CODE_TO_NAME.get(meta.datatype, str(meta.datatype))
                dtype_counts[dt_label] = dtype_counts.get(dt_label, 0) + 1

                if key not in modified_keys:
                    if out_store is not None:
                        raw_res = base_store.read_raw_tensor(key)
                        if raw_res is None:
                            raise ValueError(f"Missing raw base tensor payload: {key!r}")
                        _, raw_payload = raw_res
                        _write_raw_unmodified_tensor(out_store, meta, raw_payload)
                else:
                    base_res = base_store.read_f32(key)
                    if base_res is None:
                        raise ValueError(f"Unreadable base tensor: {key!r}")
                    _, base_f32 = base_res
                    if not np.isfinite(base_f32).all():
                        raise ValueError(f"Base tensor {key!r} contains non-finite values")

                    merged_f32 = np.array(base_f32, dtype=np.float32, copy=True)
                    for spec, l_store, pairs, directs in lora_plans:
                        if spec.weight == 0.0:
                            continue
                        if key in pairs:
                            up_key, down_key = pairs[key]
                            up_res = l_store.read_f32(up_key)
                            down_res = l_store.read_f32(down_key)
                            if up_res is None or down_res is None:
                                raise ValueError(f"Unreadable LoRA pair for {key!r}")
                            delta_f32 = _compute_lora_pair_delta(
                                up_res[1], down_res[1], meta.shape, spec.weight, key
                            )
                            max_abs_delta = max(max_abs_delta, float(np.max(np.abs(delta_f32))))
                            merged_f32 += delta_f32
                        elif key in directs:
                            diff_key = directs[key]
                            diff_res = l_store.read_f32(diff_key)
                            if diff_res is None:
                                raise ValueError(f"Unreadable LoRA direct delta for {key!r}")
                            diff_f32 = diff_res[1]
                            if tuple(diff_f32.shape) != tuple(meta.shape):
                                raise ValueError(
                                    f"Direct LoRA delta shape {diff_f32.shape} mismatches base {key!r} shape {meta.shape}"
                                )
                            if not np.isfinite(diff_f32).all():
                                raise ValueError(f"Non-finite LoRA direct delta for {key!r}")
                            scaled_diff = (
                                diff_f32 * np.float32(spec.weight) if spec.weight != 1.0 else diff_f32
                            )
                            max_abs_delta = max(max_abs_delta, float(np.max(np.abs(scaled_diff))))
                            merged_f32 += scaled_diff

                    if not np.isfinite(merged_f32).all():
                        raise ValueError(f"Merged tensor {key!r} overflowed to NaN/Inf")

                    if out_store is not None:
                        out_store.write_tensor(
                            name=key,
                            arr_f32=merged_f32,
                            datatype=meta.datatype,
                            tensor_type=int(meta.tensor_type) & 0xFFFFFFFF,
                            tensor_format=int(meta.tensor_format) & 0xFFFFFFFF,
                            ndim_slots=meta.ndim_slots,
                            rank_policy="preserve",
                        )

                if out_store is not None and idx % 256 == 0:
                    out_store.commit()

            if out_store is not None:
                out_store.commit()
                if set(out_store.list_keys()) != base_keys:
                    raise ValueError("Merged checkpoint keyset differs from base checkpoint")
                if out_store.con.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                    raise ValueError("Merged checkpoint failed SQLite integrity check")

        if staging_path is not None:
            with staging_path.open("rb") as staged_file:
                os.fsync(staged_file.fileno())
            os.replace(staging_path, out_path)
            dir_fd = os.open(out_path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
            staging_path = None
    finally:
        if staging_path is not None and staging_path.exists():
            try:
                staging_path.unlink()
            except OSError:
                pass

    return {
        "base_ckpt": str(base_path),
        "output_ckpt": str(out_path),
        "loras_applied": len(loras),
        "base_tensors_total": len(base_keys_list),
        "tensors_modified": len(modified_keys),
        "tensors_copied_verbatim": len(base_keys_list) - len(modified_keys),
        "pair_deltas_applied": total_pair_ops,
        "direct_deltas_applied": total_direct_ops,
        "unmatched_lora_targets": total_unmatched,
        "max_abs_delta": max_abs_delta,
        "dtype_counts": dtype_counts,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bake (merge) converted Draw Things LoRA _f16.ckpt file(s) into a base _f16.ckpt."
    )
    parser.add_argument("--base", required=True, help="Path to unquantized base _f16.ckpt")
    parser.add_argument(
        "--lora",
        action="append",
        required=True,
        help="Repeatable PATH[:WEIGHT] for converted _lora_f16.ckpt (default weight: 1.0)",
    )
    parser.add_argument("--output", required=True, help="Output merged _f16.ckpt path")
    parser.add_argument("--force", action="store_true", help="Overwrite output .ckpt if it exists")
    parser.add_argument(
        "--allow-unmatched",
        action="store_true",
        help="Allow LoRA targets not present in the base checkpoint instead of failing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and compute deltas without writing the output file",
    )

    args = parser.parse_args(argv)
    lora_specs = [parse_lora_spec(raw) for raw in args.lora]
    stats = merge_loras_into_ckpt(
        base_ckpt=args.base,
        loras=lora_specs,
        output_ckpt=args.output,
        overwrite=args.force,
        allow_unmatched=args.allow_unmatched,
        dry_run=args.dry_run,
    )
    print("== Draw Things LoRA Checkpoint Merge Summary ==")
    for key, value in stats.items():
        if key == "dtype_counts":
            formatted = ", ".join(f"{dt}:{cnt}" for dt, cnt in sorted(value.items()))
            print(f"  dtype_counts={formatted}")
        elif isinstance(value, float):
            print(f"  {key}={value:.6g}")
        else:
            print(f"  {key}={value}")
    print("RESULT=PASS" + (" (dry-run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
