#!/usr/bin/env python3
"""Pure-Python Streaming Safetensors/GGUF -> Draw Things `.ckpt` Builder & Contract Norm Restorer (W3).

Provides three operations driven by `contracts/<profile>.json`:
  1. `convert`: Stream a single-file or sharded `.safetensors` or `.gguf` directly into a
     Draw Things SQLite `.ckpt` with exact rotary de-interleaving (§17/§18), chunk slicing,
     datatype preservation (§3a BF16 / M3 F32), and rank-3 `[1, 1, N]` norm dims (§22).
  2. `restore-norms`: Contract-driven replacement for `dt_q6p_restore_f32_norms.py` and
     `dt_q6p_restore_qk_norms.py` that restores sensitive norms/modulations to F32 with `[1, 1, N]`
     dims from the model's own `_f16.ckpt` source.
  3. `diff-ckpt`: Compare two Draw Things `.ckpt` stores for key coverage, metadata, and value parity.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tools.dt_weellm.seek_io import open_seeker
from tools.dt_weellm.sqlite_store import DrawThingsCkptStore
from tools.dt_weellm.transforms import (
    DATATYPE_NAME_TO_CODE,
    DT_BFLOAT16,
    DT_FLOAT16,
    DT_FLOAT32,
    DT_FORMAT_NHWC,
    DT_TYPE_CPU,
    decode_dim_blob,
    encode_dim_blob,
    rotary_deinterleave,
    split_tensor_chunks,
)


def load_contract(contract_path: str | Path) -> Dict[str, Any]:
    path = Path(contract_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Contract file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def convert_source_to_ckpt(
    source_path: str | Path,
    contract_path: str | Path,
    output_ckpt: str | Path,
    strict: bool = True,
) -> Dict[str, int]:
    """Stream tensors from `source_path` (.safetensors or .gguf) into `output_ckpt` using `contract_path`."""
    contract = load_contract(contract_path)
    tensor_rules: Dict[str, Dict[str, Any]] = contract.get("tensors", {})
    if not tensor_rules:
        raise ValueError(f"Contract {contract_path} contains no tensor rules")

    out_path = Path(output_ckpt).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = out_path.parent / f".{out_path.name}.tmp.{os.getpid()}"
    if staging_path.exists():
        staging_path.unlink()

    stats = {
        "source_tensors_processed": 0,
        "ckpt_tensors_written": 0,
        "rotary_deinterleaved": 0,
        "split_groups": 0,
        "missing_in_source": 0,
    }

    try:
        with open_seeker(source_path) as seeker, DrawThingsCkptStore(staging_path, writable=True) as store:
            available_keys = set(seeker.keys())
            for src_key, rule in tensor_rules.items():
                resolved_key: Optional[str] = None
                if hasattr(seeker, "resolve_key"):
                    resolved_key = seeker.resolve_key(src_key)
                elif src_key in available_keys:
                    resolved_key = src_key

                if resolved_key is None:
                    stats["missing_in_source"] += 1
                    if strict:
                        raise KeyError(f"Required contract tensor missing from source: {src_key!r}")
                    continue

                arr_f32 = seeker.read_f32(resolved_key)
                if not np.all(np.isfinite(arr_f32)):
                    raise ValueError(f"Source tensor {resolved_key!r} contains NaN or Inf values")

                if rule.get("rotary_interleaved", False):
                    heads = int(rule["num_heads"])
                    head_dim = int(rule["head_dim"])
                    arr_f32 = rotary_deinterleave(arr_f32, num_heads=heads, head_dim=head_dim)
                    stats["rotary_deinterleaved"] += 1

                dt_names: List[str] = rule["dt_names"]
                split_axis = int(rule.get("split_axis", 0))
                if len(dt_names) > 1:
                    chunks = split_tensor_chunks(arr_f32, count=len(dt_names), axis=split_axis)
                    stats["split_groups"] += 1
                else:
                    chunks = [arr_f32]

                dtype_str = str(rule.get("f16_store_datatype", "F16")).upper()
                dt_code = DATATYPE_NAME_TO_CODE[dtype_str]
                rank_policy = str(rule.get("dim_rank_policy", "preserve"))

                for target_name, chunk in zip(dt_names, chunks):
                    store.write_tensor(
                        name=target_name,
                        arr_f32=chunk,
                        datatype=dt_code,
                        tensor_type=DT_TYPE_CPU,
                        tensor_format=DT_FORMAT_NHWC,
                        rank_policy=rank_policy,
                    )
                    stats["ckpt_tensors_written"] += 1

                stats["source_tensors_processed"] += 1
                if stats["source_tensors_processed"] % 256 == 0:
                    store.commit()

            store.commit()

        os.replace(staging_path, out_path)
    finally:
        if staging_path.exists():
            try:
                staging_path.unlink()
            except OSError:
                pass

    return stats


def restore_norms_from_contract(
    quantized_ckpt: str | Path,
    f16_source_ckpt: str | Path,
    contract_path: str | Path,
    ref_ckpt: Optional[str | Path] = None,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Restore F32 norms/modulations and `[1, 1, N]` dims into `quantized_ckpt` from `f16_source_ckpt`."""
    contract = load_contract(contract_path)
    patterns = contract.get("restore_patterns", {})
    ada_ln_re = re.compile(patterns.get("f32_ada_ln_regex", r"ada_ln"))
    qk_norm_re = re.compile(patterns.get("f32_qk_norm_regex", r"(norm_q|norm_k)-\d+-0\]$"))

    q_path = Path(quantized_ckpt).expanduser().resolve()
    src_path = Path(f16_source_ckpt).expanduser().resolve()
    if not q_path.is_file():
        raise FileNotFoundError(f"Quantized ckpt not found: {q_path}")
    if not src_path.is_file():
        raise FileNotFoundError(f"F16 source ckpt not found: {src_path}")

    # Collect target tensor names from contract rules + pattern matching
    contract_f32_targets = set()
    for rule in contract.get("tensors", {}).values():
        if rule.get("quant_restore_policy") == "f32_from_source":
            for n in rule.get("dt_names", []):
                contract_f32_targets.add(n)

    con_q = sqlite3.connect(str(q_path))
    con_q.execute("PRAGMA journal_mode=DELETE")
    cur_q = con_q.cursor()
    cur_q.execute("SELECT name FROM tensors")
    q_keys = [str(r[0]) for r in cur_q.fetchall()]

    target_names = [
        k
        for k in q_keys
        if k in contract_f32_targets or ada_ln_re.search(k) or qk_norm_re.search(k)
    ]

    con_src = sqlite3.connect(str(src_path))
    cur_src = con_src.cursor()

    stats = {
        "candidates": len(target_names),
        "restored_f32": 0,
        "already_f32_rank3": 0,
        "skipped_missing": 0,
        "skipped_badtype": 0,
    }

    pending = 0
    for name in target_names:
        try:
            s_row = cur_src.execute(
                "SELECT datatype, dim, data FROM tensors WHERE name=?", (name,)
            ).fetchone()
        except sqlite3.DataError:
            stats["skipped_missing"] += 1
            continue
        if s_row is None:
            stats["skipped_missing"] += 1
            continue

        s_dt, s_dim, s_data = int(s_row[0]), bytes(s_row[1]), bytes(s_row[2])
        if s_dt == DT_FLOAT16:
            f32_bytes = np.frombuffer(s_data, dtype="<f2").astype("<f4").tobytes()
        elif s_dt == DT_FLOAT32:
            f32_bytes = np.frombuffer(s_data, dtype="<f4").tobytes()
        else:
            stats["skipped_badtype"] += 1
            continue

        ndim_slots = len(s_dim) // 4
        n_elem = len(f32_bytes) // 4
        target_dim = encode_dim_blob([n_elem], ndim_slots=ndim_slots, rank_policy="norm_1_1_n")

        cur_q.execute("SELECT datatype, dim FROM tensors WHERE name=?", (name,))
        q_row = cur_q.fetchone()
        if q_row and int(q_row[0]) == DT_FLOAT32 and bytes(q_row[1]) == target_dim:
            stats["already_f32_rank3"] += 1
            continue

        if not dry_run:
            cur_q.execute(
                "UPDATE tensors SET datatype=?, dim=?, data=? WHERE name=?",
                (DT_FLOAT32, sqlite3.Binary(target_dim), sqlite3.Binary(f32_bytes), name),
            )
            pending += 1
            if pending >= 256:
                con_q.commit()
                pending = 0
        stats["restored_f32"] += 1

    if not dry_run:
        con_q.commit()
    con_src.close()
    con_q.close()
    return stats


def cmd_convert(args: argparse.Namespace) -> int:
    stats = convert_source_to_ckpt(
        source_path=args.input,
        contract_path=args.contract,
        output_ckpt=args.output,
        strict=not args.allow_missing,
    )
    print("== Direct Checkpoint Conversion Summary ==")
    for k, v in stats.items():
        print(f"  {k}={v}")
    print(f"RESULT=PASS output={args.output}")
    return 0


def cmd_restore_norms(args: argparse.Namespace) -> int:
    stats = restore_norms_from_contract(
        quantized_ckpt=args.quantized_ckpt,
        f16_source_ckpt=args.f16_source,
        contract_path=args.contract,
        dry_run=args.dry_run,
    )
    print("== Contract Norm Restore Summary ==")
    for k, v in stats.items():
        print(f"  {k}={v}")
    print("RESULT=PASS" + (" (dry-run)" if args.dry_run else ""))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Direct Safetensors/GGUF -> Draw Things .ckpt Builder")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_conv = sub.add_parser("convert", help="Convert safetensors/GGUF to Draw Things .ckpt")
    p_conv.add_argument("--input", required=True, help="Source .safetensors (file/dir/index) or .gguf")
    p_conv.add_argument("--contract", required=True, help="Path to contracts/<profile>.json")
    p_conv.add_argument("--output", required=True, help="Output Draw Things .ckpt path")
    p_conv.add_argument("--allow-missing", action="store_true", help="Do not fail on missing source tensors")

    p_rest = sub.add_parser("restore-norms", help="Restore F32 norms and [1,1,N] dims using contract")
    p_rest.add_argument("--quantized-ckpt", required=True, help="Quantized .ckpt to update in-place")
    p_rest.add_argument("--f16-source", required=True, help="Source _f16.ckpt with model's own norm values")
    p_rest.add_argument("--contract", required=True, help="Path to contracts/<profile>.json")
    p_rest.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.subcommand == "convert":
        return cmd_convert(args)
    if args.subcommand == "restore-norms":
        return cmd_restore_norms(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
