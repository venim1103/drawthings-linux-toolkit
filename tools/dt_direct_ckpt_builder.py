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
from contextlib import ExitStack
import json
import math
import os
import re
import sqlite3
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import jsonschema

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.dt_weellm.seek_io import open_seeker
from tools.dt_weellm.sqlite_store import DrawThingsCkptStore
from tools.dt_weellm.transforms import (
    DATATYPE_NAME_TO_CODE,
    DT_BFLOAT16,
    DT_FLOAT16,
    DT_FLOAT32,
    DT_FORMAT_NHWC,
    DT_TYPE_CPU,
    apply_contract_rule_to_tensor,
    decode_dim_blob,
    encode_dim_blob,
    rotary_deinterleave,
    split_tensor_chunks,
)


def load_contract(contract_path: str | Path) -> Dict[str, Any]:
    path = Path(contract_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Contract file not found: {path}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("Contract must be a JSON object")
    if not contract.get("tensors") and contract.get("dynamic_compiler"):
        from tools.dt_keymap_compiler import build_ltx23_contract, build_moss_image_contract

        prof = contract.get("profile")
        if prof == "ltx2_3":
            compiled = build_ltx23_contract()
            contract["tensors"] = compiled["tensors"]
        elif prof == "moss_image":
            compiled = build_moss_image_contract()
        else:
            raise ValueError(f"Unknown built-in contract compiler: {prof}")
        contract = {**compiled, **contract, "tensors": compiled["tensors"]}
    schema_path = Path(__file__).resolve().parents[1] / "contracts" / "schema.json"
    try:
        jsonschema.validate(contract, json.loads(schema_path.read_text(encoding="utf-8")))
    except jsonschema.ValidationError as error:
        raise ValueError(f"Invalid contract at {list(error.absolute_path)}: {error.message}") from error
    targets = [name for rule in contract["tensors"].values() for name in rule["dt_names"]]
    if any(not name for name in targets) or len(targets) != len(set(targets)):
        raise ValueError("Contract target names must be nonempty and globally unique")
    if contract.get("expected_ckpt_tensor_count", len(targets)) != len(targets):
        raise ValueError("Contract expected tensor count disagrees with its target rules")
    for source, rule in contract["tensors"].items():
        if "target_shapes" in rule and len(rule["target_shapes"]) != len(rule["dt_names"]):
            raise ValueError(f"Target shape count mismatch: {source}")
        if "rotary_interleaved_slices" in rule and len(rule["rotary_interleaved_slices"]) != len(rule["dt_names"]):
            raise ValueError(f"Rotary slice count mismatch: {source}")
    return contract


def convert_source_to_ckpt(
    source_path: str | Path,
    contract_path: str | Path,
    output_ckpt: str | Path,
    strict: bool = True,
    overwrite: bool = False,
) -> Dict[str, int]:
    """Stream tensors from `source_path` (.safetensors or .gguf) into `output_ckpt` using `contract_path`."""
    contract = load_contract(contract_path)
    tensor_rules: Dict[str, Dict[str, Any]] = contract.get("tensors", {})
    if not tensor_rules:
        raise ValueError(f"Contract {contract_path} contains no tensor rules")

    out_path = Path(output_ckpt).expanduser().resolve()
    if out_path == Path(source_path).expanduser().resolve():
        raise ValueError("Input and output must be different files")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; explicit overwrite required: {out_path}")
    if Path(str(out_path) + "-tensordata").exists():
        raise ValueError("Refusing to replace a checkpoint with an existing sidecar")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staging_name = tempfile.mkstemp(prefix=f".{out_path.name}.tmp.", dir=out_path.parent)
    os.close(descriptor)
    staging_path = Path(staging_name)

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
            resolved_plan = []
            consumed_keys = set()
            for src_key, rule in tensor_rules.items():
                resolved_key: Optional[str] = None
                if hasattr(seeker, "resolve_key"):
                    resolved_key = seeker.resolve_key(
                        src_key, aliases=rule.get("aliases"), profile=contract["profile"])
                elif src_key in available_keys:
                    resolved_key = src_key

                if resolved_key is None:
                    stats["missing_in_source"] += 1
                    if strict:
                        raise KeyError(f"Required contract tensor missing from source: {src_key!r}")
                    continue

                if resolved_key in consumed_keys:
                    raise ValueError(f"Multiple rules consume the same source tensor: {resolved_key}")
                consumed_keys.add(resolved_key)
                if "source_shape" in rule and list(seeker.tensors[resolved_key].shape) != rule["source_shape"]:
                    raise ValueError(f"Source shape mismatch for {resolved_key}")
                for suffix in (".scale_weight", ".weight_scale"):
                    if resolved_key.endswith(".weight"):
                        auxiliary = resolved_key[:-7] + suffix
                        if auxiliary in available_keys:
                            consumed_keys.add(auxiliary)
                resolved_plan.append((resolved_key, rule))
            ignored_patterns = [re.compile(pattern) for pattern in contract.get("ignored_source_patterns", [])]
            unmapped = sorted(key for key in available_keys - consumed_keys
                             if not any(pattern.match(key) for pattern in ignored_patterns))
            if unmapped:
                raise ValueError(f"Unmapped source tensors: {unmapped[:10]}")
            if stats["missing_in_source"] or not resolved_plan:
                raise ValueError("Refusing to publish an incomplete or empty conversion")

            for resolved_key, rule in resolved_plan:
                arr_f32 = seeker.read_f32(resolved_key)
                if not np.all(np.isfinite(arr_f32)):
                    raise ValueError(f"Source tensor {resolved_key!r} contains NaN or Inf values")

                if rule.get("rotary_interleaved", False) or any(rule.get("rotary_interleaved_slices", [])):
                    stats["rotary_deinterleaved"] += 1
                if len(rule["dt_names"]) > 1:
                    stats["split_groups"] += 1

                named_chunks = apply_contract_rule_to_tensor(arr_f32, rule)
                dtype_str = str(rule.get("f16_store_datatype", "F16")).upper()
                dt_code = DATATYPE_NAME_TO_CODE[dtype_str]
                rank_policy = str(rule.get("dim_rank_policy", "preserve"))

                for target_name, chunk in named_chunks:
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
            expected_names = {name for rule in tensor_rules.values() for name in rule["dt_names"]}
            if set(store.list_keys()) != expected_names:
                raise ValueError("Converted checkpoint target coverage is incomplete")
            if store.con.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise ValueError("Staged checkpoint failed SQLite integrity checking")

        with staging_path.open("rb") as staged:
            os.fsync(staged.fileno())
        os.replace(staging_path, out_path)
        directory_descriptor = os.open(out_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
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
    q_path = Path(quantized_ckpt).expanduser().resolve()
    src_path = Path(f16_source_ckpt).expanduser().resolve()
    if not q_path.is_file() or not src_path.is_file():
        raise FileNotFoundError("Both quantized checkpoint and own-value source must exist")
    if os.path.samefile(q_path, src_path) or ref_ckpt is not None:
        raise ValueError("Restoration requires a separate own-value source, not a reference model")
    if any(Path(str(q_path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
        raise ValueError("Refusing to replace a checkpoint with an active SQLite journal")
    original_stat = q_path.stat()
    targets = [
        (name, rule, index)
        for rule in contract["tensors"].values()
        if rule.get("quant_restore_policy", "none") != "none"
        for index, name in enumerate(rule["dt_names"])
    ]
    if not targets:
        raise ValueError("Contract contains no restoration policies")
    stats = {"candidates": len(targets), "restored_f32": 0, "restored_bf16": 0,
             "already_f32_rank3": 0, "skipped_missing": 0, "skipped_badtype": 0}
    staging_path = None
    try:
        with ExitStack() as stack:
            source = stack.enter_context(DrawThingsCkptStore(src_path))
            quantized = stack.enter_context(DrawThingsCkptStore(q_path))
            source_keys, quantized_keys = set(source.list_keys()), set(quantized.list_keys())
            required = {name for name, _, _ in targets}
            if not required <= source_keys or not required <= quantized_keys:
                raise ValueError("Required restoration tensors are missing from source or destination")
            staged = None
            if not dry_run:
                descriptor, staging_name = tempfile.mkstemp(prefix=f".{q_path.name}.restore.", dir=q_path.parent)
                os.close(descriptor)
                staging_path = Path(staging_name)
                backup = sqlite3.connect(staging_path)
                try:
                    quantized.con.backup(backup)
                finally:
                    backup.close()
                staged = stack.enter_context(DrawThingsCkptStore(staging_path, writable=True))
            for name, rule, index in targets:
                source_result = source.read_f32(name)
                if source_result is None:
                    raise ValueError(f"Required source tensor is unreadable: {name}")
                source_meta, values = source_result
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite restoration source: {name}")
                datatype = DT_FLOAT32 if rule["quant_restore_policy"] == "f32_from_source" else DT_BFLOAT16
                rank_policy = rule.get("dim_rank_policy", "preserve")
                if "target_shapes" in rule:
                    values = values.reshape(rule["target_shapes"][index])
                desired_shape, _ = decode_dim_blob(encode_dim_blob(values.shape, rank_policy=rank_policy))
                current_meta = quantized.read_metadata(name)
                if current_meta is None:
                    raise ValueError(f"Unreadable destination tensor: {name}")
                if current_meta.datatype == datatype and current_meta.shape == desired_shape and current_meta.codec_identifier == 0:
                    current = quantized.read_f32(name)
                    if current is not None and np.array_equal(current[1].reshape(-1), values.reshape(-1)):
                        stats["already_f32_rank3"] += 1
                        continue
                if staged is not None:
                    staged.write_tensor(name, values, datatype=datatype,
                                        tensor_type=source_meta.tensor_type & 0xFFFFFFFF,
                                        tensor_format=source_meta.tensor_format, rank_policy=rank_policy)
                stats["restored_f32" if datatype == DT_FLOAT32 else "restored_bf16"] += 1
            if staged is not None:
                staged.commit()
                if staged.con.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                    raise ValueError("Restored checkpoint failed integrity checking")
        if staging_path is not None:
            current_stat = q_path.stat()
            if (current_stat.st_ino, current_stat.st_mtime_ns, current_stat.st_size) != (
                original_stat.st_ino, original_stat.st_mtime_ns, original_stat.st_size
            ):
                raise ValueError("Destination changed during restoration")
            with staging_path.open("rb") as staged_file:
                os.fsync(staged_file.fileno())
            os.replace(staging_path, q_path)
    finally:
        if staging_path is not None and staging_path.exists():
            staging_path.unlink()
    return stats


def cmd_convert(args: argparse.Namespace) -> int:
    stats = convert_source_to_ckpt(
        source_path=args.input,
        contract_path=args.contract,
        output_ckpt=args.output,
        strict=not args.allow_missing,
        overwrite=args.force,
    )
    print("== Direct Checkpoint Conversion Summary ==")
    for k, v in stats.items():
        print(f"  {k}={v}")
    print(f"RESULT=PASS output={args.output}")
    return 0


def diff_ckpt_stores(
    candidate_path: str | Path,
    reference_path: str | Path,
    contract_path: Optional[str | Path] = None,
    atol: float = 1e-2,
    report_limit: int = 15,
) -> Dict[str, Any]:
    """Compare two Draw Things .ckpt stores for key coverage, serialization metadata, and value parity."""
    if not math.isfinite(atol) or atol < 0 or report_limit < 1:
        raise ValueError("Tolerance must be finite/nonnegative and report limit positive")
    c_path = Path(candidate_path).expanduser().resolve()
    r_path = Path(reference_path).expanduser().resolve()
    contract = load_contract(contract_path) if contract_path is not None else None
    contract_targets = {
        name: (rule, index) for rule in contract["tensors"].values()
        for index, name in enumerate(rule["dt_names"])
    } if contract else {}

    with DrawThingsCkptStore(c_path, writable=False) as c_store, DrawThingsCkptStore(r_path, writable=False) as r_store:
        c_keys = set(c_store.list_keys())
        r_keys = set(r_store.list_keys())

        missing_in_candidate = sorted(r_keys - c_keys)
        extra_in_candidate = sorted(c_keys - r_keys)
        shared_keys = sorted(c_keys & r_keys)

        metadata_mismatches: List[str] = []
        value_mismatches: List[Tuple[str, float]] = []
        unreadable_tensors: List[str] = []
        value_comparison_count = 0
        exact_value_matches = 0
        max_diff = 0.0
        if contract and (c_keys != set(contract_targets) or r_keys != set(contract_targets)):
            metadata_mismatches.append("Candidate/reference keysets do not match the explicit contract")

        for key in shared_keys:
            c_meta = c_store.read_metadata(key)
            r_meta = r_store.read_metadata(key)
            if c_meta is None or r_meta is None:
                metadata_mismatches.append(f"{key}: unreadable metadata")
                continue

            if c_meta.datatype != r_meta.datatype:
                metadata_mismatches.append(
                    f"{key} datatype mismatch: cand={c_meta.datatype} ref={r_meta.datatype}"
                )
            if c_meta.shape != r_meta.shape:
                metadata_mismatches.append(
                    f"{key} shape mismatch: cand={c_meta.shape} ref={r_meta.shape}"
                )
            if c_meta.tensor_format != r_meta.tensor_format:
                metadata_mismatches.append(
                    f"{key} format mismatch: cand={c_meta.tensor_format} ref={r_meta.tensor_format}"
                )
            if key in contract_targets:
                rule, index = contract_targets[key]
                if "target_shapes" in rule and (c_meta.shape != rule["target_shapes"][index] or r_meta.shape != rule["target_shapes"][index]):
                    metadata_mismatches.append(f"{key}: explicit contract target shape mismatch")

            try:
                c_res = c_store.read_f32(key)
                r_res = r_store.read_f32(key)
                if c_res is None or r_res is None:
                    raise ValueError("tensor payload is missing")
                _, c_arr = c_res
                _, r_arr = r_res
                if c_arr.shape != r_arr.shape:
                    raise ValueError("tensor shapes differ")
                if not c_arr.size or not np.isfinite(c_arr).all() or not np.isfinite(r_arr).all():
                    raise ValueError("tensor is empty or contains non-finite values")
                diff = float(np.max(np.abs(c_arr.astype(np.float64) - r_arr.astype(np.float64))))
                value_comparison_count += 1
                max_diff = max(max_diff, diff)
                if diff > atol:
                    value_mismatches.append((key, diff))
                else:
                    exact_value_matches += 1
            except (ValueError, sqlite3.Error, OSError) as error:
                unreadable_tensors.append(f"{key}: {error}")

    return {
        "candidate": str(c_path),
        "reference": str(r_path),
        "candidate_tensor_count": len(c_keys),
        "reference_tensor_count": len(r_keys),
        "shared_tensor_count": len(shared_keys),
        "missing_in_candidate": missing_in_candidate,
        "extra_in_candidate": extra_in_candidate,
        "metadata_mismatch_count": len(metadata_mismatches),
        "metadata_mismatches": metadata_mismatches[:report_limit],
        "value_mismatch_count": len(value_mismatches),
        "value_mismatches": value_mismatches[:report_limit],
        "value_comparison_count": value_comparison_count,
        "unreadable_tensor_count": len(unreadable_tensors),
        "unreadable_tensors": unreadable_tensors[:report_limit],
        "exact_value_matches": exact_value_matches,
        "max_observed_diff": max_diff,
        "is_equivalent": (
            bool(c_keys) and bool(r_keys)
            and len(missing_in_candidate) == 0
            and len(extra_in_candidate) == 0
            and len(metadata_mismatches) == 0
            and len(value_mismatches) == 0
            and len(unreadable_tensors) == 0
            and value_comparison_count == len(shared_keys)
        ),
    }


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


def cmd_diff_ckpt(args: argparse.Namespace) -> int:
    report = diff_ckpt_stores(
        candidate_path=args.candidate,
        reference_path=args.reference,
        contract_path=args.contract or None,
        atol=args.atol,
        report_limit=args.report_limit,
    )
    print("== Checkpoint Diff Report ==")
    print(f"candidate={report['candidate']} ({report['candidate_tensor_count']} tensors)")
    print(f"reference={report['reference']} ({report['reference_tensor_count']} tensors)")
    print(f"shared_tensors={report['shared_tensor_count']}")
    print(f"missing_in_candidate={len(report['missing_in_candidate'])}")
    print(f"extra_in_candidate={len(report['extra_in_candidate'])}")
    print(f"metadata_mismatch_count={report['metadata_mismatch_count']}")
    print(f"value_mismatch_count={report['value_mismatch_count']}")
    print(f"value_comparison_count={report['value_comparison_count']}")
    print(f"unreadable_tensor_count={report['unreadable_tensor_count']}")
    for reason in report["unreadable_tensors"]:
        print(f"  unreadable: {reason}")
    print(f"exact_value_matches={report['exact_value_matches']}")
    print(f"max_observed_diff={report['max_observed_diff']:.6g}")
    if not report["is_equivalent"]:
        print("RESULT=FAIL")
        return 1
    print("RESULT=PASS")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Direct Safetensors/GGUF -> Draw Things .ckpt Builder")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_conv = sub.add_parser("convert", help="Convert safetensors/GGUF to Draw Things .ckpt")
    p_conv.add_argument("--input", required=True, help="Source .safetensors (file/dir/index) or .gguf")
    p_conv.add_argument("--contract", required=True, help="Path to contracts/<profile>.json")
    p_conv.add_argument("--output", required=True, help="Output Draw Things .ckpt path")
    p_conv.add_argument("--allow-missing", action="store_true", help="Do not fail on missing source tensors")
    p_conv.add_argument("--force", action="store_true", help="Explicitly replace an existing inline checkpoint")

    p_rest = sub.add_parser("restore-norms", help="Restore F32 norms and [1,1,N] dims using contract")
    p_rest.add_argument("--quantized-ckpt", required=True, help="Quantized .ckpt to update in-place")
    p_rest.add_argument("--f16-source", required=True, help="Source _f16.ckpt with model's own norm values")
    p_rest.add_argument("--contract", required=True, help="Path to contracts/<profile>.json")
    p_rest.add_argument("--dry-run", action="store_true")

    p_diff = sub.add_parser("diff-ckpt", help="Compare two Draw Things .ckpt stores for parity")
    p_diff.add_argument("--candidate", required=True, help="Candidate .ckpt to verify")
    p_diff.add_argument("--reference", required=True, help="Reference .ckpt template")
    p_diff.add_argument("--contract", default="", help="Optional contract path")
    p_diff.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for value parity")
    p_diff.add_argument("--report-limit", type=int, default=15, help="Max reported mismatch entries")

    args = parser.parse_args(argv)
    if args.subcommand == "convert":
        return cmd_convert(args)
    if args.subcommand == "restore-norms":
        return cmd_restore_norms(args)
    if args.subcommand == "diff-ckpt":
        return cmd_diff_ckpt(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
