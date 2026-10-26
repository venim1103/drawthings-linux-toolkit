#!/usr/bin/env python3
"""Validate converted Draw Things checkpoint structure and serialization policy.

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
from typing import Dict, Iterable, List, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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

MOSS_IMAGE_REQUIRED_PREFIX_MIN_COUNTS: Dict[str, int] = {
    "__dit__[": 1,
}

KNOWN_INVALID_PLACEHOLDER_KEYS = {
    "__dit__[]",
    "__text_audio_connector__[]",
    "__text_video_connector__[]",
}

SERIALIZATION_FIELDS: Sequence[str] = ("type", "format", "datatype")


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


def load_tensor_metadata(
    ckpt_path: Path,
    keys: Sequence[str],
) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    try:
        con = sqlite3.connect(str(ckpt_path))
        cur = con.cursor()
    except sqlite3.Error as exc:
        raise RuntimeError(f"failed to read tensor metadata from {ckpt_path}: {exc}") from exc

    metadata: Dict[str, Dict[str, str]] = {}
    unreadable_keys: List[str] = []

    for key in keys:
        try:
            row = cur.execute(
                "SELECT CAST(type AS TEXT), CAST(format AS TEXT), CAST(datatype AS TEXT) FROM tensors WHERE name=?",
                (key,),
            ).fetchone()
        except sqlite3.DataError:
            unreadable_keys.append(key)
            continue
        except sqlite3.Error as exc:
            con.close()
            raise RuntimeError(
                f"failed to read tensor metadata for key {key!r} from {ckpt_path}: {exc}"
            ) from exc

        if row is None:
            continue

        metadata[key] = {
            "type": "NULL" if row[0] is None else str(row[0]),
            "format": "NULL" if row[1] is None else str(row[1]),
            "datatype": "NULL" if row[2] is None else str(row[2]),
        }

    con.close()
    return metadata, unreadable_keys


def _summarize_field_mismatches(
    converted_meta: Dict[str, Dict[str, str]],
    baseline_meta: Dict[str, Dict[str, str]],
    names: Sequence[str],
    field: str,
    report_limit: int,
) -> List[str]:
    examples: List[str] = []
    for name in names[:report_limit]:
        expected = baseline_meta[name][field]
        got = converted_meta[name][field]
        examples.append(f"{name} expected={expected} got={got}")
    return examples


def validate_serialization_policy(
    ckpt_path: Path,
    baseline_path: Path,
    report_limit: int,
) -> List[str]:
    failures: List[str] = []

    try:
        converted_keys = load_tensor_keys(ckpt_path)
        baseline_keys = load_tensor_keys(baseline_path)
    except Exception as exc:
        return [str(exc)]

    converted_names = set(converted_keys)
    baseline_names = set(baseline_keys)
    missing_from_converted = sorted(baseline_names - converted_names)
    extra_in_converted = sorted(converted_names - baseline_names)

    shared_names = sorted(converted_names & baseline_names)

    try:
        converted_meta, converted_unreadable = load_tensor_metadata(ckpt_path, shared_names)
        baseline_meta, baseline_unreadable = load_tensor_metadata(baseline_path, shared_names)
    except Exception as exc:
        return [str(exc)]

    converted_unreadable_set = set(converted_unreadable)
    baseline_unreadable_set = set(baseline_unreadable)
    unreadable_both = sorted(converted_unreadable_set & baseline_unreadable_set)
    unreadable_only_converted = sorted(converted_unreadable_set - baseline_unreadable_set)
    unreadable_only_baseline = sorted(baseline_unreadable_set - converted_unreadable_set)

    print("== Serialization Policy Guard ==")
    print(f"baseline={baseline_path}")
    print(f"converted_tensors={len(converted_keys)} baseline_tensors={len(baseline_keys)}")

    if missing_from_converted:
        failures.append(
            f"serialization baseline keys missing in converted checkpoint: {len(missing_from_converted)}"
        )
        print(f"missing_from_converted={len(missing_from_converted)}")
        for name in missing_from_converted[:report_limit]:
            print(f"  missing: {name}")
    else:
        print("missing_from_converted=0")

    if extra_in_converted:
        failures.append(
            f"serialization baseline has no match for converted keys: {len(extra_in_converted)}"
        )
        print(f"extra_in_converted={len(extra_in_converted)}")
        for name in extra_in_converted[:report_limit]:
            print(f"  extra: {name}")
    else:
        print("extra_in_converted=0")

    if unreadable_both:
        print(f"unreadable_metadata_both={len(unreadable_both)}")
        for name in unreadable_both[:report_limit]:
            print(f"  unreadable_both: {name}")
    else:
        print("unreadable_metadata_both=0")

    if unreadable_only_converted:
        failures.append(
            "metadata unreadable only in converted checkpoint: "
            f"{len(unreadable_only_converted)}"
        )
        print(f"unreadable_metadata_only_converted={len(unreadable_only_converted)}")
        for name in unreadable_only_converted[:report_limit]:
            print(f"  unreadable_only_converted: {name}")
    else:
        print("unreadable_metadata_only_converted=0")

    if unreadable_only_baseline:
        failures.append(
            "metadata unreadable only in serialization baseline: "
            f"{len(unreadable_only_baseline)}"
        )
        print(f"unreadable_metadata_only_baseline={len(unreadable_only_baseline)}")
        for name in unreadable_only_baseline[:report_limit]:
            print(f"  unreadable_only_baseline: {name}")
    else:
        print("unreadable_metadata_only_baseline=0")

    readable_shared_names = [
        name
        for name in shared_names
        if name not in converted_unreadable_set and name not in baseline_unreadable_set
    ]
    print(f"readable_shared_tensors={len(readable_shared_names)}")

    field_mismatch_names: Dict[str, List[str]] = {field: [] for field in SERIALIZATION_FIELDS}

    for name in readable_shared_names:
        converted_row = converted_meta[name]
        baseline_row = baseline_meta[name]
        for field in SERIALIZATION_FIELDS:
            if converted_row[field] != baseline_row[field]:
                field_mismatch_names[field].append(name)

    for field in SERIALIZATION_FIELDS:
        mismatches = field_mismatch_names[field]
        print(f"{field}_mismatch_count={len(mismatches)}")
        if mismatches:
            failures.append(f"serialization {field} mismatches vs baseline: {len(mismatches)}")
            for example in _summarize_field_mismatches(
                converted_meta,
                baseline_meta,
                mismatches,
                field,
                report_limit,
            ):
                print(f"  {field}_mismatch: {example}")

    return failures


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
    serialization_baseline: Path | None,
    serialization_guard_profile: str,
    serialization_report_limit: int,
    contract_path: Path | None = None,
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

    if contract_path is not None:
        from tools.dt_direct_ckpt_builder import load_contract
        from tools.dt_weellm.sqlite_store import DrawThingsCkptStore
        from tools.dt_weellm.transforms import DATATYPE_NAME_TO_CODE

        try:
            contract_obj = load_contract(contract_path)
            target_rules = {
                name: (rule, index) for rule in contract_obj["tensors"].values()
                for index, name in enumerate(rule["dt_names"])
            }
            if key_set != set(target_rules):
                failures.append("checkpoint keyset differs from complete contract targets")
            with DrawThingsCkptStore(ckpt_path) as store:
                for name in key_set & set(target_rules):
                    rule, index = target_rules[name]
                    metadata = store.read_metadata(name)
                    if metadata is None:
                        failures.append(f"unreadable contract metadata: {name}")
                        continue
                    allowed_types = {DATATYPE_NAME_TO_CODE[rule.get("f16_store_datatype", "F16")]}
                    policy = rule.get("quant_restore_policy")
                    if policy in ("f32_from_source", "bf16_from_source"):
                        allowed_types.add(DATATYPE_NAME_TO_CODE["F32" if policy == "f32_from_source" else "BF16"])
                    if metadata.datatype not in allowed_types:
                        failures.append(f"contract datatype mismatch: {name}")
                    if "target_shapes" in rule and metadata.shape != rule["target_shapes"][index]:
                        failures.append(f"contract shape mismatch: {name}")
                    if rule.get("dim_rank_policy") == "norm_1_1_n" and (
                        len(metadata.shape) != 3 or metadata.shape[:2] != [1, 1]
                    ):
                        failures.append(f"contract norm rank mismatch: {name}")
        except (ValueError, OSError, sqlite3.Error) as error:
            print(f"RESULT=FAIL invalid contract/checkpoint: {error}")
            return 1
        req_prefixes = contract_obj.get("ckpt_required_prefixes", {})
        print(f"== Contract Required Prefix Checks ({contract_path.name}) ==")
        for prefix, min_count in req_prefixes.items():
            found = count_prefix(keys, prefix)
            print(f"{prefix} count={found} required>={min_count}")
            if found < int(min_count):
                failures.append(
                    f"missing required contract tensor prefix: {prefix} (found {found}, expected >= {min_count})"
                )
        exp_total = contract_obj.get("expected_ckpt_tensor_count")
        if isinstance(exp_total, int) and exp_total > 0 and len(keys) != exp_total:
            failures.append(
                f"checkpoint tensor count mismatch vs contract: got {len(keys)}, expected {exp_total}"
            )
    elif resolved_profile == "ltx2_3":
        print("== LTX2.3 Required Prefix Checks ==")
        for prefix, min_count in LTX23_REQUIRED_PREFIX_MIN_COUNTS.items():
            found = count_prefix(keys, prefix)
            print(f"{prefix} count={found} required>={min_count}")
            if found < min_count:
                failures.append(
                    f"missing required LTX2.3 tensor prefix: {prefix} (found {found}, expected >= {min_count})"
                )
    elif resolved_profile == "moss_image":
        print("== MOSS-Image Required Prefix Checks ==")
        for prefix, min_count in MOSS_IMAGE_REQUIRED_PREFIX_MIN_COUNTS.items():
            found = count_prefix(keys, prefix)
            print(f"{prefix} count={found} required>={min_count}")
            if found < min_count:
                failures.append(
                    f"missing required MOSS-Image tensor prefix: {prefix} (found {found}, expected >= {min_count})"
                )

    if serialization_baseline is not None:
        apply_guard = True
        if serialization_guard_profile != "any" and resolved_profile != serialization_guard_profile:
            apply_guard = False

        if not apply_guard:
            print(
                "== Serialization Policy Guard ==\n"
                f"SKIPPED profile mismatch (required={serialization_guard_profile}, actual={resolved_profile})"
            )
        elif not serialization_baseline.exists():
            failures.append(
                f"serialization baseline file does not exist: {serialization_baseline}"
            )
        else:
            failures.extend(
                validate_serialization_policy(
                    ckpt_path,
                    serialization_baseline,
                    serialization_report_limit,
                )
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
        choices=["auto", "none", "ltx2_3", "moss_image"],
        default="auto",
        help="Validation profile (default: auto)",
    )
    parser.add_argument(
        "--source-safetensors",
        default="",
        help="Optional source safetensors path used for profile auto-detection",
    )
    parser.add_argument(
        "--serialization-baseline",
        default="",
        help="Optional baseline .ckpt used for strict type/format/datatype parity checks",
    )
    parser.add_argument(
        "--serialization-guard-profile",
        choices=["any", "none", "ltx2_3"],
        default="any",
        help="Only apply serialization parity guard when resolved profile matches (default: any)",
    )
    parser.add_argument(
        "--serialization-report-limit",
        type=int,
        default=10,
        help="Maximum mismatch examples printed per category (default: 10)",
    )
    parser.add_argument(
        "--contract",
        default="",
        help="Optional path to contracts/<profile>.json for contract-driven prefix/count checks",
    )
    args = parser.parse_args()

    if args.serialization_report_limit < 1:
        parser.error("--serialization-report-limit must be >= 1")

    source = Path(args.source_safetensors).expanduser().resolve() if args.source_safetensors else None
    ckpt = Path(args.file).expanduser().resolve()
    baseline = (
        Path(args.serialization_baseline).expanduser().resolve()
        if args.serialization_baseline
        else None
    )
    contract_file = Path(args.contract).expanduser().resolve() if args.contract else None

    return validate_ckpt(
        ckpt,
        args.profile,
        source,
        baseline,
        args.serialization_guard_profile,
        args.serialization_report_limit,
        contract_path=contract_file,
    )


if __name__ == "__main__":
    sys.exit(main())
