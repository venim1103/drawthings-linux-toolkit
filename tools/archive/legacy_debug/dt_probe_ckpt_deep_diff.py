#!/usr/bin/env python3
"""Deep serialization diff for Draw Things .ckpt SQLite checkpoints.

Compares shared tensor rows by name across two checkpoints and reports:

- metadata parity (type/format/datatype)
- dim/data blob length parity
- dim/data head-tail signature parity
- optional small-blob SHA256 parity

Reads are done per-key and DataError-safe to handle problematic rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence


SERIALIZATION_FIELDS: Sequence[str] = ("type", "format", "datatype")


@dataclass
class TensorSignature:
    metadata: Dict[str, str]
    dim_len: int | None
    data_len: int | None
    dim_head_hex: str
    dim_tail_hex: str
    data_head_hex: str
    data_tail_hex: str
    dim_sha256: str | None
    data_sha256_small: str | None


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_cell(value: Any) -> str:
    return "NULL" if value is None else str(value)


def _prefix_for_name(name: str) -> str:
    if "[" in name:
        return name.split("[", 1)[0]
    return name


def _sha256_blob(value: bytes | None) -> str:
    if value is None:
        return "NULL"
    return hashlib.sha256(value).hexdigest()


def _load_tensor_keys(con: sqlite3.Connection) -> List[str]:
    cur = con.cursor()
    rows = cur.execute("SELECT name FROM tensors").fetchall()
    return [str(row[0]) for row in rows if row and row[0] is not None]


def _fetch_signature(
    con: sqlite3.Connection,
    key: str,
    signature_bytes: int,
    small_hash_limit: int,
    include_tail_signature: bool,
) -> tuple[TensorSignature | None, str | None]:
    cur = con.cursor()

    query = (
        "SELECT "
        "CAST(type AS TEXT), "
        "CAST(format AS TEXT), "
        "CAST(datatype AS TEXT), "
        "length(dim), "
        "length(data), "
        "hex(substr(dim, 1, ?)), "
        "hex(substr(data, 1, ?))"
    )
    params: list[Any] = [signature_bytes, signature_bytes]

    if include_tail_signature:
        query += ", hex(substr(dim, -?, ?)), hex(substr(data, -?, ?))"
        params.extend([signature_bytes, signature_bytes, signature_bytes, signature_bytes])
    else:
        query += ", 'SKIPPED', 'SKIPPED'"

    query += " FROM tensors WHERE name=?"
    params.append(key)

    try:
        row = cur.execute(query, params).fetchone()
    except sqlite3.DataError as exc:
        return None, f"DataError: {exc}"
    except sqlite3.DatabaseError as exc:
        return None, f"DatabaseError: {exc}"

    if row is None:
        return None, "RowMissing"

    (
        type_text,
        format_text,
        datatype_text,
        dim_len,
        data_len,
        dim_head_hex,
        data_head_hex,
        dim_tail_hex,
        data_tail_hex,
    ) = row

    dim_sha256: str | None = None
    data_sha256_small: str | None = None

    try:
        dim_row = cur.execute("SELECT dim FROM tensors WHERE name=?", (key,)).fetchone()
    except sqlite3.DataError as exc:
        return None, f"DataError: {exc}"
    except sqlite3.DatabaseError as exc:
        return None, f"DatabaseError: {exc}"

    if dim_row is None:
        return None, "RowMissing"

    dim_sha256 = _sha256_blob(dim_row[0])

    if data_len is not None and data_len <= small_hash_limit:
        try:
            data_row = cur.execute("SELECT data FROM tensors WHERE name=?", (key,)).fetchone()
        except sqlite3.DataError as exc:
            return None, f"DataError: {exc}"
        except sqlite3.DatabaseError as exc:
            return None, f"DatabaseError: {exc}"

        if data_row is None:
            return None, "RowMissing"
        data_sha256_small = _sha256_blob(data_row[0])

    sig = TensorSignature(
        metadata={
            "type": _normalize_cell(type_text),
            "format": _normalize_cell(format_text),
            "datatype": _normalize_cell(datatype_text),
        },
        dim_len=dim_len,
        data_len=data_len,
        dim_head_hex=_normalize_cell(dim_head_hex),
        dim_tail_hex=_normalize_cell(dim_tail_hex),
        data_head_hex=_normalize_cell(data_head_hex),
        data_tail_hex=_normalize_cell(data_tail_hex),
        dim_sha256=dim_sha256,
        data_sha256_small=data_sha256_small,
    )
    return sig, None


def _append_sample(
    samples: Dict[str, List[dict[str, Any]]],
    category: str,
    payload: dict[str, Any],
    sample_limit: int,
) -> None:
    bucket = samples.setdefault(category, [])
    if len(bucket) < sample_limit:
        bucket.append(payload)


def _md_block_for_samples(title: str, rows: List[dict[str, Any]]) -> List[str]:
    lines: List[str] = [f"### {title}", ""]
    if not rows:
        lines.append("- (none)")
        lines.append("")
        return lines
    for row in rows:
        lines.append(f"- {json.dumps(row, sort_keys=True)}")
    lines.append("")
    return lines


def _write_markdown(report: dict[str, Any], out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# Checkpoint Deep Diff")
    lines.append("")
    lines.append(f"Generated: {report['generated_at_utc']}")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- file: {report['inputs']['file']}")
    lines.append(f"- baseline: {report['inputs']['baseline']}")
    lines.append("")
    lines.append("## Settings")
    lines.append("")
    for key, value in report["settings"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")

    lines.append("## Stats")
    lines.append("")
    for key, value in report["stats"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")

    lines.append("## Top Prefixes")
    lines.append("")
    for row in report["top_prefixes"]:
        lines.append(f"- {row['prefix']}: total={row['total']} details={row['details']}")
    if not report["top_prefixes"]:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Sample Differences")
    lines.append("")
    examples = report.get("examples", {})
    for category in sorted(examples.keys()):
        lines.extend(_md_block_for_samples(category, examples[category]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run(
    file_path: Path,
    baseline_path: Path,
    signature_bytes: int,
    small_hash_limit: int,
    include_tail_signature: bool,
    sample_limit: int,
    top_prefix_count: int,
) -> dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(f"missing file: {file_path}")
    if not baseline_path.exists():
        raise FileNotFoundError(f"missing baseline: {baseline_path}")

    con_file = sqlite3.connect(str(file_path))
    con_baseline = sqlite3.connect(str(baseline_path))
    try:
        file_keys = _load_tensor_keys(con_file)
        baseline_keys = _load_tensor_keys(con_baseline)

        file_set = set(file_keys)
        baseline_set = set(baseline_keys)
        shared = sorted(file_set & baseline_set)

        stats: dict[str, int] = {
            "tensor_count_file": len(file_keys),
            "tensor_count_baseline": len(baseline_keys),
            "missing_in_file": len(baseline_set - file_set),
            "extra_in_file": len(file_set - baseline_set),
            "shared_tensors": len(shared),
            "unreadable_both": 0,
            "unreadable_only_file": 0,
            "unreadable_only_baseline": 0,
            "metadata_mismatch_type": 0,
            "metadata_mismatch_format": 0,
            "metadata_mismatch_datatype": 0,
            "dim_len_mismatch": 0,
            "data_len_mismatch": 0,
            "dim_head_mismatch": 0,
            "dim_tail_mismatch": 0,
            "data_head_mismatch": 0,
            "data_tail_mismatch": 0,
            "dim_sha256_mismatch": 0,
            "data_small_sha256_compared": 0,
            "data_small_sha256_mismatch": 0,
            "full_signature_match": 0,
        }

        examples: Dict[str, List[dict[str, Any]]] = {}
        prefix_counters: dict[str, Counter[str]] = defaultdict(Counter)

        for name in shared:
            file_sig, file_err = _fetch_signature(
                con_file,
                name,
                signature_bytes=signature_bytes,
                small_hash_limit=small_hash_limit,
                include_tail_signature=include_tail_signature,
            )
            baseline_sig, baseline_err = _fetch_signature(
                con_baseline,
                name,
                signature_bytes=signature_bytes,
                small_hash_limit=small_hash_limit,
                include_tail_signature=include_tail_signature,
            )

            prefix = _prefix_for_name(name)
            mismatch_hit = False

            if file_err and baseline_err:
                stats["unreadable_both"] += 1
                prefix_counters[prefix]["unreadable_both"] += 1
                _append_sample(
                    examples,
                    "unreadable_both",
                    {
                        "name": name,
                        "file_error": file_err,
                        "baseline_error": baseline_err,
                    },
                    sample_limit,
                )
                continue
            if file_err and not baseline_err:
                stats["unreadable_only_file"] += 1
                prefix_counters[prefix]["unreadable_only_file"] += 1
                _append_sample(
                    examples,
                    "unreadable_only_file",
                    {"name": name, "file_error": file_err},
                    sample_limit,
                )
                continue
            if baseline_err and not file_err:
                stats["unreadable_only_baseline"] += 1
                prefix_counters[prefix]["unreadable_only_baseline"] += 1
                _append_sample(
                    examples,
                    "unreadable_only_baseline",
                    {"name": name, "baseline_error": baseline_err},
                    sample_limit,
                )
                continue

            assert file_sig is not None
            assert baseline_sig is not None

            for field in SERIALIZATION_FIELDS:
                if file_sig.metadata[field] != baseline_sig.metadata[field]:
                    category = f"metadata_mismatch_{field}"
                    stats[category] += 1
                    prefix_counters[prefix][category] += 1
                    mismatch_hit = True
                    _append_sample(
                        examples,
                        category,
                        {
                            "name": name,
                            "file": file_sig.metadata[field],
                            "baseline": baseline_sig.metadata[field],
                        },
                        sample_limit,
                    )

            if file_sig.dim_len != baseline_sig.dim_len:
                stats["dim_len_mismatch"] += 1
                prefix_counters[prefix]["dim_len_mismatch"] += 1
                mismatch_hit = True
                _append_sample(
                    examples,
                    "dim_len_mismatch",
                    {
                        "name": name,
                        "file_len": file_sig.dim_len,
                        "baseline_len": baseline_sig.dim_len,
                    },
                    sample_limit,
                )

            if file_sig.data_len != baseline_sig.data_len:
                stats["data_len_mismatch"] += 1
                prefix_counters[prefix]["data_len_mismatch"] += 1
                mismatch_hit = True
                _append_sample(
                    examples,
                    "data_len_mismatch",
                    {
                        "name": name,
                        "file_len": file_sig.data_len,
                        "baseline_len": baseline_sig.data_len,
                    },
                    sample_limit,
                )

            if file_sig.dim_head_hex != baseline_sig.dim_head_hex:
                stats["dim_head_mismatch"] += 1
                prefix_counters[prefix]["dim_head_mismatch"] += 1
                mismatch_hit = True
                _append_sample(
                    examples,
                    "dim_head_mismatch",
                    {
                        "name": name,
                        "file": file_sig.dim_head_hex,
                        "baseline": baseline_sig.dim_head_hex,
                    },
                    sample_limit,
                )

            if file_sig.dim_tail_hex != baseline_sig.dim_tail_hex:
                stats["dim_tail_mismatch"] += 1
                prefix_counters[prefix]["dim_tail_mismatch"] += 1
                mismatch_hit = True
                _append_sample(
                    examples,
                    "dim_tail_mismatch",
                    {
                        "name": name,
                        "file": file_sig.dim_tail_hex,
                        "baseline": baseline_sig.dim_tail_hex,
                    },
                    sample_limit,
                )

            if file_sig.data_head_hex != baseline_sig.data_head_hex:
                stats["data_head_mismatch"] += 1
                prefix_counters[prefix]["data_head_mismatch"] += 1
                mismatch_hit = True
                _append_sample(
                    examples,
                    "data_head_mismatch",
                    {
                        "name": name,
                        "file": file_sig.data_head_hex,
                        "baseline": baseline_sig.data_head_hex,
                    },
                    sample_limit,
                )

            if file_sig.data_tail_hex != baseline_sig.data_tail_hex:
                stats["data_tail_mismatch"] += 1
                prefix_counters[prefix]["data_tail_mismatch"] += 1
                mismatch_hit = True
                _append_sample(
                    examples,
                    "data_tail_mismatch",
                    {
                        "name": name,
                        "file": file_sig.data_tail_hex,
                        "baseline": baseline_sig.data_tail_hex,
                    },
                    sample_limit,
                )

            if file_sig.dim_sha256 != baseline_sig.dim_sha256:
                stats["dim_sha256_mismatch"] += 1
                prefix_counters[prefix]["dim_sha256_mismatch"] += 1
                mismatch_hit = True
                _append_sample(
                    examples,
                    "dim_sha256_mismatch",
                    {
                        "name": name,
                        "file": file_sig.dim_sha256,
                        "baseline": baseline_sig.dim_sha256,
                    },
                    sample_limit,
                )

            if file_sig.data_sha256_small is not None and baseline_sig.data_sha256_small is not None:
                stats["data_small_sha256_compared"] += 1
                if file_sig.data_sha256_small != baseline_sig.data_sha256_small:
                    stats["data_small_sha256_mismatch"] += 1
                    prefix_counters[prefix]["data_small_sha256_mismatch"] += 1
                    mismatch_hit = True
                    _append_sample(
                        examples,
                        "data_small_sha256_mismatch",
                        {
                            "name": name,
                            "file": file_sig.data_sha256_small,
                            "baseline": baseline_sig.data_sha256_small,
                        },
                        sample_limit,
                    )

            if not mismatch_hit:
                stats["full_signature_match"] += 1

        ranked_prefixes = []
        for prefix, counter in prefix_counters.items():
            total = sum(counter.values())
            if total <= 0:
                continue
            ranked_prefixes.append(
                {
                    "prefix": prefix,
                    "total": total,
                    "details": dict(counter),
                }
            )
        ranked_prefixes.sort(key=lambda row: row["total"], reverse=True)

        report = {
            "generated_at_utc": _now_utc_iso(),
            "inputs": {
                "file": str(file_path),
                "baseline": str(baseline_path),
            },
            "settings": {
                "signature_bytes": signature_bytes,
                "small_hash_limit": small_hash_limit,
                "include_tail_signature": include_tail_signature,
                "sample_limit": sample_limit,
                "top_prefix_count": top_prefix_count,
            },
            "stats": stats,
            "top_prefixes": ranked_prefixes[:top_prefix_count],
            "examples": examples,
        }
        return report
    finally:
        con_file.close()
        con_baseline.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Deep diff two Draw Things .ckpt tensors tables")
    parser.add_argument("--file", required=True, help="Candidate checkpoint")
    parser.add_argument("--baseline", required=True, help="Baseline checkpoint")
    parser.add_argument(
        "--signature-bytes",
        type=int,
        default=64,
        help="Byte count for head/tail blob signature comparison (default: 64)",
    )
    parser.add_argument(
        "--small-hash-limit",
        type=int,
        default=262144,
        help="Compute full SHA256 for data blobs up to this size (bytes); default: 262144",
    )
    parser.add_argument(
        "--include-tail-signature",
        action="store_true",
        help="Include tail-byte signatures for dim/data blobs (slower, disabled by default)",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=12,
        help="Max samples recorded per mismatch category (default: 12)",
    )
    parser.add_argument(
        "--top-prefix-count",
        type=int,
        default=8,
        help="Number of top mismatch prefixes to report (default: 8)",
    )
    parser.add_argument("--out-json", default="", help="Optional JSON output path")
    parser.add_argument("--out-md", default="", help="Optional markdown output path")

    args = parser.parse_args()

    if args.signature_bytes < 1:
        parser.error("--signature-bytes must be >= 1")
    if args.small_hash_limit < 0:
        parser.error("--small-hash-limit must be >= 0")
    if args.sample_limit < 1:
        parser.error("--sample-limit must be >= 1")
    if args.top_prefix_count < 1:
        parser.error("--top-prefix-count must be >= 1")

    file_path = Path(args.file).expanduser().resolve()
    baseline_path = Path(args.baseline).expanduser().resolve()

    try:
        report = run(
            file_path=file_path,
            baseline_path=baseline_path,
            signature_bytes=args.signature_bytes,
            small_hash_limit=args.small_hash_limit,
            include_tail_signature=args.include_tail_signature,
            sample_limit=args.sample_limit,
            top_prefix_count=args.top_prefix_count,
        )
    except Exception as exc:
        print(f"RESULT=FAIL {exc}")
        return 2

    print("== Deep Checkpoint Diff ==")
    print(f"file={report['inputs']['file']}")
    print(f"baseline={report['inputs']['baseline']}")
    for key, value in report["stats"].items():
        print(f"{key}={value}")

    if args.out_json:
        out_json = Path(args.out_json).expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"json_report={out_json}")

    if args.out_md:
        out_md = Path(args.out_md).expanduser().resolve()
        _write_markdown(report, out_md)
        print(f"markdown_report={out_md}")

    print("RESULT=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
