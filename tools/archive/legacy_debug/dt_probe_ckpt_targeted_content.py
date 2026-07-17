#!/usr/bin/env python3
"""Targeted content parity probe for Draw Things checkpoint tensors.

Compares shared tensor rows across two .ckpt SQLite files with DataError-safe,
row-wise reads. Designed for focused mismatch mapping by family/prefix without
full heavy signature sweeps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence


SERIALIZATION_FIELDS: Sequence[str] = ("type", "format", "datatype")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_cell(value: Any) -> str:
    return "NULL" if value is None else str(value)


def _prefix_for_name(name: str) -> str:
    if "[" in name:
        return name.split("[", 1)[0]
    return name


def _dit_subgroup_for_name(name: str) -> str:
    if not (name.startswith("__dit__[") and name.endswith("]")):
        return _prefix_for_name(name)
    inner = name[len("__dit__[") : -1]
    parts = inner.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:-2])
    return inner


def _family_key(name: str, family_mode: str) -> str:
    if family_mode == "dit_subgroup":
        return _dit_subgroup_for_name(name)
    return _prefix_for_name(name)


def _sha256_blob(value: bytes | None) -> str:
    if value is None:
        return "NULL"
    return hashlib.sha256(value).hexdigest()


def _blob_to_hex(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytes):
        return value.hex().upper()
    return _normalize_cell(value)


def _load_tensor_keys(con: sqlite3.Connection) -> List[str]:
    cur = con.cursor()
    rows = cur.execute("SELECT name FROM tensors").fetchall()
    return [str(row[0]) for row in rows if row and row[0] is not None]


def _parse_names_file(path: Path) -> List[str]:
    names: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        names.append(line)

    deduped: List[str] = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def _parse_subgroups_file(path: Path) -> List[str]:
    subgroups: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^\d+\.\s+([^|]+?)\s+\|", line)
        if m:
            subgroups.append(m.group(1).strip())
            continue
        if "|" not in line and " " not in line:
            subgroups.append(line)

    deduped: List[str] = []
    seen = set()
    for subgroup in subgroups:
        if subgroup in seen:
            continue
        seen.add(subgroup)
        deduped.append(subgroup)
    return deduped


def _name_matches_dit_subgroups(name: str, subgroups: Sequence[str]) -> bool:
    if not (name.startswith("__dit__[") and name.endswith("]")):
        return False
    inner = name[len("__dit__[") : -1]
    for subgroup in subgroups:
        if inner.startswith(subgroup + "-"):
            return True
    return False


def _fetch_signature(
    con: sqlite3.Connection,
    key: str,
    head_bytes: int,
    small_hash_limit: int,
) -> tuple[dict[str, Any] | None, str | None]:
    cur = con.cursor()
    query = (
        "SELECT "
        "CAST(type AS TEXT), "
        "CAST(format AS TEXT), "
        "CAST(datatype AS TEXT), "
        "length(dim), "
        "length(data), "
        "substr(dim, 1, ?), "
        "substr(data, 1, ?) "
        "FROM tensors WHERE name=?"
    )

    try:
        row = cur.execute(query, (head_bytes, head_bytes, key)).fetchone()
    except sqlite3.DataError as exc:
        return None, f"DataError: {exc}"
    except sqlite3.DatabaseError as exc:
        return None, f"DatabaseError: {exc}"

    if row is None:
        return None, "RowMissing"

    data_small_sha256: str | None = None
    data_len = row[4]
    if data_len is not None and data_len <= small_hash_limit:
        try:
            data_row = cur.execute("SELECT data FROM tensors WHERE name=?", (key,)).fetchone()
        except sqlite3.DataError as exc:
            return None, f"DataError: {exc}"
        except sqlite3.DatabaseError as exc:
            return None, f"DatabaseError: {exc}"

        if data_row is None:
            return None, "RowMissing"
        data_value = data_row[0]
        if isinstance(data_value, memoryview):
            data_value = data_value.tobytes()
        data_small_sha256 = _sha256_blob(data_value)

    sig = {
        "metadata": {
            "type": _normalize_cell(row[0]),
            "format": _normalize_cell(row[1]),
            "datatype": _normalize_cell(row[2]),
        },
        "dim_len": row[3],
        "data_len": row[4],
        "dim_head_hex": _blob_to_hex(row[5]),
        "data_head_hex": _blob_to_hex(row[6]),
        "data_small_sha256": data_small_sha256,
    }
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


def _write_markdown(report: dict[str, Any], out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# Targeted Content Probe")
    lines.append("")
    lines.append(f"Generated: {report['generated_at_utc']}")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- file: {report['inputs']['file']}")
    lines.append(f"- baseline: {report['inputs']['baseline']}")
    lines.append(f"- names_file: {report['inputs']['names_file']}")
    lines.append(f"- subgroup_file: {report['inputs']['subgroup_file']}")
    lines.append(f"- subgroup_start: {report['inputs']['subgroup_start']}")
    lines.append(f"- max_subgroups: {report['inputs']['max_subgroups']}")
    lines.append(f"- subgroups_selected_count: {report['inputs']['subgroups_selected_count']}")
    lines.append(f"- prefixes: {report['inputs']['prefixes']}")
    lines.append(f"- exclude_names_count: {report['inputs']['exclude_names_count']}")
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

    lines.append("## Family Map")
    lines.append("")
    if report["family_map"]:
        for row in report["family_map"]:
            lines.append(
                "- "
                f"{row['prefix']}: selected={row['selected']} "
                f"mismatch_any={row['mismatch_any']} "
                f"details={row['details']}"
            )
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Samples")
    lines.append("")
    examples = report.get("examples", {})
    for category in sorted(examples.keys()):
        lines.append(f"### {category}")
        lines.append("")
        if not examples[category]:
            lines.append("- (none)")
            lines.append("")
            continue
        for row in examples[category]:
            lines.append(f"- {json.dumps(row, sort_keys=True)}")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run(
    file_path: Path,
    baseline_path: Path,
    names_file: Path | None,
    subgroup_file: Path | None,
    subgroup_start: int,
    max_subgroups: int,
    prefixes: Sequence[str],
    exclude_names: Sequence[str],
    head_bytes: int,
    small_hash_limit: int,
    sample_limit: int,
    top_prefix_count: int,
    progress_every: int,
    family_mode: str,
) -> dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(f"missing file: {file_path}")
    if not baseline_path.exists():
        raise FileNotFoundError(f"missing baseline: {baseline_path}")
    if names_file is not None and not names_file.exists():
        raise FileNotFoundError(f"missing names file: {names_file}")
    if subgroup_file is not None and not subgroup_file.exists():
        raise FileNotFoundError(f"missing subgroup file: {subgroup_file}")

    names_filter: set[str] | None = None
    if names_file is not None:
        names_filter = set(_parse_names_file(names_file))

    subgroup_filters: List[str] = []
    if subgroup_file is not None:
        parsed = _parse_subgroups_file(subgroup_file)
        if not parsed:
            raise ValueError("no subgroups parsed from subgroup file")
        start_pos = subgroup_start - 1
        if start_pos < len(parsed):
            if max_subgroups > 0:
                subgroup_filters = parsed[start_pos : start_pos + max_subgroups]
            else:
                subgroup_filters = parsed[start_pos:]

    prefix_filters = tuple(prefixes)
    excluded = set(exclude_names)

    con_file = sqlite3.connect(str(file_path))
    con_baseline = sqlite3.connect(str(baseline_path))
    try:
        file_keys = _load_tensor_keys(con_file)
        baseline_keys = _load_tensor_keys(con_baseline)
        shared = sorted(set(file_keys) & set(baseline_keys))

        selected: List[str] = []
        for name in shared:
            if names_filter is not None and name not in names_filter:
                continue
            if subgroup_filters and not _name_matches_dit_subgroups(name, subgroup_filters):
                continue
            if prefix_filters and not any(name.startswith(prefix) for prefix in prefix_filters):
                continue
            if name in excluded:
                continue
            selected.append(name)

        stats: dict[str, int] = {
            "tensor_count_file": len(file_keys),
            "tensor_count_baseline": len(baseline_keys),
            "shared_tensors": len(shared),
            "selected_tensors": len(selected),
            "unreadable_both": 0,
            "unreadable_only_file": 0,
            "unreadable_only_baseline": 0,
            "readable_selected": 0,
            "metadata_mismatch_type": 0,
            "metadata_mismatch_format": 0,
            "metadata_mismatch_datatype": 0,
            "dim_len_mismatch": 0,
            "data_len_mismatch": 0,
            "dim_head_mismatch": 0,
            "data_head_mismatch": 0,
            "data_small_sha256_compared": 0,
            "data_small_sha256_mismatch": 0,
            "mismatch_any": 0,
            "full_match": 0,
        }

        examples: Dict[str, List[dict[str, Any]]] = {}
        family_selected: Counter[str] = Counter()
        family_mismatch_any: Counter[str] = Counter()
        family_details: dict[str, Counter[str]] = defaultdict(Counter)

        for idx, name in enumerate(selected, start=1):
            if progress_every > 0 and idx % progress_every == 0:
                print(f"progress={idx}/{len(selected)}")

            file_sig, file_err = _fetch_signature(
                con_file,
                name,
                head_bytes=head_bytes,
                small_hash_limit=small_hash_limit,
            )
            baseline_sig, baseline_err = _fetch_signature(
                con_baseline,
                name,
                head_bytes=head_bytes,
                small_hash_limit=small_hash_limit,
            )

            family = _family_key(name, family_mode)
            family_selected[family] += 1
            mismatch_hit = False

            if file_err and baseline_err:
                stats["unreadable_both"] += 1
                family_details[family]["unreadable_both"] += 1
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
                family_details[family]["unreadable_only_file"] += 1
                _append_sample(
                    examples,
                    "unreadable_only_file",
                    {"name": name, "file_error": file_err},
                    sample_limit,
                )
                continue

            if baseline_err and not file_err:
                stats["unreadable_only_baseline"] += 1
                family_details[family]["unreadable_only_baseline"] += 1
                _append_sample(
                    examples,
                    "unreadable_only_baseline",
                    {"name": name, "baseline_error": baseline_err},
                    sample_limit,
                )
                continue

            assert file_sig is not None
            assert baseline_sig is not None
            stats["readable_selected"] += 1

            for field in SERIALIZATION_FIELDS:
                if file_sig["metadata"][field] != baseline_sig["metadata"][field]:
                    category = f"metadata_mismatch_{field}"
                    stats[category] += 1
                    family_details[family][category] += 1
                    mismatch_hit = True
                    _append_sample(
                        examples,
                        category,
                        {
                            "name": name,
                            "file": file_sig["metadata"][field],
                            "baseline": baseline_sig["metadata"][field],
                        },
                        sample_limit,
                    )

            if file_sig["dim_len"] != baseline_sig["dim_len"]:
                stats["dim_len_mismatch"] += 1
                family_details[family]["dim_len_mismatch"] += 1
                mismatch_hit = True
                _append_sample(
                    examples,
                    "dim_len_mismatch",
                    {
                        "name": name,
                        "file_len": file_sig["dim_len"],
                        "baseline_len": baseline_sig["dim_len"],
                    },
                    sample_limit,
                )

            if file_sig["data_len"] != baseline_sig["data_len"]:
                stats["data_len_mismatch"] += 1
                family_details[family]["data_len_mismatch"] += 1
                mismatch_hit = True
                _append_sample(
                    examples,
                    "data_len_mismatch",
                    {
                        "name": name,
                        "file_len": file_sig["data_len"],
                        "baseline_len": baseline_sig["data_len"],
                    },
                    sample_limit,
                )

            if file_sig["dim_head_hex"] != baseline_sig["dim_head_hex"]:
                stats["dim_head_mismatch"] += 1
                family_details[family]["dim_head_mismatch"] += 1
                mismatch_hit = True
                _append_sample(
                    examples,
                    "dim_head_mismatch",
                    {
                        "name": name,
                        "file": file_sig["dim_head_hex"],
                        "baseline": baseline_sig["dim_head_hex"],
                    },
                    sample_limit,
                )

            if file_sig["data_head_hex"] != baseline_sig["data_head_hex"]:
                stats["data_head_mismatch"] += 1
                family_details[family]["data_head_mismatch"] += 1
                mismatch_hit = True
                _append_sample(
                    examples,
                    "data_head_mismatch",
                    {
                        "name": name,
                        "file": file_sig["data_head_hex"],
                        "baseline": baseline_sig["data_head_hex"],
                    },
                    sample_limit,
                )

            file_small = file_sig["data_small_sha256"]
            baseline_small = baseline_sig["data_small_sha256"]
            if file_small is not None and baseline_small is not None:
                stats["data_small_sha256_compared"] += 1
                if file_small != baseline_small:
                    stats["data_small_sha256_mismatch"] += 1
                    family_details[family]["data_small_sha256_mismatch"] += 1
                    mismatch_hit = True
                    _append_sample(
                        examples,
                        "data_small_sha256_mismatch",
                        {
                            "name": name,
                            "file": file_small,
                            "baseline": baseline_small,
                        },
                        sample_limit,
                    )

            if mismatch_hit:
                stats["mismatch_any"] += 1
                family_mismatch_any[family] += 1
            else:
                stats["full_match"] += 1

        family_rows: List[dict[str, Any]] = []
        for family, selected_count in family_selected.items():
            row = {
                "prefix": family,
                "selected": selected_count,
                "mismatch_any": int(family_mismatch_any.get(family, 0)),
                "details": dict(family_details[family]),
            }
            family_rows.append(row)

        family_rows.sort(
            key=lambda row: (
                row["mismatch_any"],
                row["selected"],
                row["prefix"],
            ),
            reverse=True,
        )

        report = {
            "generated_at_utc": _now_utc_iso(),
            "inputs": {
                "file": str(file_path),
                "baseline": str(baseline_path),
                "names_file": str(names_file) if names_file is not None else "",
                "subgroup_file": str(subgroup_file) if subgroup_file is not None else "",
                "subgroup_start": subgroup_start,
                "max_subgroups": max_subgroups,
                "subgroups_selected_count": len(subgroup_filters),
                "subgroups_selected": subgroup_filters,
                "prefixes": list(prefix_filters),
                "exclude_names_count": len(excluded),
            },
            "settings": {
                "head_bytes": head_bytes,
                "small_hash_limit": small_hash_limit,
                "sample_limit": sample_limit,
                "top_prefix_count": top_prefix_count,
                "progress_every": progress_every,
                "family_mode": family_mode,
            },
            "stats": stats,
            "family_map": family_rows[:top_prefix_count],
            "examples": examples,
        }
        return report
    finally:
        con_file.close()
        con_baseline.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Targeted content probe for Draw Things .ckpt files")
    parser.add_argument("--file", required=True, help="Candidate checkpoint")
    parser.add_argument("--baseline", required=True, help="Baseline checkpoint")
    parser.add_argument(
        "--names-file",
        default="",
        help="Optional explicit tensor-name list file (one name per line)",
    )
    parser.add_argument(
        "--subgroup-file",
        default="",
        help="Optional DIT subgroup list file (e.g. probe_dit_subset_80pct_*.txt)",
    )
    parser.add_argument(
        "--subgroup-start",
        type=int,
        default=1,
        help="1-based subgroup start index when using --subgroup-file (default: 1)",
    )
    parser.add_argument(
        "--max-subgroups",
        type=int,
        default=0,
        help="Max subgroup count when using --subgroup-file; 0 means all from start (default: 0)",
    )
    parser.add_argument(
        "--prefix",
        action="append",
        default=[],
        help="Optional prefix filter (repeatable), e.g. --prefix __dit__",
    )
    parser.add_argument(
        "--exclude-name",
        action="append",
        default=[],
        help="Optional explicit tensor name to exclude (repeatable)",
    )
    parser.add_argument(
        "--head-bytes",
        type=int,
        default=32,
        help="Head signature byte count for dim/data compare (default: 32)",
    )
    parser.add_argument(
        "--small-hash-limit",
        type=int,
        default=4096,
        help="Compute full data SHA256 for blobs <= this size (default: 4096)",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=12,
        help="Max samples per mismatch category (default: 12)",
    )
    parser.add_argument(
        "--top-prefix-count",
        type=int,
        default=16,
        help="Max family rows to include (default: 16)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Print progress every N selected rows (default: 200; 0 disables)",
    )
    parser.add_argument(
        "--family-mode",
        choices=["top_prefix", "dit_subgroup"],
        default="top_prefix",
        help="Grouping for family_map output (default: top_prefix)",
    )
    parser.add_argument("--out-json", default="", help="Optional JSON output path")
    parser.add_argument("--out-md", default="", help="Optional markdown output path")

    args = parser.parse_args()

    if args.head_bytes < 1:
        parser.error("--head-bytes must be >= 1")
    if args.small_hash_limit < 0:
        parser.error("--small-hash-limit must be >= 0")
    if args.sample_limit < 1:
        parser.error("--sample-limit must be >= 1")
    if args.top_prefix_count < 1:
        parser.error("--top-prefix-count must be >= 1")
    if args.progress_every < 0:
        parser.error("--progress-every must be >= 0")
    if args.subgroup_start < 1:
        parser.error("--subgroup-start must be >= 1")
    if args.max_subgroups < 0:
        parser.error("--max-subgroups must be >= 0")

    file_path = Path(args.file).expanduser().resolve()
    baseline_path = Path(args.baseline).expanduser().resolve()
    names_file = Path(args.names_file).expanduser().resolve() if args.names_file else None
    subgroup_file = Path(args.subgroup_file).expanduser().resolve() if args.subgroup_file else None

    try:
        report = run(
            file_path=file_path,
            baseline_path=baseline_path,
            names_file=names_file,
            subgroup_file=subgroup_file,
            subgroup_start=args.subgroup_start,
            max_subgroups=args.max_subgroups,
            prefixes=args.prefix,
            exclude_names=args.exclude_name,
            head_bytes=args.head_bytes,
            small_hash_limit=args.small_hash_limit,
            sample_limit=args.sample_limit,
            top_prefix_count=args.top_prefix_count,
            progress_every=args.progress_every,
            family_mode=args.family_mode,
        )
    except Exception as exc:
        print(f"RESULT=FAIL {exc}")
        return 2

    print("== Targeted Content Probe ==")
    print(f"file={report['inputs']['file']}")
    print(f"baseline={report['inputs']['baseline']}")
    print(f"names_file={report['inputs']['names_file']}")
    print(f"subgroup_file={report['inputs']['subgroup_file']}")
    print(f"subgroup_start={report['inputs']['subgroup_start']}")
    print(f"max_subgroups={report['inputs']['max_subgroups']}")
    print(f"subgroups_selected_count={report['inputs']['subgroups_selected_count']}")
    print(f"prefixes={report['inputs']['prefixes']}")
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
