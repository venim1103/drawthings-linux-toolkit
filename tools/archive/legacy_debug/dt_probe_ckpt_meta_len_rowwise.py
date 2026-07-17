#!/usr/bin/env python3
"""Row-wise metadata/length mismatch probe for Draw Things checkpoints.

This probe intentionally avoids dim/data head-byte reads so it remains stable
when full-table blob signature scans trigger sqlite DataError or process kills.
"""

from __future__ import annotations

import argparse
import json
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


def _parse_names_file(path: Path) -> List[str]:
    names: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        names.append(line)
    return names


def _load_tensor_keys(con: sqlite3.Connection) -> set[str]:
    cur = con.cursor()
    rows = cur.execute("SELECT name FROM tensors").fetchall()
    return {str(row[0]) for row in rows if row and row[0] is not None}


def _fetch_meta_len(con: sqlite3.Connection, key: str) -> tuple[dict[str, Any] | None, str | None]:
    cur = con.cursor()
    query = (
        "SELECT "
        "CAST(type AS TEXT), "
        "CAST(format AS TEXT), "
        "CAST(datatype AS TEXT), "
        "length(dim), "
        "length(data) "
        "FROM tensors WHERE name=?"
    )
    try:
        row = cur.execute(query, (key,)).fetchone()
    except sqlite3.DataError as exc:
        return None, f"DataError: {exc}"
    except sqlite3.DatabaseError as exc:
        return None, f"DatabaseError: {exc}"

    if row is None:
        return None, "RowMissing"

    return (
        {
            "metadata": {
                "type": _normalize_cell(row[0]),
                "format": _normalize_cell(row[1]),
                "datatype": _normalize_cell(row[2]),
            },
            "dim_len": row[3],
            "data_len": row[4],
        },
        None,
    )


def _append_sample(
    samples: Dict[str, List[dict[str, Any]]],
    category: str,
    payload: dict[str, Any],
    sample_limit: int,
) -> None:
    bucket = samples.setdefault(category, [])
    if len(bucket) < sample_limit:
        bucket.append(payload)


def _write_tsv(out_path: Path, rows: List[dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "prefix\tselected\tmismatch_any\tunreadable_both\tunreadable_only_file"
        "\tunreadable_only_baseline\tmetadata_mismatch_type\tmetadata_mismatch_format"
        "\tmetadata_mismatch_datatype\tdim_len_mismatch\tdata_len_mismatch\n"
    )
    with out_path.open("w", encoding="utf-8") as f:
        f.write(header)
        for row in rows:
            details = row["details"]
            f.write(
                "\t".join(
                    [
                        row["prefix"],
                        str(row["selected"]),
                        str(row["mismatch_any"]),
                        str(details.get("unreadable_both", 0)),
                        str(details.get("unreadable_only_file", 0)),
                        str(details.get("unreadable_only_baseline", 0)),
                        str(details.get("metadata_mismatch_type", 0)),
                        str(details.get("metadata_mismatch_format", 0)),
                        str(details.get("metadata_mismatch_datatype", 0)),
                        str(details.get("dim_len_mismatch", 0)),
                        str(details.get("data_len_mismatch", 0)),
                    ]
                )
                + "\n"
            )


def run(
    file_path: Path,
    baseline_path: Path,
    names_file: Path | None,
    sample_limit: int,
    top_prefix_count: int,
    progress_every: int,
) -> dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(f"missing file: {file_path}")
    if not baseline_path.exists():
        raise FileNotFoundError(f"missing baseline: {baseline_path}")
    if names_file is not None and not names_file.exists():
        raise FileNotFoundError(f"missing names file: {names_file}")

    con_file = sqlite3.connect(str(file_path))
    con_baseline = sqlite3.connect(str(baseline_path))
    try:
        file_keys = _load_tensor_keys(con_file)
        baseline_keys = _load_tensor_keys(con_baseline)
        shared = file_keys & baseline_keys

        if names_file is not None:
            names = _parse_names_file(names_file)
            selected = sorted(name for name in names if name in shared)
        else:
            selected = sorted(shared)

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
            "mismatch_any": 0,
            "full_match": 0,
        }

        examples: Dict[str, List[dict[str, Any]]] = {}
        family_selected: Counter[str] = Counter()
        family_mismatch_any: Counter[str] = Counter()
        family_details: dict[str, Counter[str]] = defaultdict(Counter)
        mismatch_names: List[str] = []

        for idx, name in enumerate(selected, start=1):
            if progress_every > 0 and idx % progress_every == 0:
                print(f"progress={idx}/{len(selected)}")

            file_sig, file_err = _fetch_meta_len(con_file, name)
            baseline_sig, baseline_err = _fetch_meta_len(con_baseline, name)

            family = _prefix_for_name(name)
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

            if mismatch_hit:
                stats["mismatch_any"] += 1
                family_mismatch_any[family] += 1
                mismatch_names.append(name)
            else:
                stats["full_match"] += 1

        family_rows: List[dict[str, Any]] = []
        for family, selected_count in family_selected.items():
            family_rows.append(
                {
                    "prefix": family,
                    "selected": selected_count,
                    "mismatch_any": int(family_mismatch_any.get(family, 0)),
                    "details": dict(family_details[family]),
                }
            )

        family_rows.sort(
            key=lambda row: (
                row["mismatch_any"],
                row["selected"],
                row["prefix"],
            ),
            reverse=True,
        )

        return {
            "generated_at_utc": _now_utc_iso(),
            "inputs": {
                "file": str(file_path),
                "baseline": str(baseline_path),
                "names_file": str(names_file) if names_file is not None else "",
            },
            "settings": {
                "sample_limit": sample_limit,
                "top_prefix_count": top_prefix_count,
                "progress_every": progress_every,
            },
            "stats": stats,
            "family_map": family_rows[:top_prefix_count],
            "examples": examples,
            "mismatch_names": sorted(set(mismatch_names)),
        }
    finally:
        con_file.close()
        con_baseline.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Row-wise metadata/length probe for Draw Things .ckpt files")
    parser.add_argument("--file", required=True, help="Candidate checkpoint")
    parser.add_argument("--baseline", required=True, help="Baseline checkpoint")
    parser.add_argument(
        "--names-file",
        default="",
        help="Optional explicit tensor-name list file (one name per line)",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=16,
        help="Max samples per mismatch category (default: 16)",
    )
    parser.add_argument(
        "--top-prefix-count",
        type=int,
        default=24,
        help="Max family rows to include in JSON (default: 24)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=400,
        help="Print progress every N selected rows (default: 400; 0 disables)",
    )
    parser.add_argument("--out-json", default="", help="Optional JSON output path")
    parser.add_argument("--out-tsv", default="", help="Optional TSV output path")
    parser.add_argument("--out-mismatch-names", default="", help="Optional output tensor-name list path")

    args = parser.parse_args()

    if args.sample_limit < 1:
        parser.error("--sample-limit must be >= 1")
    if args.top_prefix_count < 1:
        parser.error("--top-prefix-count must be >= 1")
    if args.progress_every < 0:
        parser.error("--progress-every must be >= 0")

    file_path = Path(args.file).expanduser().resolve()
    baseline_path = Path(args.baseline).expanduser().resolve()
    names_file = Path(args.names_file).expanduser().resolve() if args.names_file else None

    try:
        report = run(
            file_path=file_path,
            baseline_path=baseline_path,
            names_file=names_file,
            sample_limit=args.sample_limit,
            top_prefix_count=args.top_prefix_count,
            progress_every=args.progress_every,
        )
    except Exception as exc:
        print(f"RESULT=FAIL {exc}")
        return 2

    print("== Row-wise Meta/Len Probe ==")
    print(f"file={report['inputs']['file']}")
    print(f"baseline={report['inputs']['baseline']}")
    print(f"names_file={report['inputs']['names_file']}")
    for key, value in report["stats"].items():
        print(f"{key}={value}")

    if args.out_json:
        out_json = Path(args.out_json).expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"json_report={out_json}")

    if args.out_tsv:
        out_tsv = Path(args.out_tsv).expanduser().resolve()
        _write_tsv(out_tsv, report["family_map"])
        print(f"tsv_report={out_tsv}")

    if args.out_mismatch_names:
        out_names = Path(args.out_mismatch_names).expanduser().resolve()
        out_names.parent.mkdir(parents=True, exist_ok=True)
        lines = report["mismatch_names"]
        out_names.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"mismatch_names_file={out_names}")

    print("RESULT=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
