#!/usr/bin/env python3
"""Align Draw Things checkpoint metadata to an official baseline.

This tool updates tensor metadata columns (type, format, datatype) in a target
checkpoint to match a baseline checkpoint by tensor name.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


SERIALIZATION_FIELDS: Sequence[str] = ("type", "format", "datatype")


def _load_tensor_keys(con: sqlite3.Connection) -> List[str]:
    cur = con.cursor()
    rows = cur.execute("SELECT name FROM tensors").fetchall()
    return [str(r[0]) for r in rows if r and r[0] is not None]


def _load_tensor_metadata_for_keys(
    con: sqlite3.Connection,
    keys: Sequence[str],
) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    cur = con.cursor()
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

        if row is None:
            continue

        metadata[key] = {
            "type": "NULL" if row[0] is None else str(row[0]),
            "format": "NULL" if row[1] is None else str(row[1]),
            "datatype": "NULL" if row[2] is None else str(row[2]),
        }

    return metadata, unreadable_keys


def _compute_summary(target: Path, baseline: Path, sample_limit: int) -> dict[str, int | list[str]]:
    target_con = sqlite3.connect(str(target))
    baseline_con = sqlite3.connect(str(baseline))
    try:
        target_keys = _load_tensor_keys(target_con)
        baseline_keys = _load_tensor_keys(baseline_con)

        target_set = set(target_keys)
        baseline_set = set(baseline_keys)
        shared_names = sorted(target_set & baseline_set)

        target_meta, target_unreadable = _load_tensor_metadata_for_keys(target_con, shared_names)
        baseline_meta, baseline_unreadable = _load_tensor_metadata_for_keys(baseline_con, shared_names)

        target_unreadable_set = set(target_unreadable)
        baseline_unreadable_set = set(baseline_unreadable)
        unreadable_both = sorted(target_unreadable_set & baseline_unreadable_set)
        unreadable_only_target = sorted(target_unreadable_set - baseline_unreadable_set)
        unreadable_only_baseline = sorted(baseline_unreadable_set - target_unreadable_set)

        readable_shared = [
            name
            for name in shared_names
            if name not in target_unreadable_set and name not in baseline_unreadable_set
        ]

        mismatch_names: Dict[str, List[str]] = {field: [] for field in SERIALIZATION_FIELDS}
        for name in readable_shared:
            t = target_meta[name]
            b = baseline_meta[name]
            for field in SERIALIZATION_FIELDS:
                if t[field] != b[field]:
                    mismatch_names[field].append(name)

        summary: dict[str, int | list[str]] = {
            "target_count": len(target_keys),
            "baseline_count": len(baseline_keys),
            "missing_in_target": len(baseline_set - target_set),
            "extra_in_target": len(target_set - baseline_set),
            "unreadable_metadata_both": len(unreadable_both),
            "unreadable_metadata_only_target": len(unreadable_only_target),
            "unreadable_metadata_only_baseline": len(unreadable_only_baseline),
            "readable_shared_tensors": len(readable_shared),
            "type_mismatch": len(mismatch_names["type"]),
            "format_mismatch": len(mismatch_names["format"]),
            "datatype_mismatch": len(mismatch_names["datatype"]),
            "type_mismatch_samples": mismatch_names["type"][:sample_limit],
            "format_mismatch_samples": mismatch_names["format"][:sample_limit],
            "datatype_mismatch_samples": mismatch_names["datatype"][:sample_limit],
            "unreadable_metadata_only_target_samples": unreadable_only_target[:sample_limit],
            "unreadable_metadata_only_baseline_samples": unreadable_only_baseline[:sample_limit],
        }

        return summary
    finally:
        target_con.close()
        baseline_con.close()


def _coerce_value(value_text: str) -> int | str | None:
    if value_text == "NULL":
        return None

    try:
        return int(value_text)
    except ValueError:
        return value_text


def _print_summary(label: str, summary: dict[str, int | list[str]]) -> None:
    print(f"== {label} ==")
    print(f"target_count={summary['target_count']}")
    print(f"baseline_count={summary['baseline_count']}")
    print(f"missing_in_target={summary['missing_in_target']}")
    print(f"extra_in_target={summary['extra_in_target']}")
    print(f"unreadable_metadata_both={summary['unreadable_metadata_both']}")
    print(f"unreadable_metadata_only_target={summary['unreadable_metadata_only_target']}")
    print(f"unreadable_metadata_only_baseline={summary['unreadable_metadata_only_baseline']}")
    print(f"readable_shared_tensors={summary['readable_shared_tensors']}")
    print(f"type_mismatch={summary['type_mismatch']}")
    print(f"format_mismatch={summary['format_mismatch']}")
    print(f"datatype_mismatch={summary['datatype_mismatch']}")

    for key in (
        "type_mismatch_samples",
        "format_mismatch_samples",
        "datatype_mismatch_samples",
        "unreadable_metadata_only_target_samples",
        "unreadable_metadata_only_baseline_samples",
    ):
        samples = summary[key]
        print(f"{key}={len(samples)}")
        for name in samples:
            print(f"  {name}")


def _build_update_plan(target: Path, baseline: Path) -> Tuple[List[Tuple[int | str | None, int | str | None, int | str | None, str]], List[str], List[str], List[str]]:
    target_con = sqlite3.connect(str(target))
    baseline_con = sqlite3.connect(str(baseline))
    try:
        target_keys = set(_load_tensor_keys(target_con))
        baseline_keys = set(_load_tensor_keys(baseline_con))
        shared_names = sorted(target_keys & baseline_keys)

        target_meta, target_unreadable = _load_tensor_metadata_for_keys(target_con, shared_names)
        baseline_meta, baseline_unreadable = _load_tensor_metadata_for_keys(baseline_con, shared_names)

        target_unreadable_set = set(target_unreadable)
        baseline_unreadable_set = set(baseline_unreadable)
        unreadable_both = sorted(target_unreadable_set & baseline_unreadable_set)
        unreadable_only_target = sorted(target_unreadable_set - baseline_unreadable_set)
        unreadable_only_baseline = sorted(baseline_unreadable_set - target_unreadable_set)

        updates: List[Tuple[int | str | None, int | str | None, int | str | None, str]] = []
        for name in shared_names:
            if name in target_unreadable_set or name in baseline_unreadable_set:
                continue

            t = target_meta[name]
            b = baseline_meta[name]
            if all(t[field] == b[field] for field in SERIALIZATION_FIELDS):
                continue

            updates.append(
                (
                    _coerce_value(b["type"]),
                    _coerce_value(b["format"]),
                    _coerce_value(b["datatype"]),
                    name,
                )
            )

        return updates, unreadable_both, unreadable_only_target, unreadable_only_baseline
    finally:
        target_con.close()
        baseline_con.close()


def _apply_updates(target: Path, updates: List[Tuple[int | str | None, int | str | None, int | str | None, str]]) -> None:
    con = sqlite3.connect(str(target))
    try:
        cur = con.cursor()
        con.execute("BEGIN IMMEDIATE")
        cur.executemany(
            "UPDATE tensors SET type=?, format=?, datatype=? WHERE name=?",
            updates,
        )
        con.commit()
    finally:
        con.close()


def run(
    target: Path,
    baseline: Path,
    mode: str,
    allow_keyset_mismatch: bool,
    sample_limit: int,
) -> int:
    if not target.exists():
        print(f"RESULT=FAIL target file not found: {target}")
        return 2
    if not baseline.exists():
        print(f"RESULT=FAIL baseline file not found: {baseline}")
        return 2

    before = _compute_summary(target, baseline, sample_limit)
    _print_summary("Metadata Alignment (Before)", before)

    missing = int(before["missing_in_target"])
    extra = int(before["extra_in_target"])
    if not allow_keyset_mismatch and (missing > 0 or extra > 0):
        print("RESULT=FAIL keyset mismatch (use --allow-keyset-mismatch to override)")
        return 1

    if mode == "dry-run":
        print("RESULT=DRY_RUN")
        return 0

    print("== Applying metadata alignment ==")
    updates, unreadable_both, unreadable_only_target, unreadable_only_baseline = _build_update_plan(target, baseline)
    print(f"planned_updates={len(updates)}")
    print(f"planned_unreadable_both={len(unreadable_both)}")
    print(f"planned_unreadable_only_target={len(unreadable_only_target)}")
    print(f"planned_unreadable_only_baseline={len(unreadable_only_baseline)}")

    _apply_updates(target, updates)

    after = _compute_summary(target, baseline, sample_limit)
    _print_summary("Metadata Alignment (After)", after)

    remaining = (
        int(after["unreadable_metadata_only_target"])
        + int(after["unreadable_metadata_only_baseline"])
        + int(after["type_mismatch"])
        + int(after["format_mismatch"])
        + int(after["datatype_mismatch"])
    )
    if remaining == 0:
        print("RESULT=PASS")
        return 0

    print("RESULT=FAIL mismatches remain after alignment")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Align .ckpt metadata columns to a baseline")
    parser.add_argument("--file", required=True, help="Target checkpoint to update")
    parser.add_argument("--baseline", required=True, help="Baseline checkpoint to copy metadata from")
    parser.add_argument(
        "--mode",
        choices=["dry-run", "apply"],
        default="dry-run",
        help="dry-run (default) only reports; apply updates target metadata in-place",
    )
    parser.add_argument(
        "--allow-keyset-mismatch",
        action="store_true",
        help="Allow alignment to continue even when keysets differ",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=12,
        help="Number of mismatch sample tensor names to print per category (default: 12)",
    )
    args = parser.parse_args()

    if args.sample_limit < 1:
        parser.error("--sample-limit must be >= 1")

    target = Path(args.file).expanduser().resolve()
    baseline = Path(args.baseline).expanduser().resolve()
    return run(target, baseline, args.mode, args.allow_keyset_mismatch, args.sample_limit)


if __name__ == "__main__":
    sys.exit(main())
