#!/usr/bin/env python3
"""Reorder readable tensor rows to baseline order using rowid remapping.

This tool avoids reading blob columns directly. It derives readable baseline
order by scanning rowids and catching DataError rows, then remaps target rowids
for readable names only. Unreadable rows are left untouched.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, TypeVar


T = TypeVar("T")


def _load_baseline_readable_order(path: Path) -> Tuple[List[str], List[int]]:
    con = sqlite3.connect(str(path))
    try:
        rowids = [int(r[0]) for r in con.execute("SELECT rowid FROM tensors ORDER BY rowid").fetchall()]
        readable_names: List[str] = []
        unreadable_rowids: List[int] = []

        for rid in rowids:
            try:
                row = con.execute("SELECT name FROM tensors WHERE rowid=?", (rid,)).fetchone()
            except sqlite3.DataError:
                unreadable_rowids.append(rid)
                continue

            if row is None or row[0] is None:
                continue
            readable_names.append(str(row[0]))

        return readable_names, unreadable_rowids
    finally:
        con.close()


def _load_target_rowids_by_name(path: Path, names: Sequence[str]) -> Dict[str, int]:
    con = sqlite3.connect(str(path))
    try:
        mapping: Dict[str, int] = {}
        cur = con.cursor()
        for name in names:
            row = cur.execute("SELECT rowid FROM tensors WHERE name=?", (name,)).fetchone()
            if row is None:
                continue
            mapping[name] = int(row[0])
        return mapping
    finally:
        con.close()


def _compute_mismatches(
    names_in_desired_order: Sequence[str],
    current_rowid_by_name: Dict[str, int],
) -> List[Tuple[int, str, int]]:
    mismatches: List[Tuple[int, str, int]] = []
    for idx, name in enumerate(names_in_desired_order, start=1):
        current = current_rowid_by_name.get(name)
        if current is None:
            continue
        if current != idx:
            mismatches.append((idx, name, current))
    return mismatches


def _batched(items: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _apply_reorder(
    target_path: Path,
    names_in_desired_order: Sequence[str],
    rowid_offset: int,
    chunk_size: int,
) -> None:
    con = sqlite3.connect(str(target_path))
    try:
        cur = con.cursor()
        cur.execute("PRAGMA busy_timeout=15000")
        # Keep write amplification bounded in this constrained container/WSL path.
        journal_mode = cur.execute("PRAGMA journal_mode=WAL").fetchone()
        if journal_mode:
            print(f"journal_mode={journal_mode[0]}")

        total = len(names_in_desired_order)
        total_chunks = (total + chunk_size - 1) // chunk_size

        # Phase 1: move readable rows out of low rowid range in small chunks.
        # Condition rowid < rowid_offset makes this phase safely resumable.
        for idx, chunk in enumerate(_batched(names_in_desired_order, chunk_size), start=1):
            con.execute("BEGIN IMMEDIATE")
            cur.executemany(
                "UPDATE tensors SET rowid = rowid + ? WHERE name = ? AND rowid < ?",
                [(rowid_offset, name, rowid_offset) for name in chunk],
            )
            con.commit()
            cur.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            print(f"phase1_chunk={idx}/{total_chunks} rows={len(chunk)}")

        # Phase 2: assign desired compressed rowids in small chunks.
        desired_pairs = [(name, i) for i, name in enumerate(names_in_desired_order, start=1)]
        for idx, chunk in enumerate(_batched(desired_pairs, chunk_size), start=1):
            con.execute("BEGIN IMMEDIATE")
            cur.executemany(
                "UPDATE tensors SET rowid = ? WHERE name = ?",
                [(desired, name) for name, desired in chunk],
            )
            con.commit()
            cur.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            print(f"phase2_chunk={idx}/{total_chunks} rows={len(chunk)}")
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def run(target: Path, baseline: Path, mode: str, rowid_offset: int, sample_limit: int, chunk_size: int) -> int:
    if not target.exists():
        print(f"RESULT=FAIL missing target file: {target}")
        return 2
    if not baseline.exists():
        print(f"RESULT=FAIL missing baseline file: {baseline}")
        return 2

    baseline_readable_order, baseline_unreadable_rowids = _load_baseline_readable_order(baseline)
    target_rowid_by_name = _load_target_rowids_by_name(target, baseline_readable_order)

    missing_names = [name for name in baseline_readable_order if name not in target_rowid_by_name]
    pre_mismatches = _compute_mismatches(baseline_readable_order, target_rowid_by_name)

    print("== Readable Row Reorder ==")
    print(f"target={target}")
    print(f"baseline={baseline}")
    print(f"mode={mode}")
    print(f"rowid_offset={rowid_offset}")
    print(f"baseline_readable_names={len(baseline_readable_order)}")
    print(f"baseline_unreadable_rowids={baseline_unreadable_rowids}")
    print(f"missing_names_in_target={len(missing_names)}")
    print(f"pre_position_mismatch={len(pre_mismatches)}")
    print(f"chunk_size={chunk_size}")

    for desired_rowid, name, current_rowid in pre_mismatches[:sample_limit]:
        print(f"  pre_mismatch desired_rowid={desired_rowid} current_rowid={current_rowid} name={name}")

    if missing_names:
        print("RESULT=FAIL missing baseline readable names in target")
        return 1

    if mode == "dry-run":
        print("RESULT=DRY_RUN")
        return 0

    _apply_reorder(
        target_path=target,
        names_in_desired_order=baseline_readable_order,
        rowid_offset=rowid_offset,
        chunk_size=chunk_size,
    )

    post_rowid_by_name = _load_target_rowids_by_name(target, baseline_readable_order)
    post_mismatches = _compute_mismatches(baseline_readable_order, post_rowid_by_name)
    print(f"post_position_mismatch={len(post_mismatches)}")

    for desired_rowid, name, current_rowid in post_mismatches[:sample_limit]:
        print(f"  post_mismatch desired_rowid={desired_rowid} current_rowid={current_rowid} name={name}")

    if post_mismatches:
        print("RESULT=FAIL mismatches remain")
        return 1

    print("RESULT=PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Reorder readable tensors to compressed baseline row order")
    parser.add_argument("--file", required=True, help="Target checkpoint path")
    parser.add_argument("--baseline", required=True, help="Baseline checkpoint path")
    parser.add_argument(
        "--mode",
        choices=("dry-run", "apply"),
        default="dry-run",
        help="Operation mode",
    )
    parser.add_argument(
        "--rowid-offset",
        type=int,
        default=10_000_000,
        help="Temporary rowid offset used during remap (default: 10000000)",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=10,
        help="Mismatch sample count to print (default: 10)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Rows per write transaction during apply mode (default: 256)",
    )
    args = parser.parse_args()

    if args.rowid_offset < 1:
        parser.error("--rowid-offset must be >= 1")
    if args.sample_limit < 1:
        parser.error("--sample-limit must be >= 1")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be >= 1")

    return run(
        target=Path(args.file).expanduser().resolve(),
        baseline=Path(args.baseline).expanduser().resolve(),
        mode=args.mode,
        rowid_offset=args.rowid_offset,
        sample_limit=args.sample_limit,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    sys.exit(main())
