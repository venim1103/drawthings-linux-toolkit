#!/usr/bin/env python3
"""Patch selected tensor metadata fields from baseline into target checkpoint.

Fields patched per selected name:
- type
- format
- datatype
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


def _parse_names(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    seen: set[str] = set()
    for line in raw:
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch checkpoint metadata subset")
    parser.add_argument("--file", required=True, help="Target checkpoint")
    parser.add_argument("--baseline", required=True, help="Baseline checkpoint")
    parser.add_argument("--names-file", required=True, help="Tensor name list")
    parser.add_argument("--journal-mode", choices=("wal", "delete", "preserve"), default="wal")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    target = Path(args.file).expanduser().resolve()
    baseline = Path(args.baseline).expanduser().resolve()
    names_file = Path(args.names_file).expanduser().resolve()

    if not target.exists():
        print(f"RESULT=FAIL missing target file: {target}")
        return 2
    if not baseline.exists():
        print(f"RESULT=FAIL missing baseline file: {baseline}")
        return 2
    if not names_file.exists():
        print(f"RESULT=FAIL missing names file: {names_file}")
        return 2
    if args.progress_every < 1:
        print("RESULT=FAIL --progress-every must be >= 1")
        return 2

    names = _parse_names(names_file)
    if not names:
        print("RESULT=FAIL names file is empty")
        return 2

    con = sqlite3.connect(str(target))
    cur = con.cursor()
    try:
        cur.execute("PRAGMA busy_timeout=15000")
        if args.journal_mode == "preserve":
            row = cur.execute("PRAGMA journal_mode").fetchone()
        else:
            row = cur.execute(f"PRAGMA journal_mode={args.journal_mode.upper()}").fetchone()
        mode = str(row[0]).lower() if row and row[0] is not None else "unknown"
        print(f"metadata_mutation_journal_mode={mode}")

        cur.execute("ATTACH DATABASE ? AS baseline_db", (str(baseline),))

        updated = 0
        missing_target = 0
        missing_baseline = 0
        skipped_dataerror = 0

        total = len(names)
        for idx, name in enumerate(names, start=1):
            row = cur.execute("SELECT 1 FROM tensors WHERE name=?", (name,)).fetchone()
            if row is None:
                missing_target += 1
                continue

            row = cur.execute("SELECT 1 FROM baseline_db.tensors WHERE name=?", (name,)).fetchone()
            if row is None:
                missing_baseline += 1
                continue

            try:
                cur.execute(
                    "UPDATE tensors "
                    "SET type=(SELECT b.type FROM baseline_db.tensors b WHERE b.name=tensors.name), "
                    "format=(SELECT b.format FROM baseline_db.tensors b WHERE b.name=tensors.name), "
                    "datatype=(SELECT b.datatype FROM baseline_db.tensors b WHERE b.name=tensors.name) "
                    "WHERE name=?",
                    (name,),
                )
            except sqlite3.DataError:
                skipped_dataerror += 1
                continue

            if cur.rowcount and cur.rowcount > 0:
                updated += int(cur.rowcount)

            if idx % args.progress_every == 0 or idx == total:
                print(f"metadata_progress={idx}/{total}")

        con.commit()
        if mode == "wal":
            cur.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        print(f"metadata_rows_requested={total}")
        print(f"metadata_rows_updated={updated}")
        print(f"metadata_rows_missing_target={missing_target}")
        print(f"metadata_rows_missing_baseline={missing_baseline}")
        print(f"metadata_rows_skipped_dataerror={skipped_dataerror}")

        if skipped_dataerror > 0:
            print("RESULT=FAIL rows_skipped_dataerror > 0")
            return 1
        print("RESULT=PASS")
        return 0
    finally:
        try:
            cur.execute("DETACH DATABASE baseline_db")
        except sqlite3.DatabaseError:
            pass
        con.close()


if __name__ == "__main__":
    sys.exit(main())
