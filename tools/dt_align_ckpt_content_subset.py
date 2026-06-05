#!/usr/bin/env python3
"""Copy selected tensor content blobs (dim/data) from baseline checkpoint.

This tool is intended for focused remediation experiments where specific tensor
name lists should receive baseline `dim` and/or `data` blobs.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Sequence


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


def _load_target_names(con: sqlite3.Connection) -> set[str]:
    cur = con.cursor()
    rows = cur.execute("SELECT name FROM tensors").fetchall()
    return {str(row[0]) for row in rows if row and row[0] is not None}


def _free_gb(path: Path) -> float:
    usage = shutil.disk_usage(str(path))
    return usage.free / (1024**3)


def _enforce_min_free(path: Path, min_free_gb: float) -> float:
    free_gb = _free_gb(path)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"free space below threshold: {free_gb:.2f} GiB < {min_free_gb:.2f} GiB"
        )
    return free_gb


def _configure_journal_mode(cur: sqlite3.Cursor, journal_mode: str) -> str:
    if journal_mode == "preserve":
        row = cur.execute("PRAGMA journal_mode").fetchone()
    else:
        row = cur.execute(f"PRAGMA journal_mode={journal_mode.upper()}").fetchone()

    if row and row[0] is not None:
        return str(row[0]).lower()
    return "unknown"


def _chunks(values: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]


def _head_hex(con: sqlite3.Connection, name: str, column: str, head_bytes: int) -> str | None:
    cur = con.cursor()
    query = f"SELECT substr({column}, 1, ?) FROM tensors WHERE name=?"
    try:
        row = cur.execute(query, (head_bytes, name)).fetchone()
    except sqlite3.DataError:
        return None
    if row is None:
        return None
    value = row[0]
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytes):
        return value.hex().upper()
    if value is None:
        return "NULL"
    return str(value)


def _count_head_mismatch(
    target_con: sqlite3.Connection,
    baseline_con: sqlite3.Connection,
    names: Sequence[str],
    column: str,
    head_bytes: int,
) -> int:
    mismatch = 0
    for name in names:
        t = _head_hex(target_con, name, column, head_bytes)
        b = _head_hex(baseline_con, name, column, head_bytes)
        if t is None or b is None:
            continue
        if t != b:
            mismatch += 1
    return mismatch


def _apply_updates(
    target_con: sqlite3.Connection,
    baseline_path: Path,
    dim_names: Sequence[str],
    data_names: Sequence[str],
    chunk_size: int,
    journal_mode: str,
    min_free_gb: float,
    disk_guard_path: Path,
) -> Dict[str, int]:
    dim_set = set(dim_names)
    data_set = set(data_names)
    all_names = sorted(dim_set | data_set)

    tcur = target_con.cursor()
    stats = {
        "selected_union": len(all_names),
        "rows_missing_baseline": 0,
        "rows_missing_target": 0,
        "rows_updated": 0,
        "dim_rows_updated": 0,
        "data_rows_updated": 0,
        "rows_skipped_dataerror": 0,
    }

    tcur.execute("PRAGMA busy_timeout=15000")
    actual_journal_mode = _configure_journal_mode(tcur, journal_mode)
    print(f"mutation_journal_mode={actual_journal_mode}")

    # Use SQLite engine-side copying via ATTACH to avoid Python sqlite3 DataError
    # when selected rows contain very large BLOBs.
    target_con.execute("ATTACH DATABASE ? AS baseline_db", (str(baseline_path),))

    start_free_gb = _enforce_min_free(disk_guard_path, min_free_gb)
    print(f"start_free_gb={start_free_gb:.2f}")

    total_chunks = (len(all_names) + chunk_size - 1) // chunk_size
    try:
        for idx, chunk in enumerate(_chunks(all_names, chunk_size), start=1):
            free_gb = _enforce_min_free(disk_guard_path, min_free_gb)
            target_con.execute("BEGIN IMMEDIATE")
            try:
                for name in chunk:
                    copy_dim = name in dim_set
                    copy_data = name in data_set

                    row = tcur.execute(
                        "SELECT 1 FROM baseline_db.tensors WHERE name=?",
                        (name,),
                    ).fetchone()
                    if row is None:
                        stats["rows_missing_baseline"] += 1
                        continue

                    row = tcur.execute(
                        "SELECT 1 FROM tensors WHERE name=?",
                        (name,),
                    ).fetchone()
                    if row is None:
                        stats["rows_missing_target"] += 1
                        continue

                    try:
                        if copy_dim and copy_data:
                            tcur.execute(
                                "UPDATE tensors "
                                "SET dim=(SELECT b.dim FROM baseline_db.tensors b WHERE b.name=?), "
                                "data=(SELECT b.data FROM baseline_db.tensors b WHERE b.name=?) "
                                "WHERE name=?",
                                (name, name, name),
                            )
                        elif copy_dim:
                            tcur.execute(
                                "UPDATE tensors "
                                "SET dim=(SELECT b.dim FROM baseline_db.tensors b WHERE b.name=?) "
                                "WHERE name=?",
                                (name, name),
                            )
                        else:
                            tcur.execute(
                                "UPDATE tensors "
                                "SET data=(SELECT b.data FROM baseline_db.tensors b WHERE b.name=?) "
                                "WHERE name=?",
                                (name, name),
                            )
                    except sqlite3.DataError:
                        stats["rows_skipped_dataerror"] += 1
                        continue

                    if tcur.rowcount and tcur.rowcount > 0:
                        stats["rows_updated"] += 1
                        if copy_dim:
                            stats["dim_rows_updated"] += 1
                        if copy_data:
                            stats["data_rows_updated"] += 1
                    else:
                        stats["rows_missing_target"] += 1

                target_con.commit()
                if actual_journal_mode == "wal":
                    tcur.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                print(f"apply_chunk={idx}/{total_chunks} rows={len(chunk)} free_gb={free_gb:.2f}")
            except Exception:
                target_con.rollback()
                raise
    finally:
        try:
            tcur.execute("DETACH DATABASE baseline_db")
        except sqlite3.DatabaseError:
            pass

    return stats


def run(
    target_path: Path,
    baseline_path: Path,
    dim_names_file: Path | None,
    data_names_file: Path | None,
    mode: str,
    head_bytes: int,
    chunk_size: int,
    journal_mode: str,
    min_free_gb: float,
) -> int:
    if not target_path.exists():
        print(f"RESULT=FAIL missing target file: {target_path}")
        return 2
    if not baseline_path.exists():
        print(f"RESULT=FAIL missing baseline file: {baseline_path}")
        return 2
    if dim_names_file is None and data_names_file is None:
        print("RESULT=FAIL at least one of --dim-names-file or --data-names-file is required")
        return 2
    if dim_names_file is not None and not dim_names_file.exists():
        print(f"RESULT=FAIL missing dim names file: {dim_names_file}")
        return 2
    if data_names_file is not None and not data_names_file.exists():
        print(f"RESULT=FAIL missing data names file: {data_names_file}")
        return 2

    dim_names = _parse_names_file(dim_names_file) if dim_names_file is not None else []
    data_names = _parse_names_file(data_names_file) if data_names_file is not None else []

    target_con = sqlite3.connect(str(target_path))
    baseline_con = sqlite3.connect(str(baseline_path))
    try:
        target_names = _load_target_names(target_con)
        dim_selected = sorted(name for name in dim_names if name in target_names)
        data_selected = sorted(name for name in data_names if name in target_names)
        union_selected = sorted(set(dim_selected) | set(data_selected))

        pre_dim_mismatch = _count_head_mismatch(target_con, baseline_con, dim_selected, "dim", head_bytes)
        pre_data_mismatch = _count_head_mismatch(target_con, baseline_con, data_selected, "data", head_bytes)

        print("== Content Subset Alignment ==")
        print(f"target={target_path}")
        print(f"baseline={baseline_path}")
        print(f"mode={mode}")
        print(f"head_bytes={head_bytes}")
        print(f"chunk_size={chunk_size}")
        print(f"journal_mode_request={journal_mode}")
        print(f"min_free_gb={min_free_gb}")
        print(f"dim_names_file={dim_names_file if dim_names_file is not None else ''}")
        print(f"data_names_file={data_names_file if data_names_file is not None else ''}")
        print(f"dim_names_selected={len(dim_selected)}")
        print(f"data_names_selected={len(data_selected)}")
        print(f"union_selected={len(union_selected)}")
        print(f"pre_dim_head_mismatch={pre_dim_mismatch}")
        print(f"pre_data_head_mismatch={pre_data_mismatch}")

        if mode == "dry-run":
            print("RESULT=DRY_RUN")
            return 0

        try:
            apply_stats = _apply_updates(
                target_con=target_con,
                baseline_path=baseline_path,
                dim_names=dim_selected,
                data_names=data_selected,
                chunk_size=chunk_size,
                journal_mode=journal_mode,
                min_free_gb=min_free_gb,
                disk_guard_path=target_path.parent,
            )
        except RuntimeError as exc:
            print(f"RESULT=FAIL {exc}")
            return 1

        post_dim_mismatch = _count_head_mismatch(target_con, baseline_con, dim_selected, "dim", head_bytes)
        post_data_mismatch = _count_head_mismatch(target_con, baseline_con, data_selected, "data", head_bytes)

        for key, value in apply_stats.items():
            print(f"{key}={value}")
        print(f"post_dim_head_mismatch={post_dim_mismatch}")
        print(f"post_data_head_mismatch={post_data_mismatch}")
        if apply_stats["rows_skipped_dataerror"] > 0:
            print("RESULT=FAIL rows_skipped_dataerror > 0 (selected rows could not be copied)")
            return 1
        print("RESULT=PASS")
        return 0
    finally:
        target_con.close()
        baseline_con.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Align selected dim/data blobs from baseline")
    parser.add_argument("--file", required=True, help="Target checkpoint path")
    parser.add_argument("--baseline", required=True, help="Baseline checkpoint path")
    parser.add_argument(
        "--dim-names-file",
        default="",
        help="Optional tensor-name list for dim copy",
    )
    parser.add_argument(
        "--data-names-file",
        default="",
        help="Optional tensor-name list for data copy",
    )
    parser.add_argument(
        "--mode",
        choices=("dry-run", "apply"),
        default="dry-run",
        help="Operation mode",
    )
    parser.add_argument(
        "--head-bytes",
        type=int,
        default=32,
        help="Head signature bytes for pre/post mismatch checks (default: 32)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8,
        help="Rows per write transaction during apply mode (default: 8)",
    )
    parser.add_argument(
        "--journal-mode",
        choices=("delete", "wal", "preserve"),
        default="delete",
        help="SQLite journal mode for write phase (default: delete)",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=180.0,
        help="Abort apply if free workspace disk drops below this GiB threshold (default: 180)",
    )
    args = parser.parse_args()

    if args.head_bytes < 1:
        parser.error("--head-bytes must be >= 1")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be >= 1")
    if args.min_free_gb < 0:
        parser.error("--min-free-gb must be >= 0")

    target = Path(args.file).expanduser().resolve()
    baseline = Path(args.baseline).expanduser().resolve()
    dim_file = Path(args.dim_names_file).expanduser().resolve() if args.dim_names_file else None
    data_file = Path(args.data_names_file).expanduser().resolve() if args.data_names_file else None

    return run(
        target_path=target,
        baseline_path=baseline,
        dim_names_file=dim_file,
        data_names_file=data_file,
        mode=args.mode,
        head_bytes=args.head_bytes,
        chunk_size=args.chunk_size,
        journal_mode=args.journal_mode,
        min_free_gb=args.min_free_gb,
    )


if __name__ == "__main__":
    sys.exit(main())
