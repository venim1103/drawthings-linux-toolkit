#!/usr/bin/env python3
"""Align selected checkpoint payload blobs to a baseline checkpoint.

This tool is intended for targeted experiments where only a subset of tensor
payloads should be copied from a known-good baseline.

Current selection mode uses DIT subgroup names (from probe artifacts like
probe_dit_subset_80pct_*.txt) and matches tensor names in form:

  __dit__[<subgroup>-...]
"""

from __future__ import annotations

import argparse
import re
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Iterable, List, Sequence


def _parse_subgroups(path: Path) -> List[str]:
    subgroups: List[str] = []
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    for line in raw_lines:
        text = line.strip()
        if not text:
            continue
        # Expected format from probe_dit_subset_80pct_*.txt:
        # 01. t-attn1_ada_ln_0 | count=48 | top_pair=8192->16384 x48
        m = re.match(r"^\d+\.\s+([^|]+?)\s+\|", text)
        if m:
            subgroups.append(m.group(1).strip())
            continue

        # Fallback: allow raw subgroup line without numbering.
        if "|" not in text and " " not in text:
            subgroups.append(text)

    # Preserve order, dedupe.
    deduped: List[str] = []
    seen = set()
    for subgroup in subgroups:
        if subgroup in seen:
            continue
        seen.add(subgroup)
        deduped.append(subgroup)
    return deduped


def _parse_tensor_names(path: Path) -> List[str]:
    names: List[str] = []
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    for line in raw_lines:
        text = line.strip()
        if not text:
            continue
        if text.startswith("#"):
            continue
        names.append(text)

    deduped: List[str] = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def _load_tensor_names(con: sqlite3.Connection) -> List[str]:
    cur = con.cursor()
    rows = cur.execute("SELECT name FROM tensors").fetchall()
    return [str(row[0]) for row in rows if row and row[0] is not None]


def _build_selected_dit_tensor_names(all_names: Sequence[str], subgroups: Sequence[str]) -> List[str]:
    # Match __dit__[<subgroup>-...]
    selected: List[str] = []
    for name in all_names:
        if not (name.startswith("__dit__[") and name.endswith("]")):
            continue
        inner = name[len("__dit__[") : -1]
        for subgroup in subgroups:
            prefix = subgroup + "-"
            if inner.startswith(prefix):
                selected.append(name)
                break

    selected.sort()
    return selected


def _chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]


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


def _count_data_len_mismatch(
    target_con: sqlite3.Connection,
    baseline_con: sqlite3.Connection,
    names: Sequence[str],
) -> int:
    if not names:
        return 0

    tcur = target_con.cursor()
    bcur = baseline_con.cursor()
    mismatch = 0

    for chunk in _chunks(names, 256):
        placeholders = ",".join("?" for _ in chunk)
        target_rows = tcur.execute(
            f"SELECT name, length(data) FROM tensors WHERE name IN ({placeholders})",
            tuple(chunk),
        ).fetchall()
        baseline_rows = bcur.execute(
            f"SELECT name, length(data) FROM tensors WHERE name IN ({placeholders})",
            tuple(chunk),
        ).fetchall()

        t_map = {str(name): length for name, length in target_rows}
        b_map = {str(name): length for name, length in baseline_rows}

        for name in chunk:
            if name not in t_map or name not in b_map:
                continue
            if t_map[name] != b_map[name]:
                mismatch += 1

    return mismatch


def _apply_payload_updates(
    target_con: sqlite3.Connection,
    baseline_con: sqlite3.Connection,
    names: Sequence[str],
    chunk_size: int,
    journal_mode: str,
    min_free_gb: float,
    disk_guard_path: Path,
) -> tuple[int, int]:
    """Copy baseline data blobs for selected names.

    Returns tuple: (rows_updated, rows_missing_baseline)
    """
    if not names:
        return 0, 0

    tcur = target_con.cursor()
    bcur = baseline_con.cursor()
    updated = 0
    missing_baseline = 0

    tcur.execute("PRAGMA busy_timeout=15000")
    actual_journal_mode = _configure_journal_mode(tcur, journal_mode)
    print(f"mutation_journal_mode={actual_journal_mode}")

    start_free_gb = _enforce_min_free(disk_guard_path, min_free_gb)
    print(f"start_free_gb={start_free_gb:.2f}")

    total_chunks = (len(names) + chunk_size - 1) // chunk_size
    for idx, chunk in enumerate(_chunks(names, chunk_size), start=1):
        free_gb = _enforce_min_free(disk_guard_path, min_free_gb)
        target_con.execute("BEGIN IMMEDIATE")
        try:
            for name in chunk:
                row = bcur.execute("SELECT data FROM tensors WHERE name=?", (name,)).fetchone()
                if row is None:
                    missing_baseline += 1
                    continue

                tcur.execute("UPDATE tensors SET data=? WHERE name=?", (row[0], name))
                if tcur.rowcount > 0:
                    updated += 1

            target_con.commit()
            if actual_journal_mode == "wal":
                tcur.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            print(f"apply_chunk={idx}/{total_chunks} rows={len(chunk)} free_gb={free_gb:.2f}")
        except Exception:
            target_con.rollback()
            raise

    return updated, missing_baseline


def run(
    target_path: Path,
    baseline_path: Path,
    subgroup_file: Path | None,
    tensor_list_file: Path | None,
    mode: str,
    dump_selected_names_path: Path | None,
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
    if subgroup_file is None and tensor_list_file is None:
        print("RESULT=FAIL one of --subgroup-file or --tensor-list-file is required")
        return 2
    if subgroup_file is not None and tensor_list_file is not None:
        print("RESULT=FAIL --subgroup-file and --tensor-list-file are mutually exclusive")
        return 2

    if subgroup_file is not None and not subgroup_file.exists():
        print(f"RESULT=FAIL missing subgroup file: {subgroup_file}")
        return 2
    if tensor_list_file is not None and not tensor_list_file.exists():
        print(f"RESULT=FAIL missing tensor list file: {tensor_list_file}")
        return 2

    subgroups: List[str] = []
    explicit_tensor_names: List[str] = []
    if subgroup_file is not None:
        subgroups = _parse_subgroups(subgroup_file)
        if not subgroups:
            print("RESULT=FAIL no subgroups parsed from subgroup file")
            return 2
    else:
        assert tensor_list_file is not None
        explicit_tensor_names = _parse_tensor_names(tensor_list_file)
        if not explicit_tensor_names:
            print("RESULT=FAIL no tensor names parsed from tensor list file")
            return 2

    target_con = sqlite3.connect(str(target_path))
    baseline_con = sqlite3.connect(str(baseline_path))

    try:
        all_target_names = _load_tensor_names(target_con)
        all_target_set = set(all_target_names)

        if subgroup_file is not None:
            selected_names = _build_selected_dit_tensor_names(all_target_names, subgroups)
        else:
            selected_names = [name for name in explicit_tensor_names if name in all_target_set]
            selected_names.sort()
            missing_names = [name for name in explicit_tensor_names if name not in all_target_set]

        print("== Payload Subset Alignment ==")
        print(f"target={target_path}")
        print(f"baseline={baseline_path}")
        print(f"mode={mode}")
        print(f"chunk_size={chunk_size}")
        print(f"journal_mode_request={journal_mode}")
        print(f"min_free_gb={min_free_gb}")
        if subgroup_file is not None:
            print(f"subgroup_file={subgroup_file}")
            print(f"subgroups_parsed={len(subgroups)}")
        else:
            assert tensor_list_file is not None
            print(f"tensor_list_file={tensor_list_file}")
            print(f"tensor_names_parsed={len(explicit_tensor_names)}")
            print(f"tensor_names_missing_in_target={len(missing_names)}")
        print(f"selected_tensor_names={len(selected_names)}")

        if dump_selected_names_path is not None:
            dump_selected_names_path.parent.mkdir(parents=True, exist_ok=True)
            dump_selected_names_path.write_text("\n".join(selected_names) + "\n", encoding="utf-8")
            print(f"selected_names_file={dump_selected_names_path}")

        if not selected_names:
            print("RESULT=FAIL zero tensors selected")
            return 1

        pre_mismatch = _count_data_len_mismatch(target_con, baseline_con, selected_names)
        print(f"pre_selected_data_len_mismatch={pre_mismatch}")

        if mode == "dry-run":
            print("RESULT=DRY_RUN")
            return 0

        try:
            updated, missing_baseline = _apply_payload_updates(
                target_con=target_con,
                baseline_con=baseline_con,
                names=selected_names,
                chunk_size=chunk_size,
                journal_mode=journal_mode,
                min_free_gb=min_free_gb,
                disk_guard_path=target_path.parent,
            )
        except RuntimeError as exc:
            print(f"RESULT=FAIL {exc}")
            return 1
        print(f"rows_updated={updated}")
        print(f"rows_missing_baseline={missing_baseline}")

        post_mismatch = _count_data_len_mismatch(target_con, baseline_con, selected_names)
        print(f"post_selected_data_len_mismatch={post_mismatch}")

        if missing_baseline > 0:
            print("RESULT=FAIL baseline missing selected rows")
            return 1

        if post_mismatch == 0:
            print("RESULT=PASS")
            return 0

        print("RESULT=FAIL selected mismatches remain")
        return 1
    finally:
        target_con.close()
        baseline_con.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy baseline payload blobs for selected DIT subgroups")
    parser.add_argument("--file", required=True, help="Target checkpoint file to modify")
    parser.add_argument("--baseline", required=True, help="Baseline checkpoint file")
    parser.add_argument(
        "--subgroup-file",
        default="",
        help="Subgroup list file (e.g. probe_dit_subset_80pct_*.txt)",
    )
    parser.add_argument(
        "--tensor-list-file",
        default="",
        help="Explicit tensor-name list file (one tensor name per line)",
    )
    parser.add_argument(
        "--mode",
        choices=["dry-run", "apply"],
        default="dry-run",
        help="dry-run only reports selection/mismatch counts, apply copies payload blobs",
    )
    parser.add_argument(
        "--dump-selected-names",
        default="",
        help="Optional path to write resolved selected tensor names",
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

    target_path = Path(args.file).expanduser().resolve()
    baseline_path = Path(args.baseline).expanduser().resolve()
    subgroup_file = Path(args.subgroup_file).expanduser().resolve() if args.subgroup_file else None
    tensor_list_file = Path(args.tensor_list_file).expanduser().resolve() if args.tensor_list_file else None
    dump_selected = Path(args.dump_selected_names).expanduser().resolve() if args.dump_selected_names else None
    if args.chunk_size < 1:
        parser.error("--chunk-size must be >= 1")
    if args.min_free_gb < 0:
        parser.error("--min-free-gb must be >= 0")

    return run(
        target_path=target_path,
        baseline_path=baseline_path,
        subgroup_file=subgroup_file,
        tensor_list_file=tensor_list_file,
        mode=args.mode,
        dump_selected_names_path=dump_selected,
        chunk_size=args.chunk_size,
        journal_mode=args.journal_mode,
        min_free_gb=args.min_free_gb,
    )


if __name__ == "__main__":
    sys.exit(main())
