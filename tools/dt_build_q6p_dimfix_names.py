#!/usr/bin/env python3
"""Build a tensor-name list for in-place q6p dimfix remediation.

Candidate set definition:

  meta_mismatch(target_q6p vs baseline_q6p)
    intersect
  (dim_mismatch(target_q6p vs baseline_q6p) - dim_mismatch(reference_q6p vs baseline_q6p))

This mirrors the investigative heuristic used for 10_e_v1 where failing q6p had
an extra dimension-mismatch family (vs a known working q6p reference), and only
the overlapping metadata-quantized subset was selected for dequantization.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Optional


def _load_keys(con: sqlite3.Connection) -> set[str]:
    rows = con.execute("SELECT name FROM tensors").fetchall()
    return {str(row[0]) for row in rows if row and row[0] is not None}


def _fetch_meta(con: sqlite3.Connection, name: str) -> tuple[Optional[tuple[str, str, str]], Optional[str]]:
    try:
        row = con.execute(
            "SELECT CAST(type AS TEXT), CAST(format AS TEXT), CAST(datatype AS TEXT) "
            "FROM tensors WHERE name=?",
            (name,),
        ).fetchone()
    except sqlite3.DataError as exc:
        return None, f"DataError: {exc}"
    except sqlite3.DatabaseError as exc:
        return None, f"DatabaseError: {exc}"

    if row is None:
        return None, "RowMissing"

    return tuple("NULL" if v is None else str(v) for v in row), None


def _fetch_dim_head(con: sqlite3.Connection, name: str, head_bytes: int) -> tuple[Optional[str], Optional[str]]:
    try:
        row = con.execute(
            "SELECT hex(substr(dim, 1, ?)) FROM tensors WHERE name=?",
            (head_bytes, name),
        ).fetchone()
    except sqlite3.DataError as exc:
        return None, f"DataError: {exc}"
    except sqlite3.DatabaseError as exc:
        return None, f"DatabaseError: {exc}"

    if row is None:
        return None, "RowMissing"

    return "NULL" if row[0] is None else str(row[0]), None


def _prefix(name: str) -> str:
    if "[" in name:
        return name.split("[", 1)[0]
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description="Build q6p in-place dimfix tensor-name list")
    parser.add_argument("--target-q6p", required=True, help="Failing target q6p checkpoint")
    parser.add_argument("--baseline-q6p", required=True, help="Official q6p baseline checkpoint")
    parser.add_argument(
        "--reference-q6p",
        required=True,
        help="Known working q6p reference checkpoint (for extra-dim filtering)",
    )
    parser.add_argument("--out", required=True, help="Output names file path")
    parser.add_argument(
        "--bad-row-name",
        default="__text_feature_extractor__[t-video_aggregate_embed-0-0]",
        help="Row to skip if known unreadable in both files",
    )
    parser.add_argument(
        "--head-bytes",
        type=int,
        default=32,
        help="Head bytes used for dim-signature comparison (default: 32)",
    )
    parser.add_argument(
        "--selection-mode",
        choices=(
            "meta_and_extra_dim",
            "meta_and_dim",
            "extra_dim_only",
            "dim_only",
            "meta_only",
        ),
        default="meta_and_extra_dim",
        help=(
            "Candidate selection mode: "
            "meta_and_extra_dim=meta∩(dim-ref_dim), "
            "meta_and_dim=meta∩dim, "
            "extra_dim_only=(dim-ref_dim), "
            "dim_only=dim, "
            "meta_only=meta"
        ),
    )
    args = parser.parse_args()

    target_path = Path(args.target_q6p).expanduser().resolve()
    baseline_path = Path(args.baseline_q6p).expanduser().resolve()
    reference_path = Path(args.reference_q6p).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    for path in (target_path, baseline_path, reference_path):
        if not path.exists():
            print(f"RESULT=FAIL missing file: {path}")
            return 2

    if args.head_bytes < 1:
        print("RESULT=FAIL --head-bytes must be >= 1")
        return 2

    bad = args.bad_row_name

    con_target = sqlite3.connect(str(target_path))
    con_base = sqlite3.connect(str(baseline_path))
    con_ref = sqlite3.connect(str(reference_path))
    try:
        target_keys = _load_keys(con_target)
        base_keys = _load_keys(con_base)
        ref_keys = _load_keys(con_ref)

        shared_target = sorted(target_keys & base_keys)
        shared_ref = sorted(ref_keys & base_keys)

        target_meta_mismatch: set[str] = set()
        target_dim_mismatch: set[str] = set()
        ref_dim_mismatch: set[str] = set()

        skipped_meta = 0
        skipped_target_dim = 0
        skipped_ref_dim = 0

        for name in shared_target:
            if name == bad:
                continue

            t_meta, t_meta_err = _fetch_meta(con_target, name)
            b_meta, b_meta_err = _fetch_meta(con_base, name)
            if t_meta_err or b_meta_err:
                skipped_meta += 1
            elif t_meta != b_meta:
                target_meta_mismatch.add(name)

            t_dim, t_dim_err = _fetch_dim_head(con_target, name, args.head_bytes)
            b_dim, b_dim_err = _fetch_dim_head(con_base, name, args.head_bytes)
            if t_dim_err or b_dim_err:
                skipped_target_dim += 1
            elif t_dim != b_dim:
                target_dim_mismatch.add(name)

        for name in shared_ref:
            if name == bad:
                continue

            r_dim, r_dim_err = _fetch_dim_head(con_ref, name, args.head_bytes)
            b_dim, b_dim_err = _fetch_dim_head(con_base, name, args.head_bytes)
            if r_dim_err or b_dim_err:
                skipped_ref_dim += 1
                continue
            if r_dim != b_dim:
                ref_dim_mismatch.add(name)

        extra_dim_mismatch = target_dim_mismatch - ref_dim_mismatch

        if args.selection_mode == "meta_and_extra_dim":
            candidate_set = target_meta_mismatch & extra_dim_mismatch
        elif args.selection_mode == "meta_and_dim":
            candidate_set = target_meta_mismatch & target_dim_mismatch
        elif args.selection_mode == "extra_dim_only":
            candidate_set = extra_dim_mismatch
        elif args.selection_mode == "dim_only":
            candidate_set = target_dim_mismatch
        else:
            candidate_set = target_meta_mismatch

        candidate = sorted(candidate_set)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(candidate) + ("\n" if candidate else ""), encoding="utf-8")

        prefix_counter = Counter(_prefix(name) for name in candidate)

        print("== Q6P Dimfix Name Builder ==")
        print(f"target_q6p={target_path}")
        print(f"baseline_q6p={baseline_path}")
        print(f"reference_q6p={reference_path}")
        print(f"shared_target_baseline={len(shared_target)}")
        print(f"shared_reference_baseline={len(shared_ref)}")
        print(f"target_meta_mismatch={len(target_meta_mismatch)}")
        print(f"target_dim_mismatch={len(target_dim_mismatch)}")
        print(f"reference_dim_mismatch={len(ref_dim_mismatch)}")
        print(f"extra_dim_mismatch={len(extra_dim_mismatch)}")
        print(f"selection_mode={args.selection_mode}")
        print(f"candidate_count={len(candidate)}")
        print(f"skipped_meta_unreadable={skipped_meta}")
        print(f"skipped_target_dim_unreadable={skipped_target_dim}")
        print(f"skipped_reference_dim_unreadable={skipped_ref_dim}")
        for pref, count in prefix_counter.most_common(12):
            print(f"candidate_prefix[{pref}]={count}")
        print(f"names_file={out_path}")

        if not candidate:
            print("RESULT=FAIL no candidate rows selected")
            return 1
        print("RESULT=PASS")
        return 0
    finally:
        con_target.close()
        con_base.close()
        con_ref.close()


if __name__ == "__main__":
    sys.exit(main())
