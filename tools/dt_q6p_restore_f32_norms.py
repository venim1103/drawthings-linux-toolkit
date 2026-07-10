#!/usr/bin/env python3
"""Restore F32 norm/ada_ln tensors into a q6p checkpoint.

The Draw Things quantizer keeps LTX2.3 norm/ada_ln/modulation tensors at full
precision (F32) instead of palettizing them to 6-bit. Our quantizer palettizes
them, which corrupts the sensitive norm weights (6-bit) and stalls sampling.

This tool restores those tensors to F32 by copying the F32 values from OUR OWN
f16 source (not from any official weights) into the q6p, using the reference q6p
only as a template for which tensors must be F32 and how their dim is encoded.

Scope: only tensors that the reference q6p stores as F32 (datatype 16384). All
palettized weights are left untouched.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

DT_F32 = 16384

# use the high-limit sqlite python? No - these norm rows are small; default is fine.


def load_f32_names_and_dims(ref_q6p: Path):
    """From the reference q6p: name -> dim blob, for tensors stored as F32."""
    con = sqlite3.connect(str(ref_q6p))
    cur = con.cursor()
    cur.execute("SELECT name FROM tensors")
    names = [r[0] for r in cur.fetchall()]
    out = {}
    for n in names:
        try:
            cur.execute("SELECT datatype, dim, length(data) FROM tensors WHERE name=?", (n,))
            dt, dim, dlen = cur.fetchone()
            if dt == DT_F32:
                out[n] = (bytes(dim), dlen)
        except sqlite3.DataError:
            continue
    con.close()
    return out


def load_source_f32(src_f16: Path, names):
    """From our f16 source: name -> (dim blob, F32 data bytes)."""
    con = sqlite3.connect(str(src_f16))
    cur = con.cursor()
    out = {}
    for n in names:
        try:
            cur.execute("SELECT datatype, dim, data FROM tensors WHERE name=?", (n,))
            row = cur.fetchone()
            if row is None:
                continue
            dt, dim, data = row
            out[n] = (dt, bytes(dim), bytes(data))
        except sqlite3.DataError:
            continue
    con.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q6p", required=True, help="q6p checkpoint to fix in place")
    ap.add_argument("--f16-source", required=True, help="our f16 with correct F32 norms")
    ap.add_argument("--ref-q6p", required=True, help="official q6p (template for F32 tensor set + dim)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    q6p = Path(args.q6p)
    src = Path(args.f16_source)
    ref = Path(args.ref_q6p)
    for p in (q6p, src, ref):
        if not p.is_file():
            print(f"error: not found: {p}", file=sys.stderr)
            return 2

    f32_set = load_f32_names_and_dims(ref)
    print(f"reference F32 tensors: {len(f32_set)}")
    srcd = load_source_f32(src, list(f32_set.keys()))
    print(f"source tensors found: {len(srcd)}")

    con = sqlite3.connect(str(q6p))
    con.execute("PRAGMA journal_mode=DELETE")
    cur = con.cursor()

    restored = skipped_len = skipped_missing = already = 0
    pending = 0
    for name, (ref_dim, ref_dlen) in f32_set.items():
        s = srcd.get(name)
        if s is None:
            skipped_missing += 1
            continue
        s_dt, s_dim, s_data = s
        # our f16 norm should already be F32 (datatype 16384) after post-process
        if len(s_data) != ref_dlen:
            skipped_len += 1
            continue
        # check current state
        cur.execute("SELECT datatype FROM tensors WHERE name=?", (name,))
        cur_dt = cur.fetchone()
        if cur_dt and cur_dt[0] == DT_F32:
            already += 1
            continue
        if args.dry_run:
            restored += 1
            continue
        cur.execute(
            "UPDATE tensors SET datatype=?, dim=?, data=? WHERE name=?",
            (DT_F32, sqlite3.Binary(ref_dim), sqlite3.Binary(s_data), name),
        )
        restored += 1
        pending += 1
        if pending >= 256:
            con.commit(); pending = 0
    if not args.dry_run:
        con.commit()
    con.close()

    print("\n== summary ==")
    print(f"restored to F32   : {restored}")
    print(f"already F32       : {already}")
    print(f"skipped (len diff): {skipped_len}")
    print(f"skipped (missing) : {skipped_missing}")
    print("done." + (" (dry-run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
