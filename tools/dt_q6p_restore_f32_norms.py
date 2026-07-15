#!/usr/bin/env python3
"""Restore F32 norm/ada_ln tensors into a q6p checkpoint.

The Draw Things quantizer keeps LTX2.3 norm/ada_ln/modulation tensors at full
precision (F32) instead of palettizing them to 6-bit. Our quantizer palettizes
them, which corrupts the sensitive norm weights (6-bit) and stalls sampling.

This tool restores those tensors to F32 by copying the F32 values from OUR OWN
f16 source (not from any official weights) into the q6p.

Two ways to decide which tensors are F32 and how their dim is encoded:
  * reference-free (default): the ada_ln modulation set is reconstructed from the
    source by name pattern and reshaped to [1,1,N] -- no reference checkpoint needed.
  * --ref-q6p PATH: use an official q6p as the template for the F32 tensor set and
    their dim blobs (restores whatever the reference marks F32).

Scope: only the norm/ada_ln tensors; all palettized weights are left untouched.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np

DT_F32 = 16384
DT_F16 = 131072

# The ada_ln modulation weights are the tensors Draw Things keeps at F32 in the
# official q6p (1540 in LTX2.3). They are named "*_ada_ln_<n>" and stored [1,N];
# the runtime needs them [1,1,N]. The literal "ada_ln" (with underscore) matches
# these while EXCLUDING the "adaln_single" linear layers (2-D [N,N] matrices that
# stay palettized). This lets us restore them WITHOUT any reference checkpoint.
ADA_LN_RE = re.compile(r"ada_ln")

# use the high-limit sqlite python? No - these norm rows are small; default is fine.


def discover_ada_ln_from_source(src_f16: Path):
    """Reference-free F32 set: ada_ln modulation tensor names from our f16 source."""
    con = sqlite3.connect(str(src_f16))
    cur = con.cursor()
    cur.execute("SELECT name FROM tensors")
    names = [r[0] for r in cur.fetchall() if ADA_LN_RE.search(r[0])]
    con.close()
    return names


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
    ap.add_argument("--ref-q6p", default=None,
                    help="optional official q6p template; omit for reference-free ada_ln restore")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    q6p = Path(args.q6p)
    src = Path(args.f16_source)
    ref = Path(args.ref_q6p) if args.ref_q6p else None
    must_exist = [q6p, src] + ([ref] if ref is not None else [])
    for p in must_exist:
        if not p.is_file():
            print(f"error: not found: {p}", file=sys.stderr)
            return 2

    if ref is not None:
        # reference mode: F32 set + dims come from the official q6p template
        ref_map = load_f32_names_and_dims(ref)
        names = list(ref_map.keys())
        print(f"reference F32 tensors: {len(names)}")
    else:
        # reference-free mode: F32 set = ada_ln modulation tensors from the source
        ref_map = None
        names = discover_ada_ln_from_source(src)
        print(f"ada_ln F32 tensors (reference-free): {len(names)}")
    srcd = load_source_f32(src, names)
    print(f"source tensors found: {len(srcd)}")

    con = sqlite3.connect(str(q6p))
    con.execute("PRAGMA journal_mode=DELETE")
    cur = con.cursor()

    restored = skipped_len = skipped_missing = skipped_badtype = already = 0
    pending = 0
    for name in names:
        s = srcd.get(name)
        if s is None:
            skipped_missing += 1
            continue
        s_dt, s_dim, s_data = s
        # Widen the source norm to F32 (custom fine-tunes keep norms at F16; the
        # official base stores them F32). Use the model's OWN values -- never borrow
        # official norms for a fine-tune.
        if s_dt == DT_F16:
            f32_bytes = np.frombuffer(s_data, dtype=np.float16).astype("<f4").tobytes()
        elif s_dt == DT_F32:
            f32_bytes = np.frombuffer(s_data, dtype="<f4").tobytes()
        else:
            skipped_badtype += 1
            continue
        n_elem = len(f32_bytes) // 4
        if ref_map is not None:
            ref_dim, ref_dlen = ref_map[name]
            if len(f32_bytes) != ref_dlen:
                skipped_len += 1
                continue
            target_dim = ref_dim
        else:
            # ada_ln stored [1,N] -> the runtime needs [1,1,N]; rebuild the dim blob
            # with the same width (identical to the QK-norm restore).
            ndim = len(bytes(s_dim)) // 4
            target_dim = struct.pack("<%di" % ndim, *([1, 1, n_elem] + [0] * (ndim - 3)))
        # idempotent: skip if already F32 with the intended dim
        cur.execute("SELECT datatype, dim FROM tensors WHERE name=?", (name,))
        row = cur.fetchone()
        if row and row[0] == DT_F32 and bytes(row[1]) == target_dim:
            already += 1
            continue
        if args.dry_run:
            restored += 1
            continue
        cur.execute(
            "UPDATE tensors SET datatype=?, dim=?, data=? WHERE name=?",
            (DT_F32, sqlite3.Binary(target_dim), sqlite3.Binary(f32_bytes), name),
        )
        restored += 1
        pending += 1
        if pending >= 256:
            con.commit(); pending = 0
    if not args.dry_run:
        con.commit()
    con.close()

    print("\n== summary ==")
    print(f"restored to F32    : {restored}")
    print(f"already F32        : {already}")
    print(f"skipped (len diff) : {skipped_len}")
    print(f"skipped (bad type) : {skipped_badtype}")
    print(f"skipped (missing)  : {skipped_missing}")
    print("done." + (" (dry-run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
