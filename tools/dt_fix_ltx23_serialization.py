#!/usr/bin/env python3
"""Fix LTX2.3 converter serialization policy on a converted checkpoint.

Root cause (see CUSTOM_MODEL_CONVERTER_FINDINGS_2026-07-09.md):
  The patched LTX2.3 importer writes raw tensors with the wrong RANK
  (e.g. ada_ln [1,N] instead of [1,1,N]; bias [N,1] instead of [N]) and keeps
  norm/ada_ln tensors in F16 while the runtime expects F32. It also tags tensors
  CPU/NCHW instead of GPU/NHWC.

This tool applies the architecture-defined serialization POLICY to OUR data,
using an official reference checkpoint purely as a shape/dtype TEMPLATE. It does
NOT copy any weight values from the reference:
  - reshape our tensor's dim metadata to the reference rank (element count is
    verified identical; data bytes are unchanged for contiguous reshape),
  - widen our F16 norm data to F32 where the reference stores F32 (converts OUR
    values, not the reference's),
  - optionally set type/format to the reference policy (metadata only; safe here
    because there are no 4D tensors, so no transpose is required).

Only tensors that actually differ are rewritten, so big weight tensors (which
already match) are left untouched -> fast, low disk usage.
"""

from __future__ import annotations

import argparse
import sqlite3
import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ccv datatype codes observed empirically in these checkpoints
DT_F16 = 131072  # 0x20000
DT_F32 = 16384  # 0x4000

DT_SIZE = {DT_F16: 2, DT_F32: 4}


def decode_dim(blob: bytes) -> List[int]:
    ints = struct.unpack("<%di" % (len(blob) // 4), blob)
    return [x for x in ints if x != 0]


def elem_count(dims: List[int]) -> int:
    n = 1
    for d in dims:
        n *= d
    return n


def load_reference(path: Path) -> Dict[str, Tuple[int, int, int, bytes]]:
    """name -> (type, format, datatype, dim_blob) for all readable rows."""
    con = sqlite3.connect(str(path))
    cur = con.cursor()
    cur.execute("SELECT name FROM tensors ORDER BY name")
    names = [r[0] for r in cur.fetchall()]
    ref: Dict[str, Tuple[int, int, int, bytes]] = {}
    for name in names:
        try:
            cur.execute(
                "SELECT type, format, datatype, dim FROM tensors WHERE name=?", (name,)
            )
            row = cur.fetchone()
            if row is None:
                continue
            ref[name] = (row[0], row[1], row[2], bytes(row[3]))
        except sqlite3.DataError:
            # oversized row (e.g. video_aggregate_embed) - skip; not in fix set
            continue
    con.close()
    return ref


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="checkpoint to fix in place")
    ap.add_argument("--reference", required=True, help="official ckpt used as shape/dtype template")
    ap.add_argument("--set-type-format", action="store_true",
                    help="also set type/format to reference policy (rewrites all rows)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    target = Path(args.file)
    reference = Path(args.reference)
    if not target.is_file():
        print(f"error: target not found: {target}", file=sys.stderr)
        return 2
    if not reference.is_file():
        print(f"error: reference not found: {reference}", file=sys.stderr)
        return 2

    print(f"== LTX2.3 serialization fix ==")
    print(f"target   : {target}")
    print(f"reference: {reference}")
    print(f"set_type_format={args.set_type_format} dry_run={args.dry_run}")

    ref = load_reference(reference)
    print(f"reference readable rows: {len(ref)}")

    con = sqlite3.connect(str(target))
    con.execute("PRAGMA journal_mode=DELETE")
    cur = con.cursor()

    # our tensor names (avoid full scan of data)
    cur.execute("SELECT name FROM tensors ORDER BY name")
    our_names = [r[0] for r in cur.fetchall()]

    n_dim = n_widen = n_typefmt = n_skip = n_elem_mismatch = 0
    pending = 0

    for name in our_names:
        rref = ref.get(name)
        if rref is None:
            continue
        r_type, r_fmt, r_dt, r_dim = rref
        try:
            cur.execute("SELECT type, format, datatype, dim FROM tensors WHERE name=?", (name,))
            o_type, o_fmt, o_dt, o_dim = cur.fetchone()
            o_dim = bytes(o_dim)
        except sqlite3.DataError:
            n_skip += 1
            continue

        need_dim = o_dim != r_dim
        need_widen = (o_dt == DT_F16 and r_dt == DT_F32)
        need_typefmt = args.set_type_format and (o_type != r_type or o_fmt != r_fmt)

        if not (need_dim or need_widen or need_typefmt):
            continue

        # verify element-count parity before any reshape
        if need_dim:
            if elem_count(decode_dim(o_dim)) != elem_count(decode_dim(r_dim)):
                n_elem_mismatch += 1
                continue

        sets = []
        params: list = []
        new_dt = o_dt

        if need_widen:
            n_widen += 1
        if need_dim:
            n_dim += 1
        if need_typefmt:
            n_typefmt += 1

        if args.dry_run:
            continue

        if need_widen:
            cur.execute("SELECT data FROM tensors WHERE name=?", (name,))
            data = bytes(cur.fetchone()[0])
            f32 = np.frombuffer(data, dtype=np.float16).astype(np.float32)
            sets.append("data=?")
            params.append(sqlite3.Binary(f32.tobytes()))
            sets.append("datatype=?")
            params.append(DT_F32)

        if need_dim:
            sets.append("dim=?")
            params.append(sqlite3.Binary(r_dim))

        if need_typefmt:
            sets.append("type=?")
            params.append(r_type)
            sets.append("format=?")
            params.append(r_fmt)

        params.append(name)
        cur.execute(f"UPDATE tensors SET {', '.join(sets)} WHERE name=?", params)
        pending += 1
        if pending >= args.batch:
            con.commit()
            pending = 0

    if not args.dry_run:
        con.commit()
    con.close()

    print("\n== summary ==")
    print(f"dim reshaped     : {n_dim}")
    print(f"widened F16->F32 : {n_widen}")
    print(f"type/format set  : {n_typefmt}")
    print(f"skipped oversized: {n_skip}")
    print(f"elem mismatch    : {n_elem_mismatch}")
    print("done." + (" (dry-run, no writes)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
