#!/usr/bin/env python3
"""Write de-interleaved QK-norm (q_norm/k_norm) weights into a q6p as F32.

Context (CUSTOM_MODEL_CONVERTER_FINDINGS §18): the LTX2.3 converter de-interleaves
the attention Q/K *projections* but NOT their RMSNorm weights (q_norm/k_norm), because
those mappings lack `interleaved: true`. At runtime the norm is applied to the
already-de-interleaved q/k, so the norm weight must be de-interleaved too. Our q6p has
these norms palettized in raw (interleaved) order -> incoherent output.

This tool overwrites every QK-norm tensor in the q6p with the F32-widened value from a
source f16 that already has them de-interleaved (Draw Things' official f16). It is the
QK-norm analogue of dt_q6p_restore_f32_norms.py (which only covers tensors the official
q6p keeps at F32; the QK-norms are palettized there and thus skipped).

This validates the fix WITHOUT a 2h requantize. The permanent fix is in the converter
mapping (flag q_norm/k_norm interleaved).

Usage:
  python3 tools/dt_q6p_restore_qk_norms.py \
      --q6p dt-models/ltx23_control_q6p.ckpt \
      --f16-source dt-models/ltx_2.3_22b_distilled_f16.ckpt
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys

import numpy as np

DT_F32 = 16384
DT_F16 = 131072
QK_NORM_RE = re.compile(r"(norm_q|norm_k)-\d+-0\]$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q6p", required=True)
    ap.add_argument("--f16-source", required=True, help="f16 with DE-INTERLEAVED QK-norms (official)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # names in the q6p that are QK-norms
    con = sqlite3.connect(args.q6p)
    con.execute("PRAGMA journal_mode=DELETE")
    cur = con.cursor()
    cur.execute("SELECT name FROM tensors")
    qk_names = [r[0] for r in cur.fetchall() if QK_NORM_RE.search(r[0])]
    print(f"QK-norm tensors in q6p: {len(qk_names)}")

    src = sqlite3.connect(args.f16_source)
    scur = src.cursor()

    restored = missing = badtype = 0
    pending = 0
    for name in qk_names:
        scur.execute("SELECT datatype, dim, data FROM tensors WHERE name=?", (name,))
        row = scur.fetchone()
        if row is None:
            missing += 1
            continue
        s_dt, s_dim, s_data = row
        s_data = bytes(s_data)
        if s_dt == DT_F16:
            vals = np.frombuffer(s_data, dtype=np.float16).astype(np.float32)
        elif s_dt == DT_F32:
            vals = np.frombuffer(s_data, dtype=np.float32)
        else:
            badtype += 1
            continue
        f32_bytes = vals.astype("<f4").tobytes()
        if args.dry_run:
            restored += 1
            continue
        # dim blob from the source (encodes shape, dtype-independent); datatype F32.
        cur.execute(
            "UPDATE tensors SET datatype=?, dim=?, data=? WHERE name=?",
            (DT_F32, sqlite3.Binary(bytes(s_dim)), sqlite3.Binary(f32_bytes), name),
        )
        restored += 1
        pending += 1
        if pending >= 256:
            con.commit()
            pending = 0
    if not args.dry_run:
        con.commit()
    src.close()
    con.close()

    print("\n== summary ==")
    print(f"restored QK-norms to F32 : {restored}")
    print(f"missing in source        : {missing}")
    print(f"bad datatype in source   : {badtype}")
    print("done." + (" (dry-run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
