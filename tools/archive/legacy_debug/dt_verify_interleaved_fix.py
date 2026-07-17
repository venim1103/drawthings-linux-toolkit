#!/usr/bin/env python3
"""Cheap gate for the interleaved-rotary converter fix (see CUSTOM_MODEL_CONVERTER_FINDINGS §17).

Compares a set of attention Q/K tensors between OUR converted f16 ckpt and the
official f16 ckpt. The fix is confirmed when the interleaved Q-bias magnitudes are
small (~ official) instead of the pre-fix garbage (zeros interleaved with 65024, the
largest normal F16).

Usage:
  python3 tools/dt_verify_interleaved_fix.py \
      --ours dt-models/ltx23_control_f16.ckpt \
      --official dt-models/ltx_2.3_22b_distilled_f16.ckpt

Exit code 0 => fix looks good; 1 => still garbage / mismatch.
"""
import argparse
import sqlite3
import sys

import numpy as np

# Tensors that exercise the interleaved:true path (must become sane after the fix)
INTERLEAVED = [
    "__dit__[t-x_q-0-1]",  # DiT self-attn Q bias
    "__text_video_connector__[t-to_q-0-1]",  # connector Q bias
]
# Non-interleaved norms: should already match (sanity that we compare same model)
NON_INTERLEAVED = [
    "__dit__[t-x_norm_q-0-0]",
    "__text_video_connector__[t-norm_q-0-0]",
]

# Pre-fix garbage fingerprint: F16 overflow to the largest normal half (65504/65024).
GARBAGE_ABS_THRESHOLD = 100.0


def read_tensor(path, name):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT datatype, length(data), data FROM tensors WHERE name=?", (name,)
        )
        row = cur.fetchone()
    except sqlite3.DataError:
        conn.close()
        return None
    conn.close()
    if row is None:
        return None
    dt, _ln, data = row
    data = bytes(data)
    if dt == 16384:  # F32
        return np.frombuffer(data, dtype=np.float32)
    if dt == 131072:  # F16
        return np.frombuffer(data, dtype=np.float16).astype(np.float32)
    if dt == 524288:  # BF16
        u16 = np.frombuffer(data, dtype=np.uint16).astype(np.uint32) << 16
        return u16.view(np.float32)
    return None


def describe(a):
    return f"n={a.size} min={a.min():.4f} max={a.max():.4f} mean={a.mean():.5f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--official", required=True)
    args = ap.parse_args()

    ok = True

    print("== INTERLEAVED Q/K (must be sane after fix) ==")
    for name in INTERLEAVED:
        ours = read_tensor(args.ours, name)
        off = read_tensor(args.official, name)
        if ours is None or off is None:
            print(f"  {name}: MISSING (ours={ours is not None} off={off is not None})")
            ok = False
            continue
        ours_max = float(np.abs(ours).max())
        off_max = float(np.abs(off).max())
        garbage = ours_max > GARBAGE_ABS_THRESHOLD
        # After the fix the magnitudes should be in the same ballpark as official.
        ratio = ours_max / off_max if off_max > 0 else float("inf")
        sane = (not garbage) and (ratio < 5.0)
        status = "OK" if sane else "GARBAGE"
        if not sane:
            ok = False
        print(f"  [{status}] {name}")
        print(f"      ours: {describe(ours)}")
        print(f"      off : {describe(off)}")
        print(f"      |max| ours={ours_max:.4f} off={off_max:.4f} ratio={ratio:.2f}")

    print("== NON-INTERLEAVED norms (same-model sanity) ==")
    for name in NON_INTERLEAVED:
        ours = read_tensor(args.ours, name)
        off = read_tensor(args.official, name)
        if ours is None or off is None:
            print(f"  {name}: MISSING")
            continue
        same_set = np.isclose(np.sort(ours), np.sort(off), atol=1e-2).mean()
        print(
            f"  {name}: ours[{describe(ours)}] off[{describe(off)}] "
            f"sorted-match={same_set*100:.1f}%"
        )

    print()
    if ok:
        print("RESULT=PASS  interleaved Q/K weights are sane -> proceed to requantize")
        return 0
    print("RESULT=FAIL  interleaved Q/K still garbage -> do NOT requantize yet")
    return 1


if __name__ == "__main__":
    sys.exit(main())
