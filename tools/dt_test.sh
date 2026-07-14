#!/usr/bin/env bash
set -euo pipefail

# dt_test.sh — Step 3/3: render a test frame from a quantized custom model
# (GPU inference) and decode it to a viewable PNG.
#
#   bash tools/dt_test.sh <NAME> [options]
#
# Required:
#   <NAME>                Alias registered by dt_quantize (custom.json entry).
# Optional:
#   --prompt TEXT         Text prompt. Default: a red sports car at sunset.
#   --neg-prompt TEXT     Negative prompt.
#   --steps N             Sampling steps. Default: 8.
#   --width N             Default: 384.   --height N   Default: 384.
#   --seed N              Default: 4242.
#   --frames N            Frames to sample (video). Default: 9.
#   --timeout-sec N       Default: 1800.
#   -h, --help            Show this help.
#
# NOTE: GPU inference. On a laptop, run on AC power — inference on battery has
# triggered a Windows/WSL VIDEO_MEMORY_MANAGEMENT BSOD.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT/.venv/bin/python"
LD_PATH="${DRAWTHINGS_LD_LIBRARY_PATH:-/usr/local/swift/usr/lib/swift/linux:/usr/lib/wsl/lib:/usr/local/cuda/targets/x86_64-linux/lib}"

NAME=""
STEPS=8; WIDTH=384; HEIGHT=384; SEED=4242; FRAMES=9; TIMEOUT_SEC=1800
PROMPT=""; NEG_PROMPT=""

usage() { awk '/^#!/{next} /^#/{sub(/^# ?/,"");print;c=1;next} c{exit}' "$0"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt) PROMPT="${2:-}"; shift 2 ;;
    --neg-prompt) NEG_PROMPT="${2:-}"; shift 2 ;;
    --steps) STEPS="${2:-}"; shift 2 ;;
    --width) WIDTH="${2:-}"; shift 2 ;;
    --height) HEIGHT="${2:-}"; shift 2 ;;
    --seed) SEED="${2:-}"; shift 2 ;;
    --frames) FRAMES="${2:-}"; shift 2 ;;
    --timeout-sec) TIMEOUT_SEC="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "error: unknown option: $1" >&2; usage; exit 1 ;;
    *)
      if [[ -z "$NAME" ]]; then NAME="$1"; shift
      else echo "error: unexpected argument: $1" >&2; exit 1; fi ;;
  esac
done

if [[ -z "$NAME" ]]; then echo "error: <NAME> is required" >&2; usage; exit 1; fi

TAG="${NAME}_test_$(date +%Y%m%d_%H%M%S)"
echo "== dt_test =="
echo "  model : $NAME   ${WIDTH}x${HEIGHT}  steps=$STEPS  seed=$SEED  frames=$FRAMES"

echo "==> GPU render (tag=$TAG)"
env LD_LIBRARY_PATH="$LD_PATH" \
    ${PROMPT:+DT_TEST_PROMPT="$PROMPT"} \
    ${NEG_PROMPT:+DT_TEST_NEG_PROMPT="$NEG_PROMPT"} \
    DT_TEST_NUM_FRAMES="$FRAMES" \
  bash "$ROOT/tools/run_q6p_canary_once.sh" \
    --model "$NAME" --tag "$TAG" \
    --timeout-sec "$TIMEOUT_SEC" --max-responses 0 \
    --require-complete-stream --require-final-output \
    --width "$WIDTH" --height "$HEIGHT" --steps "$STEPS" --seed "$SEED" \
    --allow-missing-model --soft-fail 2>&1 | tail -8 || true

OUT="$ROOT/output/q6p_canary_${TAG}"
echo "==> Decode frames -> PNG ($OUT)"
"$PYTHON_BIN" - "$OUT" <<'PY'
import sys, struct, glob, os
import numpy as np
try:
    from PIL import Image
    have_pil = True
except Exception:
    have_pil = False
out = sys.argv[1]
imgs = sorted(glob.glob(os.path.join(out, "image_r*.bin")))
if not imgs:
    print("  RESULT=FAIL: no image payloads (model may have diverged or errored)")
    sys.exit(0)

def decode(p):
    b = open(p, "rb").read()
    h = struct.unpack("<17I", b[:68])
    return h[6], h[7], h[8], np.frombuffer(b[68:], dtype="<f2").astype(np.float32)

last = imgs[-1]
w, ht, ch, px = decode(last)
fin = px[np.isfinite(px)]
nan = int(np.isnan(px).sum())
std = float(fin.std()) if fin.size else 0.0
verdict = "COHERENT" if (nan == 0 and std > 0.15) else ("FLAT/GRAY" if std <= 0.15 else "DIVERGED(NaN)")
print(f"  images={len(imgs)}  {w}x{ht}x{ch}  std={std:.3f}  NaN={nan}  -> {verdict}")

if have_pil:
    arr = px.reshape(ht, w, ch)
    f = arr[np.isfinite(arr)]
    p1, p99 = np.percentile(f, [1, 99]) if f.size else (0.0, 1.0)
    if p1 >= -0.05 and p99 <= 1.05:
        s = arr * 255.0
    elif p1 >= -1.05 and p99 <= 1.05:
        s = (arr + 1.0) * 127.5
    else:
        s = (arr - p1) * (255.0 / (p99 - p1 if p99 > p1 else 1.0))
    png = last.replace(".bin", ".png")
    Image.fromarray(np.clip(np.nan_to_num(s), 0, 255).astype(np.uint8)).save(png)
    print(f"  wrote {png}")
else:
    print("  (install pillow in .venv to auto-save a PNG)")
PY

echo ""
echo "Done. Output: $OUT"
