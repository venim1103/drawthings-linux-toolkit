#!/usr/bin/env bash
set -euo pipefail

# dt_test_distilled_lora.sh distilled-LoRA preset for LTX-2.3
#
# Usage:
#   bash tools/dt_test_distilled_lora.sh [MODEL_ALIAS] [LORA_SPEC] [FRAMES]
#
# Defaults:
#   MODEL_ALIAS = official_q6p_via_custom
#   LORA_SPEC   = ltx-2.3-22b-distilled-lora-1.1_fro90_ceil52_condsafe:1.0
#   FRAMES      = 9
#
# Preset rationale:
#   Distilled acceleration LoRAs are schedule-sensitive. This preset uses
#   TCDTrailing with low step count to avoid divergence observed under the
#   default sampler path.
#
# IMPORTANT — frame count:
#   LTX-2.3 uses 8x temporal compression: latentFrames = ((numFrames - 1) / 8) + 1.
#   numFrames MUST be of the form 8k+1 (9, 17, 25, ...). Non-conforming values
#   like 5 produce an invalid temporal latent size -> NaN / cuDNN crash on the
#   FIRST sampling step (no final frame). Do NOT set FRAMES to 5. Use 9+.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_ALIAS="${1:-official_q6p_via_custom}"
LORA_SPEC="${2:-ltx-2.3-22b-distilled-lora-1.1_fro90_ceil52_condsafe:1.0}"
FRAMES="${3:-9}"

# Guard: LTX-2.3 requires numFrames == 8k+1 (>=9 for real video).
if (( (FRAMES - 1) % 8 != 0 )); then
  echo "error: FRAMES=$FRAMES is invalid for LTX-2.3 (must be 8k+1: 9, 17, 25, ...)." >&2
  echo "       numFrames not of the form 8k+1 diverges to NaN / crashes on step 1." >&2
  exit 1
fi

echo "== distilled lora preset =="
echo "  model  : $MODEL_ALIAS"
echo "  lora   : $LORA_SPEC"
echo "  steps  : 2"
echo "  sampler: 19 (TCDTrailing)"
echo "  frames : $FRAMES"

bash "$ROOT/tools/dt_test.sh" "$MODEL_ALIAS" \
  --lora "$LORA_SPEC" \
  --steps 2 \
  --sampler 19 \
  --frames "$FRAMES" \
  --timeout-sec 2700
