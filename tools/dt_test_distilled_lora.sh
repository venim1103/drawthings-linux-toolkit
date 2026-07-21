#!/usr/bin/env bash
set -euo pipefail

# dt_test_distilled_lora.sh — distilled-LoRA preset for LTX-2.3
#
# Usage:
#   bash tools/dt_test_distilled_lora.sh [MODEL_ALIAS] [LORA_SPEC]
#
# Defaults:
#   MODEL_ALIAS = official_q6p_via_custom
#   LORA_SPEC   = ltx-2.3-22b-distilled-lora-1.1_fro90_ceil52_condsafe:1.0
#
# Preset rationale:
#   Distilled acceleration LoRAs are schedule-sensitive. This preset uses
#   TCDTrailing with low step count to avoid divergence observed under the
#   default sampler path.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_ALIAS="${1:-official_q6p_via_custom}"
LORA_SPEC="${2:-ltx-2.3-22b-distilled-lora-1.1_fro90_ceil52_condsafe:1.0}"

echo "== distilled lora preset =="
echo "  model  : $MODEL_ALIAS"
echo "  lora   : $LORA_SPEC"
echo "  steps  : 2"
echo "  sampler: 19 (TCDTrailing)"

bash "$ROOT/tools/dt_test.sh" "$MODEL_ALIAS" \
  --lora "$LORA_SPEC" \
  --steps 2 \
  --sampler 19 \
  --frames 5 \
  --timeout-sec 2700
