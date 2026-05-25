#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${DT_HOST:-127.0.0.1:7861}"
MODEL="${DT_MODEL:-ltx_2.3_22b_distilled_1.1_q6p.ckpt}"
WIDTH="${DT_WIDTH:-384}"
HEIGHT="${DT_HEIGHT:-704}"
DT_ALLOW_LOW_STEPS="${DT_ALLOW_LOW_STEPS:-1}"
STEPS="${DT_STEPS:-2}"
NUM_FRAMES="${DT_NUM_FRAMES:-81}"
FPS_ID="${DT_FPS_ID:-5}"
SAMPLER="${DT_SAMPLER:-19}"
GUIDANCE="${DT_GUIDANCE:-1.0}"
SHIFT="${DT_SHIFT:-3.0}"
HIRES_FIX="${DT_HIRES_FIX:-false}"
HIRES_FIX_WIDTH="${DT_HIRES_FIX_WIDTH:-0}"
HIRES_FIX_HEIGHT="${DT_HIRES_FIX_HEIGHT:-0}"
SEED="${DT_SEED:--1}"
TEST_ONE_FRAME="${DT_TEST_ONE_FRAME:-0}"
ONE_FRAME_SECONDS="${DT_ONE_FRAME_SECONDS:-1.0}"
OUTPUT_ROOT="${DT_OUTPUT_ROOT:-${ROOT_DIR}/output}"
TOOLS_REQ_FILE="${ROOT_DIR}/requirements-drawthings-tools.txt"

HAS_DT_SHIFT=0
if [[ -n "${DT_SHIFT+x}" ]]; then
  HAS_DT_SHIFT=1
fi

HAS_DT_HIRES_FIX=0
if [[ -n "${DT_HIRES_FIX+x}" ]]; then
  HAS_DT_HIRES_FIX=1
fi

now_epoch() {
  date +%s
}

format_duration() {
  local total_secs="$1"
  local hours=$(( total_secs / 3600 ))
  local mins=$(( (total_secs % 3600) / 60 ))
  local secs=$(( total_secs % 60 ))

  if (( hours > 0 )); then
    printf "%dh %02dm %02ds" "${hours}" "${mins}" "${secs}"
  else
    printf "%dm %02ds" "${mins}" "${secs}"
  fi
}

print_timer() {
  local label="$1"
  local elapsed="$2"
  printf "[timer] %-10s %s\n" "${label}:" "$(format_duration "${elapsed}")"
}

if [[ -n "${DT_PYTHON:-}" ]]; then
  PYTHON_BIN="${DT_PYTHON}"
elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 \"prompt\" [negative_prompt]"
  exit 1
fi

PROMPT="$1"
NEG_PROMPT="${2:-blurry, noisy, distorted, low quality}"

if ! [[ "${STEPS}" =~ ^[0-9]+$ ]] || (( STEPS < 1 )); then
  echo "DT_STEPS must be a positive integer, got: ${STEPS}"
  exit 1
fi

if ! [[ "${NUM_FRAMES}" =~ ^[0-9]+$ ]] || (( NUM_FRAMES < 1 )); then
  echo "DT_NUM_FRAMES must be a positive integer, got: ${NUM_FRAMES}"
  exit 1
fi

if ! [[ "${FPS_ID}" =~ ^[0-9]+$ ]] || (( FPS_ID < 1 )); then
  echo "DT_FPS_ID must be a positive integer, got: ${FPS_ID}"
  exit 1
fi

if ! [[ "${SAMPLER}" =~ ^[0-9]+$ ]] || (( SAMPLER < 0 )); then
  echo "DT_SAMPLER must be a non-negative integer, got: ${SAMPLER}"
  exit 1
fi

if ! [[ "${WIDTH}" =~ ^[0-9]+$ ]] || (( WIDTH < 64 )) || (( WIDTH % 64 != 0 )); then
  echo "DT_WIDTH must be a positive multiple of 64, got: ${WIDTH}"
  exit 1
fi

if ! [[ "${HEIGHT}" =~ ^[0-9]+$ ]] || (( HEIGHT < 64 )) || (( HEIGHT % 64 != 0 )); then
  echo "DT_HEIGHT must be a positive multiple of 64, got: ${HEIGHT}"
  exit 1
fi

if [[ "${TEST_ONE_FRAME}" == "1" ]]; then
  NUM_FRAMES="1"
  echo "Testing mode enabled: forcing NUM_FRAMES=1"
fi

# LTX-2.3 profiles use model-specific defaults in Draw Things presets.
if [[ "${MODEL}" == ltx_2.3_* ]]; then
  if [[ "${HAS_DT_SHIFT}" == "0" ]]; then
    SHIFT="5.0"
  fi
  if [[ "${HAS_DT_HIRES_FIX}" == "0" ]]; then
    HIRES_FIX="true"
  fi
fi

# LTX-2.3 distilled checkpoints are sensitive to guidance / step settings.
if [[ "${MODEL}" == ltx_2.3_*distilled* ]]; then
  if [[ "${GUIDANCE}" != "1" && "${GUIDANCE}" != "1.0" ]]; then
    if [[ "${DT_ALLOW_NONSTANDARD_GUIDANCE:-0}" == "1" ]]; then
      echo "Warning: using nonstandard guidance ${GUIDANCE} for ${MODEL}"
    else
      echo "Adjusting guidance from ${GUIDANCE} to 1.0 for ${MODEL}"
      GUIDANCE="1.0"
    fi
  fi

  if (( STEPS < 8 )); then
    if [[ "${DT_ALLOW_LOW_STEPS:-0}" == "1" ]]; then
      echo "Warning: using low steps ${STEPS} for ${MODEL} (recommended: 8+)"
    else
      echo "Adjusting steps from ${STEPS} to 8 for ${MODEL}"
      STEPS="8"
    fi
  fi
fi

OUT_DIR="${OUTPUT_ROOT}/dt_video_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_DIR}"

TOTAL_START="$(now_epoch)"

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import grpc  # noqa: F401
import flatbuffers  # noqa: F401
import numpy  # noqa: F401
from PIL import Image  # noqa: F401
PY
then
  if [[ ! -f "${TOOLS_REQ_FILE}" ]]; then
    echo "Missing dependency file: ${TOOLS_REQ_FILE}"
    exit 1
  fi
  echo "Installing missing Python tool dependencies..."
  "${PYTHON_BIN}" -m pip install --upgrade pip
  "${PYTHON_BIN}" -m pip install -r "${TOOLS_REQ_FILE}"
fi

if ! "${PYTHON_BIN}" "${ROOT_DIR}/tools/dt_api_client.py" --host "${HOST}" echo --name "video-gen-check" >/dev/null 2>&1; then
  echo "Draw Things server is not reachable at ${HOST}."
  echo "Start it in another terminal with:"
  echo "drawthings-grpc --address 127.0.0.1 --port 7861 --gpu 0 --no-tls --model-browser --no-response-compression ${ROOT_DIR}/dt-models"
  exit 1
fi

echo "Phase 1/3: building generation config..."
echo "Config: model=${MODEL} sampler=${SAMPLER} steps=${STEPS} guidance=${GUIDANCE} shift=${SHIFT} hiresFix=${HIRES_FIX} frames=${NUM_FRAMES} size=${WIDTH}x${HEIGHT}"
CONFIG_START="$(now_epoch)"

"${PYTHON_BIN}" "${ROOT_DIR}/tools/dt_make_config.py" \
  --out "${OUT_DIR}/config.bin" \
  --model "${MODEL}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --steps "${STEPS}" \
  --sampler "${SAMPLER}" \
  --guidance-scale "${GUIDANCE}" \
  --shift "${SHIFT}" \
  --hires-fix "${HIRES_FIX}" \
  --hires-fix-width "${HIRES_FIX_WIDTH}" \
  --hires-fix-height "${HIRES_FIX_HEIGHT}" \
  --num-frames "${NUM_FRAMES}" \
  --fps-id "${FPS_ID}" \
  --seed "${SEED}"

CONFIG_ELAPSED=$(( $(now_epoch) - CONFIG_START ))
print_timer "config" "${CONFIG_ELAPSED}"

echo "Phase 2/3: generating tensors via Draw Things gRPC..."
GEN_START="$(now_epoch)"

"${PYTHON_BIN}" "${ROOT_DIR}/tools/dt_api_client.py" \
  --host "${HOST}" \
  --max-recv-bytes 134217728 \
  generate-raw \
  --config-bin "${OUT_DIR}/config.bin" \
  --prompt "${PROMPT}" \
  --negative-prompt "${NEG_PROMPT}" \
  --output-dir "${OUT_DIR}" \
  --chunked

GEN_ELAPSED=$(( $(now_epoch) - GEN_START ))
print_timer "generation" "${GEN_ELAPSED}"

AUDIO_BIN="$(ls -1 "${OUT_DIR}"/audio_*.bin 2>/dev/null | tail -n1 || true)"
mapfile -t IMAGE_BINS < <(ls -1 "${OUT_DIR}"/image_*.bin 2>/dev/null | sort || true)

if [[ "${#IMAGE_BINS[@]}" -eq 0 ]]; then
  echo "No generated image tensor found in ${OUT_DIR}."
  exit 1
fi

CONVERT_CMD=(
  "${PYTHON_BIN}" "${ROOT_DIR}/tools/dt_tensor_to_playable.py"
  --out-dir "${OUT_DIR}"
  --base-name playable
  --gif-fps "${FPS_ID}"
  --mp4-fps "${FPS_ID}"
)

for image_bin in "${IMAGE_BINS[@]}"; do
  CONVERT_CMD+=(--image-bin "${image_bin}")
done

if [[ -n "${AUDIO_BIN}" ]]; then
  CONVERT_CMD+=(--audio-bin "${AUDIO_BIN}")
fi

if [[ "${NUM_FRAMES}" == "1" ]]; then
  CONVERT_CMD+=(--gif-seconds "${ONE_FRAME_SECONDS}" --mp4-seconds "${ONE_FRAME_SECONDS}")
fi

echo "Phase 3/3: converting tensors to playable media..."
CONVERT_START="$(now_epoch)"

"${CONVERT_CMD[@]}"

CONVERT_ELAPSED=$(( $(now_epoch) - CONVERT_START ))
print_timer "convert" "${CONVERT_ELAPSED}"

TOTAL_ELAPSED=$(( $(now_epoch) - TOTAL_START ))

echo
if [[ -f "${OUT_DIR}/playable.mp4" ]]; then
  echo "Video: ${OUT_DIR}/playable.mp4"
fi
if [[ -f "${OUT_DIR}/playable.wav" ]]; then
  echo "Audio: ${OUT_DIR}/playable.wav"
fi
print_timer "total" "${TOTAL_ELAPSED}"
echo "Output folder: ${OUT_DIR}"
