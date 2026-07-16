#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${DT_HOST:-127.0.0.1:7861}"
MODEL="${DT_MODEL:-ltx_2.3_22b_distilled_f16.ckpt}"
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
ALLOW_PREVIEW_FALLBACK="${DT_ALLOW_PREVIEW_FALLBACK:-1}"
LORA_ENV_RAW="${DT_LORA:-}"

HAS_DT_SHIFT=0
if [[ -n "${DT_SHIFT+x}" ]]; then
  HAS_DT_SHIFT=1
fi

HAS_DT_GUIDANCE=0
if [[ -n "${DT_GUIDANCE+x}" ]]; then
  HAS_DT_GUIDANCE=1
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

is_ltx23_model() {
  local model_name="$1"
  local custom_json="${ROOT_DIR}/dt-models/custom.json"

  if [[ "${model_name}" == ltx_2.3_* ]]; then
    return 0
  fi

  if [[ ! -f "${custom_json}" ]]; then
    return 1
  fi

  "${PYTHON_BIN}" - "${model_name}" "${custom_json}" <<'PY' >/dev/null 2>&1
import json
import sys

model_name = sys.argv[1]
json_path = sys.argv[2]

try:
    payload = json.load(open(json_path, "r", encoding="utf-8"))
except Exception:
    sys.exit(1)

if not isinstance(payload, list):
    sys.exit(1)

for entry in payload:
    if not isinstance(entry, dict):
        continue
    if entry.get("file") != model_name and entry.get("name") != model_name:
        continue

    version = str(entry.get("version", "")).replace("_", ".").lower()
    if version == "ltx2.3":
        sys.exit(0)

sys.exit(1)
PY
}

resolve_custom_model_file() {
  local model_name="$1"
  local custom_json="${ROOT_DIR}/dt-models/custom.json"

  if [[ -z "${model_name}" || ! -f "${custom_json}" ]]; then
  echo "${model_name}"
  return 0
  fi

  "${PYTHON_BIN}" - "${model_name}" "${custom_json}" <<'PY'
import json
import sys

model_name = sys.argv[1]
json_path = sys.argv[2]

try:
  payload = json.load(open(json_path, "r", encoding="utf-8"))
except Exception:
  print(model_name)
  sys.exit(0)

if not isinstance(payload, list):
  print(model_name)
  sys.exit(0)

for entry in payload:
  if not isinstance(entry, dict):
    continue
  if str(entry.get("name", "")).strip() != model_name:
    continue

  model_file = str(entry.get("file", "")).strip()
  if model_file:
    print(model_file)
    sys.exit(0)

print(model_name)
PY
}

get_custom_model_numeric_field() {
  local model_name="$1"
  local field_name="$2"
  local custom_json="${ROOT_DIR}/dt-models/custom.json"

  if [[ ! -f "${custom_json}" ]]; then
  return 1
  fi

  "${PYTHON_BIN}" - "${model_name}" "${custom_json}" "${field_name}" <<'PY'
import json
import sys

model_name = sys.argv[1]
json_path = sys.argv[2]
field_name = sys.argv[3]

try:
  payload = json.load(open(json_path, "r", encoding="utf-8"))
except Exception:
  sys.exit(1)

if not isinstance(payload, list):
  sys.exit(1)

for entry in payload:
  if not isinstance(entry, dict):
    continue
  if entry.get("file") != model_name and entry.get("name") != model_name:
    continue

  value = entry.get(field_name)
  if isinstance(value, (int, float)):
    print(value)
    sys.exit(0)
  if isinstance(value, str):
    text = value.strip()
    if not text:
      sys.exit(1)
    try:
      float(text)
    except ValueError:
      sys.exit(1)
    print(text)
    sys.exit(0)

sys.exit(1)
PY
}

validate_custom_model_dependencies() {
  local model_name="$1"
  local custom_json="${ROOT_DIR}/dt-models/custom.json"
  local model_dir="${ROOT_DIR}/dt-models"

  if [[ -z "${model_name}" || ! -f "${custom_json}" ]]; then
  return 0
  fi

  "${PYTHON_BIN}" - "${model_name}" "${custom_json}" "${model_dir}" <<'PY'
import json
import sys
from pathlib import Path

model_name = sys.argv[1]
json_path = Path(sys.argv[2])
model_dir = Path(sys.argv[3])

try:
  payload = json.loads(json_path.read_text(encoding="utf-8"))
except Exception:
  sys.exit(0)

if not isinstance(payload, list):
  sys.exit(0)

entry = None
for candidate in payload:
  if not isinstance(candidate, dict):
    continue
  if candidate.get("name") == model_name or candidate.get("file") == model_name:
    entry = candidate
    break

if entry is None:
  sys.exit(0)

missing = []
for field in ("file", "text_encoder", "clip_encoder", "autoencoder"):
  value = entry.get(field)
  if not isinstance(value, str):
    continue
  value = value.strip()
  if not value:
    continue
  if not (model_dir / value).is_file():
    missing.append((field, value))

if not missing:
  sys.exit(0)

print(f"Model '{model_name}' has missing custom.json dependencies:")
for field, value in missing:
  print(f"  - {field}: {value}")
print("Update dt-models/custom.json or restore the missing files before running generation.")
sys.exit(1)
PY
}

ltx23_upscalers_present() {
  local model_dir="${ROOT_DIR}/dt-models"
  local missing=0

  for upscaler in \
    "ltx_2.3_spatial_upscaler_x2_1.1_f16.ckpt" \
    "ltx_2.3_spatial_upscaler_x1.5_f16.ckpt"
  do
    if [[ ! -f "${model_dir}/${upscaler}" ]]; then
      missing=1
    fi
  done

  if [[ "${missing}" == "1" ]]; then
    return 1
  fi
  return 0
}

IS_LTX23_MODEL=0
MODEL_EFFECTIVE="$(resolve_custom_model_file "${MODEL}")"
if [[ "${MODEL_EFFECTIVE}" != "${MODEL}" ]]; then
  echo "Resolved custom alias for profile checks: ${MODEL} -> ${MODEL_EFFECTIVE}"
fi

if ! validate_custom_model_dependencies "${MODEL}"; then
  exit 1
fi

if is_ltx23_model "${MODEL_EFFECTIVE}"; then
  IS_LTX23_MODEL=1
fi

# LTX-2.3 profiles use model-specific defaults in Draw Things presets.
if [[ "${IS_LTX23_MODEL}" == "1" ]]; then
  if [[ "${HAS_DT_GUIDANCE}" == "0" ]]; then
    model_default_scale="$(get_custom_model_numeric_field "${MODEL}" "default_scale" || true)"
    if [[ "${MODEL_EFFECTIVE}" == ltx_2.3_*distilled* ]]; then
      if [[ -n "${model_default_scale}" && "${model_default_scale}" != "1" && "${model_default_scale}" != "1.0" ]]; then
        echo "Ignoring custom default guidance ${model_default_scale} for distilled model compatibility."
      fi
    elif [[ -n "${model_default_scale}" ]]; then
      GUIDANCE="${model_default_scale}"
      echo "Using model default guidance scale ${GUIDANCE} from dt-models/custom.json"
    fi
  fi

  if [[ "${HAS_DT_SHIFT}" == "0" ]]; then
    SHIFT="5.0"
  fi
  if [[ "${HAS_DT_HIRES_FIX}" == "0" ]]; then
    if ltx23_upscalers_present; then
      HIRES_FIX="true"
    else
      HIRES_FIX="false"
      echo "LTX-2.3 spatial upscalers not found; defaulting hires-fix to false."
      echo "Install ltx_2.3_spatial_upscaler_x2_1.1_f16.ckpt and ltx_2.3_spatial_upscaler_x1.5_f16.ckpt to enable hires-fix."
    fi
  elif [[ "${HIRES_FIX}" == "true" ]] && ! ltx23_upscalers_present; then
    echo "Warning: hires-fix is enabled but required LTX-2.3 spatial upscalers are missing."
    echo "Generation may fail before final image decode."
  fi
fi

# LTX-2.3 distilled checkpoints are sensitive to guidance / step settings.
if [[ "${MODEL_EFFECTIVE}" == ltx_2.3_*distilled* ]]; then
  if [[ "${GUIDANCE}" != "1" && "${GUIDANCE}" != "1.0" ]]; then
    if [[ "${DT_ALLOW_NONSTANDARD_GUIDANCE:-0}" == "1" ]]; then
      echo "Warning: using nonstandard guidance ${GUIDANCE} for ${MODEL_EFFECTIVE}"
    else
      echo "Adjusting guidance from ${GUIDANCE} to 1.0 for ${MODEL_EFFECTIVE}"
      GUIDANCE="1.0"
    fi
  fi

  if (( STEPS < 8 )); then
    if [[ "${DT_ALLOW_LOW_STEPS:-0}" == "1" ]]; then
      echo "Warning: using low steps ${STEPS} for ${MODEL_EFFECTIVE} (recommended: 8+)"
    else
      echo "Adjusting steps from ${STEPS} to 8 for ${MODEL_EFFECTIVE}"
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

LORA_ARGS=()
if [[ -n "${LORA_ENV_RAW}" ]]; then
  IFS=',' read -r -a LORA_TOKENS <<< "${LORA_ENV_RAW}"
  for token in "${LORA_TOKENS[@]}"; do
    spec="${token//[[:space:]]/}"
    if [[ -z "${spec}" ]]; then
      continue
    fi
    LORA_ARGS+=(--lora "${spec}")
  done
fi

if [[ "${#LORA_ARGS[@]}" -gt 0 ]]; then
  echo "LoRAs: ${LORA_ENV_RAW}"
fi

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
  --seed "${SEED}" \
  "${LORA_ARGS[@]}"

CONFIG_ELAPSED=$(( $(now_epoch) - CONFIG_START ))
print_timer "config" "${CONFIG_ELAPSED}"

echo "Phase 2/3: generating tensors via Draw Things gRPC..."
GEN_START="$(now_epoch)"

CLIENT_CMD=(
  "${PYTHON_BIN}" "${ROOT_DIR}/tools/dt_api_client.py"
  --host "${HOST}"
  --max-recv-bytes 134217728
  generate-raw
  --config-bin "${OUT_DIR}/config.bin"
  --prompt "${PROMPT}"
  --negative-prompt "${NEG_PROMPT}"
  --output-dir "${OUT_DIR}"
  --chunked
)

if [[ "${ALLOW_PREVIEW_FALLBACK}" == "1" ]]; then
  CLIENT_CMD+=(--save-preview)
fi

"${CLIENT_CMD[@]}"

GEN_ELAPSED=$(( $(now_epoch) - GEN_START ))
print_timer "generation" "${GEN_ELAPSED}"

AUDIO_BIN="$(ls -1 "${OUT_DIR}"/audio_*.bin 2>/dev/null | tail -n1 || true)"
mapfile -t IMAGE_BINS < <(ls -1 "${OUT_DIR}"/image_*.bin 2>/dev/null | sort || true)
mapfile -t PREVIEW_FILES < <(ls -1 "${OUT_DIR}"/preview_* 2>/dev/null | sort || true)

if [[ "${#IMAGE_BINS[@]}" -eq 0 ]]; then
  if [[ "${TEST_ONE_FRAME}" == "1" && "${ALLOW_PREVIEW_FALLBACK}" == "1" && "${#PREVIEW_FILES[@]}" -gt 0 ]]; then
    last_preview_index=$(( ${#PREVIEW_FILES[@]} - 1 ))
    latest_preview="${PREVIEW_FILES[$last_preview_index]}"

    PREVIEW_CONVERT_CMD=(
      "${PYTHON_BIN}" "${ROOT_DIR}/tools/dt_tensor_to_playable.py"
      --image-bin "${latest_preview}"
      --out-dir "${OUT_DIR}"
      --base-name playable
      --gif-fps "${FPS_ID}"
      --mp4-fps "${FPS_ID}"
      --gif-seconds "${ONE_FRAME_SECONDS}"
      --mp4-seconds "${ONE_FRAME_SECONDS}"
    )

    echo "No generated image tensor found; converting latest preview frame fallback."
    if "${PREVIEW_CONVERT_CMD[@]}"; then
      TOTAL_ELAPSED=$(( $(now_epoch) - TOTAL_START ))
      if [[ -f "${OUT_DIR}/playable.png" ]]; then
        echo "Preview image: ${OUT_DIR}/playable.png"
      fi
      if [[ -f "${OUT_DIR}/playable.mp4" ]]; then
        echo "Preview video: ${OUT_DIR}/playable.mp4"
      fi
      print_timer "total" "${TOTAL_ELAPSED}"
      echo "Output folder: ${OUT_DIR}"
      exit 0
    fi

    echo "Preview fallback conversion failed for ${latest_preview}."
  fi

  if [[ "${#PREVIEW_FILES[@]}" -gt 0 ]]; then
    echo "No generated image tensor found in ${OUT_DIR}."
    echo "Preview frames were received (${#PREVIEW_FILES[@]}), but no final tensor payload was returned."
  else
    echo "No generated image tensor found in ${OUT_DIR}."
  fi
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
