#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${DT_PYTHON:-}" ]]; then
  PYTHON_BIN="${DT_PYTHON}"
elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT}/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi

HOST="${DT_HOST:-127.0.0.1:7861}"
SERVER_CMD="${DT_SERVER_CMD:-drawthings-grpc}"
MODEL_NAME="10_e_v1"
ALIAS_FILE="${ROOT}/dt-models/10_e_v1_bf16_regen_0_q6p.ckpt"
CANDIDATE_FILE=""

WIDTH=256
HEIGHT=256
STEPS=4
SAMPLER=19
GUIDANCE=1.0
SHIFT=5.0
NUM_FRAMES=1
FPS_ID=5
SEED=4242
MAX_RESPONSES=1
CANARY_TIMEOUT_SEC="${DT_CANARY_TIMEOUT_SEC:-180}"

PROMPT="a cinematic shot of a red sports car driving on a mountain road at sunset, detailed, realistic"
NEG_PROMPT="blurry, distorted, low quality, artifacts"

RESTART_SERVER=1
TAG="$(date +%Y%m%d_%H%M%S)"
WORK_DIR=""
CONFIG_BIN=""
CANARY_LOG=""
SERVER_LOG=""

usage() {
  cat <<'EOF'
Safely activate a candidate checkpoint behind an alias with canary + rollback.

Usage:
  tools/dt_safe_activate_model.sh --candidate <ckpt_path> [options]

Required:
  --candidate <path>          Candidate ckpt to activate behind alias file path.

Options:
  --alias-file <path>         Alias file path to switch (default: dt-models/10_e_v1_bf16_regen_0_q6p.ckpt).
  --model-name <name>         Model name passed to dt_make_config.py (default: 10_e_v1).
  --host <host:port>          gRPC host (default: 127.0.0.1:7861).
  --server-cmd <command>      Server start command (default: drawthings-grpc).
  --no-restart-server         Reuse currently running server instead of restart.
  --width <n>                 Canary width, multiple of 64 (default: 256).
  --height <n>                Canary height, multiple of 64 (default: 256).
  --steps <n>                 Canary steps (default: 4).
  --sampler <n>               Sampler enum id (default: 19).
  --guidance <v>              Guidance scale (default: 1.0).
  --shift <v>                 Shift (default: 5.0).
  --num-frames <n>            Frames (default: 1).
  --fps-id <n>                FPS id (default: 5).
  --seed <n>                  Seed (default: 4242).
  --max-responses <n>         Early stop response cap (default: 1).
  --canary-timeout-sec <n>    Timeout for canary request in seconds (default: 180, 0 disables).
  --prompt <text>             Prompt text.
  --negative-prompt <text>    Negative prompt text.
  --tag <value>               Artifact suffix tag.
  -h, --help                  Show this help text.

Behavior:
1) Captures previous alias symlink target.
2) Stops current server (if requested).
3) Switches alias symlink to candidate and starts server.
4) Runs bounded canary generation.
5) On failure, restores previous alias symlink and restarts server.
6) On success, keeps candidate active.

Notes:
- Alias file must already be a symlink.
- This guard cannot prevent every crash forever, but it blocks promoting obvious crashers.
EOF
}

abs_path() {
  local path="$1"
  if [[ "$path" == /* ]]; then
    echo "$path"
  else
    echo "${ROOT}/${path}"
  fi
}

is_pos_int() {
  local value="$1"
  [[ "$value" =~ ^[0-9]+$ ]] && (( value > 0 ))
}

is_multiple_64() {
  local value="$1"
  [[ "$value" =~ ^[0-9]+$ ]] && (( value >= 64 )) && (( value % 64 == 0 ))
}

server_pgrep() {
  pgrep -f "gRPCServerCLI.*--port 7861" >/dev/null 2>&1
}

kill_server() {
  pkill -9 -f "gRPCServerCLI.*--port 7861" >/dev/null 2>&1 || true
  local i
  for i in $(seq 1 20); do
    if ! server_pgrep; then
      return 0
    fi
    sleep 1
  done
  return 1
}

start_server() {
  if ! command -v "${SERVER_CMD}" >/dev/null 2>&1; then
    echo "error: server command not found on PATH: ${SERVER_CMD}" >&2
    return 1
  fi

  mkdir -p "$(dirname "${SERVER_LOG}")"
  nohup "${SERVER_CMD}" >"${SERVER_LOG}" 2>&1 &
}

wait_for_health() {
  local name="$1"
  local i
  for i in $(seq 1 90); do
    if "${PYTHON_BIN}" "${ROOT}/tools/dt_api_client.py" --host "${HOST}" echo --name "${name}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

run_canary() {
  mkdir -p "${WORK_DIR}"

  "${PYTHON_BIN}" "${ROOT}/tools/dt_make_config.py" \
    --out "${CONFIG_BIN}" \
    --model "${MODEL_NAME}" \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --steps "${STEPS}" \
    --sampler "${SAMPLER}" \
    --guidance-scale "${GUIDANCE}" \
    --shift "${SHIFT}" \
    --hires-fix false \
    --hires-fix-width 0 \
    --hires-fix-height 0 \
    --num-frames "${NUM_FRAMES}" \
    --fps-id "${FPS_ID}" \
    --seed "${SEED}"

  local -a canary_cmd=(
    "${PYTHON_BIN}" "${ROOT}/tools/dt_api_client.py"
    --host "${HOST}"
    --max-recv-bytes 134217728
    generate-raw
    --config-bin "${CONFIG_BIN}"
    --prompt "${PROMPT}"
    --negative-prompt "${NEG_PROMPT}"
    --output-dir "${WORK_DIR}"
    --chunked
    --save-preview
    --max-responses "${MAX_RESPONSES}"
  )

  set +e
  if (( CANARY_TIMEOUT_SEC > 0 )) && command -v timeout >/dev/null 2>&1; then
    timeout "${CANARY_TIMEOUT_SEC}" "${canary_cmd[@]}" 2>&1 | tee "${CANARY_LOG}"
  else
    "${canary_cmd[@]}" 2>&1 | tee "${CANARY_LOG}"
  fi
  local rc=${PIPESTATUS[0]}
  set -e

  if (( rc == 124 )); then
    echo "canary_timeout=true seconds=${CANARY_TIMEOUT_SEC}" >>"${CANARY_LOG}"
    return 1
  fi

  if (( rc != 0 )); then
    return 1
  fi

  if ! grep -q "current signpost: textEncoded" "${CANARY_LOG}"; then
    return 1
  fi

  if (( MAX_RESPONSES >= 2 )) && ! grep -q "current signpost: imageEncoded" "${CANARY_LOG}"; then
    return 1
  fi

  return 0
}

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --candidate)
      CANDIDATE_FILE="${2:-}"
      shift 2
      ;;
    --alias-file)
      ALIAS_FILE="${2:-}"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="${2:-}"
      shift 2
      ;;
    --host)
      HOST="${2:-}"
      shift 2
      ;;
    --server-cmd)
      SERVER_CMD="${2:-}"
      shift 2
      ;;
    --no-restart-server)
      RESTART_SERVER=0
      shift
      ;;
    --width)
      WIDTH="${2:-}"
      shift 2
      ;;
    --height)
      HEIGHT="${2:-}"
      shift 2
      ;;
    --steps)
      STEPS="${2:-}"
      shift 2
      ;;
    --sampler)
      SAMPLER="${2:-}"
      shift 2
      ;;
    --guidance)
      GUIDANCE="${2:-}"
      shift 2
      ;;
    --shift)
      SHIFT="${2:-}"
      shift 2
      ;;
    --num-frames)
      NUM_FRAMES="${2:-}"
      shift 2
      ;;
    --fps-id)
      FPS_ID="${2:-}"
      shift 2
      ;;
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --max-responses)
      MAX_RESPONSES="${2:-}"
      shift 2
      ;;
    --canary-timeout-sec)
      CANARY_TIMEOUT_SEC="${2:-}"
      shift 2
      ;;
    --prompt)
      PROMPT="${2:-}"
      shift 2
      ;;
    --negative-prompt)
      NEG_PROMPT="${2:-}"
      shift 2
      ;;
    --tag)
      TAG="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${CANDIDATE_FILE}" ]]; then
  echo "error: --candidate is required" >&2
  usage
  exit 1
fi

ALIAS_FILE="$(abs_path "${ALIAS_FILE}")"
CANDIDATE_FILE="$(abs_path "${CANDIDATE_FILE}")"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "error: python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${CANDIDATE_FILE}" ]]; then
  echo "error: candidate file does not exist: ${CANDIDATE_FILE}" >&2
  exit 1
fi

if [[ ! -L "${ALIAS_FILE}" ]]; then
  echo "error: alias file must be a symlink for safe rollback: ${ALIAS_FILE}" >&2
  exit 1
fi

if ! is_multiple_64 "${WIDTH}"; then
  echo "error: --width must be a positive multiple of 64" >&2
  exit 1
fi

if ! is_multiple_64 "${HEIGHT}"; then
  echo "error: --height must be a positive multiple of 64" >&2
  exit 1
fi

if ! is_pos_int "${STEPS}" || ! is_pos_int "${NUM_FRAMES}" || ! is_pos_int "${FPS_ID}" || ! is_pos_int "${MAX_RESPONSES}"; then
  echo "error: --steps/--num-frames/--fps-id/--max-responses must be positive integers" >&2
  exit 1
fi

if ! [[ "${CANARY_TIMEOUT_SEC}" =~ ^[0-9]+$ ]]; then
  echo "error: --canary-timeout-sec must be a non-negative integer" >&2
  exit 1
fi

PREV_TARGET="$(readlink "${ALIAS_FILE}")"
PREV_TARGET_ABS="$(readlink -f "${ALIAS_FILE}")"

WORK_DIR="${ROOT}/output/safe_activate_${MODEL_NAME}_${TAG}"
CONFIG_BIN="${WORK_DIR}/config.bin"
CANARY_LOG="${WORK_DIR}/canary.log"
SERVER_LOG="${WORK_DIR}/server.log"

echo "==> Safe activation start"
echo "    alias_file : ${ALIAS_FILE}"
echo "    prev_target: ${PREV_TARGET_ABS}"
echo "    candidate  : ${CANDIDATE_FILE}"
echo "    model_name : ${MODEL_NAME}"
echo "    work_dir   : ${WORK_DIR}"

if [[ "${RESTART_SERVER}" == "1" ]]; then
  echo "==> Stopping server before switch"
  if ! kill_server; then
    echo "error: failed to stop existing gRPCServerCLI process cleanly" >&2
    exit 1
  fi
fi

echo "==> Switching alias to candidate"
ln -sfn "${CANDIDATE_FILE}" "${ALIAS_FILE}"

if [[ "${RESTART_SERVER}" == "1" ]]; then
  echo "==> Starting server after switch"
  start_server
  if ! wait_for_health "safe-activate-post-switch"; then
    echo "error: server health check failed after switch" >&2
    ln -sfn "${PREV_TARGET}" "${ALIAS_FILE}"
    kill_server || true
    start_server || true
    wait_for_health "safe-activate-health-restore" || true
    exit 1
  fi
fi

echo "==> Running bounded canary"
if run_canary; then
  echo "==> Canary passed. Candidate remains active."
  echo "    canary_log: ${CANARY_LOG}"
  echo "    server_log: ${SERVER_LOG}"
  exit 0
fi

echo "==> Canary failed. Rolling back alias target." >&2
ln -sfn "${PREV_TARGET}" "${ALIAS_FILE}"

if [[ "${RESTART_SERVER}" == "1" ]]; then
  echo "==> Restarting server after rollback" >&2
  kill_server || true
  start_server
  wait_for_health "safe-activate-post-rollback" || true
fi

echo "error: activation rejected by canary; previous target restored" >&2
echo "       canary_log: ${CANARY_LOG}" >&2
echo "       server_log: ${SERVER_LOG}" >&2
exit 1
