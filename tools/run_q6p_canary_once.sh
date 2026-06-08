#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT/.venv/bin/python"

MODEL_KEY="10_e_v1_bf16_regen_0_q6p.ckpt"
HOST="127.0.0.1:7861"
TIMEOUT_SEC=90
MAX_RESPONSES=10
TAG="$(date +%Y%m%d_%H%M%S)"
SOFT_FAIL=0

WIDTH=256
HEIGHT=256
STEPS=4
SAMPLER=17
GUIDANCE=1.0
SHIFT=3.0
NUM_FRAMES=9
FPS_ID=5
SEED=4242

PROMPT="a cinematic shot of a red sports car driving on a mountain road at sunset, detailed, realistic"
NEG_PROMPT="blurry, distorted, low quality, artifacts"

usage() {
  cat <<'EOF'
Usage:
  tools/run_q6p_canary_once.sh [options]

Options:
  --model <file>            Model key from dt-models directory.
  --host <host:port>        gRPC host (default: 127.0.0.1:7861).
  --timeout-sec <n>         timeout(1) seconds for generate-raw (default: 90).
  --max-responses <n>       Max streamed responses before client exits (default: 10).
  --tag <value>             Output folder tag (default: current timestamp).
  --soft-fail               Always exit 0; still prints RESULT=FAIL on failures.
  -h, --help                Show this help.

Outputs:
  output/q6p_canary_<tag>/{config.bin,client.log,server.log}
EOF
}

if [[ $# -gt 0 ]]; then
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model)
        MODEL_KEY="${2:-}"
        shift 2
        ;;
      --host)
        HOST="${2:-}"
        shift 2
        ;;
      --timeout-sec)
        TIMEOUT_SEC="${2:-}"
        shift 2
        ;;
      --max-responses)
        MAX_RESPONSES="${2:-}"
        shift 2
        ;;
      --tag)
        TAG="${2:-}"
        shift 2
        ;;
      --soft-fail)
        SOFT_FAIL=1
        shift
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
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "error: missing python env at $PYTHON_BIN" >&2
  exit 1
fi

if [[ "$TIMEOUT_SEC" -lt 1 ]]; then
  echo "error: --timeout-sec must be >= 1" >&2
  exit 1
fi

if [[ "$MAX_RESPONSES" -lt 1 ]]; then
  echo "error: --max-responses must be >= 1" >&2
  exit 1
fi

if ! command -v timeout >/dev/null 2>&1; then
  echo "error: timeout command not found" >&2
  exit 1
fi

WORK_DIR="$ROOT/output/q6p_canary_${TAG}"
SERVER_LOG="$WORK_DIR/server.log"
CLIENT_LOG="$WORK_DIR/client.log"
CONFIG_BIN="$WORK_DIR/config.bin"
mkdir -p "$WORK_DIR"

echo "== q6p canary once =="
echo "model=$MODEL_KEY"
echo "host=$HOST"
echo "timeout_sec=$TIMEOUT_SEC"
echo "max_responses=$MAX_RESPONSES"
echo "work_dir=$WORK_DIR"

pkill -9 -f 'gRPCServerCLI --address 127.0.0.1 --port 7861' || true
pkill -9 -f 'dt_api_client.py .*generate-raw' || true

nohup drawthings-grpc --address 127.0.0.1 --port 7861 --gpu 0 --no-tls --model-browser --no-response-compression "$ROOT/dt-models" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "server_pid=$SERVER_PID"

READY=0
for _ in $(seq 1 200); do
  if "$PYTHON_BIN" "$ROOT/tools/dt_api_client.py" --host "$HOST" echo --name "q6p-canary" >/dev/null 2>&1; then
    READY=1
    break
  fi
done
echo "server_ready=$READY"
if [[ "$READY" != "1" ]]; then
  pkill -9 -f 'gRPCServerCLI --address 127.0.0.1 --port 7861' || true
  echo "RESULT=FAIL server did not become ready"
  exit 1
fi

"$PYTHON_BIN" "$ROOT/tools/dt_make_config.py" \
  --out "$CONFIG_BIN" \
  --model "$MODEL_KEY" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --steps "$STEPS" \
  --sampler "$SAMPLER" \
  --guidance-scale "$GUIDANCE" \
  --shift "$SHIFT" \
  --hires-fix false \
  --num-frames "$NUM_FRAMES" \
  --fps-id "$FPS_ID" \
  --seed "$SEED"

set +e
timeout -k 10s "${TIMEOUT_SEC}s" \
  "$PYTHON_BIN" "$ROOT/tools/dt_api_client.py" \
    --host "$HOST" \
    --max-recv-bytes 134217728 \
    generate-raw \
    --config-bin "$CONFIG_BIN" \
    --prompt "$PROMPT" \
    --negative-prompt "$NEG_PROMPT" \
    --output-dir "$WORK_DIR" \
    --chunked \
    --save-preview \
    --max-responses "$MAX_RESPONSES" > "$CLIENT_LOG" 2>&1
CANARY_RC=$?
timeout -k 5s 15s \
  "$PYTHON_BIN" "$ROOT/tools/dt_api_client.py" --host "$HOST" echo --name "q6p-canary-post" >/dev/null 2>&1
POST_ECHO_RC=$?
set -e

pkill -9 -f 'gRPCServerCLI --address 127.0.0.1 --port 7861' || true
pkill -9 -f 'dt_api_client.py .*generate-raw' || true

echo "canary_rc=$CANARY_RC"
echo "post_echo_rc=$POST_ECHO_RC"
echo "client_log=$CLIENT_LOG"
echo "server_log=$SERVER_LOG"
echo "--- client.log (tail) ---"
tail -n 120 "$CLIENT_LOG" || true
echo "--- server.log (tail) ---"
tail -n 120 "$SERVER_LOG" || true

if [[ "$CANARY_RC" == "124" ]]; then
  echo "RESULT=FAIL canary timed out (${TIMEOUT_SEC}s)"
  if [[ "$SOFT_FAIL" == "1" ]]; then
    exit 0
  fi
  exit 1
fi

if [[ "$CANARY_RC" != "0" ]]; then
  echo "RESULT=FAIL canary rc=$CANARY_RC"
  if [[ "$SOFT_FAIL" == "1" ]]; then
    exit 0
  fi
  exit 1
fi

if ! grep -q 'response #1' "$CLIENT_LOG"; then
  echo "RESULT=FAIL no streamed responses observed"
  if [[ "$SOFT_FAIL" == "1" ]]; then
    exit 0
  fi
  exit 1
fi

echo "RESULT=PASS"
