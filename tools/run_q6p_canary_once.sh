#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT/.venv/bin/python"
GRPC_BIN="${DRAWTHINGS_GRPC_BIN:-drawthings-grpc}"
PROTO_FILE="$ROOT/draw-things-community/Libraries/GRPC/Models/Sources/imageService/imageService.proto"
PROTO_CACHE_DIR="$ROOT/.cache/dt_proto"
BINDINGS_DIR="$ROOT/.cache/dt_flatbuf_py"
MODELS_DIR="$ROOT/dt-models"
DRY_RUN=0
IMAGE_MODE=0
REQUIRED_MODEL_FILES=()

MODEL_KEY="10_e_v1_bf16_regen_0_q6p.ckpt"
HOST="127.0.0.1:7861"
TIMEOUT_SEC=90
TIMEOUT_EXPLICIT=0
MAX_RESPONSES=10
MAX_RESPONSES_EXPLICIT=0
TAG="$(date +%Y%m%d_%H%M%S)"
SOFT_FAIL=0
FINAL_MODE=0
NO_TIMEOUT=0
REQUIRE_COMPLETE_STREAM=0
REQUIRE_FINAL_OUTPUT=0
ALLOW_MISSING_MODEL=0
SERVER_GPU=0
SERVER_CPU_OFFLOAD=0
SERVER_FREAD_PREFERRED=0
SERVER_NO_FLASH_ATTENTION=0
SERVER_WEIGHTS_CACHE=""
GPU_METRICS=0

WIDTH=256
HEIGHT=256
STEPS=4
SAMPLER=17
GUIDANCE=1.0
SHIFT=3.0
NUM_FRAMES="${DT_TEST_NUM_FRAMES:-9}"
FPS_ID=5
SEED=4242
LORA_SPECS=()

PROMPT="${DT_TEST_PROMPT:-a cinematic shot of a red sports car driving on a mountain road at sunset, detailed, realistic}"
NEG_PROMPT="${DT_TEST_NEG_PROMPT:-blurry, distorted, low quality, artifacts}"

usage() {
  cat <<'EOF'
Usage:
  tools/run_q6p_canary_once.sh [options]

Options:
  --model <file>            Model key from dt-models directory.
  --host <host:port>        gRPC host (default: 127.0.0.1:7861).
  --grpc-bin <path|name>    gRPC server launcher binary
                            (default: DRAWTHINGS_GRPC_BIN or drawthings-grpc).
  --proto <path>            Explicit imageService.proto for the selected server.
  --cache-dir <path>        Directory for generated protobuf bindings.
  --bindings-dir <path>     Generated FlatBuffer bindings for the selected server.
  --models-dir <path>       Model files and catalogs (default: dt-models).
                            Servers may rewrite checkpoint storage layouts;
                            use private copies for upgrade regression tests.
  --require-model-file <file>
                            Require a companion file in --models-dir; repeatable.
  --image-mode              Require one finite final image at the requested size,
                            no audio, and post-Echo; export image.png and metadata.
                            Uses one frame and disables resolution-dependent shift.
  --dry-run                 Generate config and print the server command only.
  --server-gpu <n>          GPU id passed to server launcher (default: 0).
  --server-cpu-offload      Pass --cpu-offload to server launcher.
  --server-fread-preferred  Pass --fread-preferred to server launcher.
  --server-no-flash-attention
                            Pass --no-flash-attention to server launcher.
  --server-weights-cache <gib>
                            Pass --weights-cache <gib> to server launcher.
  --gpu-metrics             Sample whole-device memory/utilization every 500 ms into
                            gpu.csv; requires nvidia-smi.
  --width <n>               Pixel width for request config (default: 256).
  --height <n>              Pixel height for request config (default: 256).
  --steps <n>               Sampling steps for request config (default: 4).
  --sampler <name|n>        Selected schema's sampler name or id (default: 17).
  --guidance-scale <n>      Text guidance (default: 1).
  --shift <n>               Sampling shift (default: 3).
  --prompt <text>           Override the generation prompt.
  --negative-prompt <text>  Override the negative prompt; an empty string is valid.
  --seed <n>                Seed for request config (default: 4242).
  --lora <file[:weight[:mode]]>
                            Repeatable LoRA spec passed into config.
                            mode: all/base/refiner or 0/1/2.
  --timeout-sec <n>         timeout(1) seconds for generate-raw (default: 90).
  --final-mode              Use longer final validation timeout (900s) unless
                            --timeout-sec was explicitly set.
  --no-timeout              Disable timeout wrapper for generate-raw.
  --max-responses <n>       Max streamed responses before client exits (default: 10).
                            Use 0 for unlimited.
  --require-complete-stream Require the stream to finish without early max-response stop.
                            If --max-responses is not explicitly set, this auto-sets
                            max responses to 0 (unlimited).
  --require-final-output    Require at least one non-preview output payload
                            (generated image or audio).
  --allow-missing-model     Allow missing file-like model keys (e.g. *.ckpt) and
                            continue. By default, missing file-like model keys
                            fail fast to prevent silent fallback confounds.
  --tag <value>             Output folder tag (default: current timestamp).
  --soft-fail               Always exit 0; still prints RESULT=FAIL on failures.
  -h, --help                Show this help.

Outputs:
  output/q6p_canary_<tag>/{config.bin,client.log,server.log}
  With --gpu-metrics: gpu.csv and gpu-monitor.log in the same directory.
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
      --grpc-bin)
        GRPC_BIN="${2:-}"
        shift 2
        ;;
      --proto)
        PROTO_FILE="${2:-}"
        shift 2
        ;;
      --cache-dir)
        PROTO_CACHE_DIR="${2:-}"
        shift 2
        ;;
      --bindings-dir)
        BINDINGS_DIR="${2:-}"
        shift 2
        ;;
      --models-dir)
        MODELS_DIR="${2:-}"
        shift 2
        ;;
      --require-model-file)
        REQUIRED_MODEL_FILES+=("${2:-}")
        shift 2
        ;;
      --image-mode)
        IMAGE_MODE=1
        shift
        ;;
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      --server-gpu)
        SERVER_GPU="${2:-}"
        shift 2
        ;;
      --server-cpu-offload)
        SERVER_CPU_OFFLOAD=1
        shift
        ;;
      --server-fread-preferred)
        SERVER_FREAD_PREFERRED=1
        shift
        ;;
      --server-no-flash-attention)
        SERVER_NO_FLASH_ATTENTION=1
        shift
        ;;
      --server-weights-cache)
        SERVER_WEIGHTS_CACHE="${2:-}"
        shift 2
        ;;
      --gpu-metrics)
        GPU_METRICS=1
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
      --guidance-scale)
        GUIDANCE="${2:-}"
        shift 2
        ;;
      --shift)
        SHIFT="${2:-}"
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
      --seed)
        SEED="${2:-}"
        shift 2
        ;;
      --lora)
        LORA_SPECS+=("${2:-}")
        shift 2
        ;;
      --timeout-sec)
        TIMEOUT_SEC="${2:-}"
        TIMEOUT_EXPLICIT=1
        shift 2
        ;;
      --final-mode)
        FINAL_MODE=1
        shift
        ;;
      --no-timeout)
        NO_TIMEOUT=1
        shift
        ;;
      --max-responses)
        MAX_RESPONSES="${2:-}"
        MAX_RESPONSES_EXPLICIT=1
        shift 2
        ;;
      --require-complete-stream)
        REQUIRE_COMPLETE_STREAM=1
        shift
        ;;
      --require-final-output)
        REQUIRE_FINAL_OUTPUT=1
        shift
        ;;
      --allow-missing-model)
        ALLOW_MISSING_MODEL=1
        shift
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

CONFIG_EXTRA_ARGS=()
if [[ "$IMAGE_MODE" == "1" ]]; then
  NUM_FRAMES=1
  REQUIRE_COMPLETE_STREAM=1
  REQUIRE_FINAL_OUTPUT=1
  CONFIG_EXTRA_ARGS=(--resolution-dependent-shift false)
fi

if [[ "$FINAL_MODE" == "1" && "$TIMEOUT_EXPLICIT" != "1" && "$NO_TIMEOUT" != "1" ]]; then
  TIMEOUT_SEC=900
fi

if [[ "$REQUIRE_COMPLETE_STREAM" == "1" && "$MAX_RESPONSES_EXPLICIT" != "1" ]]; then
  MAX_RESPONSES=0
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "error: missing python env at $PYTHON_BIN" >&2
  exit 1
fi

if [[ -z "$GRPC_BIN" ]]; then
  echo "error: --grpc-bin cannot be empty" >&2
  exit 1
fi

if [[ "$HOST" =~ ^(\[[^]]+\]|[^:[:space:]]+):([0-9]{1,5})$ && "$HOST" != *[[:space:]]* ]]; then
  SERVER_ADDRESS="${BASH_REMATCH[1]}"
  SERVER_PORT="$((10#${BASH_REMATCH[2]}))"
  SERVER_ADDRESS="${SERVER_ADDRESS#[}"
  SERVER_ADDRESS="${SERVER_ADDRESS%]}"
else
  echo "error: --host must be host:port or [IPv6]:port" >&2
  exit 1
fi
if [[ "$SERVER_PORT" -lt 1 || "$SERVER_PORT" -gt 65535 ]]; then
  echo "error: --host port must be between 1 and 65535" >&2
  exit 1
fi

if [[ ! -f "$PROTO_FILE" || ! -d "$BINDINGS_DIR" || -z "$PROTO_CACHE_DIR" ]]; then
  echo "error: selected schema or bindings are missing, or cache directory is empty" >&2
  exit 1
fi
CLIENT_SCHEMA_ARGS=(--proto "$PROTO_FILE" --cache-dir "$PROTO_CACHE_DIR")

if [[ ! -d "$MODELS_DIR" ]]; then
  echo "error: model directory not found: $MODELS_DIR" >&2
  exit 1
fi
MODELS_DIR="$(cd "$MODELS_DIR" && pwd)"

for required_file in "${REQUIRED_MODEL_FILES[@]}"; do
  if [[ -z "$required_file" || "$required_file" != "$(basename "$required_file")" || ! -f "$MODELS_DIR/$required_file" ]]; then
    echo "error: required companion file missing or invalid: $required_file" >&2
    exit 1
  fi
done

if [[ "$GRPC_BIN" == */* ]]; then
  if [[ ! -x "$GRPC_BIN" ]]; then
    echo "error: gRPC binary is not executable: $GRPC_BIN" >&2
    exit 1
  fi
else
  if ! command -v "$GRPC_BIN" >/dev/null 2>&1; then
    echo "error: gRPC binary not found on PATH: $GRPC_BIN" >&2
    exit 1
  fi
fi

if ! [[ "$SERVER_GPU" =~ ^[0-9]+$ ]]; then
  echo "error: --server-gpu must be a non-negative integer" >&2
  exit 1
fi

if [[ -n "$SERVER_WEIGHTS_CACHE" ]] && ! [[ "$SERVER_WEIGHTS_CACHE" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "error: --server-weights-cache must be a non-negative number" >&2
  exit 1
fi

if [[ "$NO_TIMEOUT" != "1" && "$TIMEOUT_SEC" -lt 1 ]]; then
  echo "error: --timeout-sec must be >= 1 unless --no-timeout is used" >&2
  exit 1
fi

if [[ "$MAX_RESPONSES" -lt 0 ]]; then
  echo "error: --max-responses must be >= 0" >&2
  exit 1
fi

if [[ "$ALLOW_MISSING_MODEL" != "1" ]]; then
  case "$MODEL_KEY" in
    *.ckpt|*.safetensors)
      MODEL_FILE="$MODELS_DIR/$MODEL_KEY"
      if [[ ! -f "$MODEL_FILE" ]]; then
        echo "error: model file not found: $MODEL_FILE" >&2
        echo "hint: use --allow-missing-model to permit fallback-resolution tests" >&2
        exit 1
      fi
      ;;
  esac
fi

resolve_custom_lora_file() {
  local lora_name="$1"
  local custom_lora_json="$MODELS_DIR/custom_lora.json"

  if [[ -z "$lora_name" || ! -f "$custom_lora_json" ]]; then
    echo "$lora_name"
    return 0
  fi

  "$PYTHON_BIN" - "$lora_name" "$custom_lora_json" <<'PY'
import json
import os
import sys

query = sys.argv[1].strip()
json_path = sys.argv[2]

query = os.path.basename(query)

try:
    payload = json.load(open(json_path, "r", encoding="utf-8"))
except Exception:
    print(query)
    sys.exit(0)

if not isinstance(payload, list):
    print(query)
    sys.exit(0)

for entry in payload:
    if not isinstance(entry, dict):
        continue
    name = str(entry.get("name", "")).strip()
    file_name = str(entry.get("file", "")).strip()
    if query == name and file_name:
        print(file_name)
        sys.exit(0)
    if query == file_name:
        print(file_name)
        sys.exit(0)

print(query)
PY
}

NORMALIZED_LORA_SPECS=()
if [[ "${#LORA_SPECS[@]}" -gt 0 ]]; then
  for raw_spec in "${LORA_SPECS[@]}"; do
    spec="$raw_spec"
    if [[ -z "${spec//[[:space:]]/}" ]]; then
      echo "error: --lora value must not be empty" >&2
      exit 1
    fi

    lora_token="${spec%%:*}"
    lora_tail=""
    if [[ "$spec" == *:* ]]; then
      lora_tail="${spec#"$lora_token"}"
    fi

    lora_token="$(basename "$lora_token")"
    lora_file="$(resolve_custom_lora_file "$lora_token")"
    normalized_spec="${lora_file}${lora_tail}"
    NORMALIZED_LORA_SPECS+=("$normalized_spec")

    case "$lora_file" in
      *.ckpt|*.safetensors)
        lora_path="$MODELS_DIR/$lora_file"
        if [[ ! -f "$lora_path" ]]; then
          echo "error: lora file not found: $lora_path" >&2
          exit 1
        fi
        ;;
    esac
  done
fi

LORA_SPECS=("${NORMALIZED_LORA_SPECS[@]}")

LORA_ARGS=()
for spec in "${LORA_SPECS[@]}"; do
  LORA_ARGS+=(--lora "$spec")
done

if ! [[ "$WIDTH" =~ ^[0-9]+$ ]] || [[ "$WIDTH" -lt 64 ]]; then
  echo "error: --width must be an integer >= 64" >&2
  exit 1
fi

if ! [[ "$HEIGHT" =~ ^[0-9]+$ ]] || [[ "$HEIGHT" -lt 64 ]]; then
  echo "error: --height must be an integer >= 64" >&2
  exit 1
fi

if ! [[ "$STEPS" =~ ^[0-9]+$ ]] || [[ "$STEPS" -lt 1 ]]; then
  echo "error: --steps must be an integer >= 1" >&2
  exit 1
fi

if ! [[ "$SAMPLER" =~ ^([0-9]+|[A-Za-z][A-Za-z0-9_]*)$ ]]; then
  echo "error: --sampler must be an enum name or non-negative integer" >&2
  exit 1
fi

if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
  echo "error: --seed must be a non-negative integer" >&2
  exit 1
fi

if [[ "$NO_TIMEOUT" != "1" ]]; then
  if ! command -v timeout >/dev/null 2>&1; then
    echo "error: timeout command not found" >&2
    exit 1
  fi
fi

WORK_DIR="$ROOT/output/q6p_canary_${TAG}"
SERVER_LOG="$WORK_DIR/server.log"
CLIENT_LOG="$WORK_DIR/client.log"
CONFIG_BIN="$WORK_DIR/config.bin"
mkdir -p "$WORK_DIR"

echo "== q6p canary once =="
echo "model=$MODEL_KEY"
echo "host=$HOST"
echo "grpc_bin=$GRPC_BIN"
echo "proto=$PROTO_FILE"
echo "proto_cache_dir=$PROTO_CACHE_DIR"
echo "bindings_dir=$BINDINGS_DIR"
echo "models_dir=$MODELS_DIR"
echo "image_mode=$IMAGE_MODE"
echo "server_gpu=$SERVER_GPU"
echo "server_cpu_offload=$SERVER_CPU_OFFLOAD"
echo "server_fread_preferred=$SERVER_FREAD_PREFERRED"
echo "server_no_flash_attention=$SERVER_NO_FLASH_ATTENTION"
echo "server_weights_cache=${SERVER_WEIGHTS_CACHE:-default}"
echo "gpu_metrics_enabled=$GPU_METRICS"
echo "sampler=$SAMPLER"
echo "final_mode=$FINAL_MODE"
if [[ "$NO_TIMEOUT" == "1" ]]; then
  echo "timeout_sec=none"
else
  echo "timeout_sec=$TIMEOUT_SEC"
fi
echo "max_responses=$MAX_RESPONSES"
echo "require_complete_stream=$REQUIRE_COMPLETE_STREAM"
echo "require_final_output=$REQUIRE_FINAL_OUTPUT"
if [[ "${#LORA_SPECS[@]}" -gt 0 ]]; then
  echo "loras=${LORA_SPECS[*]}"
else
  echo "loras=none"
fi
echo "work_dir=$WORK_DIR"

server_args=(
  --address "$SERVER_ADDRESS"
  --port "$SERVER_PORT"
  --gpu "$SERVER_GPU"
  --no-tls
  --model-browser
  --no-response-compression
)
if [[ "$SERVER_CPU_OFFLOAD" == "1" ]]; then
  server_args+=(--cpu-offload)
fi
if [[ "$SERVER_FREAD_PREFERRED" == "1" ]]; then
  server_args+=(--fread-preferred)
fi
if [[ "$SERVER_NO_FLASH_ATTENTION" == "1" ]]; then
  server_args+=(--no-flash-attention)
fi
if [[ -n "$SERVER_WEIGHTS_CACHE" ]]; then
  server_args+=(--weights-cache "$SERVER_WEIGHTS_CACHE")
fi
server_args+=("$MODELS_DIR")

"$PYTHON_BIN" "$ROOT/tools/dt_make_config.py" \
  --bindings-dir "$BINDINGS_DIR" \
  --models-dir "$MODELS_DIR" \
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
  --seed "$SEED" \
  "${CONFIG_EXTRA_ARGS[@]}" \
  "${LORA_ARGS[@]}"

printf 'server_command='
printf '%q ' "$GRPC_BIN" "${server_args[@]}"
printf '\n'
if [[ "$DRY_RUN" == "1" ]]; then
  echo "RESULT=DRY_RUN"
  exit 0
fi

"$PYTHON_BIN" - "$SERVER_ADDRESS" "$SERVER_PORT" <<'PY'
import socket
import sys

address, port = sys.argv[1], int(sys.argv[2])
family = socket.AF_INET6 if ":" in address else socket.AF_INET
with socket.socket(family, socket.SOCK_STREAM) as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind((address, port))
    except OSError as error:
        raise SystemExit(f"error: server endpoint unavailable: {address}:{port}: {error}")
PY

SERVER_PID=""
GPU_MONITOR_PID=""
stop_server() {
  if [[ -n "$SERVER_PID" ]]; then
    kill -9 "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
  if [[ -n "$GPU_MONITOR_PID" ]]; then
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
    GPU_MONITOR_PID=""
  fi
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "$GPU_METRICS" == "1" ]]; then
  GPU_QUERY="timestamp,index,memory.used,utilization.gpu,temperature.gpu,power.draw"
  nvidia-smi --id="$SERVER_GPU" --query-gpu="$GPU_QUERY" --format=csv,nounits \
    > "$WORK_DIR/gpu.csv"
  nvidia-smi --id="$SERVER_GPU" --query-gpu="$GPU_QUERY" --format=csv,noheader,nounits \
    --loop-ms=500 >> "$WORK_DIR/gpu.csv" 2> "$WORK_DIR/gpu-monitor.log" &
  GPU_MONITOR_PID=$!
  echo "gpu_monitor_pid=$GPU_MONITOR_PID"
  echo "gpu_metrics=$WORK_DIR/gpu.csv"
fi

nohup "$GRPC_BIN" "${server_args[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "server_pid=$SERVER_PID"

READY=0
for _ in $(seq 1 200); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  if "$PYTHON_BIN" "$ROOT/tools/dt_api_client.py" "${CLIENT_SCHEMA_ARGS[@]}" --host "$HOST" echo --name "q6p-canary" >/dev/null 2>&1 && kill -0 "$SERVER_PID" 2>/dev/null; then
    READY=1
    break
  fi
done
echo "server_ready=$READY"
if [[ "$READY" != "1" ]]; then
  echo "RESULT=FAIL server did not become ready"
  exit 1
fi

set +e
if [[ "$NO_TIMEOUT" == "1" ]]; then
  "$PYTHON_BIN" "$ROOT/tools/dt_api_client.py" \
      "${CLIENT_SCHEMA_ARGS[@]}" \
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
else
  timeout -k 10s "${TIMEOUT_SEC}s" \
    "$PYTHON_BIN" "$ROOT/tools/dt_api_client.py" \
      "${CLIENT_SCHEMA_ARGS[@]}" \
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
fi
timeout -k 5s 15s \
  "$PYTHON_BIN" "$ROOT/tools/dt_api_client.py" "${CLIENT_SCHEMA_ARGS[@]}" --host "$HOST" echo --name "q6p-canary-post" >/dev/null 2>&1
POST_ECHO_RC=$?
set -e

stop_server

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

if [[ "$REQUIRE_COMPLETE_STREAM" == "1" ]]; then
  if ! grep -q '^generation stream finished$' "$CLIENT_LOG"; then
    echo "RESULT=FAIL stream did not finish cleanly"
    if [[ "$SOFT_FAIL" == "1" ]]; then
      exit 0
    fi
    exit 1
  fi

  if grep -q '^stopping early at max responses:' "$CLIENT_LOG"; then
    echo "RESULT=FAIL stream stopped early by max response cap"
    if [[ "$SOFT_FAIL" == "1" ]]; then
      exit 0
    fi
    exit 1
  fi

  if grep -q '^warning: leftover partial .* chunk bytes:' "$CLIENT_LOG"; then
    echo "RESULT=FAIL incomplete output chunks remain"
    if [[ "$SOFT_FAIL" == "1" ]]; then exit 0; fi
    exit 1
  fi
fi

if [[ "$REQUIRE_FINAL_OUTPUT" == "1" ]]; then
  IMAGE_COUNT="$(awk -F': ' '/^images written:/{value=$2} END{if (value == "") value = "0"; print value}' "$CLIENT_LOG")"
  AUDIO_COUNT="$(awk -F': ' '/^audio written:/{value=$2} END{if (value == "") value = "0"; print value}' "$CLIENT_LOG")"

  if [[ "$IMAGE_COUNT" -lt 1 && "$AUDIO_COUNT" -lt 1 ]]; then
    echo "RESULT=FAIL no final generated output payloads observed"
    if [[ "$SOFT_FAIL" == "1" ]]; then
      exit 0
    fi
    exit 1
  fi
fi

if [[ "$IMAGE_MODE" == "1" ]]; then
  if [[ "$POST_ECHO_RC" != "0" ]]; then
    echo "RESULT=FAIL image post-generation Echo failed"
    if [[ "$SOFT_FAIL" == "1" ]]; then exit 0; fi
    exit 1
  fi
  set +e
  "$PYTHON_BIN" - "$ROOT" "$WORK_DIR" "$WIDTH" "$HEIGHT" "$PROMPT" "$NEG_PROMPT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, sys.argv[1])
from tools.dt_tensor_to_playable import validate_image_tensor

work = Path(sys.argv[2])
width, height = int(sys.argv[3]), int(sys.argv[4])
images = sorted(work.glob("image_*.bin"))
if len(images) != 1 or list(work.glob("audio_*.bin")):
    raise SystemExit("Expected exactly one final image payload and no audio")
image = validate_image_tensor(images[0].read_bytes(), width, height)
output = work / "image.png"
Image.fromarray(image).save(output)
with Image.open(output) as decoded:
    decoded.load()
    if decoded.size != (width, height):
        raise SystemExit("PNG dimensions do not match the request")
metadata = {
    "result": "PASS", "images": 1, "audio": 0, "nonfinite": 0,
    "width": width, "height": height, "prompt": sys.argv[5],
    "negative_prompt": sys.argv[6], "raw_image": images[0].name,
    "config_sha256": hashlib.sha256((work / "config.bin").read_bytes()).hexdigest(),
    "complete_stream": True, "complete_chunks": True, "post_echo": True,
}
(work / "image_acceptance.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(f"validated_image={output}")
PY
  IMAGE_RC=$?
  set -e
  if [[ "$IMAGE_RC" != "0" ]]; then
    echo "RESULT=FAIL image acceptance failed"
    if [[ "$SOFT_FAIL" == "1" ]]; then exit 0; fi
    exit 1
  fi
fi

echo "RESULT=PASS"
