#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"

MODEL_KEY="10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt"
TAG="run076_warm_server_mod_auto"
REPEATS=8
TIMEOUT_SEC=75
HOST="127.0.0.1:7861"
GRPC_BIN="$ROOT/draw-things-community/.build/release/gRPCServerCLI"
CUSTOM_JSON="$ROOT/dt-models/custom.json"
RESTART_PER_REPEAT=0
ENTRY_VERSION="ltx2.3"
ENTRY_TEXT_ENCODER="gemma_3_12b_it_qat_q8p.ckpt"
ENTRY_CLIP_ENCODER="10_e_v1_bf16_regen_0_q6p.ckpt"
ENTRY_MODIFIER="kontext"
ENTRY_AUTOENCODER="ltx_2.3_audio_video_vae_f16.ckpt"

WIDTH=256
HEIGHT=256
STEPS=4
SEED=4242
PROMPT="a cinematic shot of a red sports car driving on a mountain road at sunset, detailed, realistic"
NEG_PROMPT="blurry, distorted, low quality, artifacts"

usage() {
  cat <<'EOF'
Usage:
  tools/run_q6p_warm_server_mod_auto_repeats.sh [options]

Options:
  --model <file>         Model key (default: trace021 key).
  --tag <value>          Output tag prefix (default: run076_warm_server_mod_auto).
  --repeats <n>          Repeat count (default: 8).
  --timeout-sec <n>      Per-repeat timeout in seconds (default: 75).
  --host <host:port>     gRPC host:port (default: 127.0.0.1:7861).
  --grpc-bin <path>      Source runtime binary path.
  --restart-per-repeat   Restart server before each repeat (cold-per-repeat mode).
  --entry-version <v>    Mapped entry version (default: ltx2.3).
  --text-encoder <file|none>
                         Mapped text encoder (default: gemma_3_12b_it_qat_q8p.ckpt).
                         Use none to omit the field.
  --clip-encoder <file|none>
                         Mapped clip encoder (default: 10_e_v1_bf16_regen_0_q6p.ckpt).
                         Use none to omit the field.
  --modifier <value|none>
                         Mapped modifier (default: kontext). Use none to omit.
  --autoencoder <file|none>
                         Mapped autoencoder (default: ltx_2.3_audio_video_vae_f16.ckpt).
                         Use none to omit the field.
  --width <n>            Request width (default: 256).
  --height <n>           Request height (default: 256).
  --steps <n>            Request steps (default: 4).
  --seed <n>             Request seed (default: 4242).
  -h, --help             Show this help.

Outputs:
  output/<tag>/
    - server.log
    - <case>_server_slice.log
  output/<tag>_console.log
  output/<tag>_repeats_summary.tsv
  output/<tag>_repeats_aggregate.txt

Notes:
  - Default mode starts one long-lived gRPCServerCLI process and reuses it across repeats.
  - --restart-per-repeat switches to per-repeat server restart mode.
  - Injects a temporary mapped custom.json entry matching the model key.
  - Entry fields can be toggled with --modifier/--autoencoder/--text-encoder/--clip-encoder for control runs.
  - Always restores custom.json on exit.
EOF
}

if [[ $# -gt 0 ]]; then
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model)
        MODEL_KEY="${2:-}"
        shift 2
        ;;
      --tag)
        TAG="${2:-}"
        shift 2
        ;;
      --repeats)
        REPEATS="${2:-}"
        shift 2
        ;;
      --timeout-sec)
        TIMEOUT_SEC="${2:-}"
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
      --restart-per-repeat)
        RESTART_PER_REPEAT=1
        shift
        ;;
      --entry-version)
        ENTRY_VERSION="${2:-}"
        shift 2
        ;;
      --text-encoder)
        ENTRY_TEXT_ENCODER="${2:-}"
        shift 2
        ;;
      --clip-encoder)
        ENTRY_CLIP_ENCODER="${2:-}"
        shift 2
        ;;
      --modifier)
        ENTRY_MODIFIER="${2:-}"
        shift 2
        ;;
      --autoencoder)
        ENTRY_AUTOENCODER="${2:-}"
        shift 2
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
      --seed)
        SEED="${2:-}"
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
fi

if [[ ! -x "$PY" ]]; then
  echo "error: missing python env at $PY" >&2
  exit 1
fi

if [[ ! -x "$GRPC_BIN" ]]; then
  echo "error: gRPC binary is not executable: $GRPC_BIN" >&2
  exit 1
fi

if [[ ! -f "$CUSTOM_JSON" ]]; then
  echo "error: custom.json not found: $CUSTOM_JSON" >&2
  exit 1
fi

if [[ -z "$ENTRY_VERSION" ]]; then
  echo "error: --entry-version cannot be empty" >&2
  exit 1
fi

if ! [[ "$REPEATS" =~ ^[0-9]+$ ]] || [[ "$REPEATS" -lt 1 ]]; then
  echo "error: --repeats must be an integer >= 1" >&2
  exit 1
fi

if ! [[ "$TIMEOUT_SEC" =~ ^[0-9]+$ ]] || [[ "$TIMEOUT_SEC" -lt 1 ]]; then
  echo "error: --timeout-sec must be an integer >= 1" >&2
  exit 1
fi

for value in "$WIDTH" "$HEIGHT" "$STEPS" "$SEED"; do
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "error: width/height/steps/seed must be non-negative integers" >&2
    exit 1
  fi
done

if [[ "$WIDTH" -lt 64 || "$HEIGHT" -lt 64 ]]; then
  echo "error: --width and --height must be >= 64" >&2
  exit 1
fi

host_addr="${HOST%%:*}"
host_port="${HOST##*:}"
if [[ -z "$host_addr" || -z "$host_port" || "$host_addr" == "$HOST" ]]; then
  echo "error: --host must be host:port" >&2
  exit 1
fi

if ! [[ "$host_port" =~ ^[0-9]+$ ]]; then
  echo "error: host port must be numeric" >&2
  exit 1
fi

OUT_ROOT="$ROOT/output/$TAG"
SUMMARY_TSV="$ROOT/output/${TAG}_repeats_summary.tsv"
AGG_TXT="$ROOT/output/${TAG}_repeats_aggregate.txt"
CONSOLE_LOG="$ROOT/output/${TAG}_console.log"
BACKUP_JSON="$OUT_ROOT/custom.json.backup"
SERVER_LOG="$OUT_ROOT/server.log"
CONFIG_BIN="$OUT_ROOT/config.bin"

mkdir -p "$OUT_ROOT"
cp "$CUSTOM_JSON" "$BACKUP_JSON"

cleanup() {
  set +e
  pkill -9 -f "gRPCServerCLI --address $host_addr --port $host_port" >/dev/null 2>&1 || true
  pkill -9 -f "dt_api_client.py .*generate-raw" >/dev/null 2>&1 || true
  if [[ -f "$BACKUP_JSON" ]]; then
    cp "$BACKUP_JSON" "$CUSTOM_JSON"
    rm -f "$BACKUP_JSON"
  fi
}
trap cleanup EXIT

python3 - "$CUSTOM_JSON" "$MODEL_KEY" "$ENTRY_VERSION" "$ENTRY_TEXT_ENCODER" "$ENTRY_CLIP_ENCODER" "$ENTRY_MODIFIER" "$ENTRY_AUTOENCODER" <<'PY'
import json
import sys

path = sys.argv[1]
model_key = sys.argv[2]
entry_version = sys.argv[3]
entry_text_encoder = sys.argv[4]
entry_clip_encoder = sys.argv[5]
entry_modifier = sys.argv[6]
entry_autoencoder = sys.argv[7]

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, list):
    raise SystemExit("custom.json root must be a list")

entry = {
    "name": model_key,
  "version": entry_version,
    "prefix": "",
    "default_scale": 1,
    "hires_fix_scale": 24,
    "latents_upscalers": [],
    "file": model_key,
    "upcast_attention": False,
    "high_precision_autoencoder": False,
    "objective": {"u": {"condition_scale": 1000}},
}

if entry_text_encoder.lower() != "none":
  entry["text_encoder"] = entry_text_encoder
if entry_clip_encoder.lower() != "none":
  entry["clip_encoder"] = entry_clip_encoder
if entry_modifier.lower() != "none":
  entry["modifier"] = entry_modifier
if entry_autoencoder.lower() != "none":
  entry["autoencoder"] = entry_autoencoder

out = [x for x in data if not (isinstance(x, dict) and x.get("name") == model_key)]
out.append(entry)

with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
    f.write("\n")
PY

python3 -m json.tool "$CUSTOM_JSON" >/dev/null

pkill -9 -f "gRPCServerCLI --address $host_addr --port $host_port" >/dev/null 2>&1 || true
pkill -9 -f "dt_api_client.py .*generate-raw" >/dev/null 2>&1 || true

rm -f "$SERVER_LOG"

start_server() {
  local label="$1"
  local append_mode="$2"

  pkill -9 -f "gRPCServerCLI --address $host_addr --port $host_port" >/dev/null 2>&1 || true
  pkill -9 -f "dt_api_client.py .*generate-raw" >/dev/null 2>&1 || true

  if [[ "$append_mode" == "1" ]]; then
    nohup env DT_LTX23_TRACE=1 "$GRPC_BIN" \
      --address "$host_addr" \
      --port "$host_port" \
      --gpu 0 \
      --no-tls \
      --model-browser \
      --no-response-compression \
      "$ROOT/dt-models" >> "$SERVER_LOG" 2>&1 &
  else
    nohup env DT_LTX23_TRACE=1 "$GRPC_BIN" \
      --address "$host_addr" \
      --port "$host_port" \
      --gpu 0 \
      --no-tls \
      --model-browser \
      --no-response-compression \
      "$ROOT/dt-models" > "$SERVER_LOG" 2>&1 &
  fi

  SERVER_PID=$!
  # Detach background job bookkeeping to avoid noisy "Killed" job notices
  # when we intentionally terminate previous server instances between repeats.
  disown %% >/dev/null 2>&1 || true
  echo "server_pid=${SERVER_PID} (${label})" | tee -a "$CONSOLE_LOG"

  READY=0
  for _ in $(seq 1 240); do
    if "$PY" "$ROOT/tools/dt_api_client.py" --host "$HOST" echo --name "${TAG}-${label}-ready" >/dev/null 2>&1; then
      READY=1
      break
    fi
  done

  if [[ "$READY" != "1" ]]; then
    echo "ERROR: server did not become ready (${label})" | tee -a "$CONSOLE_LOG"
    return 1
  fi

  echo "server_ready=1 (${label})" | tee -a "$CONSOLE_LOG"
  return 0
}

{
  echo "== q6p warm-server repeats =="
  echo "tag=$TAG"
  echo "model=$MODEL_KEY"
  echo "host=$HOST"
  echo "repeats=$REPEATS"
  echo "timeout_sec=$TIMEOUT_SEC"
  echo "restart_per_repeat=$RESTART_PER_REPEAT"
  echo "entry_version=$ENTRY_VERSION"
  echo "entry_text_encoder=$ENTRY_TEXT_ENCODER"
  echo "entry_clip_encoder=$ENTRY_CLIP_ENCODER"
  echo "entry_modifier=$ENTRY_MODIFIER"
  echo "entry_autoencoder=$ENTRY_AUTOENCODER"
} | tee "$CONSOLE_LOG"

if [[ "$RESTART_PER_REPEAT" == "0" ]]; then
  start_server "initial" 0
fi

"$PY" "$ROOT/tools/dt_make_config.py" \
  --out "$CONFIG_BIN" \
  --model "$MODEL_KEY" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --steps "$STEPS" \
  --sampler 17 \
  --guidance-scale 1.0 \
  --shift 3.0 \
  --hires-fix false \
  --num-frames 9 \
  --fps-id 5 \
  --seed "$SEED"

echo -e "repeat\tcanary_rc\tpost_echo_rc\tresult\tsource\ttext_init\tencode_begin\tencode_loading\tresponses\timages\taudio\tcase_tag" > "$SUMMARY_TSV"

for i in $(seq 1 "$REPEATS"); do
  case_tag="${TAG}_r${i}"
  case_dir="$ROOT/output/q6p_canary_${case_tag}"
  client_log="$ROOT/output/${case_tag}_client.log"
  slice_log="$OUT_ROOT/${case_tag}_server_slice.log"

  mkdir -p "$case_dir"
  touch "$SERVER_LOG"
  start_line=$(( $(wc -l < "$SERVER_LOG") + 1 ))

  if [[ "$RESTART_PER_REPEAT" == "1" ]]; then
    start_server "r${i}" 1
  fi

  echo "starting repeat=r${i}" | tee -a "$CONSOLE_LOG"

  set +e
  timeout -k 10s "${TIMEOUT_SEC}s" "$PY" "$ROOT/tools/dt_api_client.py" \
    --host "$HOST" \
    --max-recv-bytes 134217728 \
    generate-raw \
    --config-bin "$CONFIG_BIN" \
    --prompt "$PROMPT" \
    --negative-prompt "$NEG_PROMPT" \
    --output-dir "$case_dir" \
    --chunked \
    --save-preview \
    --max-responses 0 > "$client_log" 2>&1
  canary_rc=$?

  timeout -k 5s 15s "$PY" "$ROOT/tools/dt_api_client.py" --host "$HOST" echo --name "${TAG}-post-${i}" >/dev/null 2>&1
  post_echo_rc=$?
  set -e

  end_line=$(wc -l < "$SERVER_LOG")
  if [[ "$end_line" -ge "$start_line" ]]; then
    sed -n "${start_line},${end_line}p" "$SERVER_LOG" > "$slice_log"
  else
    : > "$slice_log"
  fi

  if rg -q "source=mapping" "$slice_log"; then
    source="mapping"
  elif rg -q "source=miss" "$slice_log"; then
    source="miss"
  else
    source="unknown"
  fi

  if rg -q "LocalImageGenerator.textEncoderInit" "$slice_log"; then text_init=1; else text_init=0; fi
  if rg -q "encodeLTX2.begin" "$slice_log"; then encode_begin=1; else encode_begin=0; fi
  if rg -q "encodeLTX2.loading_text_model" "$slice_log"; then encode_loading=1; else encode_loading=0; fi

  responses=$(awk '/^response #/{c++} END{print c+0}' "$client_log")
  images=$(awk -F': ' '/^images written:/{v=$2} END{if (v=="") v="0"; print v}' "$client_log")
  audio=$(awk -F': ' '/^audio written:/{v=$2} END{if (v=="") v="0"; print v}' "$client_log")

  if [[ "$canary_rc" == "0" ]]; then
    result="PASS"
  elif [[ "$canary_rc" == "124" ]]; then
    result="FAIL timeout"
  else
    result="FAIL rc=${canary_rc}"
  fi

  printf "r%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$i" "$canary_rc" "$post_echo_rc" "$result" "$source" "$text_init" "$encode_begin" "$encode_loading" "$responses" "$images" "$audio" "$case_tag" >> "$SUMMARY_TSV"

  echo "repeat=r${i} canary_rc=${canary_rc} post_echo_rc=${post_echo_rc} source=${source}" | tee -a "$CONSOLE_LOG"
done

canary_124=$(awk -F'\t' 'NR>1 && $2==124{c++} END{print c+0}' "$SUMMARY_TSV")
post0=$(awk -F'\t' 'NR>1 && $3==0{c++} END{print c+0}' "$SUMMARY_TSV")
post124=$(awk -F'\t' 'NR>1 && $3==124{c++} END{print c+0}' "$SUMMARY_TSV")
post_other=$(awk -F'\t' 'NR>1 && $3!=0 && $3!=124{c++} END{print c+0}' "$SUMMARY_TSV")
source_map=$(awk -F'\t' 'NR>1 && $5=="mapping"{c++} END{print c+0}' "$SUMMARY_TSV")

echo "${TAG}_repeats=${REPEATS}" > "$AGG_TXT"
echo "canary_rc_124=${canary_124}" >> "$AGG_TXT"
echo "post_echo_rc_0=${post0}" >> "$AGG_TXT"
echo "post_echo_rc_124=${post124}" >> "$AGG_TXT"
echo "post_echo_rc_other=${post_other}" >> "$AGG_TXT"
echo "source_mapping=${source_map}" >> "$AGG_TXT"

echo "== summary tsv ==" | tee -a "$CONSOLE_LOG"
cat "$SUMMARY_TSV" | tee -a "$CONSOLE_LOG"
echo "== aggregate ==" | tee -a "$CONSOLE_LOG"
cat "$AGG_TXT" | tee -a "$CONSOLE_LOG"

echo "RESULT=PASS" | tee -a "$CONSOLE_LOG"