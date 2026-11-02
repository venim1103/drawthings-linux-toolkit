#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANDIDATE="$ROOT/output/upgrades/community-0621d819-qwen35-cuda"
GRPC_BIN="$CANDIDATE/bazel-bin/Apps/gRPCServerCLI"
GRPC_BIN_EXPLICIT=0
SCHEMA="$ROOT/output/upgrades/schemas/0621d819"
MODELS="$ROOT/output/upgrades/moss-image-stateful-20261027/models"
MODEL="sft-i4x"
MODEL_EXPLICIT=0
BUNDLE=""
PRESET=""
MODE=""
PROMPT=""
PROMPT_SET=0
NEGATIVE_PROMPT=""
WIDTH=512
HEIGHT=512
STEPS=""
GUIDANCE_SCALE=""
TIMEOUT_SEC=""
LORAS=()
SEED=4242
PORT=7862
TAG=""
OUTPUT_DIR=""
DRY_RUN=0
METRICS=0
PREFLIGHT_ONLY=0
SESSION=""
WEIGHTS_CACHE=""

usage() {
  cat <<'EOF'
Usage: bash tools/dt_image.sh [options] "Your prompt"

  --preset mosskr2-lora4           Tested MOSSkr2 i4x + four-step LoRA, GPU policy,
                                  guarded patched server, CFG 1, 180s timeout,
                                  and GPU metrics. Requires prepared local files.
                                  Cannot combine with --model/--bundle/--grpc-bin.
  --model sft-i4x|sft-q8p|base-q8p  Model (default: sft-i4x).
  --bundle PATH                    Converted MOSS inference bundle under output/.
                                  Cannot be combined with --model.
                                  Only verified i4x bundles support gpu mode;
                                  other bundles require offload.
  --grpc-bin PATH                  Explicit native server or resource-guard executable.
  --mode offload|gpu               Weight policy (default: offload).
                                  gpu omits forced CPU weight offload; it is
                                  not a promise of zero CPU or host-memory use.
  --size N                        Square size (default: 512).
  --width N --height N            Non-square dimensions; multiples of 64.
  --steps N                       Sampling steps (default: 50).
  --guidance-scale N              Nonnegative finite CFG scale (default: 3).
  --lora FILE[:WEIGHT[:MODE]]      Adapter from the selected model directory;
                                  repeatable; replaces preset adapters if given.
  --timeout-sec N                 Generation timeout in seconds (default: 3600).
  --seed N                        Seed (default: 4242).
  --negative-prompt TEXT           Negative prompt (default: empty).
  --port N                        Local temporary server port (default: 7862).
  --session start|use|status|stop  Persistent local server; start/use require
                                  --preset mosskr2-lora4. Start needs no prompt.
                                  use submits one request; status/stop need no preset.
  --weights-cache N               Session-start RAM cache bound, 0..8 integer GiB
                                  (default: 0). Not persistent GPU model retention.
  --tag NAME                      Output tag (default: automatic timestamp).
  --output-dir PATH               Exact new directory for image, config and logs.
                                  Relative to the current directory; must not exist.
  --metrics                       Save sampled GPU telemetry.
  --dry-run                       Generate config and show the server command.
  --check                         Check native model selection without generation.
  -h, --help                      Show help.

Starts the verified private candidate, writes a validated PNG and run metadata
under output/ unless --output-dir is given, then stops the temporary server.
Canonical models are not used.
Explicit sampling options override preset defaults regardless of argument order.
EOF
}

fail() {
  printf 'error: %s\n' "$1" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --metrics) METRICS=1; shift ;;
    --check) PREFLIGHT_ONLY=1; shift ;;
    --preset|--model|--bundle|--grpc-bin|--mode|--size|--width|--height|--steps|--guidance-scale|--lora|--timeout-sec|--seed|--negative-prompt|--port|--tag|--output-dir|--session|--weights-cache)
      [[ $# -ge 2 ]] || fail "missing value for $1"
      case "$1" in
        --preset) PRESET="$2"; [[ "$PRESET" == "mosskr2-lora4" ]] || fail "preset must be mosskr2-lora4" ;;
        --model) MODEL="$2"; MODEL_EXPLICIT=1 ;;
        --bundle) BUNDLE="$2"; [[ -n "$BUNDLE" ]] || fail "bundle path cannot be empty" ;;
        --grpc-bin)
          [[ -f "$2" && -x "$2" ]] || fail "grpc-bin must be an executable file"
          GRPC_BIN="$(realpath "$2")"
          GRPC_BIN_EXPLICIT=1
          ;;
        --mode) MODE="$2"; [[ -n "$MODE" ]] || fail "mode cannot be empty" ;;
        --size) WIDTH="$2"; HEIGHT="$2" ;;
        --width) WIDTH="$2" ;;
        --height) HEIGHT="$2" ;;
        --steps) STEPS="$2"; [[ -n "$STEPS" ]] || fail "steps cannot be empty" ;;
        --guidance-scale) GUIDANCE_SCALE="$2"; [[ -n "$GUIDANCE_SCALE" ]] || fail "guidance-scale cannot be empty" ;;
        --lora) [[ -n "$2" ]] || fail "lora cannot be empty"; LORAS+=("$2") ;;
        --timeout-sec) TIMEOUT_SEC="$2"; [[ -n "$TIMEOUT_SEC" ]] || fail "timeout-sec cannot be empty" ;;
        --seed) SEED="$2" ;;
        --negative-prompt) NEGATIVE_PROMPT="$2" ;;
        --port) PORT="$2" ;;
        --session) SESSION="$2"; [[ "$SESSION" =~ ^(start|use|status|stop)$ ]] || fail "session must be start, use, status or stop" ;;
        --weights-cache) WEIGHTS_CACHE="$2"; [[ "$WEIGHTS_CACHE" =~ ^[0-8]$ ]] || fail "weights-cache must be 0..8 integer GiB" ;;
        --tag) TAG="$2"; [[ -n "$TAG" ]] || fail "tag cannot be empty" ;;
        --output-dir) OUTPUT_DIR="$2"; [[ -n "$OUTPUT_DIR" ]] || fail "output-dir cannot be empty" ;;
      esac
      shift 2
      ;;
    --) shift; [[ $# -eq 1 && "$PROMPT_SET" == "0" ]] || fail "provide one quoted prompt"; PROMPT="$1"; PROMPT_SET=1; shift ;;
    -*) fail "unknown option: $1" ;;
    *) [[ "$PROMPT_SET" == "0" ]] || fail "provide one quoted prompt"; PROMPT="$1"; PROMPT_SET=1; shift ;;
  esac
done

if [[ "$SESSION" == "status" || "$SESSION" == "stop" ]]; then
  exec "$ROOT/.venv/bin/python" "$ROOT/tools/dt_image_session.py" "$SESSION" --port "$PORT"
fi
if [[ -n "$SESSION" ]]; then
  [[ "$PRESET" == "mosskr2-lora4" ]] || fail "session start/use requires --preset mosskr2-lora4"
  [[ "$DRY_RUN" == "0" && "$PREFLIGHT_ONLY" == "0" ]] || fail "session mode cannot combine with --dry-run or --check"
  if [[ "$SESSION" == "start" && "$PROMPT_SET" == "0" ]]; then PROMPT="Session model preflight"; fi
elif [[ -n "$WEIGHTS_CACHE" ]]; then
  fail "weights-cache requires --session start (or a matching --session use)"
fi
[[ -n "${PROMPT//[[:space:]]/}" ]] || fail "provide a nonempty quoted prompt"
if [[ -n "$PRESET" ]]; then
  [[ "$MODEL_EXPLICIT" == "0" && -z "$BUNDLE" && "$GRPC_BIN_EXPLICIT" == "0" ]] || fail "preset cannot be combined with --model, --bundle or --grpc-bin; use those options without --preset for a custom setup"
  BUNDLE="$ROOT/output/moss-models/mosskr2-i4x-4step-lora-test-20261030"
  GRPC_BIN="$ROOT/output/upgrades/moss-image-stateful-20261027/guard-moss-converted-server.sh"
  PRESET_NATIVE="$ROOT/output/upgrades/community-0621d819-qwen35-cuda-prefetch-guard/bazel-bin/Apps/gRPCServerCLI"
  [[ -f "$GRPC_BIN" && -x "$GRPC_BIN" ]] || fail "preset requires the prepared local resource guard: $GRPC_BIN"
  [[ -f "$PRESET_NATIVE" && -x "$PRESET_NATIVE" ]] || fail "preset requires the prepared patched native server: $PRESET_NATIVE"
  MODE="${MODE:-gpu}"
  STEPS="${STEPS:-4}"
  GUIDANCE_SCALE="${GUIDANCE_SCALE:-1}"
  TIMEOUT_SEC="${TIMEOUT_SEC:-180}"
  if [[ "${#LORAS[@]}" == "0" ]]; then
    LORAS=(moss_image_sft_4_steps_lora_v2_600_avg_rank_72_bf16_lora_f16.ckpt:1.0)
  fi
  METRICS=1
  export DT_VALIDATION_NATIVE="$PRESET_NATIVE"
  export DT_VALIDATION_WEIGHT_POLICY="$MODE"
  unset LD_PRELOAD DT_SAMPLE_CPU_STACKS DT_TARGETED_PROFILE_CLIENT_LOG
fi
MODE="${MODE:-offload}"
STEPS="${STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3}"
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || fail "steps must be a positive integer"
[[ "$TIMEOUT_SEC" =~ ^[1-9][0-9]*$ ]] || fail "timeout-sec must be a positive integer"
"$ROOT/.venv/bin/python" -c 'import math, sys; value = float(sys.argv[1]); sys.exit(not (math.isfinite(value) and value >= 0))' "$GUIDANCE_SCALE" 2>/dev/null || fail "guidance-scale must be finite and nonnegative"
if [[ -n "$BUNDLE" ]]; then
  [[ "$MODEL_EXPLICIT" == "0" ]] || fail "--bundle cannot be combined with --model"
  [[ -d "$BUNDLE" ]] || fail "bundle directory does not exist: $BUNDLE"
  BUNDLE="$(cd "$BUNDLE" && pwd)"
  if [[ "$SESSION" == "use" ]]; then
    MODEL_FILE="$("$ROOT/.venv/bin/python" "$ROOT/tools/dt_image_session.py" model --port "$PORT")"
  elif [[ "$SESSION" == "start" ]]; then
    MODEL_FILE="$("$ROOT/.venv/bin/python" -c 'import json, sys; print(json.load(open(sys.argv[1]))["model"])' "$BUNDLE/bundle.json")"
  else
    MODEL_FILE="$("$ROOT/.venv/bin/python" "$ROOT/tools/dt_moss_convert.py" --bundle-model "$BUNDLE")"
  fi
  if [[ "$MODE" == "gpu" ]]; then
    BUNDLE_QUANTIZATION="$("$ROOT/.venv/bin/python" -c 'import json, sys; print(json.load(open(sys.argv[1])).get("quantization", ""))' "$BUNDLE/bundle.json")"
    [[ "$BUNDLE_QUANTIZATION" == "i4x" ]] || fail "only verified i4x bundles support --mode gpu; other bundles require --mode offload"
  fi
  MODELS="$BUNDLE/models"
  MODEL="bundle:$MODEL_FILE"
else
  case "$MODEL" in
    sft-i4x) MODEL_FILE="moss_image_sft_i4x.ckpt" ;;
    sft-q8p) MODEL_FILE="moss_image_sft_q8p.ckpt" ;;
    base-q8p) MODEL_FILE="moss_image_base_q8p.ckpt" ;;
    *) fail "model must be sft-i4x, sft-q8p, or base-q8p" ;;
  esac
fi
case "$MODE" in
  offload) MODE_ARGUMENTS=(--server-cpu-offload) ;;
  gpu) MODE_ARGUMENTS=() ;;
  *) fail "mode must be offload or gpu" ;;
esac
for dimension in "$WIDTH" "$HEIGHT"; do
  [[ "$dimension" =~ ^[1-9][0-9]{0,3}$ ]] || fail "dimensions must be positive multiples of 64"
  (( dimension >= 64 && dimension % 64 == 0 )) || fail "dimensions must be positive multiples of 64"
done
if [[ -z "$TAG" ]]; then TAG="image_$(date +%Y%m%d_%H%M%S_%N)"; fi
[[ "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "tag must contain only letters, digits, dot, underscore, or hyphen"
if [[ -n "$OUTPUT_DIR" ]]; then
  [[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || fail "output directory already exists: $OUTPUT_DIR"
  OUTPUT_DIR="$(realpath -m -- "$OUTPUT_DIR")"
  [[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || fail "output directory already exists: $OUTPUT_DIR"
  RUN_DIR="$OUTPUT_DIR"
else
  RUN_DIR="$ROOT/output/q6p_canary_$TAG"
  [[ ! -e "$RUN_DIR" && ! -L "$RUN_DIR" ]] || fail "output tag already exists: $TAG"
fi
if [[ -n "$PRESET" ]]; then export DT_VALIDATION_CLIENT_LOG="$RUN_DIR/client.log"; fi
EXTRA_ARGUMENTS=()
if [[ -n "$OUTPUT_DIR" ]]; then EXTRA_ARGUMENTS+=(--output-dir "$OUTPUT_DIR"); fi
if [[ "$SESSION" == "use" && -z "$OUTPUT_DIR" ]]; then EXTRA_ARGUMENTS+=(--output-dir "$RUN_DIR"); fi
for lora in "${LORAS[@]}"; do EXTRA_ARGUMENTS+=(--lora "$lora"); done
if [[ -n "$BUNDLE" ]]; then EXTRA_ARGUMENTS+=(--require-model-version moss_image); fi
if [[ "$PREFLIGHT_ONLY" == "1" ]]; then EXTRA_ARGUMENTS+=(--preflight-only); fi
if [[ "$DRY_RUN" == "1" ]]; then EXTRA_ARGUMENTS+=(--dry-run); fi
if [[ "$METRICS" == "1" ]]; then EXTRA_ARGUMENTS+=(--gpu-metrics); fi
if [[ -n "$TAG" ]]; then EXTRA_ARGUMENTS+=(--tag "$TAG"); fi

printf 'image_model=%s weight_mode=%s dimensions=%sx%s\n' "$MODEL" "$MODE" "$WIDTH" "$HEIGHT"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
CANARY_ARGUMENTS=( \
  --grpc-bin "$GRPC_BIN" \
  --proto "$CANDIDATE/Libraries/GRPC/Models/Sources/imageService/imageService.proto" \
  --cache-dir "$SCHEMA" --bindings-dir "$SCHEMA/flatbuffers" \
  --models-dir "$MODELS" --model "$MODEL_FILE" \
  --require-model-file qwen_3.5_9b_i5x.ckpt --require-model-file flux_2_vae_f16.ckpt \
  --host "127.0.0.1:$PORT" --image-mode --width "$WIDTH" --height "$HEIGHT" \
  --steps "$STEPS" --sampler DDIMTrailing --guidance-scale "$GUIDANCE_SCALE" --shift 5 --seed "$SEED" \
  --prompt "$PROMPT" --negative-prompt "$NEGATIVE_PROMPT" \
  --server-fread-preferred --server-weights-cache "${WEIGHTS_CACHE:-0}" --final-mode --timeout-sec "$TIMEOUT_SEC" \
  "${MODE_ARGUMENTS[@]}" "${EXTRA_ARGUMENTS[@]}" )
if [[ -n "$SESSION" ]]; then
  SESSION_ARGUMENTS=("$SESSION" --port "$PORT" --bundle "$BUNDLE" --mode "$MODE")
  if [[ -n "$WEIGHTS_CACHE" ]]; then SESSION_ARGUMENTS+=(--cache-gib "$WEIGHTS_CACHE"); fi
  exec "$ROOT/.venv/bin/python" "$ROOT/tools/dt_image_session.py" "${SESSION_ARGUMENTS[@]}" -- "${CANARY_ARGUMENTS[@]}"
fi
exec bash "$ROOT/tools/run_q6p_canary_once.sh" "${CANARY_ARGUMENTS[@]}"