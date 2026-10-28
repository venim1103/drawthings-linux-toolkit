#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANDIDATE="$ROOT/output/upgrades/community-0621d819-qwen35-cuda"
SCHEMA="$ROOT/output/upgrades/schemas/0621d819"
MODELS="$ROOT/output/upgrades/moss-image-stateful-20261027/models"
MODEL="sft-i4x"
MODE="offload"
PROMPT=""
PROMPT_SET=0
NEGATIVE_PROMPT=""
WIDTH=512
HEIGHT=512
STEPS=50
SEED=4242
PORT=7862
TAG=""
DRY_RUN=0
METRICS=0

usage() {
  cat <<'EOF'
Usage: bash tools/dt_image.sh [options] "Your prompt"

  --model sft-i4x|sft-q8p|base-q8p  Model (default: sft-i4x).
  --mode offload|gpu               Weight policy (default: offload).
                                  gpu omits forced CPU weight offload; it is
                                  not a promise of zero CPU or host-memory use.
  --size N                        Square size (default: 512).
  --width N --height N            Non-square dimensions; multiples of 64.
  --steps N                       Sampling steps (default: 50).
  --seed N                        Seed (default: 4242).
  --negative-prompt TEXT           Negative prompt (default: empty).
  --port N                        Local temporary server port (default: 7862).
  --tag NAME                      Output tag (default: automatic timestamp).
  --metrics                       Save sampled GPU telemetry.
  --dry-run                       Generate config and show the server command.
  -h, --help                      Show help.

Starts the verified private candidate, writes a validated PNG and run metadata
under output/, then stops the temporary server. Canonical models are not used.
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
    --model|--mode|--size|--width|--height|--steps|--seed|--negative-prompt|--port|--tag)
      [[ $# -ge 2 ]] || fail "missing value for $1"
      case "$1" in
        --model) MODEL="$2" ;;
        --mode) MODE="$2" ;;
        --size) WIDTH="$2"; HEIGHT="$2" ;;
        --width) WIDTH="$2" ;;
        --height) HEIGHT="$2" ;;
        --steps) STEPS="$2" ;;
        --seed) SEED="$2" ;;
        --negative-prompt) NEGATIVE_PROMPT="$2" ;;
        --port) PORT="$2" ;;
        --tag) TAG="$2"; [[ -n "$TAG" ]] || fail "tag cannot be empty" ;;
      esac
      shift 2
      ;;
    --) shift; [[ $# -eq 1 && "$PROMPT_SET" == "0" ]] || fail "provide one quoted prompt"; PROMPT="$1"; PROMPT_SET=1; shift ;;
    -*) fail "unknown option: $1" ;;
    *) [[ "$PROMPT_SET" == "0" ]] || fail "provide one quoted prompt"; PROMPT="$1"; PROMPT_SET=1; shift ;;
  esac
done

[[ -n "${PROMPT//[[:space:]]/}" ]] || fail "provide a nonempty quoted prompt"
case "$MODEL" in
  sft-i4x) MODEL_FILE="moss_image_sft_i4x.ckpt" ;;
  sft-q8p) MODEL_FILE="moss_image_sft_q8p.ckpt" ;;
  base-q8p) MODEL_FILE="moss_image_base_q8p.ckpt" ;;
  *) fail "model must be sft-i4x, sft-q8p, or base-q8p" ;;
esac
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
[[ ! -e "$ROOT/output/q6p_canary_$TAG" && ! -L "$ROOT/output/q6p_canary_$TAG" ]] || fail "output tag already exists: $TAG"
EXTRA_ARGUMENTS=()
if [[ "$DRY_RUN" == "1" ]]; then EXTRA_ARGUMENTS+=(--dry-run); fi
if [[ "$METRICS" == "1" ]]; then EXTRA_ARGUMENTS+=(--gpu-metrics); fi
if [[ -n "$TAG" ]]; then EXTRA_ARGUMENTS+=(--tag "$TAG"); fi

printf 'image_model=%s weight_mode=%s dimensions=%sx%s\n' "$MODEL" "$MODE" "$WIDTH" "$HEIGHT"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
exec bash "$ROOT/tools/run_q6p_canary_once.sh" \
  --grpc-bin "$CANDIDATE/bazel-bin/Apps/gRPCServerCLI" \
  --proto "$CANDIDATE/Libraries/GRPC/Models/Sources/imageService/imageService.proto" \
  --cache-dir "$SCHEMA" --bindings-dir "$SCHEMA/flatbuffers" \
  --models-dir "$MODELS" --model "$MODEL_FILE" \
  --require-model-file qwen_3.5_9b_i5x.ckpt --require-model-file flux_2_vae_f16.ckpt \
  --host "127.0.0.1:$PORT" --image-mode --width "$WIDTH" --height "$HEIGHT" \
  --steps "$STEPS" --sampler DDIMTrailing --guidance-scale 3 --shift 5 --seed "$SEED" \
  --prompt "$PROMPT" --negative-prompt "$NEGATIVE_PROMPT" \
  --server-fread-preferred --server-weights-cache 0 --final-mode --timeout-sec 3600 \
  "${MODE_ARGUMENTS[@]}" "${EXTRA_ARGUMENTS[@]}"