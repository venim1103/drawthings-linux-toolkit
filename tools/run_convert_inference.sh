#!/usr/bin/env bash
set -euo pipefail

# End-to-end convert + first-frame inference orchestrator.
#
# Purpose:
#   Chain the EXISTING toolkit primitives into one control-first flow:
#     preflight -> convert (raw by default) -> validate -> [quantize]
#     -> upsert custom.json alias -> first-frame canary -> render PNG
#
#   It adds no new conversion/inference logic. It only sequences existing tools
#   and wires a correct ltx2.3 custom.json alias so inference has its companions.
#
# Why "raw by default":
#   run_convert_safetensors_to_f16.sh force-aligns metadata to the official
#   baseline. For a true positive control we first want the converter's RAW
#   output, with no align-to-official step. Use --aligned to opt back in.
#
# Typical positive-control usage (prove the pipeline on a KNOWN source):
#   tools/run_convert_inference.sh \
#     --safetensors dt-models/ltx_2.3_22b_distilled_1.1.safetensors \
#     --name ltx23_control \
#     --render
#
# Then the custom-model usage (only after the control passes):
#   tools/run_convert_inference.sh \
#     --safetensors dt-models/10_e_v1_bf16.safetensors \
#     --name 10_e_v1_ctrl \
#     --render

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT/.venv/bin/python"
MODELS_DIR="$ROOT/dt-models"
CUSTOM_JSON="$MODELS_DIR/custom.json"

SAFETENSORS=""
NAME=""
ALIGNED=0
QUANTIZE=""          # empty = skip; e.g. q6p
RAW_ONLY=0           # stop after convert/validate (no inference)
DO_RENDER=0
FORCE_CONVERT=0      # re-convert even if f16 output already exists

# ltx2.3 inference companions (proven-working template = official_q6p_via_custom)
AUTOENCODER="ltx_2.3_audio_video_vae_f16.ckpt"
CLIP_ENCODER=""      # default: the model file itself (matches official template)
TEXT_ENCODER="gemma_3_12b_it_qat_q8p.ckpt"
MODIFIER="kontext"

# first-frame canary knobs (small + bounded; still expect long waits)
WIDTH=384
HEIGHT=384
STEPS=8
SEED=4242
NUM_FRAMES=1
TIMEOUT_SEC=1200

usage() {
  cat <<'EOF'
Usage:
  tools/run_convert_inference.sh --safetensors <path> --name <alias> [options]

Required:
  --safetensors <path>   Source .safetensors to convert.
  --name <alias>         Base name/alias (also custom.json entry name).

Convert options:
  --aligned              Use metadata align-to-official convert wrapper
                         (default is RAW: no align-to-official).
  --quantize <codec>     Also quantize f16 -> codec (e.g. q6p). Default: skip.
  --raw-only             Stop after convert + validate (no inference).
  --force-convert        Re-convert even if the f16 output already exists
                         (default: reuse existing f16 to allow resuming).

Inference alias options (ltx2.3):
  --autoencoder <ckpt>   Default: ltx_2.3_audio_video_vae_f16.ckpt
  --clip-encoder <ckpt>  Default: the converted model file itself.
  --text-encoder <ckpt>  Default: gemma_3_12b_it_qat_q8p.ckpt
  --modifier <name>      Default: kontext

Canary options:
  --width <n>            Default: 384
  --height <n>           Default: 384
  --steps <n>            Default: 8
  --seed <n>             Default: 4242
  --num-frames <n>       Default: 1 (first-frame test)
  --timeout-sec <n>      Default: 1200 (expect 15+ min waits)
  --render               Render first output bin to PNG after inference.

  -h, --help             Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --safetensors) SAFETENSORS="${2:-}"; shift 2 ;;
    --name) NAME="${2:-}"; shift 2 ;;
    --aligned) ALIGNED=1; shift ;;
    --quantize) QUANTIZE="${2:-}"; shift 2 ;;
    --raw-only) RAW_ONLY=1; shift ;;
    --force-convert) FORCE_CONVERT=1; shift ;;
    --autoencoder) AUTOENCODER="${2:-}"; shift 2 ;;
    --clip-encoder) CLIP_ENCODER="${2:-}"; shift 2 ;;
    --text-encoder) TEXT_ENCODER="${2:-}"; shift 2 ;;
    --modifier) MODIFIER="${2:-}"; shift 2 ;;
    --width) WIDTH="${2:-}"; shift 2 ;;
    --height) HEIGHT="${2:-}"; shift 2 ;;
    --steps) STEPS="${2:-}"; shift 2 ;;
    --seed) SEED="${2:-}"; shift 2 ;;
    --num-frames) NUM_FRAMES="${2:-}"; shift 2 ;;
    --timeout-sec) TIMEOUT_SEC="${2:-}"; shift 2 ;;
    --render) DO_RENDER=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$SAFETENSORS" || -z "$NAME" ]]; then
  echo "error: --safetensors and --name are required" >&2
  usage
  exit 1
fi

[[ "$SAFETENSORS" != /* ]] && SAFETENSORS="$ROOT/$SAFETENSORS"
if [[ ! -f "$SAFETENSORS" ]]; then
  echo "error: safetensors not found: $SAFETENSORS" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "error: python venv not found at $PYTHON_BIN" >&2
  exit 1
fi

F16="$MODELS_DIR/${NAME}_f16.ckpt"
TS="$(date +%Y%m%d_%H%M%S)"

echo "=============================================="
echo "convert+inference orchestrator"
echo "  safetensors : $SAFETENSORS"
echo "  name        : $NAME"
echo "  mode        : $([[ $ALIGNED == 1 ]] && echo aligned || echo raw)"
echo "  quantize    : ${QUANTIZE:-<skip>}"
echo "  f16 output  : $F16"
echo "=============================================="

# -------------------------------------------------------------------
# Stage 1: preflight
# -------------------------------------------------------------------
echo "==> [1/6] Preflight safetensors"
"$PYTHON_BIN" "$ROOT/tools/dt_preflight_safetensors.py" --file "$SAFETENSORS" || {
  echo "error: preflight failed" >&2; exit 1; }

# -------------------------------------------------------------------
# Stage 2: convert to f16
# -------------------------------------------------------------------
if [[ -f "$F16" && "$FORCE_CONVERT" != "1" ]]; then
  echo "==> [2/6] Convert SKIPPED (f16 already exists: $F16)"
  echo "    pass --force-convert to rebuild it."
elif [[ "$ALIGNED" == "1" ]]; then
  echo "==> [2/6] Convert (aligned-to-official)"
  bash "$ROOT/tools/run_convert_safetensors_to_f16.sh" "$SAFETENSORS" "$F16" "$NAME"
else
  echo "==> [2/6] Convert (RAW, no align-to-official)"
  DRAWTHINGS_CONVERTER_VALIDATE_SERIALIZATION_GUARD=0 \
  DRAWTHINGS_CONVERTER_ALIGN_METADATA_MODE=0 \
  bash "$ROOT/tools/dt_convert_model.sh" \
    --file "$SAFETENSORS" \
    --name "$NAME" \
    --output-directory "$MODELS_DIR"
  CONVERTER_OUT="$MODELS_DIR/${NAME}_f16.ckpt"
  if [[ ! -f "$CONVERTER_OUT" ]]; then
    echo "error: converter output not found: $CONVERTER_OUT" >&2; exit 1
  fi
  echo "==> [2b/6] Validate converted ckpt against SOURCE safetensors"
  "$PYTHON_BIN" "$ROOT/tools/dt_validate_converted_ckpt.py" \
    --file "$F16" \
    --profile auto \
    --source-safetensors "$SAFETENSORS" \
    --serialization-guard-profile none \
    --serialization-report-limit 5 || {
      echo "warn: validation reported issues (continuing to record evidence)" >&2; }
fi

# -------------------------------------------------------------------
# Stage 3: optional quantize
# -------------------------------------------------------------------
INFER_FILE="$(basename "$F16")"
if [[ -n "$QUANTIZE" ]]; then
  QOUT="$MODELS_DIR/${NAME}_${QUANTIZE}.ckpt"
  echo "==> [3/6] Quantize f16 -> $QUANTIZE"
  bash "$ROOT/tools/dt_quantize_model.sh" \
    -i "$F16" \
    -m ltx2.3 \
    -o "$QOUT" \
    --target-codec "$QUANTIZE"
  if [[ ! -f "$QOUT" ]]; then
    echo "error: quantizer output not found: $QOUT" >&2; exit 1
  fi
  INFER_FILE="$(basename "$QOUT")"
else
  echo "==> [3/6] Quantize skipped (f16 inference first)"
fi

if [[ "$RAW_ONLY" == "1" ]]; then
  echo "==> raw-only mode: stopping before inference."
  echo "    convert output ready: $F16"
  [[ -n "$QUANTIZE" ]] && echo "    quantized output ready: $MODELS_DIR/${NAME}_${QUANTIZE}.ckpt"
  exit 0
fi

# -------------------------------------------------------------------
# Stage 4: upsert custom.json alias (ltx2.3 schema with companions)
# -------------------------------------------------------------------
[[ -z "$CLIP_ENCODER" ]] && CLIP_ENCODER="$INFER_FILE"
echo "==> [4/6] Upsert custom.json alias '$NAME' -> $INFER_FILE"
if [[ -f "$CUSTOM_JSON" ]]; then
  cp -f "$CUSTOM_JSON" "$CUSTOM_JSON.bak_${TS}"
fi
"$PYTHON_BIN" - "$CUSTOM_JSON" "$NAME" "$INFER_FILE" \
  "$AUTOENCODER" "$CLIP_ENCODER" "$TEXT_ENCODER" "$MODIFIER" <<'PY'
import json, sys, os
path, name, file, ae, clip, te, modifier = sys.argv[1:8]
entries = []
if os.path.exists(path):
    with open(path) as f:
        entries = json.load(f)
entry = {
    "name": name,
    "version": "ltx2.3",
    "autoencoder": ae,
    "clip_encoder": clip,
    "prefix": "",
    "modifier": modifier,
    "default_scale": 12,
    "hires_fix_scale": 24,
    "latents_upscalers": [],
    "file": file,
    "upcast_attention": False,
    "text_encoder": te,
    "high_precision_autoencoder": False,
    "objective": {"u": {"condition_scale": 1000}},
}
entries = [e for e in entries if e.get("name") != name]
entries.append(entry)
with open(path, "w") as f:
    json.dump(entries, f, indent=2)
print(f"upserted alias '{name}' (file={file}, clip={clip})")
PY

# -------------------------------------------------------------------
# Stage 5: first-frame canary inference
# -------------------------------------------------------------------
CANARY_TAG="${NAME}_firstframe_${TS}"
echo "==> [5/6] First-frame canary (timeout ${TIMEOUT_SEC}s)"
set +e
env LD_LIBRARY_PATH=/usr/local/swift/usr/lib/swift/linux:/usr/lib/wsl/lib:/usr/local/cuda/targets/x86_64-linux/lib \
bash "$ROOT/tools/run_q6p_canary_once.sh" \
  --model "$NAME" \
  --tag "$CANARY_TAG" \
  --timeout-sec "$TIMEOUT_SEC" \
  --max-responses 0 \
  --require-complete-stream \
  --require-final-output \
  --width "$WIDTH" --height "$HEIGHT" \
  --steps "$STEPS" --seed "$SEED" \
  --allow-missing-model \
  --soft-fail
CANARY_RC=$?
set -e
CANARY_DIR="$ROOT/output/q6p_canary_${CANARY_TAG}"
echo "    canary_rc=$CANARY_RC  dir=$CANARY_DIR"

# -------------------------------------------------------------------
# Stage 6: render first frame
# -------------------------------------------------------------------
if [[ "$DO_RENDER" == "1" ]]; then
  echo "==> [6/6] Render first output bin -> PNG"
  IMG_BIN="$(ls -1 "$CANARY_DIR"/image_*_01.bin 2>/dev/null | head -n1 || true)"
  if [[ -z "$IMG_BIN" ]]; then
    IMG_BIN="$(ls -1 "$CANARY_DIR"/*.bin 2>/dev/null | head -n1 || true)"
  fi
  if [[ -n "$IMG_BIN" ]]; then
    "$PYTHON_BIN" "$ROOT/tools/dt_tensor_to_playable.py" \
      --image-bin "$IMG_BIN" \
      --out-dir "$CANARY_DIR/media" \
      --base-name "${NAME}_firstframe" || true
    echo "    rendered from: $IMG_BIN"
    echo "    media dir    : $CANARY_DIR/media"
  else
    echo "    no output bin found to render (inference likely did not produce a frame)"
  fi
else
  echo "==> [6/6] Render skipped (pass --render to enable)"
fi

echo "=============================================="
echo "DONE. Acceptance = COHERENT frame, not just rc=0."
echo "  f16 model  : $F16"
[[ -n "$QUANTIZE" ]] && echo "  quant model: $MODELS_DIR/${NAME}_${QUANTIZE}.ckpt"
echo "  canary dir : $CANARY_DIR"
echo "=============================================="
