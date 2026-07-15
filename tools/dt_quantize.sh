#!/usr/bin/env bash
set -euo pipefail

# dt_quantize.sh — Step 2/3: quantize an LTX2.3 f16 checkpoint to a GPU/mobile
# friendly codec (q6p / q8p / q4p), restore the sensitive norms to full
# precision, and register a custom.json alias. The output .ckpt is
# self-contained and portable (all fixes are written into the file).
#
#   bash tools/dt_quantize.sh <NAME> [--codec q6p|q8p|q4p] [options]
#
# Required:
#   <NAME>                Base name from dt_convert (uses dt-models/<NAME>_f16.ckpt).
# Optional:
#   --codec CODEC         q6p (default, ~20GB), q8p (larger, higher quality),
#                         or q4p (smallest, lowest precision).
#   --force               Re-quantize even if <NAME>_<codec>.ckpt exists.
#   --restore-only        Skip quantization; (re)run the norm restores + alias on an
#                         existing <NAME>_<codec>.ckpt. Use this to supply --ref-q6p
#                         after a quantize that finished but had no reference.
#   --no-restore          Skip the F32 norm restores (debugging only; will diverge).
#   --ref-q6p PATH        OPTIONAL official q6p template for the F32 norm set/dims.
#                         Omitted by default: the ada_ln norms are reconstructed
#                         reference-free. Auto-used if this path exists:
#                         dt-models/ltx_2.3_22b_distilled_1.1_q6p.ckpt
#   --model-version VER   Quantizer model version. Default: ltx2.3
#   -h, --help            Show this help.
#
# Next step: bash tools/dt_test.sh <NAME>

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT/.venv/bin/python"
MODELS_DIR="$ROOT/dt-models"
CUSTOM_JSON="$MODELS_DIR/custom.json"

NAME=""
CODEC="q6p"
FORCE=0
NO_RESTORE=0
RESTORE_ONLY=0
REF_Q6P="$MODELS_DIR/ltx_2.3_22b_distilled_1.1_q6p.ckpt"
MODEL_VERSION="ltx2.3"

# ltx2.3 inference companions (defaults matching the proven-working alias)
AUTOENCODER="ltx_2.3_audio_video_vae_f16.ckpt"
TEXT_ENCODER="gemma_3_12b_it_qat_q8p.ckpt"
MODIFIER="kontext"

usage() { awk '/^#!/{next} /^#/{sub(/^# ?/,"");print;c=1;next} c{exit}' "$0"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --codec) CODEC="${2:-}"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --restore-only) RESTORE_ONLY=1; shift ;;
    --no-restore) NO_RESTORE=1; shift ;;
    --ref-q6p) REF_Q6P="${2:-}"; shift 2 ;;
    --model-version) MODEL_VERSION="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "error: unknown option: $1" >&2; usage; exit 1 ;;
    *)
      if [[ -z "$NAME" ]]; then NAME="$1"; shift
      else echo "error: unexpected argument: $1" >&2; exit 1; fi ;;
  esac
done

if [[ -z "$NAME" ]]; then echo "error: <NAME> is required" >&2; usage; exit 1; fi
# Accept either a bare name (10_e_v1_4) or a path to the f16 ckpt.
NAME="$(basename "$NAME")"; NAME="${NAME%.ckpt}"; NAME="${NAME%_f16}"
case "$CODEC" in q6p|q8p|q4p) ;; *) echo "error: --codec must be q6p, q8p, or q4p" >&2; exit 1 ;; esac

F16="$MODELS_DIR/${NAME}_f16.ckpt"
QOUT="$MODELS_DIR/${NAME}_${CODEC}.ckpt"
QFILE="$(basename "$QOUT")"

if [[ ! -f "$F16" ]]; then
  echo "error: f16 not found: $F16 (run dt_convert.sh first)" >&2; exit 1
fi

# Decide whether the ~1-2h quantization step will actually run.
WILL_QUANTIZE=1
if [[ "$RESTORE_ONLY" == "1" ]]; then
  WILL_QUANTIZE=0
elif [[ -f "$QOUT" && "$FORCE" != "1" ]]; then
  WILL_QUANTIZE=0
fi

# --restore-only needs an existing quantized model to fix.
if [[ "$RESTORE_ONLY" == "1" && ! -f "$QOUT" ]]; then
  echo "error: --restore-only given but no quantized model exists: $QOUT" >&2
  echo "       Run without --restore-only to quantize first." >&2
  exit 1
fi

# The norm restore is REFERENCE-FREE by default (it reconstructs the ada_ln F32
# set from the source). An official q6p is only an optional higher-fidelity
# template: use it if --ref-q6p was given, or if the default path is on disk.
REF_ARGS=()
REF_MODE=""
if [[ "$NO_RESTORE" != "1" ]]; then
  if [[ -f "$REF_Q6P" ]]; then
    REF_ARGS=(--ref-q6p "$REF_Q6P")
    REF_MODE="reference template ($REF_Q6P)"
  else
    REF_MODE="reference-free (ada_ln reconstruction)"
  fi
fi

echo "== dt_quantize =="
echo "  name   : $NAME"
echo "  input  : $F16"
echo "  codec  : $CODEC"
echo "  output : $QOUT"
if [[ "$WILL_QUANTIZE" == "0" && "$NO_RESTORE" != "1" ]]; then
  echo "  mode   : RESTORE-ONLY (model exists; skipping the ~1-2h quantize, just fixing norms)"
fi
if [[ "$NO_RESTORE" != "1" ]]; then
  echo "  restore: $REF_MODE"
fi

# --- quantize ---
if [[ "$WILL_QUANTIZE" == "0" ]]; then
  if [[ "$RESTORE_ONLY" == "1" ]]; then
    echo "==> [1/4] Quantize SKIPPED (--restore-only)"
  else
    echo "==> [1/4] Quantize SKIPPED (exists; pass --force to rebuild)"
  fi
else
  echo "==> [1/4] Quantize f16 -> $CODEC (CPU-bound; ~1-2h)"
  rm -f "$QOUT" "$QOUT-tensordata" "$QOUT-journal"
  # q4p is blocked for ltx2.3 by default; opt in when the user asks for it.
  DRAWTHINGS_QUANTIZER_ALLOW_Q4P_LTX23=1 \
    bash "$ROOT/tools/dt_quantize_model.sh" \
      -i "$F16" -m "$MODEL_VERSION" -o "$QOUT" --target-codec "$CODEC"
  [[ -f "$QOUT" ]] || { echo "error: quantizer output not found: $QOUT" >&2; exit 1; }
fi

# --- restore full-precision norms (the step that makes it render coherently) ---
if [[ "$NO_RESTORE" == "1" ]]; then
  echo "==> [2/4] + [3/4] Norm restores SKIPPED (--no-restore)"
else
  echo "==> [2/4] Restore F32 ada_ln/modulation norms (own values, widened)"
  "$PYTHON_BIN" "$ROOT/tools/dt_q6p_restore_f32_norms.py" \
    --q6p "$QOUT" --f16-source "$F16" "${REF_ARGS[@]}" | tail -6

  echo "==> [3/4] Restore F32 QK-norms with [1,1,N] dim fix (own values)"
  "$PYTHON_BIN" "$ROOT/tools/dt_q6p_restore_qk_norms.py" \
    --q6p "$QOUT" --f16-source "$F16" | tail -5
fi

# --- register custom.json alias ---
echo "==> [4/4] Upsert custom.json alias '$NAME' -> $QFILE"
if [[ -f "$CUSTOM_JSON" ]]; then cp -f "$CUSTOM_JSON" "$CUSTOM_JSON.bak_$(date +%Y%m%d_%H%M%S)"; fi
"$PYTHON_BIN" - "$CUSTOM_JSON" "$NAME" "$QFILE" "$AUTOENCODER" "$TEXT_ENCODER" "$MODIFIER" <<'PY'
import json, os, sys
path, name, qfile, ae, te, modifier = sys.argv[1:7]
entries = json.load(open(path)) if os.path.exists(path) else []
entries = [e for e in entries if e.get("name") != name]
entries.append({
    "name": name, "version": "ltx2.3",
    "autoencoder": ae, "clip_encoder": qfile, "prefix": "", "modifier": modifier,
    "default_scale": 12, "hires_fix_scale": 24, "latents_upscalers": [],
    "file": qfile, "upcast_attention": False,
    "text_encoder": te, "high_precision_autoencoder": False,
    "objective": {"u": {"condition_scale": 1000}},
})
json.dump(entries, open(path, "w"), indent=2)
print(f"  alias '{name}' -> {qfile}")
PY

echo ""
echo "OK: $QOUT"
echo "    Self-contained & portable: all fixes are baked into this single file."
echo "    For mobile Draw Things, copy this .ckpt + companions:"
echo "      - $MODELS_DIR/$AUTOENCODER"
echo "      - $MODELS_DIR/$TEXT_ENCODER (+ its -tensordata)"
echo "Next: bash tools/dt_test.sh $NAME"
