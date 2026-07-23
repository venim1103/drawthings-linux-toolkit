#!/usr/bin/env bash
set -euo pipefail

# dt_convert.sh — Step 1/3: convert an LTX2.3 .safetensors into a Draw Things f16
# checkpoint, with the toolkit's converter fixes baked in (§17 rotary
# de-interleave + §18 QK-norm de-interleave).
#
#   bash tools/dt_convert.sh <model.safetensors> [--name NAME] [--force]
#
# Required:
#   <model.safetensors>   Path to the source safetensors (positional).
# Optional:
#   --name NAME           Output/alias base name. Default: derived from the
#                         filename (strips .safetensors and a trailing
#                         _bf16/_fp16/_f16). The converter normalizes model
#                         filenames (non-alnum/dot -> _, lowercase); this
#                         wrapper auto-detects the produced *_f16.ckpt.
#   --force               Re-convert even if <NAME>_f16.ckpt already exists.
#   -h, --help            Show this help.
#
# Next step: bash tools/dt_quantize.sh <NAME>

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT/.venv/bin/python"
MODELS_DIR="$ROOT/dt-models"

SAFETENSORS=""
NAME=""
FORCE=0

usage() { awk '/^#!/{next} /^#/{sub(/^# ?/,"");print;c=1;next} c{exit}' "$0"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="${2:-}"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "error: unknown option: $1" >&2; usage; exit 1 ;;
    *)
      if [[ -z "$SAFETENSORS" ]]; then SAFETENSORS="$1"; shift
      else echo "error: unexpected argument: $1" >&2; exit 1; fi ;;
  esac
done

if [[ -z "$SAFETENSORS" ]]; then
  echo "error: <model.safetensors> is required" >&2; usage; exit 1
fi
[[ "$SAFETENSORS" != /* ]] && SAFETENSORS="$ROOT/$SAFETENSORS"
if [[ ! -f "$SAFETENSORS" ]]; then
  echo "error: safetensors not found: $SAFETENSORS" >&2; exit 1
fi

if [[ -z "$NAME" ]]; then
  NAME="$(basename "$SAFETENSORS")"
  NAME="${NAME%.safetensors}"
  NAME="${NAME%_bf16}"; NAME="${NAME%_fp16}"; NAME="${NAME%_f16}"
fi

CLEAN_NAME="$($PYTHON_BIN - "$NAME" <<'PY'
import sys

name = sys.argv[1]
out = []
for ch in name:
    if ch.isascii() and (ch.isalpha() or ch.isdigit() or ch == "."):
        out.append(ch)
    else:
        out.append("_")
print("".join(out).lower())
PY
)"

REQUESTED_F16="$MODELS_DIR/${NAME}_f16.ckpt"
CLEAN_F16="$MODELS_DIR/${CLEAN_NAME}_f16.ckpt"
RESOLVED_NAME="$NAME"
F16="$REQUESTED_F16"

resolve_f16_output() {
  if [[ "$NAME" != "$CLEAN_NAME" && -f "$CLEAN_F16" ]]; then
    F16="$CLEAN_F16"
    RESOLVED_NAME="$CLEAN_NAME"
  elif [[ -f "$REQUESTED_F16" ]]; then
    F16="$REQUESTED_F16"
    RESOLVED_NAME="$NAME"
  else
    F16="$REQUESTED_F16"
    RESOLVED_NAME="$NAME"
  fi
}

resolve_f16_output

echo "== dt_convert =="
echo "  safetensors : $SAFETENSORS"
echo "  name        : $NAME"
if [[ "$NAME" != "$CLEAN_NAME" ]]; then
  echo "  normalized  : $CLEAN_NAME"
fi
echo "  output      : $F16"

if [[ -f "$F16" && "$FORCE" != "1" ]]; then
  echo "==> f16 already exists; skipping (pass --force to rebuild)."
  echo "OK: $F16"
  echo "Next: bash tools/dt_quantize.sh $RESOLVED_NAME"
  exit 0
fi

echo "==> [1/3] Preflight"
"$PYTHON_BIN" "$ROOT/tools/dt_preflight_safetensors.py" --file "$SAFETENSORS" || {
  echo "warn: preflight reported issues (continuing)" >&2; }

echo "==> [2/3] Convert (raw, no align-to-official)"
DRAWTHINGS_CONVERTER_VALIDATE_SERIALIZATION_GUARD=0 \
DRAWTHINGS_CONVERTER_ALIGN_METADATA_MODE=0 \
  bash "$ROOT/tools/dt_convert_model.sh" \
    --file "$SAFETENSORS" \
    --name "$NAME" \
    --output-directory "$MODELS_DIR"

resolve_f16_output
if [[ ! -f "$F16" ]]; then
  echo "error: converter output not found: $REQUESTED_F16" >&2
  if [[ "$REQUESTED_F16" != "$CLEAN_F16" ]]; then
    echo "       also checked normalized path: $CLEAN_F16" >&2
  fi
  exit 1
fi

echo "==> [3/3] Validate converted checkpoint"
"$PYTHON_BIN" "$ROOT/tools/dt_validate_converted_ckpt.py" --file "$F16" || {
  echo "warn: validation reported issues (continuing)" >&2; }

echo ""
echo "OK: $F16"
echo "Next: bash tools/dt_quantize.sh $RESOLVED_NAME   # q6p (default); or --codec q8p / q4p"
