#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspaces/drawthings-linux-toolkit"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage:
  tools/run_convert_safetensors_to_f16.sh <source_safetensors> <output_f16_ckpt> [name] [baseline_ckpt]

Examples:
  tools/run_convert_safetensors_to_f16.sh \
    dt-models/10_e_v1_bf16.safetensors \
    dt-models/10_e_v1_bf16_regen_20260603_f16.ckpt

  tools/run_convert_safetensors_to_f16.sh \
    dt-models/other_model.safetensors \
    dt-models/other_model_f16.ckpt \
    other_model_name \
    dt-models/other_model_official_f16.ckpt

Notes:
  - If [name] is omitted, it is inferred from <output_f16_ckpt>.
  - If [baseline_ckpt] is omitted, default is:
      dt-models/ltx_2.3_22b_distilled_f16.ckpt
  - Backward compatibility:
      If 3rd arg looks like a .ckpt path and 4th arg is omitted, it is treated
      as [baseline_ckpt] and name is inferred.
  - The converter emits <name>_f16.ckpt; this script renames/moves it to the
    exact output path you pass.
EOF
}

if [[ $# -lt 2 || $# -gt 4 ]]; then
  usage
  exit 1
fi

SRC="$1"
F16_TARGET="$2"
NAME="${3:-}"
BASELINE="${4:-$ROOT/dt-models/ltx_2.3_22b_distilled_f16.ckpt}"

if [[ -n "$NAME" && -z "${4:-}" ]]; then
  if [[ "$NAME" == *.ckpt || "$NAME" == */* ]]; then
    BASELINE="$NAME"
    NAME=""
  fi
fi

if [[ "$SRC" != /* ]]; then
  SRC="$ROOT/$SRC"
fi

if [[ "$F16_TARGET" != /* ]]; then
  F16_TARGET="$ROOT/$F16_TARGET"
fi

if [[ "$BASELINE" != /* ]]; then
  BASELINE="$ROOT/$BASELINE"
fi

OUT_DIR="$(dirname "$F16_TARGET")"
mkdir -p "$OUT_DIR"

if [[ -z "$NAME" ]]; then
  inferred="$(basename "$F16_TARGET")"
  inferred="${inferred%.ckpt}"
  NAME="${inferred%_f16}"
fi

if [[ -z "$NAME" ]]; then
  echo "error: could not infer converter name from output path; pass [name] explicitly" >&2
  exit 1
fi

F16_CONVERTER_OUT="$OUT_DIR/${NAME}_f16.ckpt"
F16="$F16_TARGET"
PYTHON_BIN="$ROOT/.venv/bin/python"

if [[ ! -f "$SRC" ]]; then
  echo "error: source safetensors not found: $SRC" >&2
  exit 1
fi

if [[ ! -f "$BASELINE" ]]; then
  echo "error: baseline checkpoint not found: $BASELINE" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "error: python environment not found at: $PYTHON_BIN" >&2
  exit 1
fi

echo "==> Converting safetensors to f16 ckpt"
echo "    name: $NAME"
echo "    src : $SRC"
echo "    f16 : $F16"
echo "    base: $BASELINE"

# Disable hard-fail serialization guard during conversion.
# We run strict validation after metadata alignment below.
DRAWTHINGS_CONVERTER_VALIDATE_SERIALIZATION_GUARD=0 \
tools/dt_convert_model.sh \
  --file "$SRC" \
  --name "$NAME" \
  --output-directory "$OUT_DIR"

if [[ ! -f "$F16_CONVERTER_OUT" ]]; then
  echo "error: converter output not found: $F16_CONVERTER_OUT" >&2
  exit 1
fi

if [[ "$F16_CONVERTER_OUT" != "$F16_TARGET" ]]; then
  echo "==> Moving converter output to requested path"
  mv -f "$F16_CONVERTER_OUT" "$F16_TARGET"
fi

echo "==> Aligning metadata to official baseline"
"$PYTHON_BIN" tools/dt_align_ckpt_metadata.py \
  --file "$F16" \
  --baseline "$BASELINE" \
  --mode apply \
  --sample-limit 5

echo "==> Running strict post-align validation"
"$PYTHON_BIN" tools/dt_validate_converted_ckpt.py \
  --file "$F16" \
  --profile auto \
  --source-safetensors "$SRC" \
  --serialization-baseline "$BASELINE" \
  --serialization-guard-profile any \
  --serialization-report-limit 5

echo "==> Done"
echo "    output: $F16"