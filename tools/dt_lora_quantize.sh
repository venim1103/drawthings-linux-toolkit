#!/usr/bin/env bash
set -euo pipefail

# dt_lora_quantize.sh — Quantize an LTX2.3 LoRA f16 ckpt to a smaller codec
# (q8p / q6p / q4p) and register it in dt-models/custom_lora.json.
#
#   bash tools/dt_lora_quantize.sh <lora_f16.ckpt|NAME> [--codec q8p|q6p|q4p] [options]
#
# Required:
#   <lora_f16.ckpt|NAME>  Path/filename of a converted LoRA f16 ckpt, or a name
#                         registered in custom_lora.json.
# Optional:
#   --codec CODEC         q8p (default), q6p, or q4p.
#   --name NAME           Display name for custom_lora.json (default: derived).
#   --force               Re-quantize even if the output ckpt already exists.
#   --trace               Write the LTX per-tensor quantization decision JSONL to
#                         output/<out>.trace.jsonl.
#   -h, --help            Show this help.
#
# Notes:
#   - Uses the existing model-quantizer with -m ltx2.3. Its LTX heuristics keep
#     the LoRA's sensitive tensors safe automatically:
#       * embedder / proj_out up+down  -> fp16 (preserved)
#       * rank-1 / adaln / scalar rows -> ezm7  (near-lossless)
#       * rank>1 attention/ffn deltas  -> [codec, ezm7]
#   - Unlike the base model, LoRAs need NO separate F32 norm-restore step.
#   - All of q8p/q6p/q4p load and apply at runtime (s4nnc dequantizes the
#     self-describing palettized blobs). Quality drift grows with smaller codecs;
#     q8p is the recommended default.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT/.venv/bin/python"
MODELS_DIR="$ROOT/dt-models"
CUSTOM_LORA_JSON="$MODELS_DIR/custom_lora.json"

INPUT=""
CODEC="q8p"
NAME=""
FORCE=0
TRACE=0

usage() { awk '/^#!/{next} /^#/{sub(/^# ?/,"");print;c=1;next} c{exit}' "$0"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --codec) CODEC="${2:-}"; shift 2 ;;
    --name) NAME="${2:-}"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --trace) TRACE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "error: unknown option: $1" >&2; usage; exit 1 ;;
    *)
      if [[ -z "$INPUT" ]]; then INPUT="$1"; shift
      else echo "error: unexpected argument: $1" >&2; exit 1; fi ;;
  esac
done

if [[ -z "$INPUT" ]]; then echo "error: <lora_f16.ckpt|NAME> is required" >&2; usage; exit 1; fi
case "$CODEC" in q8p|q6p|q4p) ;; *) echo "error: --codec must be q8p, q6p, or q4p" >&2; exit 1 ;; esac
if [[ ! -x "$PYTHON_BIN" ]]; then echo "error: missing python env at $PYTHON_BIN" >&2; exit 1; fi

# Resolve INPUT -> an f16 ckpt file inside dt-models.
resolve_lora_file() {
  "$PYTHON_BIN" - "$INPUT" "$CUSTOM_LORA_JSON" <<'PY'
import json, os, sys
query, jp = sys.argv[1], sys.argv[2]
q = os.path.basename(query)
# direct file?
if q.endswith(".ckpt"):
    print(q); raise SystemExit
if os.path.exists(jp):
    try:
        for e in json.load(open(jp)):
            if isinstance(e, dict) and e.get("name") == query and e.get("file"):
                print(e["file"]); raise SystemExit
    except SystemExit:
        raise
    except Exception:
        pass
print(q)
PY
}

IN_FILE="$(resolve_lora_file)"
IN_PATH="$MODELS_DIR/$IN_FILE"
if [[ ! -f "$IN_PATH" ]]; then
  echo "error: input LoRA ckpt not found: $IN_PATH" >&2; exit 1
fi

# Output filename: swap a trailing _f16 for _<codec>, else append _<codec>.
BASE="${IN_FILE%.ckpt}"
if [[ "$BASE" == *_f16 ]]; then
  OUT_FILE="${BASE%_f16}_${CODEC}.ckpt"
else
  OUT_FILE="${BASE}_${CODEC}.ckpt"
fi
OUT_PATH="$MODELS_DIR/$OUT_FILE"

if [[ -z "$NAME" ]]; then
  NAME="${OUT_FILE%.ckpt}"
fi

echo "== dt_lora_quantize =="
echo "  input : $IN_PATH"
echo "  codec : $CODEC"
echo "  output: $OUT_PATH"
echo "  name  : $NAME"

if [[ -f "$OUT_PATH" && "$FORCE" != "1" ]]; then
  echo "==> output already exists; skipping quantization (pass --force to rebuild)."
else
  rm -f "$OUT_PATH" "$OUT_PATH-tensordata" "$OUT_PATH-journal"
  TRACE_ARGS=()
  if [[ "$TRACE" == "1" ]]; then
    mkdir -p "$ROOT/output"
    TRACE_ARGS=(--ltx-trace-output "$ROOT/output/${OUT_FILE%.ckpt}.trace.jsonl")
  fi
  echo "==> Quantizing LoRA f16 -> $CODEC (-m ltx2.3)"
  # q4p is gated for ltx2.3 base models, but LoRAs are safe: opt in here.
  DRAWTHINGS_QUANTIZER_ALLOW_Q4P_LTX23=1 \
    bash "$ROOT/tools/dt_quantize_model.sh" \
      -i "$IN_PATH" -m ltx2.3 -o "$OUT_PATH" --target-codec "$CODEC" "${TRACE_ARGS[@]}"
  [[ -f "$OUT_PATH" ]] || { echo "error: quantizer output not found: $OUT_PATH" >&2; exit 1; }
fi

# Register in custom_lora.json.
if [[ -f "$CUSTOM_LORA_JSON" ]]; then
  cp -f "$CUSTOM_LORA_JSON" "$CUSTOM_LORA_JSON.bak_$(date +%Y%m%d_%H%M%S)"
fi
"$PYTHON_BIN" - "$CUSTOM_LORA_JSON" "$NAME" "$OUT_FILE" <<'PY'
import json, os, sys
path, name, file_name = sys.argv[1:4]
entries = json.load(open(path)) if os.path.exists(path) else []
entries = [e for e in entries if not (isinstance(e, dict) and (e.get("name") == name or e.get("file") == file_name))]
entries.append({"name": name, "file": file_name, "prefix": "", "version": "ltx2.3", "is_lo_ha": False})
json.dump(entries, open(path, "w"), indent=2)
print(f"  custom_lora '{name}' -> {file_name} (ltx2.3)")
PY

echo ""
echo "OK: $OUT_PATH"
echo "Next: bash tools/dt_test_distilled_lora.sh official_q6p_via_custom $NAME:1.0"
