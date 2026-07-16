#!/usr/bin/env bash
set -euo pipefail

# dt_lora_convert.sh — Convert a LoRA safetensors file into Draw Things ckpt
# and register it in dt-models/custom_lora.json.
#
#   bash tools/dt_lora_convert.sh <lora.safetensors> [--name NAME] [options]
#
# Required:
#   <lora.safetensors>    Path to source LoRA safetensors.
# Optional:
#   --name NAME           Display name for custom_lora.json.
#                         Default: derived from source filename.
#   --version VERSION     Force model version for import (default: ltx2.3).
#   --scale-factor N      LoRA network scale factor (default: 1.0).
#   --force               Re-convert even when output ckpt already exists.
#   -h, --help            Show this help.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$ROOT/dt-models"
PKG_PATH="$ROOT/draw-things-community"
CUSTOM_LORA_JSON="$MODELS_DIR/custom_lora.json"
PYTHON_BIN="$ROOT/.venv/bin/python"

BUILD_CONFIG="${DRAWTHINGS_BUILD_CONFIG:-release}"
AUTOBUILD="${DRAWTHINGS_LORA_CONVERTER_AUTOBUILD:-${DRAWTHINGS_CONVERTER_AUTOBUILD:-1}}"

SAFETENSORS=""
NAME=""
VERSION="${DT_LORA_VERSION:-ltx2.3}"
SCALE_FACTOR="${DT_LORA_SCALE_FACTOR:-1.0}"
FORCE=0

usage() { awk '/^#!/{next} /^#/{sub(/^# ?/,"");print;c=1;next} c{exit}' "$0"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="${2:-}"; shift 2 ;;
    --version) VERSION="${2:-}"; shift 2 ;;
    --scale-factor) SCALE_FACTOR="${2:-}"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "error: unknown option: $1" >&2; usage; exit 1 ;;
    *)
      if [[ -z "$SAFETENSORS" ]]; then SAFETENSORS="$1"; shift
      else echo "error: unexpected argument: $1" >&2; exit 1; fi ;;
  esac
done

if [[ -z "$SAFETENSORS" ]]; then
  echo "error: <lora.safetensors> is required" >&2
  usage
  exit 1
fi

[[ "$SAFETENSORS" != /* ]] && SAFETENSORS="$ROOT/$SAFETENSORS"
if [[ ! -f "$SAFETENSORS" ]]; then
  echo "error: safetensors not found: $SAFETENSORS" >&2
  exit 1
fi

if [[ -z "$NAME" ]]; then
  NAME="$(basename "$SAFETENSORS")"
  NAME="${NAME%.safetensors}"
fi

if [[ -z "$VERSION" ]]; then
  echo "error: --version must not be empty" >&2
  exit 1
fi

if ! [[ "$SCALE_FACTOR" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "error: --scale-factor must be a non-negative number" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "error: missing python env at $PYTHON_BIN" >&2
  exit 1
fi

sanitize_name() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
name = sys.argv[1]
out = []
for ch in name:
    if ord(ch) < 128 and (ch.isalnum() or ch == "."):
        out.append(ch.lower())
    else:
        out.append("_")
print("".join(out))
PY
}

resolve_converter_bin() {
  local primary="$PKG_PATH/.build/$BUILD_CONFIG/lora-converter"
  local legacy="$PKG_PATH/.build/$BUILD_CONFIG/LoRAConverter"

  if [[ -x "$primary" ]]; then
    echo "$primary"
    return 0
  fi
  if [[ -x "$legacy" ]]; then
    echo "$legacy"
    return 0
  fi

  if [[ "$AUTOBUILD" != "1" ]]; then
    echo "error: lora-converter not found and auto-build disabled" >&2
    return 1
  fi

  echo "==> Building lora-converter ($BUILD_CONFIG)..." >&2
  swift build --package-path "$PKG_PATH" -c "$BUILD_CONFIG" --product lora-converter

  if [[ -x "$primary" ]]; then
    echo "$primary"
    return 0
  fi
  if [[ -x "$legacy" ]]; then
    echo "$legacy"
    return 0
  fi

  echo "error: lora-converter still not found after build" >&2
  return 1
}

SANITIZED_NAME="$(sanitize_name "$NAME")"
OUT_FILE="${SANITIZED_NAME}_lora_f16.ckpt"
OUT_PATH="$MODELS_DIR/$OUT_FILE"
TMP_SPEC=""

cleanup() {
  if [[ -n "$TMP_SPEC" && -f "$TMP_SPEC" ]]; then
    rm -f "$TMP_SPEC"
  fi
}
trap cleanup EXIT

echo "== dt_lora_convert =="
echo "  safetensors  : $SAFETENSORS"
echo "  name         : $NAME"
echo "  version      : $VERSION"
echo "  scale-factor : $SCALE_FACTOR"
echo "  output (hint): $OUT_PATH"

if [[ -f "$OUT_PATH" && "$FORCE" != "1" ]]; then
  echo "==> output already exists; skipping conversion (pass --force to rebuild)."
else
  CONVERTER_BIN="$(resolve_converter_bin)"
  TMP_SPEC="$(mktemp)"

  CONVERT_CMD=(
    "$CONVERTER_BIN"
    --file "$SAFETENSORS"
    --name "$NAME"
    --version "$VERSION"
    --scale-factor "$SCALE_FACTOR"
    --output-directory "$MODELS_DIR"
  )

  echo "==> Converting LoRA"
  "${CONVERT_CMD[@]}" > "$TMP_SPEC"

  PARSED_FILE="$($PYTHON_BIN - "$TMP_SPEC" <<'PY'
import json
import sys

spec_path = sys.argv[1]
payload = json.loads(open(spec_path, "r", encoding="utf-8").read())
file_name = str(payload.get("file", "")).strip()
print(file_name)
PY
)"

  if [[ -n "$PARSED_FILE" ]]; then
    OUT_FILE="$PARSED_FILE"
    OUT_PATH="$MODELS_DIR/$OUT_FILE"
  fi
fi

if [[ ! -f "$OUT_PATH" ]]; then
  echo "error: expected converted LoRA not found: $OUT_PATH" >&2
  exit 1
fi

if [[ -f "$CUSTOM_LORA_JSON" ]]; then
  cp -f "$CUSTOM_LORA_JSON" "$CUSTOM_LORA_JSON.bak_$(date +%Y%m%d_%H%M%S)"
fi

"$PYTHON_BIN" - "$CUSTOM_LORA_JSON" "$NAME" "$OUT_FILE" "$VERSION" "${TMP_SPEC:-}" <<'PY'
import json
import os
import sys

path, fallback_name, fallback_file, fallback_version, spec_path = sys.argv[1:6]

entries = []
if os.path.exists(path):
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    if isinstance(loaded, list):
        entries = loaded

spec = {}
if spec_path and os.path.exists(spec_path):
    with open(spec_path, "r", encoding="utf-8") as f:
        loaded_spec = json.load(f)
    if isinstance(loaded_spec, dict):
        spec = loaded_spec

name = str(spec.get("name") or fallback_name).strip() or fallback_name
file_name = str(spec.get("file") or fallback_file).strip() or fallback_file
version = str(spec.get("version") or fallback_version).strip() or fallback_version
prefix = spec.get("prefix", "")
if not isinstance(prefix, str):
    prefix = ""

entry = {
    "name": name,
    "file": file_name,
    "prefix": prefix,
    "version": version,
}

is_lo_ha = spec.get("is_lo_ha")
if isinstance(is_lo_ha, bool):
    entry["is_lo_ha"] = is_lo_ha

entries = [
    e for e in entries
    if not (isinstance(e, dict) and (e.get("name") == name or e.get("file") == file_name))
]
entries.append(entry)

with open(path, "w", encoding="utf-8") as f:
    json.dump(entries, f, indent=2)
print(f"  custom_lora '{name}' -> {file_name} ({version})")
PY

echo ""
echo "OK: $OUT_PATH"
echo "Registered: $CUSTOM_LORA_JSON"
echo "Next: bash tools/dt_test.sh <MODEL_ALIAS> --lora $OUT_FILE:0.8"
