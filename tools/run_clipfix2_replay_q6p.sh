#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT/.venv/bin/python"
QUANT_WRAPPER="$ROOT/tools/dt_quantize_model.sh"
DEFAULT_OFFICIAL_BASELINE="$ROOT/dt-models/ltx_2.3_22b_distilled_1.1_q6p.ckpt"

SRC=""
OUT=""
OFFICIAL_BASELINE="$DEFAULT_OFFICIAL_BASELINE"
BACKUP=""
SAMPLE_LIMIT=12
PROGRESS_EVERY=50
TAG="$(date +%Y%m%d)"
SKIP_BACKUP=0
SKIP_PROBES=0

usage() {
  cat <<'EOF'
Usage:
  tools/run_clipfix2_replay_q6p.sh -i <source_f16_ckpt> -o <output_q6p_ckpt> [options]

Required:
  -i, --input <path>              Source f16 ckpt.
  -o, --output <path>             Output q6p ckpt.

Options:
      --official-baseline <path>  Official q6p baseline for probe comparison.
                                  Default: dt-models/ltx_2.3_22b_distilled_1.1_q6p.ckpt
      --backup <path>             Backup path for existing output file.
                                  Default: <output>.pre_replay_<YYYYMMDD>.ckpt
      --skip-backup               Do not backup existing output.
      --skip-probes               Run build + quantize + validate only.
      --sample-limit <n>          Probe sample limit (default: 12).
      --progress-every <n>        Probe progress interval (default: 50).
      --tag <value>               Suffix tag used in probe filenames.
                                  Default: current date (YYYYMMDD)
  -h, --help                      Show this help message.

Example:
  tools/run_clipfix2_replay_q6p.sh \
    -i dt-models/10_e_v1_bf16_regen_0_f16.ckpt \
    -o dt-models/10_e_v1_bf16_regen_0_q6p.ckpt
EOF
}

abs_path() {
  local value="$1"
  if [[ "$value" == /* ]]; then
    echo "$value"
  else
    echo "$ROOT/$value"
  fi
}

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input)
      SRC="${2:-}"
      shift 2
      ;;
    -o|--output)
      OUT="${2:-}"
      shift 2
      ;;
    --official-baseline)
      OFFICIAL_BASELINE="${2:-}"
      shift 2
      ;;
    --backup)
      BACKUP="${2:-}"
      shift 2
      ;;
    --skip-backup)
      SKIP_BACKUP=1
      shift
      ;;
    --skip-probes)
      SKIP_PROBES=1
      shift
      ;;
    --sample-limit)
      SAMPLE_LIMIT="${2:-}"
      shift 2
      ;;
    --progress-every)
      PROGRESS_EVERY="${2:-}"
      shift 2
      ;;
    --tag)
      TAG="${2:-}"
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

if [[ -z "$SRC" || -z "$OUT" ]]; then
  echo "error: --input and --output are required" >&2
  usage
  exit 1
fi

if ! [[ "$SAMPLE_LIMIT" =~ ^[0-9]+$ ]] || ((SAMPLE_LIMIT < 1)); then
  echo "error: --sample-limit must be a positive integer" >&2
  exit 1
fi

if ! [[ "$PROGRESS_EVERY" =~ ^[0-9]+$ ]] || ((PROGRESS_EVERY < 1)); then
  echo "error: --progress-every must be a positive integer" >&2
  exit 1
fi

SRC="$(abs_path "$SRC")"
OUT="$(abs_path "$OUT")"
OFFICIAL_BASELINE="$(abs_path "$OFFICIAL_BASELINE")"

if [[ ! -f "$SRC" ]]; then
  echo "error: source file not found: $SRC" >&2
  exit 1
fi

if [[ ! -f "$OFFICIAL_BASELINE" ]]; then
  echo "error: official baseline not found: $OFFICIAL_BASELINE" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "error: python not found at: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -x "$QUANT_WRAPPER" ]]; then
  echo "error: quantize wrapper not executable: $QUANT_WRAPPER" >&2
  exit 1
fi

if [[ -z "$BACKUP" ]]; then
  if [[ "$OUT" == *.ckpt ]]; then
    BACKUP="${OUT%.ckpt}.pre_replay_${TAG}.ckpt"
  else
    BACKUP="${OUT}.pre_replay_${TAG}"
  fi
else
  BACKUP="$(abs_path "$BACKUP")"
fi

mkdir -p "$(dirname "$OUT")" "$ROOT/output"
if [[ "$SKIP_BACKUP" == "0" ]]; then
  mkdir -p "$(dirname "$BACKUP")"
fi

echo "==> clipfix2-style q6p replay"
echo "    src               : $SRC"
echo "    out               : $OUT"
echo "    official baseline : $OFFICIAL_BASELINE"
echo "    tag               : $TAG"

if [[ "$SKIP_BACKUP" == "0" ]]; then
  if [[ -f "$OUT" ]]; then
    echo "==> Backup existing output"
    echo "    backup: $BACKUP"
    cp -f "$OUT" "$BACKUP"
  else
    echo "==> No existing output file; skipping backup copy"
  fi
fi

echo "==> Build model-quantizer"
swift build --package-path "$ROOT/draw-things-community" -c release --product model-quantizer

echo "==> Quantize (forced q6p replay path)"
bash "$QUANT_WRAPPER" \
  -i "$SRC" \
  -m ltx2.3 \
  -o "$OUT" \
  --target-codec q6p

if [[ ! -f "$OUT" ]]; then
  echo "error: quantization did not produce output file: $OUT" >&2
  exit 1
fi

echo "==> Validate output structure"
"$PYTHON_BIN" "$ROOT/tools/dt_validate_converted_ckpt.py" --file "$OUT" --profile ltx2_3

if [[ "$SKIP_PROBES" == "1" ]]; then
  echo "==> Done (probes skipped by request)"
  echo "    output: $OUT"
  exit 0
fi

out_stem="$(basename "$OUT")"
out_stem="${out_stem%.ckpt}"
probe_stem="${out_stem//[^a-zA-Z0-9_]/_}"

OFF_JSON="$ROOT/output/probe_${probe_stem}_vs_official_clip_path_${TAG}.json"
OFF_MD="$ROOT/output/probe_${probe_stem}_vs_official_clip_path_${TAG}.md"
SRC_JSON="$ROOT/output/probe_${probe_stem}_vs_sourcef16_clip_path_${TAG}.json"
SRC_MD="$ROOT/output/probe_${probe_stem}_vs_sourcef16_clip_path_${TAG}.md"

echo "==> Probe clip-path families vs official baseline"
"$PYTHON_BIN" "$ROOT/tools/dt_probe_ckpt_targeted_content.py" \
  --file "$OUT" \
  --baseline "$OFFICIAL_BASELINE" \
  --prefix __text_feature_extractor__ \
  --prefix __text_video_connector__ \
  --prefix __text_audio_connector__ \
  --prefix text_video_connector_learnable_registers \
  --prefix text_audio_connector_learnable_registers \
  --sample-limit "$SAMPLE_LIMIT" \
  --progress-every "$PROGRESS_EVERY" \
  --out-json "$OFF_JSON" \
  --out-md "$OFF_MD"

echo "==> Probe clip-path families vs source f16"
"$PYTHON_BIN" "$ROOT/tools/dt_probe_ckpt_targeted_content.py" \
  --file "$OUT" \
  --baseline "$SRC" \
  --prefix __text_feature_extractor__ \
  --prefix __text_video_connector__ \
  --prefix __text_audio_connector__ \
  --prefix text_video_connector_learnable_registers \
  --prefix text_audio_connector_learnable_registers \
  --sample-limit "$SAMPLE_LIMIT" \
  --progress-every "$PROGRESS_EVERY" \
  --out-json "$SRC_JSON" \
  --out-md "$SRC_MD"

echo "==> Done"
echo "    output: $OUT"
echo "    probe : $OFF_MD"
echo "    probe : $SRC_MD"
