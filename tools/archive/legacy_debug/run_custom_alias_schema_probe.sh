#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUSTOM_JSON="$ROOT/dt-models/custom.json"
CANARY_SCRIPT="$ROOT/tools/run_q6p_canary_once.sh"

ENTRY_NAME="10_e_v1"
MODEL_KEY="10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt"
TAG="$(date +%Y%m%d_%H%M%S)"
TIMEOUT_SEC=90
HOST="127.0.0.1:7861"
MATRIX="core"

KEEP="__KEEP__"
DELETE="__DELETE__"

usage() {
  cat <<'EOF'
Usage:
  tools/run_custom_alias_schema_probe.sh [options]

Purpose:
  Probe custom.json entry-field combinations for 10_e_v1 while validating the
  traced q6p model key under strict final-mode canary gates.

Options:
  --entry-name <name>       custom.json entry name to mutate (default: 10_e_v1).
  --model-key <file>        Model key passed to run_q6p_canary_once.sh
                            (default: 10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt).
  --matrix <name>           Case matrix to run: core | extended-fields
                            (default: core).
  --tag <value>             Output tag suffix (default: timestamp).
  --timeout-sec <n>         Per-case canary timeout (default: 90).
  --host <host:port>        gRPC host (default: 127.0.0.1:7861).
  -h, --help                Show this help.

Outputs:
  output/custom_alias_schema_probe_<tag>/
    - results.tsv
    - summary.md
    - cases/<case>.log

Safety:
  - Backs up dt-models/custom.json and restores it on exit.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --entry-name)
      ENTRY_NAME="${2:-}"
      shift 2
      ;;
    --model-key)
      MODEL_KEY="${2:-}"
      shift 2
      ;;
    --matrix)
      MATRIX="${2:-}"
      shift 2
      ;;
    --tag)
      TAG="${2:-}"
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

if ! [[ "$TIMEOUT_SEC" =~ ^[0-9]+$ ]] || [[ "$TIMEOUT_SEC" -lt 1 ]]; then
  echo "error: --timeout-sec must be an integer >= 1" >&2
  exit 1
fi

if [[ "$MATRIX" != "core" && "$MATRIX" != "extended-fields" ]]; then
  echo "error: --matrix must be one of: core, extended-fields" >&2
  exit 1
fi

if [[ ! -f "$CUSTOM_JSON" ]]; then
  echo "error: missing custom.json: $CUSTOM_JSON" >&2
  exit 1
fi

if [[ ! -x "$CANARY_SCRIPT" ]]; then
  echo "error: missing executable canary script: $CANARY_SCRIPT" >&2
  exit 1
fi

WORK_DIR="$ROOT/output/custom_alias_schema_probe_${TAG}"
CASES_DIR="$WORK_DIR/cases"
RESULTS_TSV="$WORK_DIR/results.tsv"
SUMMARY_MD="$WORK_DIR/summary.md"
mkdir -p "$CASES_DIR"

BACKUP_JSON="$(mktemp "$ROOT/output/custom_json_probe_backup_${TAG}_XXXXXX.json")"
cp "$CUSTOM_JSON" "$BACKUP_JSON"

restore_custom_json() {
  if [[ -f "$BACKUP_JSON" ]]; then
    cp "$BACKUP_JSON" "$CUSTOM_JSON"
    rm -f "$BACKUP_JSON"
  fi
}

trap restore_custom_json EXIT INT TERM

set_entry_fields() {
  local entry_name="$1"
  local file_key="$2"
  local clip_key="$3"
  local default_scale="$4"
  local version="$5"
  local modifier="$6"
  local text_encoder="$7"
  local autoencoder="$8"
  local objective_scale="$9"

  python3 - "$CUSTOM_JSON" "$entry_name" "$file_key" "$clip_key" "$default_scale" "$version" "$modifier" "$text_encoder" "$autoencoder" "$objective_scale" "$KEEP" "$DELETE" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
entry_name = sys.argv[2]
file_key = sys.argv[3]
clip_key = sys.argv[4]
default_scale = int(sys.argv[5])
version = sys.argv[6]
modifier = sys.argv[7]
text_encoder = sys.argv[8]
autoencoder = sys.argv[9]
objective_scale = sys.argv[10]
keep = sys.argv[11]
delete = sys.argv[12]

payload = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(payload, list):
    raise SystemExit("custom.json root must be a list")

for entry in payload:
    if not isinstance(entry, dict):
        continue
    if str(entry.get("name", "")).strip() != entry_name:
        continue
    entry["file"] = file_key
    entry["clip_encoder"] = clip_key
    entry["default_scale"] = default_scale

    if version != keep:
        entry["version"] = version

    if modifier == delete:
        entry.pop("modifier", None)
    elif modifier != keep:
        entry["modifier"] = modifier

    if text_encoder == delete:
        entry.pop("text_encoder", None)
    elif text_encoder != keep:
        entry["text_encoder"] = text_encoder

    if autoencoder == delete:
        entry.pop("autoencoder", None)
    elif autoencoder != keep:
        entry["autoencoder"] = autoencoder

    if objective_scale == delete:
        entry.pop("objective", None)
    elif objective_scale != keep:
        scale = float(objective_scale)
        if scale.is_integer():
            scale = int(scale)
        entry["objective"] = {"u": {"condition_scale": scale}}

    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    break
else:
    raise SystemExit(f"entry not found: {entry_name}")
PY
}

classify_signature() {
  local case_log="$1"

  if rg -q 'TextEncoder\.encodeLTX2|Illegal instruction' "$case_log"; then
    echo "textencoder_illegal"
    return
  fi

  if rg -q 'ccv_nnc_tensor_read|ccv_cnnp_model_read|Bad pointer dereference|Segmentation fault' "$case_log"; then
    echo "loader_crash"
    return
  fi

  if rg -q 'No such file or directory|is not downloaded|Unable to read .* checkpoint|Failed to open|cannot open' "$case_log"; then
    echo "missing_file"
    return
  fi

  if rg -q 'canary timed out|canary_rc=124' "$case_log"; then
    echo "timeout"
    return
  fi

  if rg -q '^RESULT=PASS$' "$case_log"; then
    echo "pass"
    return
  fi

  echo "unknown"
}

run_case() {
  local case_id="$1"
  local file_key="$2"
  local clip_key="$3"
  local default_scale="$4"
  local version="${5:-$KEEP}"
  local modifier="${6:-$KEEP}"
  local text_encoder="${7:-$KEEP}"
  local autoencoder="${8:-$KEEP}"
  local objective_scale="${9:-$KEEP}"

  local case_tag="${TAG}_${case_id}"
  local case_log="$CASES_DIR/${case_id}.log"

  echo "== case: $case_id =="
  echo "file=$file_key clip_encoder=$clip_key default_scale=$default_scale version=$version modifier=$modifier text_encoder=$text_encoder autoencoder=$autoencoder objective_scale=$objective_scale"

  set_entry_fields "$ENTRY_NAME" "$file_key" "$clip_key" "$default_scale" "$version" "$modifier" "$text_encoder" "$autoencoder" "$objective_scale"
  python3 -m json.tool "$CUSTOM_JSON" >/dev/null

  set +e
  bash "$CANARY_SCRIPT" \
    --model "$MODEL_KEY" \
    --host "$HOST" \
    --timeout-sec "$TIMEOUT_SEC" \
    --max-responses 0 \
    --require-complete-stream \
    --require-final-output \
    --final-mode \
    --tag "$case_tag" > "$case_log" 2>&1 &
  local canary_pid=$!
  while kill -0 "$canary_pid" >/dev/null 2>&1; do
    echo "case=$case_id status=running"
    sleep 10
  done
  wait "$canary_pid"
  local rc=$?
  set -e

  local result="UNKNOWN"
  if rg -q '^RESULT=PASS$' "$case_log"; then
    result="PASS"
  elif rg -q '^RESULT=FAIL' "$case_log"; then
    result="FAIL"
  fi

  local canary_rc
  canary_rc="$(awk -F= '/^canary_rc=/{print $2; exit}' "$case_log")"
  local post_echo_rc
  post_echo_rc="$(awk -F= '/^post_echo_rc=/{print $2; exit}' "$case_log")"

  local responses
  responses="$(awk -F': ' '/^responses:/{value=$2} END{if (value=="") value="0"; print value}' "$case_log")"
  local images
  images="$(awk -F': ' '/^images written:/{value=$2} END{if (value=="") value="0"; print value}' "$case_log")"
  local audio
  audio="$(awk -F': ' '/^audio written:/{value=$2} END{if (value=="") value="0"; print value}' "$case_log")"

  local signature
  signature="$(classify_signature "$case_log")"

  echo -e "${case_id}\t${file_key}\t${clip_key}\t${default_scale}\t${version}\t${modifier}\t${text_encoder}\t${autoencoder}\t${objective_scale}\t${rc}\t${result}\t${signature}\t${canary_rc}\t${post_echo_rc}\t${responses}\t${images}\t${audio}\t${case_tag}" >> "$RESULTS_TSV"
}

echo "== custom alias schema probe =="
echo "entry_name=$ENTRY_NAME"
echo "model_key=$MODEL_KEY"
echo "matrix=$MATRIX"
echo "tag=$TAG"
echo "timeout_sec=$TIMEOUT_SEC"
echo "host=$HOST"
echo "work_dir=$WORK_DIR"

echo -e "case\tentry_file\tentry_clip_encoder\tentry_default_scale\tentry_version\tentry_modifier\tentry_text_encoder\tentry_autoencoder\tentry_objective_scale\trc\tresult\tsignature\tcanary_rc\tpost_echo_rc\tresponses\timages\taudio\tcanary_tag" > "$RESULTS_TSV"

if [[ "$MATRIX" == "core" ]]; then
  run_case "control_unmatched_oldq6p" \
    "10_e_v1_bf16_regen_0_q6p.ckpt" \
    "10_e_v1_bf16_regen_0_q6p.ckpt" \
    "1"

  run_case "match_trace_clip_trace_scale1" \
    "10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt" \
    "10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt" \
    "1"

  run_case "match_trace_clip_oldq6p_scale1" \
    "10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt" \
    "10_e_v1_bf16_regen_0_q6p.ckpt" \
    "1"

  run_case "match_trace_clip_official_scale1" \
    "10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt" \
    "ltx_2.3_22b_distilled_1.1_q6p.ckpt" \
    "1"

  run_case "match_trace_clip_trace_scale12" \
    "10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt" \
    "10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt" \
    "12"
else
  run_case "control_unmatched_oldq6p" \
    "10_e_v1_bf16_regen_0_q6p.ckpt" \
    "10_e_v1_bf16_regen_0_q6p.ckpt" \
    "1"

  run_case "match_trace_baseline" \
    "10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt" \
    "10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt" \
    "1"

  run_case "match_trace_modifier_none" \
    "10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt" \
    "10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt" \
    "1" \
    "$KEEP" "none"

  run_case "match_trace_modifier_kontext_kv" \
    "10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt" \
    "10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt" \
    "1" \
    "$KEEP" "kontext_kv"

  run_case "match_trace_version_ltx2" \
    "10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt" \
    "10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt" \
    "1" \
    "ltx2"

  run_case "match_trace_objective_u1" \
    "10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt" \
    "10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt" \
    "1" \
    "$KEEP" "$KEEP" "$KEEP" "$KEEP" "1"

  run_case "match_trace_objective_delete" \
    "10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt" \
    "10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt" \
    "1" \
    "$KEEP" "$KEEP" "$KEEP" "$KEEP" "$DELETE"

  run_case "match_trace_text_encoder_delete" \
    "10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt" \
    "10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt" \
    "1" \
    "$KEEP" "$KEEP" "$DELETE"

  run_case "match_trace_autoencoder_delete" \
    "10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt" \
    "10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt" \
    "1" \
    "$KEEP" "$KEEP" "$KEEP" "$DELETE"

  run_case "match_trace_text_auto_delete" \
    "10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt" \
    "10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt" \
    "1" \
    "$KEEP" "$KEEP" "$DELETE" "$DELETE"
fi

pass_count="$(awk -F'\t' 'NR>1 && $11=="PASS"{c++} END{print c+0}' "$RESULTS_TSV")"
fail_count="$(awk -F'\t' 'NR>1 && $11!="PASS"{c++} END{print c+0}' "$RESULTS_TSV")"
case_count="$(awk 'END{print NR-1}' "$RESULTS_TSV")"

{
  echo "# Custom Alias Schema Probe"
  echo
  echo "- entry_name: $ENTRY_NAME"
  echo "- model_key: $MODEL_KEY"
  echo "- matrix: $MATRIX"
  echo "- tag: $TAG"
  echo "- timeout_sec: $TIMEOUT_SEC"
  echo "- host: $HOST"
  echo "- work_dir: $WORK_DIR"
  echo "- cases: $case_count"
  echo "- pass: $pass_count"
  echo "- fail: $fail_count"
  echo
  echo "## Results"
  echo
  echo "| case | entry_file | entry_clip_encoder | default_scale | version | modifier | text_encoder | autoencoder | objective_scale | rc | result | signature | canary_rc | post_echo_rc | responses | images | audio |"
  echo "|---|---|---|---:|---|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|"
  awk -F'\t' 'NR>1 {printf("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17)}' "$RESULTS_TSV"
  echo
  echo "- results_tsv: $RESULTS_TSV"
  echo "- cases_dir: $CASES_DIR"
} > "$SUMMARY_MD"

echo "results_tsv=$RESULTS_TSV"
echo "summary_md=$SUMMARY_MD"
echo "RESULT=PASS"
