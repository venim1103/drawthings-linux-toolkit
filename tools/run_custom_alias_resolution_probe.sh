#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUSTOM_JSON="$ROOT/dt-models/custom.json"
CANARY_SCRIPT="$ROOT/tools/run_q6p_canary_once.sh"

BASE_ENTRY_NAME="10_e_v1"
TRACE_MODEL_KEY="10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt"
TMPKEY_MODEL_KEY="10_e_v1_bf16_regen_0_q6p_trace021_run033_tmpkey.ckpt"

TAG="$(date +%Y%m%d_%H%M%S)"
TIMEOUT_SEC=75
HOST="127.0.0.1:7861"
MATRIX="core"

PROBE_ALIAS_A="probe_trace_alias_a"
PROBE_ALIAS_B="probe_trace_alias_b"
PROBE_TMP_ALIAS="probe_tmpkey_alias"

COMPANION_CLIP_A="ltx_2.3_22b_distilled_q6p_forcedfix_clipfix2_20260602.ckpt"
COMPANION_CLIP_B="10_e_v1_bf16_regen_0_q6p.ckpt"
COMPANION_CLIP_C="ltx_2.3_22b_distilled_f16.ckpt"
TEXT_PIN_A="clip_vit_l14_f16.ckpt"
TEXT_PIN_B="gemma_3_12b_it_qat_q8p.ckpt"

usage() {
  cat <<'EOF'
Usage:
  tools/run_custom_alias_resolution_probe.sh [options]

Purpose:
  Probe custom entry key-shadowing / resolution semantics by varying:
  - model argument path (file key vs alias name)
  - probe alias entries keyed to traced file
  - duplicate custom entries for the same file (order sensitivity)
  - tmp hardlink key with and without custom entry

Options:
  --matrix <name>          Case matrix: core | cross-file | minimal-v1 | field-ladder | ltx23-encoders | ltx23-companion | ltx23-order | ltx23-textpin | ltx23-pinned-companion | ltx23-focused
                           (default: core).
  --base-entry-name <name>  Baseline custom entry to clone for probe aliases
                            (default: 10_e_v1).
  --trace-model-key <file>  Traced checkpoint key under dt-models
                            (default: 10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt).
  --tmpkey-model-key <file> Temporary hardlink key used for non-custom/custom
                            comparison (default: 10_e_v1_bf16_regen_0_q6p_trace021_run033_tmpkey.ckpt).
  --tag <value>             Output tag suffix (default: timestamp).
  --timeout-sec <n>         Per-case canary timeout (default: 75).
  --host <host:port>        gRPC host (default: 127.0.0.1:7861).
  --companion-clip-a <file> LTX2.3 companion clip candidate A
                             (default: ltx_2.3_22b_distilled_q6p_forcedfix_clipfix2_20260602.ckpt).
  --companion-clip-b <file> LTX2.3 companion clip candidate B
                             (default: 10_e_v1_bf16_regen_0_q6p.ckpt).
  --companion-clip-c <file> LTX2.3 companion clip candidate C
                             (default: ltx_2.3_22b_distilled_f16.ckpt).
  --text-pin-a <file>       Explicit text_encoder pin A for ltx23-textpin and ltx23-pinned-companion matrices
                             (default: clip_vit_l14_f16.ckpt).
  --text-pin-b <file>       Explicit text_encoder pin B for ltx23-textpin matrix
                             (default: gemma_3_12b_it_qat_q8p.ckpt).
  -h, --help                Show this help.

Outputs:
  output/custom_alias_resolution_probe_<tag>/
    - results.tsv
    - summary.md
    - cases/<case>.log

Safety:
  - Backs up dt-models/custom.json and restores it on EXIT/INT/TERM.
  - Removes temporary hardlink key on EXIT/INT/TERM.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-entry-name)
      BASE_ENTRY_NAME="${2:-}"
      shift 2
      ;;
    --matrix)
      MATRIX="${2:-}"
      shift 2
      ;;
    --trace-model-key)
      TRACE_MODEL_KEY="${2:-}"
      shift 2
      ;;
    --tmpkey-model-key)
      TMPKEY_MODEL_KEY="${2:-}"
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
    --companion-clip-a)
      COMPANION_CLIP_A="${2:-}"
      shift 2
      ;;
    --companion-clip-b)
      COMPANION_CLIP_B="${2:-}"
      shift 2
      ;;
    --companion-clip-c)
      COMPANION_CLIP_C="${2:-}"
      shift 2
      ;;
    --text-pin-a)
      TEXT_PIN_A="${2:-}"
      shift 2
      ;;
    --text-pin-b)
      TEXT_PIN_B="${2:-}"
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

if [[ "$MATRIX" != "core" && "$MATRIX" != "cross-file" && "$MATRIX" != "minimal-v1" && "$MATRIX" != "field-ladder" && "$MATRIX" != "ltx23-encoders" && "$MATRIX" != "ltx23-companion" && "$MATRIX" != "ltx23-order" && "$MATRIX" != "ltx23-textpin" && "$MATRIX" != "ltx23-pinned-companion" && "$MATRIX" != "ltx23-focused" ]]; then
  echo "error: --matrix must be one of: core, cross-file, minimal-v1, field-ladder, ltx23-encoders, ltx23-companion, ltx23-order, ltx23-textpin, ltx23-pinned-companion, ltx23-focused" >&2
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

TRACE_MODEL_PATH="$ROOT/dt-models/$TRACE_MODEL_KEY"
TMPKEY_MODEL_PATH="$ROOT/dt-models/$TMPKEY_MODEL_KEY"
if [[ ! -f "$TRACE_MODEL_PATH" ]]; then
  echo "error: missing trace model file: $TRACE_MODEL_PATH" >&2
  exit 1
fi

WORK_DIR="$ROOT/output/custom_alias_resolution_probe_${TAG}"
CASES_DIR="$WORK_DIR/cases"
RESULTS_TSV="$WORK_DIR/results.tsv"
SUMMARY_MD="$WORK_DIR/summary.md"
mkdir -p "$CASES_DIR"

BACKUP_JSON="$(mktemp "$ROOT/output/custom_json_resolution_probe_backup_${TAG}_XXXXXX.json")"
cp "$CUSTOM_JSON" "$BACKUP_JSON"

restore_state() {
  if [[ -f "$BACKUP_JSON" ]]; then
    cp "$BACKUP_JSON" "$CUSTOM_JSON"
    rm -f "$BACKUP_JSON"
  fi
  rm -f "$TMPKEY_MODEL_PATH"
}

trap restore_state EXIT INT TERM

make_tmpkey_hardlink() {
  rm -f "$TMPKEY_MODEL_PATH"
  ln "$TRACE_MODEL_PATH" "$TMPKEY_MODEL_PATH"
}

set_custom_mode() {
  local mode="$1"

  python3 - "$CUSTOM_JSON" "$BACKUP_JSON" "$BASE_ENTRY_NAME" "$TRACE_MODEL_KEY" "$TMPKEY_MODEL_KEY" "$PROBE_ALIAS_A" "$PROBE_ALIAS_B" "$PROBE_TMP_ALIAS" "$COMPANION_CLIP_A" "$COMPANION_CLIP_B" "$COMPANION_CLIP_C" "$TEXT_PIN_A" "$TEXT_PIN_B" "$mode" <<'PY'
import copy
import json
import pathlib
import sys

custom_path = pathlib.Path(sys.argv[1])
backup_path = pathlib.Path(sys.argv[2])
base_entry_name = sys.argv[3]
trace_key = sys.argv[4]
tmpkey_key = sys.argv[5]
alias_a = sys.argv[6]
alias_b = sys.argv[7]
tmp_alias = sys.argv[8]
companion_clip_a = sys.argv[9]
companion_clip_b = sys.argv[10]
companion_clip_c = sys.argv[11]
text_pin_a = sys.argv[12]
text_pin_b = sys.argv[13]
mode = sys.argv[14]

def clean(value):
  if value is None:
    return ""
  return str(value).strip()

payload = json.loads(backup_path.read_text(encoding="utf-8"))
if not isinstance(payload, list):
    raise SystemExit("custom.json root must be a list")

base_entry = None
for entry in payload:
    if isinstance(entry, dict) and str(entry.get("name", "")).strip() == base_entry_name:
        base_entry = copy.deepcopy(entry)
        break

if base_entry is None:
    raise SystemExit(f"base entry not found: {base_entry_name}")

def alias_from_base(name: str, file_key: str, clip_key: str | None = None, modifier: str | None = None):
    entry = copy.deepcopy(base_entry)
    entry["name"] = name
    entry["file"] = file_key
    entry["clip_encoder"] = clip_key if clip_key is not None else file_key
    entry["default_scale"] = 1
    entry["version"] = "ltx2.3"
    entry["modifier"] = modifier if modifier is not None else "kontext"
    return entry

def alias_min_v1(name: str, file_key: str):
    # Minimal schema to emulate specificationForModel miss defaults while still creating a custom winner.
    return {
        "name": name,
        "file": file_key,
        "prefix": "",
        "version": "v1",
        "upcast_attention": False,
        "default_scale": 8,
    }

def alias_ladder_ltx23_min(name: str, file_key: str):
    entry = alias_min_v1(name, file_key)
    entry["version"] = "ltx2.3"
    return entry

def alias_ladder_ltx23_modifier(name: str, file_key: str):
    entry = alias_ladder_ltx23_min(name, file_key)
    entry["modifier"] = "kontext"
    return entry

def alias_ladder_ltx23_modifier_scale1(name: str, file_key: str):
    entry = alias_ladder_ltx23_modifier(name, file_key)
    entry["default_scale"] = 1
    return entry

def alias_ladder_ltx23_modifier_scale1_clip(name: str, file_key: str):
    entry = alias_ladder_ltx23_modifier_scale1(name, file_key)
    entry["clip_encoder"] = file_key
    return entry

def alias_ltx23_min_text(name: str, file_key: str):
  entry = alias_ladder_ltx23_min(name, file_key)
  text_encoder = str(base_entry.get("text_encoder", "")).strip()
  if text_encoder:
    entry["text_encoder"] = text_encoder
  return entry

def alias_ltx23_min_auto(name: str, file_key: str):
  entry = alias_ladder_ltx23_min(name, file_key)
  autoencoder = str(base_entry.get("autoencoder", "")).strip()
  if autoencoder:
    entry["autoencoder"] = autoencoder
  return entry

def alias_ltx23_min_text_auto(name: str, file_key: str):
  entry = alias_ltx23_min_text(name, file_key)
  autoencoder = str(base_entry.get("autoencoder", "")).strip()
  if autoencoder:
    entry["autoencoder"] = autoencoder
  return entry

def alias_ltx23_min_clip(name: str, file_key: str):
  entry = alias_ladder_ltx23_min(name, file_key)
  entry["clip_encoder"] = file_key
  return entry

def alias_ltx23_text_clip(name: str, file_key: str, clip_key: str):
  entry = alias_ltx23_min_text(name, file_key)
  entry["clip_encoder"] = clean(clip_key)
  return entry

def alias_ltx23_text_clip_explicit(name: str, file_key: str, text_key: str, clip_key: str):
  entry = alias_ladder_ltx23_min(name, file_key)
  text_clean = clean(text_key)
  if text_clean:
    entry["text_encoder"] = text_clean
  clip_clean = clean(clip_key)
  if clip_clean:
    entry["clip_encoder"] = clip_clean
  return entry

filtered = []
probe_names = {alias_a, alias_b, tmp_alias}
for entry in payload:
    if not isinstance(entry, dict):
        continue
    name = str(entry.get("name", "")).strip()
    if name in probe_names:
        continue
    filtered.append(entry)

if mode == "baseline_only":
    pass
elif mode == "alias_trace_a":
    filtered.append(alias_from_base(alias_a, trace_key, trace_key, "kontext"))
elif mode == "alias_trace_a_clip_old":
    filtered.append(alias_from_base(alias_a, trace_key, base_entry.get("clip_encoder"), "kontext"))
elif mode == "alias_trace_dupe_ab":
    filtered.append(alias_from_base(alias_a, trace_key, trace_key, "kontext"))
    filtered.append(alias_from_base(alias_b, trace_key, trace_key, "none"))
elif mode == "alias_trace_dupe_ba":
    filtered.append(alias_from_base(alias_b, trace_key, trace_key, "none"))
    filtered.append(alias_from_base(alias_a, trace_key, trace_key, "kontext"))
elif mode == "alias_tmpkey":
    filtered.append(alias_from_base(tmp_alias, tmpkey_key, tmpkey_key, "kontext"))
elif mode == "alias_both":
    filtered.append(alias_from_base(alias_a, trace_key, trace_key, "kontext"))
    filtered.append(alias_from_base(tmp_alias, tmpkey_key, tmpkey_key, "kontext"))
elif mode == "alias_trace_min_v1":
    filtered.append(alias_min_v1(alias_a, trace_key))
elif mode == "alias_tmpkey_min_v1":
    filtered.append(alias_min_v1(tmp_alias, tmpkey_key))
elif mode == "alias_both_min_v1":
    filtered.append(alias_min_v1(alias_a, trace_key))
    filtered.append(alias_min_v1(tmp_alias, tmpkey_key))
elif mode == "alias_trace_ladder_ltx23_min":
    filtered.append(alias_ladder_ltx23_min(alias_a, trace_key))
elif mode == "alias_trace_ladder_ltx23_modifier":
    filtered.append(alias_ladder_ltx23_modifier(alias_a, trace_key))
elif mode == "alias_trace_ladder_ltx23_modifier_scale1":
    filtered.append(alias_ladder_ltx23_modifier_scale1(alias_a, trace_key))
elif mode == "alias_trace_ladder_ltx23_modifier_scale1_clip":
    filtered.append(alias_ladder_ltx23_modifier_scale1_clip(alias_a, trace_key))
elif mode == "alias_trace_ltx23_min_text":
  filtered.append(alias_ltx23_min_text(alias_a, trace_key))
elif mode == "alias_trace_ltx23_min_auto":
  filtered.append(alias_ltx23_min_auto(alias_a, trace_key))
elif mode == "alias_trace_ltx23_min_text_auto":
  filtered.append(alias_ltx23_min_text_auto(alias_a, trace_key))
elif mode == "alias_trace_ltx23_min_clip":
  filtered.append(alias_ltx23_min_clip(alias_a, trace_key))
elif mode == "alias_trace_ltx23_text_clip_trace":
  filtered.append(alias_ltx23_text_clip(alias_a, trace_key, trace_key))
elif mode == "alias_trace_ltx23_text_clip_companion_a":
  filtered.append(alias_ltx23_text_clip(alias_a, trace_key, companion_clip_a))
elif mode == "alias_trace_ltx23_text_clip_companion_b":
  filtered.append(alias_ltx23_text_clip(alias_a, trace_key, companion_clip_b))
elif mode == "alias_trace_ltx23_text_clip_companion_c":
  filtered.append(alias_ltx23_text_clip(alias_a, trace_key, companion_clip_c))
elif mode == "alias_trace_ltx23_text_swap_companion_a":
  filtered.append(alias_ltx23_text_clip_explicit(alias_a, trace_key, companion_clip_a, trace_key))
elif mode == "alias_trace_ltx23_text_swap_companion_b":
  filtered.append(alias_ltx23_text_clip_explicit(alias_a, trace_key, companion_clip_b, trace_key))
elif mode == "alias_trace_ltx23_text_swap_companion_c":
  filtered.append(alias_ltx23_text_clip_explicit(alias_a, trace_key, companion_clip_c, trace_key))
elif mode == "alias_trace_ltx23_text_only_pin_a":
  filtered.append(alias_ltx23_text_clip_explicit(alias_a, trace_key, text_pin_a, ""))
elif mode == "alias_trace_ltx23_text_only_pin_b":
  filtered.append(alias_ltx23_text_clip_explicit(alias_a, trace_key, text_pin_b, ""))
elif mode == "alias_trace_ltx23_text_clip_trace_pin_a":
  filtered.append(alias_ltx23_text_clip_explicit(alias_a, trace_key, text_pin_a, trace_key))
elif mode == "alias_trace_ltx23_text_clip_trace_pin_b":
  filtered.append(alias_ltx23_text_clip_explicit(alias_a, trace_key, text_pin_b, trace_key))
elif mode == "alias_trace_ltx23_text_clip_companion_a_pin_a":
  filtered.append(alias_ltx23_text_clip_explicit(alias_a, trace_key, text_pin_a, companion_clip_a))
elif mode == "alias_trace_ltx23_text_clip_companion_b_pin_a":
  filtered.append(alias_ltx23_text_clip_explicit(alias_a, trace_key, text_pin_a, companion_clip_b))
elif mode == "alias_trace_ltx23_text_clip_companion_c_pin_a":
  filtered.append(alias_ltx23_text_clip_explicit(alias_a, trace_key, text_pin_a, companion_clip_c))
else:
    raise SystemExit(f"unknown mode: {mode}")

custom_path.write_text(json.dumps(filtered, indent=2) + "\n", encoding="utf-8")
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

resolve_context() {
  local model_arg="$1"

  python3 - "$CUSTOM_JSON" "$model_arg" "$TRACE_MODEL_KEY" "$TMPKEY_MODEL_KEY" <<'PY'
import json
import pathlib
import sys

custom_json = pathlib.Path(sys.argv[1])
model_arg = sys.argv[2].strip()
trace_key = sys.argv[3].strip()
tmpkey_key = sys.argv[4].strip()
models_dir = custom_json.parent

DEFAULT_TEXT_ENCODER = "clip_vit_l14_f16.ckpt"

try:
    payload = json.loads(custom_json.read_text(encoding="utf-8"))
except Exception:
    payload = []

if not isinstance(payload, list):
    payload = []

entries = [e for e in payload if isinstance(e, dict)]

def clean(value):
    if value is None:
        return ""
    return str(value).replace("\t", " ").replace("\n", " ").strip()

def is_downloaded(file_key: str) -> bool:
  key = clean(file_key)
  if not key:
    return False
  return (models_dir / key).is_file()

def list_from_entry(entry: dict, snake: str, camel: str):
  value = entry.get(snake, entry.get(camel))
  if isinstance(value, list):
    return [clean(v) for v in value if clean(v)]
  return []

arg_source = "file"
arg_resolved_file = model_arg
for entry in entries:
    if clean(entry.get("name", "")) != model_arg:
        continue
    model_file = clean(entry.get("file", ""))
    if model_file:
        arg_source = "custom_name"
        arg_resolved_file = model_file
        break

def match_list(file_key: str):
    key = clean(file_key)
    return [e for e in entries if clean(e.get("file", "")) == key]

arg_matches = match_list(arg_resolved_file)
trace_matches = match_list(trace_key)
tmp_matches = match_list(tmpkey_key)

arg_winner = arg_matches[-1] if arg_matches else None
trace_winner = trace_matches[-1] if trace_matches else None
tmp_winner = tmp_matches[-1] if tmp_matches else None

winner_version = clean(arg_winner.get("version", "") if arg_winner else "")
winner_text = clean(arg_winner.get("text_encoder", "") if arg_winner else "")
winner_clip = clean(arg_winner.get("clip_encoder", "") if arg_winner else "")
winner_auto = clean(arg_winner.get("autoencoder", "") if arg_winner else "")

text_file0 = ""
text_file1 = ""
text_files = []
if arg_winner:
  primary = winner_text if is_downloaded(winner_text) else DEFAULT_TEXT_ENCODER
  text_files.append(primary)

  clips = []
  if winner_clip and is_downloaded(winner_clip):
    clips.append(winner_clip)
  for extra in list_from_entry(arg_winner, "additional_clip_encoders", "additionalClipEncoders"):
    if is_downloaded(extra):
      clips.append(extra)
  text_files.extend(clips)

  t5 = clean(arg_winner.get("t5_encoder", ""))
  if t5 and is_downloaded(t5):
    text_files.append(t5)

  if text_files:
    text_file0 = text_files[0]
  if len(text_files) >= 2:
    text_file1 = text_files[1]

version_norm = winner_version.lower().replace("_", "").replace("-", "")
is_ltx23 = "1" if version_norm in {"ltx2.3", "ltx23"} else "0"
ltx23_textfiles_ok = "1" if is_ltx23 == "0" or len(text_files) >= 2 else "0"

fields = [
    arg_source,
    arg_resolved_file,
    str(len(arg_matches)),
    clean(arg_winner.get("name", "") if arg_winner else ""),
    clean(arg_winner.get("modifier", "") if arg_winner else ""),
    str(len(trace_matches)),
    clean(trace_winner.get("name", "") if trace_winner else ""),
    str(len(tmp_matches)),
    clean(tmp_winner.get("name", "") if tmp_winner else ""),
  winner_version,
  winner_text,
  winner_clip,
  winner_auto,
  str(len(text_files)),
  text_file0,
  text_file1,
  ltx23_textfiles_ok,
]
print("\t".join(fields))
PY
}

run_case() {
  local case_id="$1"
  local custom_mode="$2"
  local model_arg="$3"
  local ensure_tmpkey="${4:-0}"

  local case_tag="${TAG}_${case_id}"
  local case_log="$CASES_DIR/${case_id}.log"

  if [[ "$ensure_tmpkey" == "1" ]]; then
    make_tmpkey_hardlink
  else
    rm -f "$TMPKEY_MODEL_PATH"
  fi

  set_custom_mode "$custom_mode"
  python3 -m json.tool "$CUSTOM_JSON" >/dev/null

  local resolution_context
  resolution_context="$(resolve_context "$model_arg")"

  echo "== case: $case_id =="
  echo "custom_mode=$custom_mode model_arg=$model_arg ensure_tmpkey=$ensure_tmpkey"

  set +e
  bash "$CANARY_SCRIPT" \
    --model "$model_arg" \
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

  echo -e "${case_id}\t${custom_mode}\t${model_arg}\t${ensure_tmpkey}\t${resolution_context}\t${rc}\t${result}\t${signature}\t${canary_rc}\t${post_echo_rc}\t${responses}\t${images}\t${audio}\t${case_tag}" >> "$RESULTS_TSV"
}

echo "== custom alias resolution probe =="
echo "base_entry_name=$BASE_ENTRY_NAME"
echo "trace_model_key=$TRACE_MODEL_KEY"
echo "tmpkey_model_key=$TMPKEY_MODEL_KEY"
echo "matrix=$MATRIX"
echo "tag=$TAG"
echo "timeout_sec=$TIMEOUT_SEC"
echo "host=$HOST"
echo "work_dir=$WORK_DIR"

echo -e "case\tcustom_mode\tmodel_arg\tensure_tmpkey\targ_source\targ_resolved_file\targ_match_count\targ_winner_name\targ_winner_modifier\ttrace_match_count\ttrace_winner_name\ttmpkey_match_count\ttmpkey_winner_name\targ_winner_version\targ_winner_text_encoder\targ_winner_clip_encoder\targ_winner_autoencoder\targ_text_files_count\targ_text_file0\targ_text_file1\targ_ltx23_textfiles_ok\trc\tresult\tsignature\tcanary_rc\tpost_echo_rc\tresponses\timages\taudio\tcanary_tag" > "$RESULTS_TSV"

if [[ "$MATRIX" == "core" ]]; then
  run_case "control_trace_noncustom" "baseline_only" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_alias_model_filearg" "alias_trace_a" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_alias_model_namearg" "alias_trace_a" "$PROBE_ALIAS_A" "0"
  run_case "probe_trace_alias_clipold_model_filearg" "alias_trace_a_clip_old" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_dupe_order_ab" "alias_trace_dupe_ab" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_dupe_order_ba" "alias_trace_dupe_ba" "$TRACE_MODEL_KEY" "0"
  run_case "control_tmpkey_noncustom" "baseline_only" "$TMPKEY_MODEL_KEY" "1"
  run_case "probe_tmpkey_alias_model_filearg" "alias_tmpkey" "$TMPKEY_MODEL_KEY" "1"
  run_case "probe_tmpkey_alias_model_namearg" "alias_tmpkey" "$PROBE_TMP_ALIAS" "1"
elif [[ "$MATRIX" == "cross-file" ]]; then
  run_case "control_trace_noncustom" "baseline_only" "$TRACE_MODEL_KEY" "0"
  run_case "control_tmpkey_noncustom" "baseline_only" "$TMPKEY_MODEL_KEY" "1"

  run_case "cross_trace_alias_active_request_tmpkey" "alias_trace_a" "$TMPKEY_MODEL_KEY" "1"
  run_case "cross_tmpkey_alias_active_request_trace" "alias_tmpkey" "$TRACE_MODEL_KEY" "1"

  run_case "cross_trace_dupe_active_request_tmpkey" "alias_trace_dupe_ab" "$TMPKEY_MODEL_KEY" "1"

  run_case "cross_both_aliases_request_trace" "alias_both" "$TRACE_MODEL_KEY" "1"
  run_case "cross_both_aliases_request_tmpkey" "alias_both" "$TMPKEY_MODEL_KEY" "1"

  run_case "cross_both_aliases_request_trace_name" "alias_both" "$PROBE_ALIAS_A" "1"
  run_case "cross_both_aliases_request_tmpkey_name" "alias_both" "$PROBE_TMP_ALIAS" "1"
elif [[ "$MATRIX" == "minimal-v1" ]]; then
  run_case "control_trace_noncustom" "baseline_only" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_minv1_model_filearg" "alias_trace_min_v1" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_minv1_model_namearg" "alias_trace_min_v1" "$PROBE_ALIAS_A" "0"
  run_case "control_tmpkey_noncustom" "baseline_only" "$TMPKEY_MODEL_KEY" "1"
  run_case "probe_tmpkey_minv1_model_filearg" "alias_tmpkey_min_v1" "$TMPKEY_MODEL_KEY" "1"
  run_case "probe_tmpkey_minv1_model_namearg" "alias_tmpkey_min_v1" "$PROBE_TMP_ALIAS" "1"
  run_case "probe_both_minv1_trace_filearg" "alias_both_min_v1" "$TRACE_MODEL_KEY" "1"
  run_case "probe_both_minv1_tmpkey_filearg" "alias_both_min_v1" "$TMPKEY_MODEL_KEY" "1"
elif [[ "$MATRIX" == "field-ladder" ]]; then
  run_case "control_trace_noncustom" "baseline_only" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_minv1" "alias_trace_min_v1" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ladder_v_ltx23" "alias_trace_ladder_ltx23_min" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ladder_v_ltx23_modifier" "alias_trace_ladder_ltx23_modifier" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ladder_v_ltx23_modifier_scale1" "alias_trace_ladder_ltx23_modifier_scale1" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ladder_v_ltx23_modifier_scale1_clip" "alias_trace_ladder_ltx23_modifier_scale1_clip" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ladder_full_base" "alias_trace_a" "$TRACE_MODEL_KEY" "0"
elif [[ "$MATRIX" == "ltx23-encoders" ]]; then
  run_case "control_trace_noncustom" "baseline_only" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_min" "alias_trace_ladder_ltx23_min" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_min_text" "alias_trace_ltx23_min_text" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_min_auto" "alias_trace_ltx23_min_auto" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_min_text_auto" "alias_trace_ltx23_min_text_auto" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_min_clip" "alias_trace_ltx23_min_clip" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_full_base" "alias_trace_a" "$TRACE_MODEL_KEY" "0"
elif [[ "$MATRIX" == "ltx23-companion" ]]; then
  run_case "control_trace_noncustom" "baseline_only" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_min_text" "alias_trace_ltx23_min_text" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_text_clip_trace" "alias_trace_ltx23_text_clip_trace" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_text_clip_companion_a" "alias_trace_ltx23_text_clip_companion_a" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_text_clip_companion_b" "alias_trace_ltx23_text_clip_companion_b" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_text_clip_companion_c" "alias_trace_ltx23_text_clip_companion_c" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_full_base" "alias_trace_a" "$TRACE_MODEL_KEY" "0"
elif [[ "$MATRIX" == "ltx23-order" ]]; then
  run_case "control_trace_noncustom" "baseline_only" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_min_text" "alias_trace_ltx23_min_text" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_text_clip_trace" "alias_trace_ltx23_text_clip_trace" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_text_clip_companion_a" "alias_trace_ltx23_text_clip_companion_a" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_text_swap_companion_a" "alias_trace_ltx23_text_swap_companion_a" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_text_swap_companion_b" "alias_trace_ltx23_text_swap_companion_b" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_text_swap_companion_c" "alias_trace_ltx23_text_swap_companion_c" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_full_base" "alias_trace_a" "$TRACE_MODEL_KEY" "0"
elif [[ "$MATRIX" == "ltx23-textpin" ]]; then
  run_case "control_trace_noncustom" "baseline_only" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_text_only_pin_a" "alias_trace_ltx23_text_only_pin_a" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_text_only_pin_b" "alias_trace_ltx23_text_only_pin_b" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_text_clip_trace_pin_a" "alias_trace_ltx23_text_clip_trace_pin_a" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_text_clip_trace_pin_b" "alias_trace_ltx23_text_clip_trace_pin_b" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_full_base" "alias_trace_a" "$TRACE_MODEL_KEY" "0"
elif [[ "$MATRIX" == "ltx23-pinned-companion" ]]; then
  run_case "control_trace_noncustom" "baseline_only" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_text_only_pin_a" "alias_trace_ltx23_text_only_pin_a" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_text_clip_trace_pin_a" "alias_trace_ltx23_text_clip_trace_pin_a" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_text_clip_companion_a_pin_a" "alias_trace_ltx23_text_clip_companion_a_pin_a" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_text_clip_companion_b_pin_a" "alias_trace_ltx23_text_clip_companion_b_pin_a" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_text_clip_companion_c_pin_a" "alias_trace_ltx23_text_clip_companion_c_pin_a" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_full_base" "alias_trace_a" "$TRACE_MODEL_KEY" "0"
elif [[ "$MATRIX" == "ltx23-focused" ]]; then
  run_case "control_trace_noncustom" "baseline_only" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_text_clip_trace_pin_a" "alias_trace_ltx23_text_clip_trace_pin_a" "$TRACE_MODEL_KEY" "0"
  run_case "probe_trace_ltx23_text_clip_companion_a_pin_a" "alias_trace_ltx23_text_clip_companion_a_pin_a" "$TRACE_MODEL_KEY" "0"
fi

pass_count="$(awk -F'\t' 'NR>1 && $23=="PASS"{c++} END{print c+0}' "$RESULTS_TSV")"
fail_count="$(awk -F'\t' 'NR>1 && $23!="PASS"{c++} END{print c+0}' "$RESULTS_TSV")"
case_count="$(awk 'END{print NR-1}' "$RESULTS_TSV")"

{
  echo "# Custom Alias Resolution Probe"
  echo
  echo "- base_entry_name: $BASE_ENTRY_NAME"
  echo "- trace_model_key: $TRACE_MODEL_KEY"
  echo "- tmpkey_model_key: $TMPKEY_MODEL_KEY"
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
  echo "| case | custom_mode | model_arg | tmpkey | arg_source | arg_resolved_file | arg_match_count | arg_winner_name | arg_winner_modifier | arg_winner_version | arg_text_files_count | arg_text_file0 | arg_text_file1 | arg_ltx23_textfiles_ok | trace_match_count | trace_winner_name | tmpkey_match_count | tmpkey_winner_name | rc | result | signature | canary_rc | post_echo_rc | responses | images | audio |"
  echo "|---|---|---|---:|---|---|---:|---|---|---|---:|---|---|---:|---:|---|---:|---|---:|---|---|---:|---:|---:|---:|---:|"
  awk -F'\t' 'NR>1 {printf("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", $1,$2,$3,$4,$5,$6,$7,$8,$9,$14,$18,$19,$20,$21,$10,$11,$12,$13,$22,$23,$24,$25,$26,$27,$28,$29)}' "$RESULTS_TSV"
  echo
  echo "- results_tsv: $RESULTS_TSV"
  echo "- cases_dir: $CASES_DIR"
} > "$SUMMARY_MD"

echo "results_tsv=$RESULTS_TSV"
echo "summary_md=$SUMMARY_MD"
echo "RESULT=PASS"
