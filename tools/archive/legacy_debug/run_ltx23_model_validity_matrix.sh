#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT/.venv/bin/python"
VALIDATE_SCRIPT="$ROOT/tools/dt_validate_converted_ckpt.py"
CANARY_SCRIPT="$ROOT/tools/run_q6p_canary_once.sh"

F16_CKPT=""
Q6P_CKPT=""
SOURCE_SAFETENSORS=""
PROFILE="auto"

HOST="127.0.0.1:7861"
TAG="$(date +%Y%m%d_%H%M%S)"

RUN_F16_CANARY=1
RUN_Q6P_CANARY=1
STRICT_ALL=1
FINAL_MODE=1
CANARY_NO_TIMEOUT=0

F16_TIMEOUT_SEC=900
Q6P_TIMEOUT_SEC=900
F16_MAX_RESPONSES=0
Q6P_MAX_RESPONSES=0

SERIALIZATION_BASELINE=""
SERIALIZATION_GUARD_PROFILE="any"
SERIALIZATION_REPORT_LIMIT=10

usage() {
  cat <<'EOF'
Usage:
  tools/run_ltx23_model_validity_matrix.sh --f16-ckpt <path> --q6p-ckpt <path> --source-safetensors <path> [options]

Purpose:
  Run a generic LTX2.3 validity matrix for custom model pairs:
    1) SQLite sanity checks
    2) f16 structure validation against source safetensors profile
    3) f16 runtime canary with full completion gates
    4) q6p runtime canary with full completion gates

Required:
  --f16-ckpt <path>             Path to f16 checkpoint.
  --q6p-ckpt <path>             Path to q6p checkpoint.
  --source-safetensors <path>   Path to source safetensors file.

Options:
  --profile <auto|none|ltx2_3>  Validation profile for f16 check (default: auto).
  --host <host:port>            gRPC host for canary runs (default: 127.0.0.1:7861).
  --tag <value>                 Tag for output folder naming.

  --skip-f16-canary             Skip runtime canary for f16.
  --skip-q6p-canary             Skip runtime canary for q6p.

  --f16-timeout-sec <n>         Timeout for f16 canary (default: 900).
  --q6p-timeout-sec <n>         Timeout for q6p canary (default: 900).
  --f16-max-responses <n>       Max responses for f16 canary (default: 0, unlimited).
  --q6p-max-responses <n>       Max responses for q6p canary (default: 0, unlimited).

  --no-final-mode               Do not pass --final-mode to canary.
  --canary-no-timeout           Disable timeout wrapper in canary runs.

  --serialization-baseline <p>  Optional strict metadata baseline for f16 validator.
  --serialization-guard-profile <any|none|ltx2_3>
                                Apply serialization guard only for matching profile.
  --serialization-report-limit <n>
                                Max mismatch examples per category (default: 10).

  --q6p-nonfatal                Report q6p canary status but do not fail overall exit code.
  -h, --help                    Show this help.

Outputs:
  output/model_validity_ltx23_<tag>/
    - f16_validator.log
    - f16_canary.log (if enabled)
    - q6p_canary.log (if enabled)
    - summary.md
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

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --f16-ckpt)
      F16_CKPT="${2:-}"
      shift 2
      ;;
    --q6p-ckpt)
      Q6P_CKPT="${2:-}"
      shift 2
      ;;
    --source-safetensors)
      SOURCE_SAFETENSORS="${2:-}"
      shift 2
      ;;
    --profile)
      PROFILE="${2:-}"
      shift 2
      ;;
    --host)
      HOST="${2:-}"
      shift 2
      ;;
    --tag)
      TAG="${2:-}"
      shift 2
      ;;
    --skip-f16-canary)
      RUN_F16_CANARY=0
      shift
      ;;
    --skip-q6p-canary)
      RUN_Q6P_CANARY=0
      shift
      ;;
    --f16-timeout-sec)
      F16_TIMEOUT_SEC="${2:-}"
      shift 2
      ;;
    --q6p-timeout-sec)
      Q6P_TIMEOUT_SEC="${2:-}"
      shift 2
      ;;
    --f16-max-responses)
      F16_MAX_RESPONSES="${2:-}"
      shift 2
      ;;
    --q6p-max-responses)
      Q6P_MAX_RESPONSES="${2:-}"
      shift 2
      ;;
    --no-final-mode)
      FINAL_MODE=0
      shift
      ;;
    --canary-no-timeout)
      CANARY_NO_TIMEOUT=1
      shift
      ;;
    --serialization-baseline)
      SERIALIZATION_BASELINE="${2:-}"
      shift 2
      ;;
    --serialization-guard-profile)
      SERIALIZATION_GUARD_PROFILE="${2:-}"
      shift 2
      ;;
    --serialization-report-limit)
      SERIALIZATION_REPORT_LIMIT="${2:-}"
      shift 2
      ;;
    --q6p-nonfatal)
      STRICT_ALL=0
      shift
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

if [[ -z "$F16_CKPT" || -z "$Q6P_CKPT" || -z "$SOURCE_SAFETENSORS" ]]; then
  echo "error: --f16-ckpt, --q6p-ckpt, and --source-safetensors are required" >&2
  usage
  exit 1
fi

F16_CKPT="$(abs_path "$F16_CKPT")"
Q6P_CKPT="$(abs_path "$Q6P_CKPT")"
SOURCE_SAFETENSORS="$(abs_path "$SOURCE_SAFETENSORS")"

if [[ -n "$SERIALIZATION_BASELINE" ]]; then
  SERIALIZATION_BASELINE="$(abs_path "$SERIALIZATION_BASELINE")"
fi

if [[ "$PROFILE" != "auto" && "$PROFILE" != "none" && "$PROFILE" != "ltx2_3" ]]; then
  echo "error: --profile must be one of auto|none|ltx2_3" >&2
  exit 1
fi

if [[ "$SERIALIZATION_GUARD_PROFILE" != "any" && "$SERIALIZATION_GUARD_PROFILE" != "none" && "$SERIALIZATION_GUARD_PROFILE" != "ltx2_3" ]]; then
  echo "error: --serialization-guard-profile must be one of any|none|ltx2_3" >&2
  exit 1
fi

for n in "$F16_TIMEOUT_SEC" "$Q6P_TIMEOUT_SEC" "$F16_MAX_RESPONSES" "$Q6P_MAX_RESPONSES" "$SERIALIZATION_REPORT_LIMIT"; do
  if ! [[ "$n" =~ ^[0-9]+$ ]]; then
    echo "error: numeric option contains non-integer value: $n" >&2
    exit 1
  fi
done

for p in "$PYTHON_BIN" "$VALIDATE_SCRIPT" "$CANARY_SCRIPT"; do
  if [[ ! -e "$p" ]]; then
    echo "error: required tool missing: $p" >&2
    exit 1
  fi
done

for p in "$F16_CKPT" "$Q6P_CKPT" "$SOURCE_SAFETENSORS"; do
  if [[ ! -f "$p" ]]; then
    echo "error: required file missing: $p" >&2
    exit 1
  fi
done

WORK_DIR="$ROOT/output/model_validity_ltx23_${TAG}"
mkdir -p "$WORK_DIR"
SUMMARY_FILE="$WORK_DIR/summary.md"

F16_MODEL_KEY="$(basename "$F16_CKPT")"
Q6P_MODEL_KEY="$(basename "$Q6P_CKPT")"

echo "== LTX2.3 model validity matrix =="
echo "tag=$TAG"
echo "work_dir=$WORK_DIR"
echo "f16_ckpt=$F16_CKPT"
echo "q6p_ckpt=$Q6P_CKPT"
echo "source_safetensors=$SOURCE_SAFETENSORS"

echo "== SQLite sanity (f16) =="
set +e
F16_TENSOR_COUNT="$(sqlite3 "$F16_CKPT" 'SELECT COUNT(*) FROM tensors;' 2>/dev/null)"
F16_TENSOR_COUNT_RC=$?
F16_NULL_NAME_COUNT="$(sqlite3 "$F16_CKPT" 'SELECT COUNT(*) FROM tensors WHERE name IS NULL OR name = "";' 2>/dev/null)"
F16_NULL_NAME_COUNT_RC=$?
set -e

echo "f16_tensor_count=${F16_TENSOR_COUNT:-<error>}"
echo "f16_null_name_count=${F16_NULL_NAME_COUNT:-<error>}"

echo "== SQLite sanity (q6p) =="
set +e
Q6P_TENSOR_COUNT="$(sqlite3 "$Q6P_CKPT" 'SELECT COUNT(*) FROM tensors;' 2>/dev/null)"
Q6P_TENSOR_COUNT_RC=$?
Q6P_NULL_NAME_COUNT="$(sqlite3 "$Q6P_CKPT" 'SELECT COUNT(*) FROM tensors WHERE name IS NULL OR name = "";' 2>/dev/null)"
Q6P_NULL_NAME_COUNT_RC=$?
set -e

echo "q6p_tensor_count=${Q6P_TENSOR_COUNT:-<error>}"
echo "q6p_null_name_count=${Q6P_NULL_NAME_COUNT:-<error>}"

echo "== f16 conversion validation =="
VALIDATE_CMD=(
  "$PYTHON_BIN" "$VALIDATE_SCRIPT"
  --file "$F16_CKPT"
  --source-safetensors "$SOURCE_SAFETENSORS"
  --profile "$PROFILE"
  --serialization-guard-profile "$SERIALIZATION_GUARD_PROFILE"
  --serialization-report-limit "$SERIALIZATION_REPORT_LIMIT"
)
if [[ -n "$SERIALIZATION_BASELINE" ]]; then
  VALIDATE_CMD+=(--serialization-baseline "$SERIALIZATION_BASELINE")
fi

set +e
"${VALIDATE_CMD[@]}" | tee "$WORK_DIR/f16_validator.log"
F16_VALIDATE_RC=${PIPESTATUS[0]}
set -e

echo "f16_validate_rc=$F16_VALIDATE_RC"

run_canary() {
  local model_key="$1"
  local timeout_sec="$2"
  local max_responses="$3"
  local tag_suffix="$4"
  local log_path="$5"

  local cmd=(
    bash "$CANARY_SCRIPT"
    --model "$model_key"
    --host "$HOST"
    --max-responses "$max_responses"
    --tag "${TAG}_${tag_suffix}"
    --require-complete-stream
    --require-final-output
  )

  if [[ "$CANARY_NO_TIMEOUT" == "1" ]]; then
    cmd+=(--no-timeout)
  else
    cmd+=(--timeout-sec "$timeout_sec")
  fi

  if [[ "$FINAL_MODE" == "1" ]]; then
    cmd+=(--final-mode)
  fi

  set +e
  "${cmd[@]}" | tee "$log_path"
  local canary_rc=${PIPESTATUS[0]}
  set -e
  return "$canary_rc"
}

F16_CANARY_RC=0
if [[ "$RUN_F16_CANARY" == "1" ]]; then
  echo "== f16 runtime canary (strict completion) =="
  if run_canary "$F16_MODEL_KEY" "$F16_TIMEOUT_SEC" "$F16_MAX_RESPONSES" "f16" "$WORK_DIR/f16_canary.log"; then
    F16_CANARY_RC=0
  else
    F16_CANARY_RC=$?
  fi
  echo "f16_canary_rc=$F16_CANARY_RC"
else
  echo "f16 runtime canary skipped"
fi

Q6P_CANARY_RC=0
if [[ "$RUN_Q6P_CANARY" == "1" ]]; then
  echo "== q6p runtime canary (strict completion) =="
  if run_canary "$Q6P_MODEL_KEY" "$Q6P_TIMEOUT_SEC" "$Q6P_MAX_RESPONSES" "q6p" "$WORK_DIR/q6p_canary.log"; then
    Q6P_CANARY_RC=0
  else
    Q6P_CANARY_RC=$?
  fi
  echo "q6p_canary_rc=$Q6P_CANARY_RC"
else
  echo "q6p runtime canary skipped"
fi

FINAL_RC=0
if [[ "$F16_TENSOR_COUNT_RC" != "0" || "$F16_NULL_NAME_COUNT_RC" != "0" ]]; then
  FINAL_RC=1
fi
if [[ "$Q6P_TENSOR_COUNT_RC" != "0" || "$Q6P_NULL_NAME_COUNT_RC" != "0" ]]; then
  FINAL_RC=1
fi
if [[ "$F16_VALIDATE_RC" != "0" ]]; then
  FINAL_RC=1
fi
if [[ "$RUN_F16_CANARY" == "1" && "$F16_CANARY_RC" != "0" ]]; then
  FINAL_RC=1
fi
if [[ "$STRICT_ALL" == "1" && "$RUN_Q6P_CANARY" == "1" && "$Q6P_CANARY_RC" != "0" ]]; then
  FINAL_RC=1
fi

{
  echo "# LTX2.3 Model Validity Matrix"
  echo
  echo "- tag: $TAG"
  echo "- work_dir: $WORK_DIR"
  echo "- f16_ckpt: $F16_CKPT"
  echo "- q6p_ckpt: $Q6P_CKPT"
  echo "- source_safetensors: $SOURCE_SAFETENSORS"
  echo "- profile: $PROFILE"
  echo
  echo "## SQLite sanity"
  echo "- f16_tensor_count: ${F16_TENSOR_COUNT:-<error>} (rc=$F16_TENSOR_COUNT_RC)"
  echo "- f16_null_name_count: ${F16_NULL_NAME_COUNT:-<error>} (rc=$F16_NULL_NAME_COUNT_RC)"
  echo "- q6p_tensor_count: ${Q6P_TENSOR_COUNT:-<error>} (rc=$Q6P_TENSOR_COUNT_RC)"
  echo "- q6p_null_name_count: ${Q6P_NULL_NAME_COUNT:-<error>} (rc=$Q6P_NULL_NAME_COUNT_RC)"
  echo
  echo "## Conversion validation (f16)"
  echo "- validate_rc: $F16_VALIDATE_RC"
  echo "- log: $WORK_DIR/f16_validator.log"
  echo
  echo "## Runtime canary (strict completion)"
  if [[ "$RUN_F16_CANARY" == "1" ]]; then
    echo "- f16_canary_rc: $F16_CANARY_RC"
    echo "- f16_canary_log: $WORK_DIR/f16_canary.log"
  else
    echo "- f16_canary: skipped"
  fi
  if [[ "$RUN_Q6P_CANARY" == "1" ]]; then
    echo "- q6p_canary_rc: $Q6P_CANARY_RC"
    echo "- q6p_canary_log: $WORK_DIR/q6p_canary.log"
  else
    echo "- q6p_canary: skipped"
  fi
  echo
  echo "## Exit policy"
  echo "- strict_all: $STRICT_ALL"
  echo "- final_mode: $FINAL_MODE"
  echo "- canary_no_timeout: $CANARY_NO_TIMEOUT"
  echo "- final_rc: $FINAL_RC"
} > "$SUMMARY_FILE"

echo "summary_file=$SUMMARY_FILE"
if [[ "$FINAL_RC" == "0" ]]; then
  echo "RESULT=PASS"
else
  echo "RESULT=FAIL"
fi

exit "$FINAL_RC"
