#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT/.venv/bin/python"
VALIDATE_SCRIPT="$ROOT/tools/dt_validate_converted_ckpt.py"
CANARY_SCRIPT="$ROOT/tools/run_q6p_canary_once.sh"

F16_CKPT="$ROOT/dt-models/10_e_v1_bf16_regen_0_f16.ckpt"
Q6P_CKPT="$ROOT/dt-models/10_e_v1_bf16_regen_0_q6p.ckpt"
SOURCE_SAFETENSORS="$ROOT/dt-models/10_e_v1_bf16.safetensors"

HOST="127.0.0.1:7861"
TAG="$(date +%Y%m%d_%H%M%S)"

RUN_F16_CANARY=1
RUN_Q6P_CANARY=1
STRICT_ALL=0

F16_TIMEOUT_SEC=240
F16_MAX_RESPONSES=30
Q6P_TIMEOUT_SEC=120
Q6P_MAX_RESPONSES=10

usage() {
  cat <<'EOF'
Usage:
  tools/run_10e_v1_model_validity_matrix.sh [options]

Purpose:
  Run a reproducible validity matrix for 10_e_v1 models:
    1) f16 SQLite sanity (tensor count + null name check)
    2) f16 structure validation against source safetensors profile
    3) f16 bounded runtime canary
    4) q6p bounded runtime canary (comparison branch)

Options:
  --f16-ckpt <path>             Path to f16 checkpoint.
  --q6p-ckpt <path>             Path to q6p checkpoint.
  --source-safetensors <path>   Path to source safetensors file.
  --host <host:port>            gRPC host for canary runs (default: 127.0.0.1:7861).
  --tag <value>                 Tag for output folder naming.

  --skip-f16-canary             Skip runtime canary for f16.
  --skip-q6p-canary             Skip runtime canary for q6p.

  --f16-timeout-sec <n>         Timeout for f16 canary (default: 240).
  --f16-max-responses <n>       Max responses for f16 canary (default: 30).
  --q6p-timeout-sec <n>         Timeout for q6p canary (default: 120).
  --q6p-max-responses <n>       Max responses for q6p canary (default: 10).

  --strict-all                  Exit non-zero if q6p canary fails.
                                By default, q6p failure is reported but not fatal.
  -h, --help                    Show this help.

Outputs:
  output/model_validity_<tag>/
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

if [[ $# -gt 0 ]]; then
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
      --f16-max-responses)
        F16_MAX_RESPONSES="${2:-}"
        shift 2
        ;;
      --q6p-timeout-sec)
        Q6P_TIMEOUT_SEC="${2:-}"
        shift 2
        ;;
      --q6p-max-responses)
        Q6P_MAX_RESPONSES="${2:-}"
        shift 2
        ;;
      --strict-all)
        STRICT_ALL=1
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
fi

F16_CKPT="$(abs_path "$F16_CKPT")"
Q6P_CKPT="$(abs_path "$Q6P_CKPT")"
SOURCE_SAFETENSORS="$(abs_path "$SOURCE_SAFETENSORS")"

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

WORK_DIR="$ROOT/output/model_validity_${TAG}"
mkdir -p "$WORK_DIR"
SUMMARY_FILE="$WORK_DIR/summary.md"

echo "== 10_e_v1 model validity matrix =="
echo "tag=$TAG"
echo "work_dir=$WORK_DIR"
echo "f16_ckpt=$F16_CKPT"
echo "q6p_ckpt=$Q6P_CKPT"
echo "source_safetensors=$SOURCE_SAFETENSORS"

echo "== f16 SQLite sanity =="
set +e
F16_TENSOR_COUNT="$(sqlite3 "$F16_CKPT" 'SELECT COUNT(*) FROM tensors;' 2>/dev/null)"
F16_TENSOR_COUNT_RC=$?
F16_NULL_NAME_COUNT="$(sqlite3 "$F16_CKPT" 'SELECT COUNT(*) FROM tensors WHERE name IS NULL OR name = "";' 2>/dev/null)"
F16_NULL_NAME_COUNT_RC=$?
set -e

echo "f16_tensor_count=${F16_TENSOR_COUNT:-<error>}"
echo "f16_null_name_count=${F16_NULL_NAME_COUNT:-<error>}"

echo "== f16 conversion validation =="
set +e
"$PYTHON_BIN" "$VALIDATE_SCRIPT" \
  --file "$F16_CKPT" \
  --source-safetensors "$SOURCE_SAFETENSORS" \
  --profile auto | tee "$WORK_DIR/f16_validator.log"
F16_VALIDATE_RC=${PIPESTATUS[0]}
set -e

echo "f16_validate_rc=$F16_VALIDATE_RC"

F16_CANARY_RC=0
if [[ "$RUN_F16_CANARY" == "1" ]]; then
  echo "== f16 runtime canary =="
  set +e
  bash "$CANARY_SCRIPT" \
    --model "$(basename "$F16_CKPT")" \
    --host "$HOST" \
    --timeout-sec "$F16_TIMEOUT_SEC" \
    --max-responses "$F16_MAX_RESPONSES" \
    --tag "${TAG}_f16" | tee "$WORK_DIR/f16_canary.log"
  F16_CANARY_RC=${PIPESTATUS[0]}
  set -e
  echo "f16_canary_rc=$F16_CANARY_RC"
else
  echo "f16 runtime canary skipped"
fi

Q6P_CANARY_RC=0
if [[ "$RUN_Q6P_CANARY" == "1" ]]; then
  echo "== q6p runtime canary =="
  set +e
  bash "$CANARY_SCRIPT" \
    --model "$(basename "$Q6P_CKPT")" \
    --host "$HOST" \
    --timeout-sec "$Q6P_TIMEOUT_SEC" \
    --max-responses "$Q6P_MAX_RESPONSES" \
    --tag "${TAG}_q6p" | tee "$WORK_DIR/q6p_canary.log"
  Q6P_CANARY_RC=${PIPESTATUS[0]}
  set -e
  echo "q6p_canary_rc=$Q6P_CANARY_RC"
else
  echo "q6p runtime canary skipped"
fi

FINAL_RC=0
if [[ "$F16_TENSOR_COUNT_RC" != "0" || "$F16_NULL_NAME_COUNT_RC" != "0" ]]; then
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
  echo "# 10_e_v1 Model Validity Matrix"
  echo
  echo "- tag: $TAG"
  echo "- work_dir: $WORK_DIR"
  echo "- f16_ckpt: $F16_CKPT"
  echo "- q6p_ckpt: $Q6P_CKPT"
  echo "- source_safetensors: $SOURCE_SAFETENSORS"
  echo
  echo "## SQLite sanity (f16)"
  echo "- tensor_count: ${F16_TENSOR_COUNT:-<error>} (rc=$F16_TENSOR_COUNT_RC)"
  echo "- null_name_count: ${F16_NULL_NAME_COUNT:-<error>} (rc=$F16_NULL_NAME_COUNT_RC)"
  echo
  echo "## Conversion validation (f16)"
  echo "- validate_rc: $F16_VALIDATE_RC"
  echo "- log: $WORK_DIR/f16_validator.log"
  echo
  echo "## Runtime canary"
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
  echo "- final_rc: $FINAL_RC"
} > "$SUMMARY_FILE"

echo "summary_file=$SUMMARY_FILE"
if [[ "$FINAL_RC" == "0" ]]; then
  echo "RESULT=PASS"
else
  echo "RESULT=FAIL"
fi

exit "$FINAL_RC"
