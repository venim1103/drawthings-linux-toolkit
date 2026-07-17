#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT/.venv/bin/python"

NAME_BUILDER="$ROOT/tools/dt_build_q6p_dimfix_names.py"
ALIGN_SUBSET="$ROOT/tools/dt_align_ckpt_content_subset.py"
PATCH_META="$ROOT/tools/dt_patch_ckpt_metadata_subset.py"
PAYLOADFIX_RUNNER="$ROOT/tools/run_q6p_payloadfix_recursive.sh"

TARGET_Q6P="$ROOT/dt-models/10_e_v1_bf16_regen_0_q6p.ckpt"
SOURCE_F16="$ROOT/dt-models/10_e_v1_bf16_regen_0_f16.ckpt"
BASELINE_Q6P="$ROOT/dt-models/ltx_2.3_22b_distilled_1.1_q6p.ckpt"
REFERENCE_Q6P="$ROOT/dt-models/ltx_2.3_22b_distilled_q6p_forcedfix_clipfix2_20260602.ckpt"
NAMES_FILE="$ROOT/tools/patch_sets/10_e_v1_q6p_dimfix770_20260608.txt"

REBUILD_NAMES=0
RUN_CANARY=1
TAG="$(date +%Y%m%d_%H%M%S)"
WORK_DIR=""

HOST="127.0.0.1:7861"
CANARY_TIMEOUT_SEC=180
MAX_RESPONSES=30
WIDTH=256
HEIGHT=256
STEPS=4
SAMPLER=17
GUIDANCE=1.0
SHIFT=3.0
NUM_FRAMES=9
FPS_ID=5
SEED=4242

PROMPT="a cinematic shot of a red sports car driving on a mountain road at sunset, detailed, realistic"
NEG_PROMPT="blurry, distorted, low quality, artifacts"

CHUNK_SIZE=8
MIN_FREE_GB=5
PROGRESS_EVERY=100

PAYLOADFIX_NAMES_FILE=""
PAYLOADFIX_INITIAL_SPLIT_LINES=200
PAYLOADFIX_JOURNAL_MODE=delete
PAYLOADFIX_SKIP_PROBE=0
PAYLOADFIX_NO_RETRY_FAILED_LEAVES=0
PAYLOADFIX_IGNORE_CANARY_FAILURE=0

usage() {
  cat <<'EOF'
Usage:
  tools/run_q6p_inplace_dimfix_from_f16.sh [options]

This script patches the target q6p checkpoint IN PLACE (no full checkpoint copy):
  1) Build or load selected tensor names.
  2) Copy dim/data for those names from source f16 into target q6p.
  3) Copy metadata(type/format/datatype) for those names from source f16.
  4) Validate structure and optionally run a bounded canary probe.

Defaults are tuned for current 10_e_v1 workflow.

Options:
  --target-q6p <path>         Target q6p file to patch in place.
  --source-f16 <path>         Source f16 file used for row copy.
  --baseline-q6p <path>       Baseline q6p for name rebuilding.
  --reference-q6p <path>      Known working q6p for name rebuilding.
  --names-file <path>         Name list path (default: tools/patch_sets/...)
  --rebuild-names             Recompute names-file using baseline/reference.
  --skip-canary               Do not run bounded runtime canary.
  --tag <value>               Tag for output folder naming.
  --host <host:port>          gRPC host for canary (default: 127.0.0.1:7861)
  --canary-timeout-sec <n>    timeout(1) duration for generate-raw.
  --max-responses <n>         max responses for canary stream.
  --width <n>                 canary width (pixels)
  --height <n>                canary height (pixels)
  --steps <n>                 canary steps
  --seed <n>                  canary seed
  --chunk-size <n>            row update chunk size for dim/data copy.
  --min-free-gb <n>           free-space floor for dim/data mutation.

  Payloadfix mode (Run-006 style recursive split-on-failure):
  --payloadfix-names-file <path>
                               Run recursive data-only payload remediation using
                               the provided names file, then probe/canary.
  --payloadfix-initial-split-lines <n>
                               Initial split size for recursive payloadfix mode.
                               Default: 200.
  --payloadfix-journal-mode <delete|wal|preserve>
                               Journal mode for recursive payloadfix dt_align.
                               Default: delete.
  --payloadfix-skip-probe      Skip row-wise probe in payloadfix mode.
  --payloadfix-no-retry-failed-leaves
                               Skip one-name retry pass in payloadfix mode.
  --payloadfix-ignore-canary-failure
                               Keep payloadfix mode exit status successful even
                               when bounded canary fails.
  -h, --help                  show this help.

Examples:
  tools/run_q6p_inplace_dimfix_from_f16.sh

  tools/run_q6p_inplace_dimfix_from_f16.sh \
    --rebuild-names \
    --baseline-q6p dt-models/ltx_2.3_22b_distilled_1.1_q6p.ckpt \
    --reference-q6p dt-models/ltx_2.3_22b_distilled_q6p_forcedfix_clipfix2_20260602.ckpt

  tools/run_q6p_inplace_dimfix_from_f16.sh \
    --payloadfix-names-file tools/patch_sets/10_e_v1_q6p_post_metachunkfix_mismatch_meta_len_all5746_20260608.txt \
    --chunk-size 4 \
    --min-free-gb 50 \
    --canary-timeout-sec 120 \
    --max-responses 10 \
    --tag 20260608_run006_payloadfix
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
      --target-q6p)
        TARGET_Q6P="${2:-}"
        shift 2
        ;;
      --source-f16)
        SOURCE_F16="${2:-}"
        shift 2
        ;;
      --baseline-q6p)
        BASELINE_Q6P="${2:-}"
        shift 2
        ;;
      --reference-q6p)
        REFERENCE_Q6P="${2:-}"
        shift 2
        ;;
      --names-file)
        NAMES_FILE="${2:-}"
        shift 2
        ;;
      --rebuild-names)
        REBUILD_NAMES=1
        shift
        ;;
      --skip-canary)
        RUN_CANARY=0
        shift
        ;;
      --tag)
        TAG="${2:-}"
        shift 2
        ;;
      --host)
        HOST="${2:-}"
        shift 2
        ;;
      --canary-timeout-sec)
        CANARY_TIMEOUT_SEC="${2:-}"
        shift 2
        ;;
      --max-responses)
        MAX_RESPONSES="${2:-}"
        shift 2
        ;;
      --width)
        WIDTH="${2:-}"
        shift 2
        ;;
      --height)
        HEIGHT="${2:-}"
        shift 2
        ;;
      --steps)
        STEPS="${2:-}"
        shift 2
        ;;
      --seed)
        SEED="${2:-}"
        shift 2
        ;;
      --chunk-size)
        CHUNK_SIZE="${2:-}"
        shift 2
        ;;
      --min-free-gb)
        MIN_FREE_GB="${2:-}"
        shift 2
        ;;
      --payloadfix-names-file)
        PAYLOADFIX_NAMES_FILE="${2:-}"
        shift 2
        ;;
      --payloadfix-initial-split-lines)
        PAYLOADFIX_INITIAL_SPLIT_LINES="${2:-}"
        shift 2
        ;;
      --payloadfix-journal-mode)
        PAYLOADFIX_JOURNAL_MODE="${2:-}"
        shift 2
        ;;
      --payloadfix-skip-probe)
        PAYLOADFIX_SKIP_PROBE=1
        shift
        ;;
      --payloadfix-no-retry-failed-leaves)
        PAYLOADFIX_NO_RETRY_FAILED_LEAVES=1
        shift
        ;;
      --payloadfix-ignore-canary-failure)
        PAYLOADFIX_IGNORE_CANARY_FAILURE=1
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

TARGET_Q6P="$(abs_path "$TARGET_Q6P")"
SOURCE_F16="$(abs_path "$SOURCE_F16")"
BASELINE_Q6P="$(abs_path "$BASELINE_Q6P")"
REFERENCE_Q6P="$(abs_path "$REFERENCE_Q6P")"
NAMES_FILE="$(abs_path "$NAMES_FILE")"
if [[ -n "$PAYLOADFIX_NAMES_FILE" ]]; then
  PAYLOADFIX_NAMES_FILE="$(abs_path "$PAYLOADFIX_NAMES_FILE")"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "error: missing python env at $PYTHON_BIN" >&2
  exit 1
fi

for p in "$NAME_BUILDER" "$ALIGN_SUBSET" "$PATCH_META"; do
  if [[ ! -f "$p" ]]; then
    echo "error: missing helper script: $p" >&2
    exit 1
  fi
done

for p in "$TARGET_Q6P" "$SOURCE_F16"; do
  if [[ ! -f "$p" ]]; then
    echo "error: required file missing: $p" >&2
    exit 1
  fi
done

if [[ -n "$PAYLOADFIX_NAMES_FILE" ]]; then
  if [[ "$REBUILD_NAMES" == "1" ]]; then
    echo "error: --payloadfix-names-file cannot be combined with --rebuild-names" >&2
    exit 1
  fi
  if [[ ! -x "$PAYLOADFIX_RUNNER" ]]; then
    echo "error: payloadfix runner is not executable: $PAYLOADFIX_RUNNER" >&2
    exit 1
  fi
  if [[ ! -f "$PAYLOADFIX_NAMES_FILE" ]]; then
    echo "error: payloadfix names file missing: $PAYLOADFIX_NAMES_FILE" >&2
    exit 1
  fi

  payloadfix_chunk_size="$CHUNK_SIZE"
  payloadfix_min_free_gb="$MIN_FREE_GB"
  # Keep historical defaults for dimfix mode, but map payloadfix mode defaults
  # to the Run-006 profile unless explicitly overridden.
  if [[ "$payloadfix_chunk_size" == "8" ]]; then
    payloadfix_chunk_size=4
  fi
  if [[ "$payloadfix_min_free_gb" == "5" ]]; then
    payloadfix_min_free_gb=50
  fi

  payloadfix_cmd=(
    bash "$PAYLOADFIX_RUNNER"
    --target-q6p "$TARGET_Q6P"
    --baseline-f16 "$SOURCE_F16"
    --names-file "$PAYLOADFIX_NAMES_FILE"
    --tag "$TAG"
    --initial-split-lines "$PAYLOADFIX_INITIAL_SPLIT_LINES"
    --chunk-size "$payloadfix_chunk_size"
    --journal-mode "$PAYLOADFIX_JOURNAL_MODE"
    --min-free-gb "$payloadfix_min_free_gb"
    --head-bytes 32
    --host "$HOST"
    --canary-timeout-sec "$CANARY_TIMEOUT_SEC"
    --max-responses "$MAX_RESPONSES"
  )

  if [[ "$RUN_CANARY" != "1" ]]; then
    payloadfix_cmd+=(--skip-canary)
  fi
  if [[ "$PAYLOADFIX_SKIP_PROBE" == "1" ]]; then
    payloadfix_cmd+=(--skip-probe)
  fi
  if [[ "$PAYLOADFIX_NO_RETRY_FAILED_LEAVES" == "1" ]]; then
    payloadfix_cmd+=(--no-retry-failed-leaves)
  fi
  if [[ "$PAYLOADFIX_IGNORE_CANARY_FAILURE" == "1" ]]; then
    payloadfix_cmd+=(--ignore-canary-failure)
  fi

  echo "== dispatch recursive payloadfix runner =="
  "${payloadfix_cmd[@]}"
  exit $?
fi

if [[ "$REBUILD_NAMES" == "1" ]]; then
  for p in "$BASELINE_Q6P" "$REFERENCE_Q6P"; do
    if [[ ! -f "$p" ]]; then
      echo "error: --rebuild-names needs existing file: $p" >&2
      exit 1
    fi
  done

  mkdir -p "$(dirname "$NAMES_FILE")"
  "$PYTHON_BIN" "$NAME_BUILDER" \
    --target-q6p "$TARGET_Q6P" \
    --baseline-q6p "$BASELINE_Q6P" \
    --reference-q6p "$REFERENCE_Q6P" \
    --out "$NAMES_FILE"
fi

if [[ ! -f "$NAMES_FILE" ]]; then
  echo "error: names file missing: $NAMES_FILE" >&2
  echo "       Either provide --names-file with an existing list or use --rebuild-names." >&2
  exit 1
fi

NAMES_COUNT="$(wc -l < "$NAMES_FILE" | tr -d '[:space:]')"
if [[ -z "$NAMES_COUNT" || "$NAMES_COUNT" == "0" ]]; then
  echo "error: names file has zero entries: $NAMES_FILE" >&2
  exit 1
fi

WORK_DIR="$ROOT/output/q6p_inplace_dimfix_${TAG}"
mkdir -p "$WORK_DIR"

echo "== q6p in-place dimfix from f16 =="
echo "target_q6p=$TARGET_Q6P"
echo "source_f16=$SOURCE_F16"
echo "names_file=$NAMES_FILE"
echo "names_count=$NAMES_COUNT"
echo "work_dir=$WORK_DIR"
echo "run_canary=$RUN_CANARY"

echo "== stop runtime processes =="
pkill -9 -f 'gRPCServerCLI --address 127.0.0.1 --port 7861' || true
pkill -9 -f 'tools/dt_api_client.py .*generate-raw' || true

echo "== apply dim/data subset (in place) =="
"$PYTHON_BIN" "$ALIGN_SUBSET" \
  --file "$TARGET_Q6P" \
  --baseline "$SOURCE_F16" \
  --dim-names-file "$NAMES_FILE" \
  --data-names-file "$NAMES_FILE" \
  --mode apply \
  --chunk-size "$CHUNK_SIZE" \
  --journal-mode wal \
  --min-free-gb "$MIN_FREE_GB" \
  --head-bytes 32

echo "== apply metadata subset (in place) =="
"$PYTHON_BIN" "$PATCH_META" \
  --file "$TARGET_Q6P" \
  --baseline "$SOURCE_F16" \
  --names-file "$NAMES_FILE" \
  --journal-mode wal \
  --progress-every "$PROGRESS_EVERY"

echo "== structural validate =="
sqlite3 "$TARGET_Q6P" 'SELECT COUNT(*) FROM tensors;'
"$PYTHON_BIN" "$ROOT/tools/dt_validate_converted_ckpt.py" --file "$TARGET_Q6P" --profile ltx2_3

if [[ "$RUN_CANARY" != "1" ]]; then
  echo "== done (canary skipped) =="
  exit 0
fi

if ! command -v timeout >/dev/null 2>&1; then
  echo "error: timeout command not found; cannot run bounded canary" >&2
  exit 1
fi

SERVER_LOG="$WORK_DIR/server.log"
CLIENT_LOG="$WORK_DIR/client.log"
CONFIG_BIN="$WORK_DIR/config.bin"
MODEL_KEY="$(basename "$TARGET_Q6P")"

echo "== start temporary server for bounded canary =="
nohup drawthings-grpc --address 127.0.0.1 --port 7861 --gpu 0 --no-tls --model-browser --no-response-compression "$ROOT/dt-models" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "server_pid=$SERVER_PID"

READY=0
for _ in $(seq 1 200); do
  if "$PYTHON_BIN" "$ROOT/tools/dt_api_client.py" --host "$HOST" echo --name "q6p-dimfix-canary" >/dev/null 2>&1; then
    READY=1
    break
  fi
done
echo "server_ready=$READY"
if [[ "$READY" != "1" ]]; then
  pkill -9 -f 'gRPCServerCLI --address 127.0.0.1 --port 7861' || true
  echo "error: server did not become ready" >&2
  exit 1
fi

"$PYTHON_BIN" "$ROOT/tools/dt_make_config.py" \
  --out "$CONFIG_BIN" \
  --model "$MODEL_KEY" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --steps "$STEPS" \
  --sampler "$SAMPLER" \
  --guidance-scale "$GUIDANCE" \
  --shift "$SHIFT" \
  --hires-fix false \
  --num-frames "$NUM_FRAMES" \
  --fps-id "$FPS_ID" \
  --seed "$SEED"

echo "== bounded canary generate-raw =="
set +e
timeout "${CANARY_TIMEOUT_SEC}s" \
  "$PYTHON_BIN" "$ROOT/tools/dt_api_client.py" \
    --host "$HOST" \
    --max-recv-bytes 134217728 \
    generate-raw \
    --config-bin "$CONFIG_BIN" \
    --prompt "$PROMPT" \
    --negative-prompt "$NEG_PROMPT" \
    --output-dir "$WORK_DIR" \
    --chunked \
    --save-preview \
    --max-responses "$MAX_RESPONSES" > "$CLIENT_LOG" 2>&1
CANARY_RC=$?
timeout -k 5s 15s \
  "$PYTHON_BIN" "$ROOT/tools/dt_api_client.py" --host "$HOST" echo --name "q6p-dimfix-post" >/dev/null 2>&1
POST_ECHO_RC=$?
set -e

pkill -9 -f 'gRPCServerCLI --address 127.0.0.1 --port 7861' || true

echo "canary_rc=$CANARY_RC"
echo "post_echo_rc=$POST_ECHO_RC"
echo "client_log=$CLIENT_LOG"
echo "server_log=$SERVER_LOG"
echo "--- client.log (tail) ---"
tail -n 120 "$CLIENT_LOG" || true
echo "--- server.log (tail) ---"
tail -n 120 "$SERVER_LOG" || true

if [[ "$CANARY_RC" == "124" ]]; then
  echo "RESULT=FAIL canary timed out (${CANARY_TIMEOUT_SEC}s)"
  exit 1
fi

if [[ "$CANARY_RC" != "0" ]]; then
  echo "RESULT=FAIL canary rc=$CANARY_RC"
  exit 1
fi

if ! grep -q 'response #1' "$CLIENT_LOG"; then
  echo "RESULT=FAIL no streamed responses observed"
  exit 1
fi

echo "RESULT=PASS"
