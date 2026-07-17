#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_KEY="10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt"
TAG="$(date +%Y%m%d_%H%M%S)"
SEEDS_CSV="4242,7777,1337"
SIZES_CSV="256x256,384x704"
STEPS=4
TIMEOUT_SEC=240
HOST="127.0.0.1:7861"

usage() {
  cat <<'EOF'
Usage:
  tools/run_q6p_strict_stability_matrix.sh [options]

Options:
  --model <file>            Model key from dt-models directory.
  --tag <value>             Run tag (default: timestamp).
  --seeds <csv>             Comma-separated seeds (default: 4242,7777,1337).
  --sizes <csv>             Comma-separated WIDTHxHEIGHT pairs (default: 256x256,384x704).
  --steps <n>               Sampling steps per case (default: 4).
  --timeout-sec <n>         Per-case timeout (default: 240).
  --host <host:port>        gRPC host (default: 127.0.0.1:7861).
  -h, --help                Show this help.

Outputs:
  output/q6p_strict_stability_<tag>/
    - cases/<case>.log
    - results.tsv
    - summary.md
EOF
}

if [[ $# -gt 0 ]]; then
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model)
        MODEL_KEY="${2:-}"
        shift 2
        ;;
      --tag)
        TAG="${2:-}"
        shift 2
        ;;
      --seeds)
        SEEDS_CSV="${2:-}"
        shift 2
        ;;
      --sizes)
        SIZES_CSV="${2:-}"
        shift 2
        ;;
      --steps)
        STEPS="${2:-}"
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
fi

if ! [[ "$STEPS" =~ ^[0-9]+$ ]] || [[ "$STEPS" -lt 1 ]]; then
  echo "error: --steps must be an integer >= 1" >&2
  exit 1
fi

if ! [[ "$TIMEOUT_SEC" =~ ^[0-9]+$ ]] || [[ "$TIMEOUT_SEC" -lt 1 ]]; then
  echo "error: --timeout-sec must be an integer >= 1" >&2
  exit 1
fi

WORK_DIR="$ROOT/output/q6p_strict_stability_${TAG}"
CASES_DIR="$WORK_DIR/cases"
RESULTS_TSV="$WORK_DIR/results.tsv"
SUMMARY_MD="$WORK_DIR/summary.md"
mkdir -p "$CASES_DIR"

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
IFS=',' read -r -a SIZES <<< "$SIZES_CSV"

echo -e "case\tseed\twidth\theight\tsteps\trc\tresult\tcanary_rc\tpost_echo_rc\tresponses\timages\taudio" > "$RESULTS_TSV"

overall_rc=0

echo "== q6p strict stability matrix =="
echo "model=$MODEL_KEY"
echo "tag=$TAG"
echo "seeds=$SEEDS_CSV"
echo "sizes=$SIZES_CSV"
echo "steps=$STEPS"
echo "timeout_sec=$TIMEOUT_SEC"
echo "host=$HOST"
echo "work_dir=$WORK_DIR"

for size in "${SIZES[@]}"; do
  width="${size%x*}"
  height="${size#*x}"

  if [[ -z "$width" || -z "$height" || "$width" == "$size" ]]; then
    echo "error: invalid size entry: $size (expected WIDTHxHEIGHT)" >&2
    exit 1
  fi

  if ! [[ "$width" =~ ^[0-9]+$ ]] || ! [[ "$height" =~ ^[0-9]+$ ]]; then
    echo "error: invalid numeric size entry: $size" >&2
    exit 1
  fi

  for seed in "${SEEDS[@]}"; do
    if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
      echo "error: invalid seed entry: $seed" >&2
      exit 1
    fi

    case_tag="${TAG}_s${seed}_${width}x${height}_st${STEPS}"
    case_log="$CASES_DIR/${case_tag}.log"

    echo "== case: $case_tag =="
    set +e
    bash "$ROOT/tools/run_q6p_canary_once.sh" \
      --model "$MODEL_KEY" \
      --host "$HOST" \
      --width "$width" \
      --height "$height" \
      --steps "$STEPS" \
      --seed "$seed" \
      --timeout-sec "$TIMEOUT_SEC" \
      --max-responses 0 \
      --require-complete-stream \
      --require-final-output \
      --tag "$case_tag" > "$case_log" 2>&1
    rc=$?
    set -e

    if [[ "$rc" -ne 0 ]]; then
      overall_rc=1
    fi

    result="UNKNOWN"
    if rg -q '^RESULT=PASS$' "$case_log"; then
      result="PASS"
    elif rg -q '^RESULT=FAIL' "$case_log"; then
      result="FAIL"
    fi

    canary_rc="$(awk -F= '/^canary_rc=/{print $2; exit}' "$case_log")"
    post_echo_rc="$(awk -F= '/^post_echo_rc=/{print $2; exit}' "$case_log")"
    responses="$(awk -F': ' '/^responses:/{value=$2} END{if (value=="") value="0"; print value}' "$case_log")"
    images="$(awk -F': ' '/^images written:/{value=$2} END{if (value=="") value="0"; print value}' "$case_log")"
    audio="$(awk -F': ' '/^audio written:/{value=$2} END{if (value=="") value="0"; print value}' "$case_log")"

    echo -e "${case_tag}\t${seed}\t${width}\t${height}\t${STEPS}\t${rc}\t${result}\t${canary_rc}\t${post_echo_rc}\t${responses}\t${images}\t${audio}" >> "$RESULTS_TSV"
  done
done

pass_count="$(awk -F'\t' 'NR>1 && $7=="PASS"{c++} END{print c+0}' "$RESULTS_TSV")"
fail_count="$(awk -F'\t' 'NR>1 && $7!="PASS"{c++} END{print c+0}' "$RESULTS_TSV")"
case_count="$(awk 'END{print NR-1}' "$RESULTS_TSV")"

{
  echo "# Q6P Strict Stability Matrix"
  echo
  echo "- model: $MODEL_KEY"
  echo "- tag: $TAG"
  echo "- work_dir: $WORK_DIR"
  echo "- seeds: $SEEDS_CSV"
  echo "- sizes: $SIZES_CSV"
  echo "- steps: $STEPS"
  echo "- timeout_sec: $TIMEOUT_SEC"
  echo "- cases: $case_count"
  echo "- pass: $pass_count"
  echo "- fail: $fail_count"
  echo
  echo "## Per-case results"
  echo
  echo "| case | seed | width | height | steps | rc | result | canary_rc | post_echo_rc | responses | images | audio |"
  echo "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|"
  awk -F'\t' 'NR>1 {printf("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)}' "$RESULTS_TSV"
  echo
  echo "- results_tsv: $RESULTS_TSV"
} > "$SUMMARY_MD"

echo "results_tsv=$RESULTS_TSV"
echo "summary_md=$SUMMARY_MD"

if [[ "$overall_rc" -eq 0 ]]; then
  echo "RESULT=PASS"
  exit 0
fi

echo "RESULT=FAIL"
exit 1
