#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_KEY="10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt"
TAG="run082_restart_per_repeat_textgate_boundary_matrix"
REPEATS=8
TIMEOUT_SEC=75
HOST="127.0.0.1:7861"
ENTRY_VERSION="ltx2.3"

# Case format: id|text_encoder|clip_encoder
CASES=(
  "text_gemma_only|gemma_3_12b_it_qat_q8p.ckpt|none"
  "text_gemma_clip_traced|gemma_3_12b_it_qat_q8p.ckpt|10_e_v1_bf16_regen_0_q6p.ckpt"
  "text_gemma_clipvit|gemma_3_12b_it_qat_q8p.ckpt|clip_vit_l14_f16.ckpt"
)

usage() {
  cat <<'EOF'
Usage:
  tools/run_q6p_restart_per_repeat_textgate_boundary_matrix.sh [options]

Options:
  --model <file>         Model key (default: trace021 key).
  --tag <value>          Output matrix tag (default: run082_restart_per_repeat_textgate_boundary_matrix).
  --repeats <n>          Repeats per case (default: 8).
  --timeout-sec <n>      Per-repeat timeout (default: 75).
  --host <host:port>     gRPC host (default: 127.0.0.1:7861).
  --entry-version <v>    Mapped entry version (default: ltx2.3).
  -h, --help             Show this help.

Outputs:
  output/<tag>/
    - cases/<case>.log
    - results.tsv
    - summary.md

Notes:
  - Uses restart-per-repeat mode for every case.
  - Forces modifier/autoencoder to none to isolate text+companion boundary behavior.
  - Focuses on the gemma boundary from run081/run081b.
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
      --repeats)
        REPEATS="${2:-}"
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
      --entry-version)
        ENTRY_VERSION="${2:-}"
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

if ! [[ "$REPEATS" =~ ^[0-9]+$ ]] || [[ "$REPEATS" -lt 1 ]]; then
  echo "error: --repeats must be an integer >= 1" >&2
  exit 1
fi

if ! [[ "$TIMEOUT_SEC" =~ ^[0-9]+$ ]] || [[ "$TIMEOUT_SEC" -lt 1 ]]; then
  echo "error: --timeout-sec must be an integer >= 1" >&2
  exit 1
fi

WORK_DIR="$ROOT/output/$TAG"
CASES_DIR="$WORK_DIR/cases"
RESULTS_TSV="$WORK_DIR/results.tsv"
SUMMARY_MD="$WORK_DIR/summary.md"
mkdir -p "$CASES_DIR"

echo -e "case\ttext_encoder\tclip_encoder\trepeats\tcanary_rc_124\tcanary_rc_1\tpost_echo_rc_124\tpost_echo_rc_0\tsource_mapping\tassert_count\tabort_count\tunavailable_count" > "$RESULTS_TSV"

overall_fail=0

echo "== restart-per-repeat text-gate boundary matrix =="
echo "tag=$TAG"
echo "model=$MODEL_KEY"
echo "repeats=$REPEATS"
echo "timeout_sec=$TIMEOUT_SEC"
echo "host=$HOST"
echo "entry_version=$ENTRY_VERSION"
echo "work_dir=$WORK_DIR"

for spec in "${CASES[@]}"; do
  IFS='|' read -r case_id text_encoder clip_encoder <<< "$spec"

  case_tag="${TAG}_${case_id}"
  case_log="$CASES_DIR/${case_id}.log"

  echo "== case: $case_id (text=$text_encoder clip=$clip_encoder) =="

  set +e
  bash "$ROOT/tools/run_q6p_warm_server_mod_auto_repeats.sh" \
    --model "$MODEL_KEY" \
    --tag "$case_tag" \
    --repeats "$REPEATS" \
    --timeout-sec "$TIMEOUT_SEC" \
    --host "$HOST" \
    --restart-per-repeat \
    --entry-version "$ENTRY_VERSION" \
    --text-encoder "$text_encoder" \
    --clip-encoder "$clip_encoder" \
    --modifier none \
    --autoencoder none > "$case_log" 2>&1
  rc=$?
  set -e

  if [[ "$rc" -ne 0 ]]; then
    overall_fail=1
  fi

  summary_tsv="$ROOT/output/${case_tag}_repeats_summary.tsv"
  server_log="$ROOT/output/${case_tag}/server.log"

  if [[ ! -f "$summary_tsv" ]]; then
    echo "error: missing summary for case $case_id: $summary_tsv" >&2
    overall_fail=1
    continue
  fi

  canary_124=$(awk -F'\t' 'NR>1 && $2==124{c++} END{print c+0}' "$summary_tsv")
  canary_1=$(awk -F'\t' 'NR>1 && $2==1{c++} END{print c+0}' "$summary_tsv")
  post_124=$(awk -F'\t' 'NR>1 && $3==124{c++} END{print c+0}' "$summary_tsv")
  post_0=$(awk -F'\t' 'NR>1 && $3==0{c++} END{print c+0}' "$summary_tsv")
  source_mapping=$(awk -F'\t' 'NR>1 && $5=="mapping"{c++} END{print c+0}' "$summary_tsv")

  assert_count=0
  abort_count=0
  unavailable_count=0

  if [[ -f "$server_log" ]]; then
    assert_count=$(rg -F -c "memory_type == CCV_TENSOR_CPU_MEMORY" "$server_log" || true)
    assert_count=${assert_count:-0}
    abort_count=$(rg -F -c "Program crashed: Aborted" "$server_log" || true)
    abort_count=${abort_count:-0}
  fi

  shopt -s nullglob
  client_logs=("$ROOT/output/${case_tag}_r"*_client.log)
  shopt -u nullglob
  for cl in "${client_logs[@]}"; do
    c=$(rg -c "UNAVAILABLE|SETTINGS frame" "$cl" || true)
    c=${c:-0}
    unavailable_count=$((unavailable_count + c))
  done

  echo -e "${case_id}\t${text_encoder}\t${clip_encoder}\t${REPEATS}\t${canary_124}\t${canary_1}\t${post_124}\t${post_0}\t${source_mapping}\t${assert_count}\t${abort_count}\t${unavailable_count}" >> "$RESULTS_TSV"
done

case_count=$(awk 'END{print NR-1}' "$RESULTS_TSV")
all_mapping=$(awk -F'\t' 'NR>1 && $9==$4{c++} END{print c+0}' "$RESULTS_TSV")
any_canary_1=$(awk -F'\t' 'NR>1 && $6>0{c++} END{print c+0}' "$RESULTS_TSV")
any_assert=$(awk -F'\t' 'NR>1 && $10>0{c++} END{print c+0}' "$RESULTS_TSV")

{
  echo "# Restart-Per-Repeat Text-Gate Boundary Matrix"
  echo
  echo "- tag: $TAG"
  echo "- model: $MODEL_KEY"
  echo "- repeats_per_case: $REPEATS"
  echo "- timeout_sec: $TIMEOUT_SEC"
  echo "- host: $HOST"
  echo "- entry_version: $ENTRY_VERSION"
  echo "- cases: $case_count"
  echo
  echo "## Summary"
  echo
  echo "- all_mapping_cases: $all_mapping/$case_count"
  echo "- cases_with_any_canary_rc_1: $any_canary_1"
  echo "- cases_with_any_assertion: $any_assert"
  echo
  echo "## Per-case results"
  echo
  echo "| case | text_encoder | clip_encoder | repeats | canary_rc_124 | canary_rc_1 | post_echo_rc_124 | post_echo_rc_0 | source_mapping | assert_count | abort_count | unavailable_count |"
  echo "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
  awk -F'\t' 'NR>1 {printf("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)}' "$RESULTS_TSV"
  echo
  echo "- results_tsv: $RESULTS_TSV"
} > "$SUMMARY_MD"

echo "results_tsv=$RESULTS_TSV"
echo "summary_md=$SUMMARY_MD"

if [[ "$overall_fail" -eq 0 ]]; then
  echo "RESULT=PASS"
  exit 0
fi

echo "RESULT=FAIL"
exit 1
