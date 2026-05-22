#!/usr/bin/env bash
set -euo pipefail

# Wrapper around model-quantizer for long-running quantization runs.
#
# Features:
# - Auto-builds model-quantizer if missing.
# - Shows a live read/rate/ETA monitor and output file growth.
# - Keeps quantizer CLI arguments unchanged (pass-through).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${DRAWTHINGS_WORKSPACE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PKG_PATH="$ROOT/draw-things-community"
BUILD_CONFIG="${DRAWTHINGS_BUILD_CONFIG:-release}"
AUTOBUILD="${DRAWTHINGS_QUANTIZER_AUTOBUILD:-1}"
MONITOR_ENABLED="${DRAWTHINGS_QUANTIZER_MONITOR:-1}"
MONITOR_INTERVAL="${DRAWTHINGS_QUANTIZER_MONITOR_INTERVAL:-5}"

usage() {
  cat <<'EOF'
Usage:
  dt_quantize_model.sh [model-quantizer args]

Pass-through wrapper for draw-things-community/.build/<config>/model-quantizer.

Environment:
  DRAWTHINGS_QUANTIZER_AUTOBUILD      Set to 0 to disable auto-build when missing.
  DRAWTHINGS_QUANTIZER_MONITOR        Set to 0 to disable live progress monitoring.
  DRAWTHINGS_QUANTIZER_MONITOR_INTERVAL
                                      Monitor refresh interval in seconds (default: 5).
  DRAWTHINGS_BUILD_CONFIG             release (default) or debug.

Examples:
  tools/dt_quantize_model.sh --help
  tools/dt_quantize_model.sh \
    -i dt-models/model_f16.ckpt \
    -m ltx2.3 \
    -o dt-models/model_q4p.ckpt \
    --target-codec q4p

Notes:
  - Progress percentage is based on input bytes read from /proc/<pid>/io.
  - On some workloads, read progress can reach 100% before final file flush completes.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

quantizer_args=("$@")

resolve_quantizer_bin() {
  local primary="$PKG_PATH/.build/$BUILD_CONFIG/model-quantizer"
  local legacy="$PKG_PATH/.build/$BUILD_CONFIG/ModelQuantizer"

  if [[ -x "$primary" ]]; then
    echo "$primary"
    return 0
  fi
  if [[ -x "$legacy" ]]; then
    echo "$legacy"
    return 0
  fi

  if [[ "$AUTOBUILD" != "1" ]]; then
    echo "error: model-quantizer not found and auto-build disabled" >&2
    return 1
  fi

  echo "==> Building model-quantizer ($BUILD_CONFIG)..." >&2
  swift build --package-path "$PKG_PATH" -c "$BUILD_CONFIG" --product model-quantizer

  if [[ -x "$primary" ]]; then
    echo "$primary"
    return 0
  fi
  if [[ -x "$legacy" ]]; then
    echo "$legacy"
    return 0
  fi

  echo "error: model-quantizer still not found after build" >&2
  return 1
}

detect_input_file() {
  local args=("$@")
  local input_file=""
  local i
  for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
      --input-file|-i)
        if ((i + 1 < ${#args[@]})); then
          input_file="${args[$((i + 1))]}"
        fi
        ;;
      --input-file=*)
        input_file="${args[$i]#--input-file=}"
        ;;
      -i=*)
        input_file="${args[$i]#-i=}"
        ;;
    esac
  done
  echo "$input_file"
}

detect_output_file() {
  local args=("$@")
  local output_file=""
  local i
  for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
      --output-file|-o)
        if ((i + 1 < ${#args[@]})); then
          output_file="${args[$((i + 1))]}"
        fi
        ;;
      --output-file=*)
        output_file="${args[$i]#--output-file=}"
        ;;
      -o=*)
        output_file="${args[$i]#-o=}"
        ;;
    esac
  done
  echo "$output_file"
}

has_help_arg() {
  local args=("$@")
  local i
  for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
      --help|-h)
        return 0
        ;;
    esac
  done
  return 1
}

fmt_bytes() {
  local n="${1:-0}"
  if command -v numfmt >/dev/null 2>&1; then
    numfmt --to=iec --suffix=B "$n"
  else
    echo "${n}B"
  fi
}

fmt_duration() {
  local total="${1:-0}"
  if ((total < 0)); then
    total=0
  fi
  local h=$((total / 3600))
  local m=$(((total % 3600) / 60))
  local s=$((total % 60))
  printf '%02d:%02d:%02d' "$h" "$m" "$s"
}

progress_bar() {
  local pct="$1"
  local width="${2:-40}"
  local filled=$((pct * width / 100))
  local empty=$((width - filled))

  printf '%*s' "$filled" '' | tr ' ' '#'
  printf '%*s' "$empty" '' | tr ' ' '-'
}

monitor_quantizer() {
  local pid="$1"
  local input_file="$2"
  local output_file="$3"

  local input_size=0
  if [[ -n "$input_file" ]]; then
    if [[ "$input_file" != /* ]]; then
      input_file="$PWD/$input_file"
    fi
    if [[ -f "$input_file" ]]; then
      input_size="$(stat -c %s "$input_file")"
    fi
  fi

  if [[ -n "$output_file" ]] && [[ "$output_file" != /* ]]; then
    output_file="$PWD/$output_file"
  fi

  local start_ts
  start_ts="$(date +%s)"
  local prev_ts="$start_ts"
  local prev_read=0

  while kill -0 "$pid" >/dev/null 2>&1; do
    local now elapsed read_bytes delta_t delta_read rate avg_rate output_size
    now="$(date +%s)"
    elapsed=$((now - start_ts))
    read_bytes="$(awk '/read_bytes/ {print $2}' "/proc/$pid/io" 2>/dev/null || echo 0)"

    delta_t=$((now - prev_ts))
    delta_read=$((read_bytes - prev_read))
    if ((delta_read < 0)); then
      delta_read=0
    fi

    rate=0
    if ((delta_t > 0)); then
      rate=$((delta_read / delta_t))
    fi
    avg_rate=0
    if ((elapsed > 0)); then
      avg_rate=$((read_bytes / elapsed))
    fi
    if ((rate <= 0)); then
      rate="$avg_rate"
    fi

    output_size=0
    if [[ -n "$output_file" ]] && [[ -f "$output_file" ]]; then
      output_size="$(stat -c %s "$output_file" 2>/dev/null || echo 0)"
    fi

    local pct=-1 bar eta
    bar="----------------------------------------"
    eta="n/a"
    if ((input_size > 0)); then
      pct=$((read_bytes * 100 / input_size))
      if ((pct > 100)); then
        pct=100
      fi
      bar="$(progress_bar "$pct" 40)"
      if ((rate > 0 && read_bytes < input_size)); then
        eta="$(fmt_duration $(((input_size - read_bytes) / rate)))"
      fi
    fi

    if ((pct >= 0)); then
      printf '\rquant progress %3d%% [%s] in=%s out=%s rate=%s/s eta=%s elapsed=%s' \
        "$pct" \
        "$bar" \
        "$(fmt_bytes "$read_bytes")" \
        "$(fmt_bytes "$output_size")" \
        "$(fmt_bytes "$rate")" \
        "$eta" \
        "$(fmt_duration "$elapsed")" >&2
    else
      printf '\rquant progress n/a [%s] in=%s out=%s rate=%s/s elapsed=%s' \
        "$bar" \
        "$(fmt_bytes "$read_bytes")" \
        "$(fmt_bytes "$output_size")" \
        "$(fmt_bytes "$rate")" \
        "$(fmt_duration "$elapsed")" >&2
    fi

    prev_ts="$now"
    prev_read="$read_bytes"
    sleep "$MONITOR_INTERVAL"
  done

  printf '\n' >&2
}

if ! [[ "$MONITOR_INTERVAL" =~ ^[0-9]+$ ]] || ((MONITOR_INTERVAL < 1)); then
  MONITOR_INTERVAL=5
fi

if [[ ${#quantizer_args[@]} -eq 1 ]] && has_help_arg "${quantizer_args[@]}"; then
  usage
  echo
  quantizer_bin="$(resolve_quantizer_bin)"
  "$quantizer_bin" --help
  exit 0
fi

quantizer_bin="$(resolve_quantizer_bin)"
input_file="$(detect_input_file "${quantizer_args[@]}")"
output_file="$(detect_output_file "${quantizer_args[@]}")"

echo "==> Using quantizer: $quantizer_bin" >&2

"$quantizer_bin" "${quantizer_args[@]}" &
quantizer_pid="$!"
monitor_pid=""

if [[ "$MONITOR_ENABLED" == "1" ]] && ! has_help_arg "${quantizer_args[@]}"; then
  monitor_quantizer "$quantizer_pid" "$input_file" "$output_file" &
  monitor_pid="$!"
fi

set +e
wait "$quantizer_pid"
exit_code="$?"
set -e

if [[ -n "$monitor_pid" ]]; then
  kill "$monitor_pid" >/dev/null 2>&1 || true
  wait "$monitor_pid" >/dev/null 2>&1 || true
fi

exit "$exit_code"