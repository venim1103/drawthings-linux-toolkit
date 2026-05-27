#!/usr/bin/env bash
set -euo pipefail

# Wrapper around model-converter for long-running conversions.
#
# Features:
# - Forces math backends to use the requested thread count (default: all CPU cores).
# - Auto-builds model-converter if missing.
# - Shows a live read/rate/ETA monitor while conversion runs.
#
# Example:
#   tools/dt_convert_model.sh \
#     --file dt-models/ltx-2.3-22b-distilled-1.1.safetensors \
#     --name ltx-2.3-22b-distilled-1.1 \
#     --output-directory dt-models

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${DRAWTHINGS_WORKSPACE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PKG_PATH="$ROOT/draw-things-community"
BUILD_CONFIG="${DRAWTHINGS_BUILD_CONFIG:-release}"
AUTOBUILD="${DRAWTHINGS_CONVERTER_AUTOBUILD:-1}"
MONITOR_ENABLED="${DRAWTHINGS_CONVERTER_MONITOR:-1}"
LOCK_FILE="${DRAWTHINGS_CONVERTER_LOCK_FILE:-$ROOT/.cache/dt_convert_model.lock}"

usage() {
  cat <<'EOF'
Usage:
  dt_convert_model.sh [model-converter args]

Pass-through wrapper for draw-things-community/.build/<config>/model-converter.

Environment:
  DRAWTHINGS_MATH_THREADS            Thread count to force for math backends.
                                     Default: nproc (all detected CPU cores).
  DRAWTHINGS_CONVERTER_MONITOR       Set to 0 to disable live /proc monitor output.
  DRAWTHINGS_CONVERTER_AUTOBUILD     Set to 0 to disable auto-build when missing.
  DRAWTHINGS_CONVERTER_LOCK_FILE     Lock path used to enforce one active conversion
                                     wrapper run at a time.
  DRAWTHINGS_BUILD_CONFIG            release (default) or debug.

Examples:
  tools/dt_convert_model.sh --help
  tools/dt_convert_model.sh \
    --file dt-models/model.safetensors \
    --name my-model \
    --output-directory dt-models

Notes:
  - Do not run multiple conversions to the same output at once.
  - Converter import progress can hit 100% before final output flush is done.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

converter_args=("$@")

acquire_lock() {
  if ! command -v flock >/dev/null 2>&1; then
    return 0
  fi

  mkdir -p "$(dirname "$LOCK_FILE")"
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "error: another conversion wrapper is already running (lock: $LOCK_FILE)" >&2
    echo "       wait for it to finish or set DRAWTHINGS_CONVERTER_LOCK_FILE to a different path" >&2
    return 1
  fi
  return 0
}

resolve_converter_bin() {
  local primary="$PKG_PATH/.build/$BUILD_CONFIG/model-converter"
  local legacy="$PKG_PATH/.build/$BUILD_CONFIG/ModelConverter"

  if [[ -x "$primary" ]]; then
    echo "$primary"
    return 0
  fi
  if [[ -x "$legacy" ]]; then
    echo "$legacy"
    return 0
  fi

  if [[ "$AUTOBUILD" != "1" ]]; then
    echo "error: model-converter not found and auto-build disabled" >&2
    return 1
  fi

  echo "==> Building model-converter ($BUILD_CONFIG)..." >&2
  swift build --package-path "$PKG_PATH" -c "$BUILD_CONFIG" --product model-converter

  if [[ -x "$primary" ]]; then
    echo "$primary"
    return 0
  fi
  if [[ -x "$legacy" ]]; then
    echo "$legacy"
    return 0
  fi

  echo "error: model-converter still not found after build" >&2
  return 1
}

detect_input_file() {
  local args=("$@")
  local input_file=""
  local i
  for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
      --file|-f)
        if ((i + 1 < ${#args[@]})); then
          input_file="${args[$((i + 1))]}"
        fi
        ;;
      --file=*)
        input_file="${args[$i]#--file=}"
        ;;
      -f=*)
        input_file="${args[$i]#-f=}"
        ;;
    esac
  done
  echo "$input_file"
}

has_math_threads_arg() {
  local args=("$@")
  local i
  for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
      --math-threads|--math-threads=*)
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

monitor_converter() {
  local pid="$1"
  local input_file="$2"
  local input_size=0

  if [[ -n "$input_file" ]]; then
    if [[ "$input_file" != /* ]]; then
      input_file="$PWD/$input_file"
    fi
    if [[ -f "$input_file" ]]; then
      input_size="$(stat -c %s "$input_file")"
    fi
  fi

  local start_ts
  start_ts="$(date +%s)"

  while kill -0 "$pid" >/dev/null 2>&1; do
    local now elapsed read_bytes rate progress eta
    now="$(date +%s)"
    elapsed=$((now - start_ts))
    read_bytes="$(awk '/read_bytes/ {print $2}' "/proc/$pid/io" 2>/dev/null || echo 0)"

    rate=0
    if ((elapsed > 0)); then
      rate=$((read_bytes / elapsed))
    fi

    progress="n/a"
    eta="n/a"
    if ((input_size > 0)); then
      local pct
      pct=$((read_bytes * 100 / input_size))
      if ((pct > 100)); then
        pct=100
      fi
      progress="${pct}%"
      if ((rate > 0 && read_bytes < input_size)); then
        eta="$(fmt_duration $(((input_size - read_bytes) / rate)))"
      fi
    fi

    printf '\rmonitor read=%s rate=%s/s progress=%s eta=%s elapsed=%s' \
      "$(fmt_bytes "$read_bytes")" \
      "$(fmt_bytes "$rate")" \
      "$progress" \
      "$eta" \
      "$(fmt_duration "$elapsed")" >&2

    sleep 5
  done

  printf '\n' >&2
}

thread_count="${DRAWTHINGS_MATH_THREADS:-$(nproc)}"
if [[ -z "$thread_count" ]] || ((thread_count < 1)); then
  thread_count=1
fi

# Force common math backends to use all available CPU threads.
export OPENBLAS_NUM_THREADS="$thread_count"
export OMP_NUM_THREADS="$thread_count"
export MKL_NUM_THREADS="$thread_count"
export GOTO_NUM_THREADS="$thread_count"
export BLIS_NUM_THREADS="$thread_count"
export VECLIB_MAXIMUM_THREADS="$thread_count"
export NUMEXPR_NUM_THREADS="$thread_count"

echo "==> Math threads set to $thread_count" >&2

if ! has_math_threads_arg "${converter_args[@]}"; then
  converter_args+=(--math-threads "$thread_count")
fi

input_file="$(detect_input_file "${converter_args[@]}")"
acquire_lock
converter_bin="$(resolve_converter_bin)"

echo "==> Using converter: $converter_bin" >&2
"$converter_bin" "${converter_args[@]}" &
converter_pid="$!"
monitor_pid=""

if [[ "$MONITOR_ENABLED" == "1" ]]; then
  monitor_converter "$converter_pid" "$input_file" &
  monitor_pid="$!"
fi

set +e
wait "$converter_pid"
exit_code="$?"
set -e

if [[ -n "$monitor_pid" ]]; then
  kill "$monitor_pid" >/dev/null 2>&1 || true
  wait "$monitor_pid" >/dev/null 2>&1 || true
fi

exit "$exit_code"
