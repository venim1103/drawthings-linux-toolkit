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
VALIDATE_ENABLED="${DRAWTHINGS_CONVERTER_VALIDATE:-1}"
VALIDATE_PROFILE="${DRAWTHINGS_CONVERTER_VALIDATE_PROFILE:-auto}"
VALIDATE_SERIALIZATION_GUARD="${DRAWTHINGS_CONVERTER_VALIDATE_SERIALIZATION_GUARD:-auto}"
VALIDATE_SERIALIZATION_BASELINE="${DRAWTHINGS_CONVERTER_VALIDATE_SERIALIZATION_BASELINE:-$ROOT/dt-models/ltx_2.3_22b_distilled_f16.ckpt}"
VALIDATE_SERIALIZATION_PROFILE="${DRAWTHINGS_CONVERTER_VALIDATE_SERIALIZATION_PROFILE:-ltx2_3}"
VALIDATE_SERIALIZATION_REPORT_LIMIT="${DRAWTHINGS_CONVERTER_VALIDATE_SERIALIZATION_REPORT_LIMIT:-12}"
ALIGN_METADATA_MODE="${DRAWTHINGS_CONVERTER_ALIGN_METADATA_MODE:-0}"
ALIGN_METADATA_BASELINE="${DRAWTHINGS_CONVERTER_ALIGN_METADATA_BASELINE:-$VALIDATE_SERIALIZATION_BASELINE}"
ALIGN_METADATA_ALLOW_KEYSET_MISMATCH="${DRAWTHINGS_CONVERTER_ALIGN_METADATA_ALLOW_KEYSET_MISMATCH:-0}"
ALIGN_METADATA_SAMPLE_LIMIT="${DRAWTHINGS_CONVERTER_ALIGN_METADATA_SAMPLE_LIMIT:-12}"
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
  DRAWTHINGS_CONVERTER_VALIDATE      Set to 0 to skip post-conversion checkpoint
                                     integrity validation.
  DRAWTHINGS_CONVERTER_VALIDATE_PROFILE
                                     Validation profile: auto (default), none,
                                     or ltx2_3.
  DRAWTHINGS_CONVERTER_VALIDATE_SERIALIZATION_GUARD
                                     Serialization parity guard mode:
                                     auto (default), 1, or 0.
                                     - auto: run parity check only when baseline exists;
                                       report mismatches as warnings without failing.
                                     - 1: always enforce (fails if baseline missing)
                                     - 0: disable serialization parity checks
  DRAWTHINGS_CONVERTER_VALIDATE_SERIALIZATION_BASELINE
                                     Baseline .ckpt used for strict type/format/
                                     datatype parity checks. Relative paths are
                                     resolved from workspace root.
                                     Default: dt-models/ltx_2.3_22b_distilled_f16.ckpt
  DRAWTHINGS_CONVERTER_VALIDATE_SERIALIZATION_PROFILE
                                     Profile scope for parity guard: any, none,
                                     or ltx2_3. Default: ltx2_3.
  DRAWTHINGS_CONVERTER_VALIDATE_SERIALIZATION_REPORT_LIMIT
                                     Max mismatch examples printed per category.
                                     Default: 12.
  DRAWTHINGS_CONVERTER_ALIGN_METADATA_MODE
                                     Metadata alignment mode for converted ckpt:
                                     0 (default), auto, or 1.
                                     - 0: disable metadata alignment step
                                     - auto: apply when baseline exists
                                     - 1: require and apply (fails if baseline missing)
  DRAWTHINGS_CONVERTER_ALIGN_METADATA_BASELINE
                                     Baseline .ckpt used for metadata alignment.
                                     Default: DRAWTHINGS_CONVERTER_VALIDATE_SERIALIZATION_BASELINE
  DRAWTHINGS_CONVERTER_ALIGN_METADATA_ALLOW_KEYSET_MISMATCH
                                     Set to 1 to allow alignment when keysets differ.
                                     Default: 0 (strict keyset match required).
  DRAWTHINGS_CONVERTER_ALIGN_METADATA_SAMPLE_LIMIT
                                     Sample mismatch names printed by alignment tool.
                                     Default: 12.
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

detect_output_directory() {
  local args=("$@")
  local output_dir=""
  local i
  for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
      --output-directory|-o)
        if ((i + 1 < ${#args[@]})); then
          output_dir="${args[$((i + 1))]}"
        fi
        ;;
      --output-directory=*)
        output_dir="${args[$i]#--output-directory=}"
        ;;
      -o=*)
        output_dir="${args[$i]#-o=}"
        ;;
    esac
  done
  echo "$output_dir"
}

resolve_python_bin() {
  if [[ -n "${DT_PYTHON:-}" ]] && command -v "$DT_PYTHON" >/dev/null 2>&1; then
    echo "$DT_PYTHON"
    return 0
  fi

  local venv_python="$ROOT/.venv/bin/python"
  if [[ -x "$venv_python" ]]; then
    echo "$venv_python"
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi

  echo "error: python3 not found for post-conversion validation" >&2
  return 1
}

resolve_path_from_root() {
  local value="$1"
  if [[ -z "$value" ]]; then
    echo ""
    return 0
  fi

  if [[ "$value" == /* ]]; then
    echo "$value"
  else
    echo "$ROOT/$value"
  fi
}

detect_output_ckpt() {
  local output_dir="$1"
  local conversion_start_ts="$2"
  local candidate=""

  candidate="$(find "$output_dir" -maxdepth 1 -type f -name '*_f16.ckpt' -printf '%T@ %s %p\n' 2>/dev/null \
    | awk -v start="$conversion_start_ts" '$1 >= start {print}' \
    | sort -nr -k2,2 -k1,1 \
    | head -n1 \
    | cut -d' ' -f3-)"

  echo "$candidate"
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

has_help_arg() {
  local args=("$@")
  local i
  for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
      -h|--help)
        return 0
        ;;
    esac
  done
  return 1
}

if [[ ${#converter_args[@]} -eq 1 ]] && has_help_arg "${converter_args[@]}"; then
  converter_bin="$(resolve_converter_bin)"
  "$converter_bin" --help
  exit 0
fi

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
output_dir="$(detect_output_directory "${converter_args[@]}")"

if [[ -n "$input_file" ]] && [[ "$input_file" != /* ]]; then
  input_file="$PWD/$input_file"
fi
if [[ -n "$output_dir" ]] && [[ "$output_dir" != /* ]]; then
  output_dir="$PWD/$output_dir"
fi

acquire_lock
converter_bin="$(resolve_converter_bin)"

echo "==> Using converter: $converter_bin" >&2
conversion_start_ts="$(date +%s)"
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

if [[ "$exit_code" -eq 0 && "$VALIDATE_ENABLED" == "1" ]] && ! has_help_arg "${converter_args[@]}"; then
  validator_script="$ROOT/tools/dt_validate_converted_ckpt.py"
  if [[ ! -f "$validator_script" ]]; then
    echo "error: validation requested but missing validator script: $validator_script" >&2
    exit_code=1
  elif [[ -z "$output_dir" ]]; then
    echo "error: validation requested but --output-directory was not provided" >&2
    exit_code=1
  else
    converted_ckpt="$(detect_output_ckpt "$output_dir" "$conversion_start_ts")"
    if [[ -z "$converted_ckpt" ]]; then
      echo "error: validation requested but no *_f16.ckpt output found in $output_dir" >&2
      exit_code=1
    else
      python_bin="$(resolve_python_bin)"
      validator_base_args=(
        "$python_bin"
        "$validator_script"
        --file "$converted_ckpt"
        --profile "$VALIDATE_PROFILE"
      )
      validator_guard_args=()
      run_guard_validator="0"

      align_script="$ROOT/tools/dt_align_ckpt_metadata.py"
      align_mode="${ALIGN_METADATA_MODE,,}"
      align_enabled=""
      case "$align_mode" in
        0|false|off|no)
          align_enabled="0"
          ;;
        1|true|on|yes)
          align_enabled="1"
          ;;
        auto)
          align_enabled="auto"
          ;;
        *)
          echo "error: invalid DRAWTHINGS_CONVERTER_ALIGN_METADATA_MODE=$ALIGN_METADATA_MODE (expected auto/1/0)" >&2
          exit_code=1
          ;;
      esac

      align_allow_keyset="${ALIGN_METADATA_ALLOW_KEYSET_MISMATCH,,}"
      case "$align_allow_keyset" in
        0|false|off|no)
          align_allow_keyset="0"
          ;;
        1|true|on|yes)
          align_allow_keyset="1"
          ;;
        *)
          echo "error: invalid DRAWTHINGS_CONVERTER_ALIGN_METADATA_ALLOW_KEYSET_MISMATCH=$ALIGN_METADATA_ALLOW_KEYSET_MISMATCH (expected 1/0)" >&2
          exit_code=1
          ;;
      esac

      if [[ "$exit_code" -eq 0 ]]; then
        if ! [[ "$ALIGN_METADATA_SAMPLE_LIMIT" =~ ^[0-9]+$ ]] || ((ALIGN_METADATA_SAMPLE_LIMIT < 1)); then
          echo "error: DRAWTHINGS_CONVERTER_ALIGN_METADATA_SAMPLE_LIMIT must be >= 1" >&2
          exit_code=1
        fi
      fi

      if [[ "$exit_code" -eq 0 && "$align_enabled" != "0" ]]; then
        if [[ ! -f "$align_script" ]]; then
          echo "error: metadata alignment requested but missing script: $align_script" >&2
          exit_code=1
        else
          align_baseline="$(resolve_path_from_root "$ALIGN_METADATA_BASELINE")"
          align_args=(
            "$python_bin"
            "$align_script"
            --file "$converted_ckpt"
            --baseline "$align_baseline"
            --mode apply
            --sample-limit "$ALIGN_METADATA_SAMPLE_LIMIT"
          )
          if [[ "$align_allow_keyset" == "1" ]]; then
            align_args+=(--allow-keyset-mismatch)
          fi

          if [[ "$align_enabled" == "1" ]]; then
            if [[ -z "$align_baseline" || ! -f "$align_baseline" ]]; then
              echo "error: metadata alignment enabled but baseline file is missing: $align_baseline" >&2
              exit_code=1
            fi
          fi

          if [[ "$exit_code" -eq 0 && -n "$align_baseline" && -f "$align_baseline" ]]; then
            echo "==> Aligning converted metadata to baseline: $align_baseline" >&2
            if ! "${align_args[@]}"; then
              echo "error: converted checkpoint metadata alignment failed" >&2
              exit_code=1
            fi
          elif [[ "$align_enabled" == "auto" ]]; then
            echo "==> Metadata alignment skipped (auto): baseline not found: $align_baseline" >&2
          fi
        fi
      fi

      guard_mode="${VALIDATE_SERIALIZATION_GUARD,,}"
      guard_enabled=""
      case "$guard_mode" in
        0|false|off|no)
          guard_enabled="0"
          ;;
        1|true|on|yes)
          guard_enabled="1"
          ;;
        auto)
          guard_enabled="auto"
          ;;
        *)
          echo "error: invalid DRAWTHINGS_CONVERTER_VALIDATE_SERIALIZATION_GUARD=$VALIDATE_SERIALIZATION_GUARD (expected auto/1/0)" >&2
          exit_code=1
          ;;
      esac

      if [[ "$exit_code" -eq 0 ]]; then
        if ! [[ "$VALIDATE_SERIALIZATION_REPORT_LIMIT" =~ ^[0-9]+$ ]] || ((VALIDATE_SERIALIZATION_REPORT_LIMIT < 1)); then
          echo "error: DRAWTHINGS_CONVERTER_VALIDATE_SERIALIZATION_REPORT_LIMIT must be >= 1" >&2
          exit_code=1
        fi
      fi

      if [[ "$exit_code" -eq 0 ]]; then
        case "$VALIDATE_SERIALIZATION_PROFILE" in
          any|none|ltx2_3)
            ;;
          *)
            echo "error: invalid DRAWTHINGS_CONVERTER_VALIDATE_SERIALIZATION_PROFILE=$VALIDATE_SERIALIZATION_PROFILE (expected any/none/ltx2_3)" >&2
            exit_code=1
            ;;
        esac
      fi

      if [[ "$exit_code" -eq 0 && "$guard_enabled" != "0" ]]; then
        serialization_baseline="$(resolve_path_from_root "$VALIDATE_SERIALIZATION_BASELINE")"

        if [[ "$guard_enabled" == "1" ]]; then
          if [[ -z "$serialization_baseline" || ! -f "$serialization_baseline" ]]; then
            echo "error: serialization parity guard enabled but baseline file is missing: $serialization_baseline" >&2
            exit_code=1
          fi
        fi

        if [[ "$exit_code" -eq 0 && -n "$serialization_baseline" && -f "$serialization_baseline" ]]; then
          validator_guard_args=(
            "${validator_base_args[@]}"
            --serialization-baseline "$serialization_baseline"
            --serialization-guard-profile "$VALIDATE_SERIALIZATION_PROFILE"
            --serialization-report-limit "$VALIDATE_SERIALIZATION_REPORT_LIMIT"
          )
          run_guard_validator="1"
          echo "==> Serialization parity guard baseline: $serialization_baseline (profile=$VALIDATE_SERIALIZATION_PROFILE)" >&2
        elif [[ "$guard_enabled" == "auto" ]]; then
          echo "==> Serialization parity guard skipped (auto): baseline not found: $serialization_baseline" >&2
        fi
      fi

      if [[ -n "$input_file" ]] && [[ "${input_file,,}" == *.safetensors ]]; then
        validator_base_args+=(--source-safetensors "$input_file")
      fi

      if [[ "$exit_code" -eq 0 ]]; then
        echo "==> Validating converted checkpoint structure: $converted_ckpt" >&2
        if ! "${validator_base_args[@]}"; then
          echo "error: converted checkpoint failed structural validation" >&2
          exit_code=1
        fi
      fi

      if [[ "$exit_code" -eq 0 && "$run_guard_validator" == "1" ]]; then
        echo "==> Validating serialization parity guard: $converted_ckpt" >&2
        if ! "${validator_guard_args[@]}"; then
          if [[ "$guard_enabled" == "auto" ]]; then
            echo "warning: serialization parity mismatches detected in auto mode; conversion output kept." >&2
            echo "         set DRAWTHINGS_CONVERTER_VALIDATE_SERIALIZATION_GUARD=1 to enforce failure." >&2
          else
            echo "error: converted checkpoint failed serialization parity guard" >&2
            exit_code=1
          fi
        fi
      fi
    fi
  fi
fi

exit "$exit_code"
