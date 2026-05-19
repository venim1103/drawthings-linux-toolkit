#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper for DrawThingsCLI model operations so we have stable, repeatable commands.
#
# Examples:
#   tools/dt_models_cli.sh list
#   tools/dt_models_cli.sh list --downloaded-only
#   tools/dt_models_cli.sh ensure ltx_2.3_22b_distilled_1.1_q6p.ckpt
#   tools/dt_models_cli.sh ensure ltx_2.3_22b_distilled_1.1_q6p.ckpt --offline

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${DRAWTHINGS_WORKSPACE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PKG_PATH="$ROOT/draw-things-community"
MODEL_REQUIREMENTS_SCRIPT="$SCRIPT_DIR/dt_model_requirements.py"
if [[ -n "${DRAWTHINGS_MODEL_DIR:-}" ]] && [[ -d "${DRAWTHINGS_MODEL_DIR}" ]]; then
  MODELS_DIR_DEFAULT="$DRAWTHINGS_MODEL_DIR"
else
  MODELS_DIR_DEFAULT="$ROOT/dt-models"
fi
BUILD_CONFIG="${DRAWTHINGS_BUILD_CONFIG:-release}"
AUTOBUILD_SWIFT_CLI="${DRAWTHINGS_MODELS_AUTOBUILD_SWIFT_CLI:-0}"
FORCE_SWIFT_CLI="${DRAWTHINGS_MODELS_FORCE_SWIFT_CLI:-0}"

resolve_python_bin() {
  if [[ -n "${DT_PYTHON:-}" ]]; then
    echo "$DT_PYTHON"
    return 0
  fi
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    echo "$ROOT/.venv/bin/python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  echo "error: python3 not found; cannot use fallback model tooling" >&2
  return 1
}

resolve_cli_bin() {
  local primary="$PKG_PATH/.build/$BUILD_CONFIG/draw-things-cli"
  local legacy="$PKG_PATH/.build/$BUILD_CONFIG/DrawThingsCLI"

  if [[ -x "$primary" ]]; then
    echo "$primary"
    return 0
  fi
  if [[ -x "$legacy" ]]; then
    echo "$legacy"
    return 0
  fi

  if [[ "$AUTOBUILD_SWIFT_CLI" != "1" ]]; then
    return 1
  fi

  echo "==> Building Draw Things CLI ($BUILD_CONFIG) once for reuse..."
  if ! swift build --package-path "$PKG_PATH" -c "$BUILD_CONFIG" --product draw-things-cli; then
    return 1
  fi

  if [[ -x "$primary" ]]; then
    echo "$primary"
    return 0
  fi
  if [[ -x "$legacy" ]]; then
    echo "$legacy"
    return 0
  fi

  return 1
}

run_list_fallback() {
  local -a extra=("$@")
  local -a fallback_args=(--list --models-dir "$models_dir")

  while [[ ${#extra[@]} -gt 0 ]]; do
    case "${extra[0]}" in
      --downloaded-only)
        fallback_args+=(--downloaded-only)
        extra=("${extra[@]:1}")
        ;;
      --offline)
        # Listing is local-only in fallback mode.
        extra=("${extra[@]:1}")
        ;;
      --filter)
        if [[ ${#extra[@]} -lt 2 ]]; then
          echo "error: --filter needs a value" >&2
          return 1
        fi
        fallback_args+=(--filter "${extra[1]}")
        extra=("${extra[@]:2}")
        ;;
      *)
        echo "error: unsupported list option in fallback mode: ${extra[0]}" >&2
        return 1
        ;;
    esac
  done

  local python_bin
  python_bin="$(resolve_python_bin)" || return 1
  "$python_bin" "$MODEL_REQUIREMENTS_SCRIPT" "${fallback_args[@]}"
}

run_ensure_fallback() {
  local model_ref="$1"
  shift
  local -a extra=("$@")
  local -a fallback_args=(--model "$model_ref" --models-dir "$models_dir")
  local offline=0

  while [[ ${#extra[@]} -gt 0 ]]; do
    case "${extra[0]}" in
      --offline)
        offline=1
        extra=("${extra[@]:1}")
        ;;
      --no-include-dependencies)
        fallback_args+=(--no-include-dependencies)
        extra=("${extra[@]:1}")
        ;;
      --include-upscalers|--no-include-upscalers)
        fallback_args+=("${extra[0]}")
        extra=("${extra[@]:1}")
        ;;
      *)
        echo "error: unsupported ensure option in fallback mode: ${extra[0]}" >&2
        return 1
        ;;
    esac
  done

  if [[ "$offline" != "1" ]]; then
    fallback_args+=(--download-missing)
  fi

  local python_bin
  python_bin="$(resolve_python_bin)" || return 1
  "$python_bin" "$MODEL_REQUIREMENTS_SCRIPT" "${fallback_args[@]}"
}

usage() {
  cat <<'EOF'
Usage:
  dt_models_cli.sh list [--downloaded-only] [--offline] [--models-dir PATH]
  dt_models_cli.sh ensure MODEL_REF [--offline] [--no-include-dependencies] [--models-dir PATH]

Notes:
- Uses prebuilt draw-things-cli under draw-things-community/.build when available.
- Falls back to tools/dt_model_requirements.py when draw-things-cli is unavailable.
- Set DRAWTHINGS_MODELS_AUTOBUILD_SWIFT_CLI=1 to try building draw-things-cli.
- Set DRAWTHINGS_MODELS_FORCE_SWIFT_CLI=1 to require draw-things-cli and disable fallback.
- MODEL_REF can be exact model file name or a resolvable display name.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

cmd="$1"
shift

models_dir="$MODELS_DIR_DEFAULT"

collect_common_args() {
  local -n _out=$1
  shift
  _out=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --models-dir)
        if [[ $# -lt 2 ]]; then
          echo "error: --models-dir needs a value" >&2
          exit 1
        fi
        models_dir="$2"
        shift 2
        ;;
      *)
        _out+=("$1")
        shift
        ;;
    esac
  done
}

case "$cmd" in
  list)
    extra=()
    collect_common_args extra "$@"
    cli_bin=""
    if cli_bin="$(resolve_cli_bin)"; then
      "$cli_bin" models list --models-dir "$models_dir" "${extra[@]}"
    elif [[ "$FORCE_SWIFT_CLI" == "1" ]]; then
      echo "error: draw-things-cli unavailable and DRAWTHINGS_MODELS_FORCE_SWIFT_CLI=1" >&2
      echo "hint: set DRAWTHINGS_MODELS_AUTOBUILD_SWIFT_CLI=1 to attempt a build" >&2
      exit 1
    else
      echo "warning: draw-things-cli unavailable; using Python fallback for list" >&2
      run_list_fallback "${extra[@]}"
    fi
    ;;
  ensure)
    if [[ $# -lt 1 ]]; then
      echo "error: ensure requires MODEL_REF" >&2
      usage
      exit 1
    fi
    model_ref="$1"
    shift
    extra=()
    collect_common_args extra "$@"
    cli_bin=""
    if cli_bin="$(resolve_cli_bin)"; then
      "$cli_bin" models ensure --model "$model_ref" --models-dir "$models_dir" "${extra[@]}"
    elif [[ "$FORCE_SWIFT_CLI" == "1" ]]; then
      echo "error: draw-things-cli unavailable and DRAWTHINGS_MODELS_FORCE_SWIFT_CLI=1" >&2
      echo "hint: set DRAWTHINGS_MODELS_AUTOBUILD_SWIFT_CLI=1 to attempt a build" >&2
      exit 1
    else
      echo "warning: draw-things-cli unavailable; using Python fallback for ensure" >&2
      run_ensure_fallback "$model_ref" "${extra[@]}"
    fi
    ;;
  *)
    echo "error: unknown command: $cmd" >&2
    usage
    exit 1
    ;;
esac
