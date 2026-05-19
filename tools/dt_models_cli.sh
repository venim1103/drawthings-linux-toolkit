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
if [[ -n "${DRAWTHINGS_MODEL_DIR:-}" ]] && [[ -d "${DRAWTHINGS_MODEL_DIR}" ]]; then
  MODELS_DIR_DEFAULT="$DRAWTHINGS_MODEL_DIR"
else
  MODELS_DIR_DEFAULT="$ROOT/dt-models"
fi
BUILD_CONFIG="${DRAWTHINGS_BUILD_CONFIG:-release}"

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

  echo "==> Building Draw Things CLI ($BUILD_CONFIG) once for reuse..."
  if ! swift build --package-path "$PKG_PATH" -c "$BUILD_CONFIG" --product draw-things-cli; then
    echo "error: failed to build draw-things-cli; see swift build errors above" >&2
    exit 1
  fi

  if [[ -x "$primary" ]]; then
    echo "$primary"
    return 0
  fi
  if [[ -x "$legacy" ]]; then
    echo "$legacy"
    return 0
  fi

  echo "error: Draw Things CLI binary not found after build" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  dt_models_cli.sh list [--downloaded-only] [--offline] [--models-dir PATH]
  dt_models_cli.sh ensure MODEL_REF [--offline] [--no-include-dependencies] [--models-dir PATH]

Notes:
- Uses a prebuilt Draw Things CLI binary under draw-things-community/.build.
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
    cli_bin="$(resolve_cli_bin)"
    "$cli_bin" models list --models-dir "$models_dir" "${extra[@]}"
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
    cli_bin="$(resolve_cli_bin)"
    "$cli_bin" models ensure --model "$model_ref" --models-dir "$models_dir" "${extra[@]}"
    ;;
  *)
    echo "error: unknown command: $cmd" >&2
    usage
    exit 1
    ;;
esac
