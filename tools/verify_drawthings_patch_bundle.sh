#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="${DRAWTHINGS_WORKSPACE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [workspace_root] [--json-report <path>]

Runs strict patch verification gate:
  tools/apply_drawthings_quant_patch.sh --verify

Options:
  --json-report <path>   Forwarded to apply script.
  -h, --help             Show this help.
EOF
}

ROOT="$DEFAULT_ROOT"
JSON_REPORT_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --json-report)
      shift
      if [[ $# -eq 0 ]]; then
        echo "error: --json-report requires a path argument" >&2
        exit 1
      fi
      JSON_REPORT_ARGS=("--json-report" "$1")
      ;;
    --json-report=*)
      JSON_REPORT_ARGS=("$1")
      ;;
    *)
      if [[ "$ROOT" != "$DEFAULT_ROOT" ]]; then
        echo "error: multiple workspace roots provided: '$ROOT' and '$1'" >&2
        exit 1
      fi
      ROOT="$1"
      ;;
  esac
  shift
done

exec "$SCRIPT_DIR/apply_drawthings_quant_patch.sh" "$ROOT" --verify "${JSON_REPORT_ARGS[@]}"
