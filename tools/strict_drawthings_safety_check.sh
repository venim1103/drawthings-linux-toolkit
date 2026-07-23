#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="${DRAWTHINGS_WORKSPACE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

ROOT="$DEFAULT_ROOT"
RUN_BUILD=1
BUILD_PRODUCT="model-quantizer"
ALLOW_DIRTY=0

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [workspace_root] [--allow-dirty] [--skip-build] [--build-product <name>]

Strict local safety gate for Draw Things patch maintenance:
  1) strict patch verify (idempotence): verify_drawthings_patch_bundle.sh
  2) manifest path integrity checks
  3) patch regeneration: generate_drawthings_quant_patches.sh
  4) strict patch verify again
  5) smoke build (default: model-quantizer)

Options:
  --allow-dirty          Skip clean-tree precheck for DRAW_THINGS_PATCH paths.
  --skip-build           Skip smoke build step.
  --build-product <name> Swift product to build (default: model-quantizer).
  -h, --help             Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --allow-dirty)
      ALLOW_DIRTY=1
      ;;
    --skip-build)
      RUN_BUILD=0
      ;;
    --build-product)
      shift
      if [[ $# -eq 0 ]]; then
        echo "error: --build-product requires a value" >&2
        exit 1
      fi
      BUILD_PRODUCT="$1"
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

REPO_ROOT="$ROOT/draw-things-community"
PATCH_ROOT="$ROOT/DRAW_THINGS_PATCH"
MANIFEST_FILE="$PATCH_ROOT/patches/manifest.sh"

VERIFY_SCRIPT="$SCRIPT_DIR/verify_drawthings_patch_bundle.sh"
GENERATE_SCRIPT="$SCRIPT_DIR/generate_drawthings_quant_patches.sh"
APPLY_SCRIPT="$SCRIPT_DIR/apply_drawthings_quant_patch.sh"
SYNC_SCRIPT="$SCRIPT_DIR/sync_drawthings_patch_bundle.sh"

if [[ ! -d "$ROOT" ]]; then
  echo "error: workspace root not found: $ROOT" >&2
  exit 1
fi
if [[ ! -d "$REPO_ROOT" ]]; then
  echo "error: draw-things-community repo not found: $REPO_ROOT" >&2
  exit 1
fi
if [[ ! -f "$MANIFEST_FILE" ]]; then
  echo "error: patch manifest not found: $MANIFEST_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$MANIFEST_FILE"

check_script_syntax() {
  echo "==> Checking shell script syntax..."
  bash -n "$APPLY_SCRIPT"
  bash -n "$VERIFY_SCRIPT"
  bash -n "$GENERATE_SCRIPT"
  bash -n "$SYNC_SCRIPT"
  bash -n "$MANIFEST_FILE"
}

check_patch_tree_clean() {
  if [[ "$ALLOW_DIRTY" == "1" ]]; then
    return 0
  fi
  echo "==> Checking DRAW_THINGS_PATCH tree cleanliness..."
  local status
  status="$(git -C "$ROOT" status --porcelain -- DRAW_THINGS_PATCH)"
  if [[ -n "$status" ]]; then
    echo "error: DRAW_THINGS_PATCH has uncommitted changes; run with --allow-dirty to override." >&2
    echo "$status" >&2
    exit 1
  fi
}

check_manifest_paths() {
  echo "==> Validating patch manifest paths..."
  local rel_path

  for rel_path in "${PATCH_ROOT_FILES[@]}"; do
    if [[ ! -f "$PATCH_ROOT/$rel_path" ]]; then
      echo "error: missing snapshot file: $PATCH_ROOT/$rel_path" >&2
      exit 1
    fi
    if [[ ! -f "$REPO_ROOT/$rel_path" ]]; then
      echo "error: missing repo file: $REPO_ROOT/$rel_path" >&2
      exit 1
    fi
  done

  for rel_path in "${PATCH_S4NNC_FILES[@]}"; do
    if [[ ! -f "$PATCH_ROOT/checkouts/s4nnc/$rel_path" ]]; then
      echo "error: missing s4nnc snapshot file: $PATCH_ROOT/checkouts/s4nnc/$rel_path" >&2
      exit 1
    fi
    if [[ ! -f "$REPO_ROOT/.build/checkouts/s4nnc/$rel_path" ]]; then
      echo "error: missing s4nnc checkout file: $REPO_ROOT/.build/checkouts/s4nnc/$rel_path" >&2
      exit 1
    fi
  done

  for rel_path in "${PATCH_CCV_FILES[@]}"; do
    if [[ ! -f "$PATCH_ROOT/checkouts/ccv/$rel_path" ]]; then
      echo "error: missing ccv snapshot file: $PATCH_ROOT/checkouts/ccv/$rel_path" >&2
      exit 1
    fi
    if [[ ! -f "$REPO_ROOT/.build/checkouts/ccv/$rel_path" ]]; then
      echo "error: missing ccv checkout file: $REPO_ROOT/.build/checkouts/ccv/$rel_path" >&2
      exit 1
    fi
  done
}

run_verify() {
  echo "==> Running strict verify gate..."
  bash "$VERIFY_SCRIPT" "$ROOT"
}

run_regenerate() {
  echo "==> Regenerating patch bundle..."
  bash "$GENERATE_SCRIPT" "$ROOT"
}

run_smoke_build() {
  if [[ "$RUN_BUILD" != "1" ]]; then
    echo "==> Skipping smoke build (--skip-build)."
    return 0
  fi
  echo "==> Running smoke build: $BUILD_PRODUCT"
  swift build --package-path "$REPO_ROOT" -c release --product "$BUILD_PRODUCT"
}

echo "==> Strict Draw Things safety check"
check_script_syntax
check_patch_tree_clean
check_manifest_paths
run_verify
run_regenerate
check_manifest_paths
run_verify
run_smoke_build
echo "==> Safety check passed."
