#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REVISION="0621d8198f3ed505bb7cefc71b84f8f0e1ce5936"
CANDIDATE="$(realpath -m "${DRAWTHINGS_CANDIDATE_DIR:-$ROOT/output/upgrades/community-${REVISION:0:8}}")"
BAZEL_ROOT="$(realpath -m "${DRAWTHINGS_BAZEL_ROOT:-$ROOT/.cache/bazel-upgrade}")"
CCV_OVERRIDE="${DRAWTHINGS_CCV_OVERRIDE:-}"
ARTIFACT_TAG="${DRAWTHINGS_CANDIDATE_TAG:-${REVISION:0:8}}"
MODE="${1:-build}"

usage() {
  cat <<EOF
Usage: bash tools/drawthings_build_candidate.sh [prepare|build|--check|--help]

prepare  Fetch the pinned detached checkout and configure upstream Bazel files.
build    Prepare and build the CUDA server without installing it (default).
--check  Verify prerequisites and the existing checkout without modifying it.

Install build prerequisites separately:
  sudo bash .devcontainer/install-build-tools.sh

Defaults: CUDA 12.8, Swift 6.0.3, SM86, 16 jobs, 8192 MiB build memory.
Overrides: DRAWTHINGS_CANDIDATE_DIR, DRAWTHINGS_BAZEL_ROOT,
           DRAWTHINGS_CCV_OVERRIDE, DRAWTHINGS_CANDIDATE_TAG,
           DT_BUILD_JOBS, DT_BUILD_MEMORY_MB, DT_CUDA_COMPUTE_CAPABILITIES.
EOF
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

[[ $# -le 1 ]] || fail "expected at most one command"
case "$MODE" in
  --help|-h) usage; exit 0 ;;
  prepare|build|--check) ;;
  *) usage >&2; fail "unknown command: $MODE" ;;
esac

[[ "$ARTIFACT_TAG" =~ ^[A-Za-z0-9_-]+$ ]] || fail "candidate artifact tag contains invalid characters"
CCV_ARGS=()
if [[ -n "$CCV_OVERRIDE" ]]; then
  CCV_OVERRIDE="$(realpath "$CCV_OVERRIDE")"
  [[ "$CCV_OVERRIDE" == "$ROOT/output/upgrades/"* ]] || fail "CUDA override must be a private copy under output/upgrades"
  [[ "$CANDIDATE" != "$ROOT/output/upgrades/community-${REVISION:0:8}" ]] || fail "patched builds require a separate candidate directory"
  [[ "$BAZEL_ROOT" != "$ROOT/.cache/bazel-upgrade" ]] || fail "patched builds require a separate Bazel output root"
  [[ "$ARTIFACT_TAG" != "${REVISION:0:8}" ]] || fail "patched builds require a distinct candidate artifact tag"
  "$ROOT/.venv/bin/python" - "$ROOT" "$CCV_OVERRIDE" <<'PY'
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from tools.drawthings_cuda_patch import BUNDLE, patch_state
manifest = json.loads((BUNDLE / "manifest.json").read_text())
if patch_state(Path(sys.argv[2]), manifest) != "applied":
    raise SystemExit("error: CCV override does not match the applied pinned CUDA profile")
PY
  CCV_ARGS=("--override_repository=ccv=$CCV_OVERRIDE")
fi

bash "$ROOT/.devcontainer/install-build-tools.sh" --check
SWIFT_VERSION="$(swift --version)"
[[ "$SWIFT_VERSION" == "Swift version 6.0.3 "* ]] || fail "Swift 6.0.3 is required"
LLVM_BINDIR="$(/usr/bin/llvm-config --bindir)"
[[ -x "$LLVM_BINDIR/clang" ]] || fail "canonical LLVM clang executable is missing"

if [[ ! -e "$CANDIDATE/.git" ]]; then
  [[ "$MODE" != --check ]] || fail "candidate checkout is missing: $CANDIDATE"
  [[ ! -e "$CANDIDATE" ]] || fail "refusing to initialize an existing directory: $CANDIDATE"
  mkdir -p "$CANDIDATE"
  git init --quiet "$CANDIDATE"
  git -C "$CANDIDATE" remote add origin https://github.com/drawthingsai/draw-things-community.git
  git -C "$CANDIDATE" fetch --depth=1 origin "$REVISION"
  git -C "$CANDIDATE" checkout --detach FETCH_HEAD
fi

[[ "$(git -C "$CANDIDATE" rev-parse HEAD)" == "$REVISION" ]] || fail "candidate HEAD does not match the pinned revision"
[[ -z "$(git -C "$CANDIDATE" status --porcelain=v1 --untracked-files=all)" ]] || fail "candidate has local source changes; refusing to proceed"

if [[ "$MODE" == --check ]]; then
  printf 'Candidate prerequisites and pinned source passed: %s\n' "$CANDIDATE"
  exit 0
fi

if [[ -n "$CCV_OVERRIDE" ]]; then
  "$ROOT/.venv/bin/python" "$ROOT/tools/drawthings_cuda_patch.py" prepare --ccv-source "$CCV_OVERRIDE"
fi

pushd "$CANDIDATE" >/dev/null
if [[ ! -e WORKSPACE ]]; then
  bash Scripts/install.sh /usr
fi
[[ -L WORKSPACE && "$(realpath WORKSPACE)" == "$CANDIDATE/WORKSPACE.linux" ]] || fail "candidate WORKSPACE is not the expected Linux symlink"
[[ -f .bazelrc && -f clang.bazelrc ]] || fail "candidate Bazel setup is incomplete"
if [[ "$MODE" == prepare ]]; then
  printf 'Prepared detached CUDA candidate: %s\n' "$CANDIDATE"
  popd >/dev/null
  exit 0
fi

LOG_DIRECTORY="$ROOT/output/upgrades"
mkdir -p "$LOG_DIRECTORY"
LOG_FILE="$LOG_DIRECTORY/build-${ARTIFACT_TAG}.log"
bazel --output_user_root="$BAZEL_ROOT" \
  --host_jvm_args=-Djavax.net.ssl.trustStore=/etc/ssl/certs/java/cacerts \
  build //Apps:gRPCServerCLI \
  "${CCV_ARGS[@]}" \
  --spawn_strategy=local --compilation_mode=opt --config=clang --config=cuda \
  --per_file_copt='.*[.]cu@-nvcc_options=compiler-options=-frelaxed-template-template-args' \
  --action_env=TF_NEED_CUDA=1 --action_env=TF_NEED_OPENCL=0 \
  --action_env=TF_CUDA_CLANG=0 \
  --action_env=HOST_CXX_COMPILER=/usr/bin/clang++ \
  --action_env=HOST_C_COMPILER=/usr/bin/clang \
  --action_env=CLANG_CUDA_COMPILER_PATH=/usr/bin/clang \
  --action_env="GCC_HOST_COMPILER_PATH=$LLVM_BINDIR/clang" \
  --action_env=CUDA_TOOLKIT_PATH=/usr/local/cuda-12.8 \
  --action_env=TF_CUDA_VERSION=12.8 \
  --action_env="TF_CUDA_COMPUTE_CAPABILITIES=${DT_CUDA_COMPUTE_CAPABILITIES:-8.6}" \
  --action_env=TF_CUDNN_VERSION=9 --action_env=CUDNN_INSTALL_PATH=/usr \
  --action_env=TF_NCCL_VERSION=2 --action_env=NCCL_INSTALL_PATH=/usr \
  --action_env=TMP=/tmp \
  --linkopt=-z --linkopt=nostart-stop-gc \
  --host_linkopt=-z --host_linkopt=nostart-stop-gc \
  --define=enable_sm80=true \
  --jobs="${DT_BUILD_JOBS:-16}" \
  --local_resources="memory=${DT_BUILD_MEMORY_MB:-8192}" \
  2>&1 | tee "$LOG_FILE"

BINARY="$CANDIDATE/bazel-bin/Apps/gRPCServerCLI"
[[ -x "$BINARY" ]] || fail "build succeeded without the expected executable: $BINARY"
sha256sum "$BINARY" | tee "$LOG_DIRECTORY/gRPCServerCLI-${ARTIFACT_TAG}.sha256"
printf 'Candidate built, not installed: %s\n' "$BINARY"
popd >/dev/null