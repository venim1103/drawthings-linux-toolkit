#!/usr/bin/env bash
set -euo pipefail

BAZEL_VERSION=7.4.1
BAZEL_SHA256=c97f02133adce63f0c28678ac1f21d65fa8255c80429b588aeeba8a1fac6202b
CUDA_ROOT=/usr/local/cuda-12.8

if [[ "${1:-}" == "--help" ]]; then
    printf 'Usage: sudo bash .devcontainer/install-build-tools.sh\n'
    printf '       bash .devcontainer/install-build-tools.sh --check\n'
    printf 'Install CUDA 12.8 development tools and Bazel 7.4.1 without changing CUDA runtime versions.\n'
    exit 0
fi

check_tools() {
    [[ "$(bazel --version)" == "bazel $BAZEL_VERSION" ]]
    "$CUDA_ROOT/bin/nvcc" --version
    /usr/bin/llvm-config --version
    command -v ruby >/dev/null
    [[ -r /etc/ssl/certs/java/cacerts ]]
    [[ -f "$(/usr/bin/clang -print-resource-dir)/include/omp.h" ]]
    for header in cuda_runtime.h cuda_profiler_api.h cublas_api.h cusolver_common.h curand.h cufft.h cusparse.h; do
        [[ -f "$CUDA_ROOT/include/$header" ]]
    done
    [[ -f "$CUDA_ROOT/include/cupti.h" || -f "$CUDA_ROOT/extras/CUPTI/include/cupti.h" ]]
    [[ -f "$CUDA_ROOT/lib64/libcufile.so" ]]
    [[ -f /usr/include/cudnn.h || -f /usr/include/x86_64-linux-gnu/cudnn.h ]]
    printf 'Native CUDA builder prerequisites passed.\n'
}

if [[ "${1:-}" == "--check" && $# == 1 ]]; then
    check_tools
    exit 0
fi
if [[ $# != 0 || "$EUID" != 0 ]]; then
    printf 'Error: installation requires root; use sudo or run --check.\n' >&2
    exit 1
fi
if [[ "$(uname -m)" != x86_64 ]]; then
    printf 'Error: this pinned builder supports Linux x86_64 only.\n' >&2
    exit 1
fi

runtime_packages=(
    cuda-cudart-12-8 libcublas-12-8 libcurand-12-8 libcudnn9-cuda-12
    libcusolver-12-8 libcufft-12-8 libcusparse-12-8
)
runtime_versions=()
for package in "${runtime_packages[@]}"; do
    runtime_versions+=("$(dpkg-query -W -f='${Version}' "$package")")
done

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-remove --no-install-recommends \
    ca-certificates-java clang llvm llvm-dev libomp-dev ruby \
    libpng-dev libjpeg-dev libatlas-base-dev libblas-dev \
    cuda-nvcc-12-8 cuda-cupti-dev-12-8 cuda-profiler-api-12-8 libcufile-dev-12-8 \
    "cuda-cudart-dev-12-8=${runtime_versions[0]}" \
    "libcublas-dev-12-8=${runtime_versions[1]}" \
    "libcurand-dev-12-8=${runtime_versions[2]}" \
    "libcudnn9-dev-cuda-12=${runtime_versions[3]}" \
    "libcusolver-dev-12-8=${runtime_versions[4]}" \
    "libcufft-dev-12-8=${runtime_versions[5]}" \
    "libcusparse-dev-12-8=${runtime_versions[6]}"

for index in "${!runtime_packages[@]}"; do
    [[ "$(dpkg-query -W -f='${Version}' "${runtime_packages[$index]}")" == "${runtime_versions[$index]}" ]] || {
        printf 'Error: runtime version changed: %s\n' "${runtime_packages[$index]}" >&2
        exit 1
    }
done

if ! command -v bazel >/dev/null 2>&1; then
    temporary="$(mktemp -d)"
    trap 'rm -f "$temporary/bazel"; rmdir "$temporary"' EXIT
    curl --fail --show-error --location --proto '=https' \
        "https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-linux-x86_64" \
        -o "$temporary/bazel"
    printf '%s  %s\n' "$BAZEL_SHA256" "$temporary/bazel" | sha256sum --check
    install -m 0755 "$temporary/bazel" /usr/local/bin/bazel
fi

check_tools