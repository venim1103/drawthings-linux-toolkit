#!/usr/bin/env bash
set -euo pipefail

# Optional helper: install grpcurl in a user-local location.
#
# Usage:
#   tools/install_grpcurl.sh
#   tools/install_grpcurl.sh 1.9.2
#
# Installs to ~/.local/bin by default. Override with INSTALL_DIR.

VERSION_RAW="${1:-1.9.2}"
VERSION="${VERSION_RAW#v}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"

arch="$(uname -m)"
case "$arch" in
  x86_64|amd64)
    asset_arch="x86_64"
    # Known checksum for grpcurl_1.9.2_linux_x86_64.tar.gz
    expected_sha256="1c7caf2628d8607d8a3bbee5ce7786bba4879abe566b075a4f129a97ccfa8465"
    ;;
  aarch64|arm64)
    asset_arch="arm64"
    expected_sha256=""
    ;;
  *)
    echo "error: unsupported architecture: $arch" >&2
    exit 1
    ;;
esac

asset="grpcurl_${VERSION}_linux_${asset_arch}.tar.gz"
url="https://github.com/fullstorydev/grpcurl/releases/download/v${VERSION}/${asset}"

tmpdir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmpdir"
}
trap cleanup EXIT

echo "==> Downloading grpcurl v${VERSION} (${asset_arch})"
curl -fsSL "$url" -o "$tmpdir/$asset"

if [[ -n "$expected_sha256" ]]; then
  actual_sha256="$(sha256sum "$tmpdir/$asset" | awk '{print $1}')"
  if [[ "$actual_sha256" != "$expected_sha256" ]]; then
    echo "error: checksum mismatch for $asset" >&2
    echo "expected: $expected_sha256" >&2
    echo "actual:   $actual_sha256" >&2
    exit 1
  fi
else
  echo "WARNING: no pinned checksum for architecture $asset_arch; skipping checksum verification."
fi

echo "==> Extracting"
tar -xzf "$tmpdir/$asset" -C "$tmpdir"

if [[ ! -f "$tmpdir/grpcurl" ]]; then
  echo "error: grpcurl binary not found in archive" >&2
  exit 1
fi

echo "==> Installing to $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
install -m 0755 "$tmpdir/grpcurl" "$INSTALL_DIR/grpcurl"

echo "==> Installed: $INSTALL_DIR/grpcurl"
"$INSTALL_DIR/grpcurl" -version || true

if ! printf '%s\n' "$PATH" | tr ':' '\n' | grep -Fx "$INSTALL_DIR" >/dev/null 2>&1; then
  echo
  echo "Add this to your shell profile if needed:"
  echo "  export PATH=\"$INSTALL_DIR:\$PATH\""
fi