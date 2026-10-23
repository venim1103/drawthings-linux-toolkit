#!/usr/bin/env bash
set -euo pipefail

CHECK_ONLY=0
if [[ "${1:-}" == "--help" ]]; then
    printf 'Usage: bash .devcontainer/install-host-certificates.sh [--check] <host-ca-bundle>\n'
    printf 'Validates exported host CAs; installation requires root.\n'
    exit 0
fi
if [[ "${1:-}" == "--check" ]]; then
    CHECK_ONLY=1
    shift
fi
if [[ $# -ne 1 || ! -r "$1" ]]; then
    printf 'Error: provide a readable PEM host CA bundle.\n' >&2
    exit 1
fi

BUNDLE="$1"
if grep -q -- 'PRIVATE KEY' "$BUNDLE"; then
    printf 'Error: the CA bundle must not contain private keys.\n' >&2
    exit 1
fi
if ! grep -q -- '-----BEGIN CERTIFICATE-----' "$BUNDLE"; then
    if [[ -s "$BUNDLE" ]]; then
        printf 'Error: nonempty CA bundle contains no PEM certificates.\n' >&2
        exit 1
    fi
    printf 'No exported host certificates; keeping the existing trust store.\n'
    exit 0
fi

CERT_TEMP_DIR="$(mktemp -d)"
trap 'rm -f "$CERT_TEMP_DIR"/host-*.crt; rmdir "$CERT_TEMP_DIR"' EXIT

awk -v directory="$CERT_TEMP_DIR" '
    { sub(/\r$/, "") }
    /-----BEGIN CERTIFICATE-----/ {
        if (path != "") exit 1
        count++
        path = sprintf("%s/host-%04d.crt", directory, count)
    }
    path != "" { print > path }
    /-----END CERTIFICATE-----/ {
        if (path == "") exit 1
        close(path)
        path = ""
    }
    END { if (path != "") exit 1 }
' "$BUNDLE"

certificates=("$CERT_TEMP_DIR"/host-*.crt)
for certificate in "${certificates[@]}"; do
    openssl x509 -noout -in "$certificate"
done
printf 'Validated %s host CA certificates.\n' "${#certificates[@]}"
if [[ "$CHECK_ONLY" == 1 ]]; then
    exit 0
fi
if [[ "$EUID" -ne 0 ]]; then
    printf 'Error: installation requires root; run this installer with sudo.\n' >&2
    exit 1
fi

CERT_DIRECTORY=/usr/local/share/ca-certificates/drawthings-host
install -d "$CERT_DIRECTORY"
for certificate in "${certificates[@]}"; do
    install -m 0644 "$certificate" "$CERT_DIRECTORY/$(basename "$certificate")"
done
for certificate in "$CERT_DIRECTORY"/host-*.crt; do
    if [[ ! -f "$CERT_TEMP_DIR/$(basename "$certificate")" ]]; then
        rm -f "$certificate"
    fi
done
update-ca-certificates