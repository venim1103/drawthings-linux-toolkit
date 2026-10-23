#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CERT_DEST="$SCRIPT_DIR/host-ca-certificates.crt"

if [[ "${1:-}" == "--help" ]]; then
    printf 'Usage: bash .devcontainer/prepare-build.sh\n'
    printf 'Run on the host before building the devcontainer.\n'
    printf 'DRAWTHINGS_CA_BUNDLE may specify a trusted PEM bundle instead of host export.\n'
    exit 0
fi
if [[ $# -ne 0 ]]; then
    printf 'Error: unexpected arguments; use --help.\n' >&2
    exit 1
fi

CERT_TEMP="$(mktemp "$CERT_DEST.XXXXXX")"
trap 'rm -f "$CERT_TEMP"' EXIT

has_certificate() {
    [[ -r "$1" && -s "$1" ]] && grep -q -- '-----BEGIN CERTIFICATE-----' "$1"
}

export_macos() {
    local keychain
    while IFS= read -r keychain; do
        keychain="${keychain#*\"}"
        keychain="${keychain%\"*}"
        if [[ -r "$keychain" ]]; then
            security find-certificate -a -p "$keychain" >> "$CERT_TEMP" || return 1
        fi
    done < <(security list-keychains -d user; security list-keychains -d system)
}

export_windows() {
    local destination powershell_command
    if command -v wslpath >/dev/null 2>&1; then
        destination="$(wslpath -w "$CERT_TEMP")" || return 1
    elif command -v cygpath >/dev/null 2>&1; then
        destination="$(cygpath -w "$CERT_TEMP")" || return 1
    else
        return 1
    fi
    read -r -d '' powershell_command <<'POWERSHELL' || true
$ErrorActionPreference = "Stop"
try {
    $destination = [Console]::In.ReadLine()
    $certificates = Get-ChildItem Cert:\CurrentUser\Root, Cert:\LocalMachine\Root
    $pem = foreach ($certificate in $certificates) {
        $base64 = [Convert]::ToBase64String(
            $certificate.RawData,
            [Base64FormattingOptions]::InsertLineBreaks
        )
        "-----BEGIN CERTIFICATE-----`n$base64`n-----END CERTIFICATE-----`n"
    }
    [IO.File]::WriteAllText($destination, ($pem -join ""), [Text.Encoding]::ASCII)
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
POWERSHELL
    printf '%s\n' "$destination" |
        powershell.exe -NoLogo -NoProfile -NonInteractive -Command "$powershell_command"
}

export_host_bundle() {
    local bundle
    for bundle in \
        /etc/ssl/certs/ca-certificates.crt \
        /etc/ssl/certs/ca-bundle.crt \
        /etc/pki/tls/certs/ca-bundle.crt \
        /mingw64/etc/ssl/certs/ca-bundle.crt; do
        if has_certificate "$bundle"; then
            cp "$bundle" "$CERT_TEMP"
            return 0
        fi
    done
    return 1
}

if [[ -n "${DRAWTHINGS_CA_BUNDLE:-}" ]]; then
    if ! has_certificate "$DRAWTHINGS_CA_BUNDLE"; then
        printf 'Error: DRAWTHINGS_CA_BUNDLE must name a readable PEM certificate bundle.\n' >&2
        exit 1
    fi
    cp "$DRAWTHINGS_CA_BUNDLE" "$CERT_TEMP"
elif command -v security >/dev/null 2>&1; then
    if ! export_macos; then
        printf 'Warning: macOS certificate export failed; trying the system bundle.\n' >&2
        : > "$CERT_TEMP"
    fi
elif command -v powershell.exe >/dev/null 2>&1; then
    if ! export_windows; then
        printf 'Warning: Windows certificate export failed; trying the WSL/system bundle.\n' >&2
        : > "$CERT_TEMP"
    fi
fi

if ! has_certificate "$CERT_TEMP"; then
    export_host_bundle || true
fi
if ! has_certificate "$CERT_TEMP"; then
    printf 'Warning: no host certificates found; only the image default trust store will be used.\n' >&2
    : > "$CERT_TEMP"
fi
if grep -q -- 'PRIVATE KEY' "$CERT_TEMP"; then
    printf 'Error: the CA bundle must not contain private keys.\n' >&2
    exit 1
fi
if has_certificate "$CERT_TEMP" && command -v openssl >/dev/null 2>&1; then
    openssl crl2pkcs7 -nocrl -certfile "$CERT_TEMP" -out /dev/null
fi

chmod 0644 "$CERT_TEMP"
mv "$CERT_TEMP" "$CERT_DEST"
count="$(grep -c -- '-----BEGIN CERTIFICATE-----' "$CERT_DEST" || true)"
printf 'Prepared %s host certificates for the container build.\n' "$count"