#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRAWTHINGS_MODEL_DIR="${DRAWTHINGS_MODEL_DIR:-$WORKSPACE_ROOT/dt-models}"
DRAWTHINGS_REPO="$WORKSPACE_ROOT/draw-things-community"
DRAWTHINGS_REPO_URL="${DRAWTHINGS_REPO_URL:-https://github.com/drawthingsai/draw-things-community.git}"
DRAWTHINGS_REPO_REF="${DRAWTHINGS_REPO_REF:-}"
DRAWTHINGS_PREBUILD_RELEASE="${DRAWTHINGS_PREBUILD_RELEASE:-0}"
DRAWTHINGS_PREBUILD_CLI="${DRAWTHINGS_PREBUILD_CLI:-0}"
QUANT_PATCH_SCRIPT="$WORKSPACE_ROOT/tools/apply_drawthings_quant_patch.sh"
VENV_DIR="$WORKSPACE_ROOT/.venv"
TOOLS_REQ_FILE="$WORKSPACE_ROOT/requirements-drawthings-tools.txt"
PROJECT_REQ_FILE="$WORKSPACE_ROOT/requirements-cuda.txt"
PROJECT_REQ_FALLBACK_FILE="$WORKSPACE_ROOT/requirements.txt"
COMFYUI_REQ_FILE="$WORKSPACE_ROOT/ComfyUI/requirements.txt"
GRPCURL_INSTALL_SCRIPT="$WORKSPACE_ROOT/tools/install_grpcurl.sh"

# Ensure user-local tools are discoverable during setup.
export PATH="$HOME/.local/bin:$PATH"

# Update CA certificates if a real Zscaler cert is mounted (not an empty placeholder)
if [ -s /usr/local/share/ca-certificates/ZscalerRootCertificate-2048-SHA256.crt ]; then
    echo "==> Updating CA certificates (Zscaler)..."
    sudo update-ca-certificates
    if command -v update-ca-certificates-java >/dev/null 2>&1; then
        sudo update-ca-certificates-java || true
    fi
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
    export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
fi

echo "==> Verifying libxml2 runtime availability..."
if ! ldconfig -p | grep -q "libxml2.so.2"; then
    echo "ERROR: libxml2.so.2 not found in linker cache."
    exit 1
fi

echo "==> Ensuring linker tooling availability (ld)..."
if command -v ld >/dev/null 2>&1; then
    echo "==> ld already available: $(command -v ld)"
else
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends binutils
fi

if ! command -v ld >/dev/null 2>&1; then
    echo "ERROR: ld not found in PATH after installing binutils."
    exit 1
fi

echo "==> Ensuring C/C++ toolchain availability..."
if command -v gcc >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
    echo "==> gcc/g++ already available."
else
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends build-essential
fi

if ! command -v gcc >/dev/null 2>&1 || ! command -v g++ >/dev/null 2>&1; then
    echo "ERROR: gcc/g++ not found in PATH after installing build-essential."
    exit 1
fi

echo "==> Verifying Python tooling availability..."
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found in PATH."
    exit 1
fi
if ! command -v pip3 >/dev/null 2>&1; then
    echo "ERROR: pip3 not found in PATH."
    exit 1
fi
if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git not found in PATH."
    exit 1
fi

echo "==> Ensuring draw-things-community source tree..."
if [[ -d "$DRAWTHINGS_REPO/.git" ]]; then
    echo "==> draw-things-community already present: $DRAWTHINGS_REPO"
elif [[ -e "$DRAWTHINGS_REPO" ]]; then
    echo "ERROR: path exists but is not a git repository: $DRAWTHINGS_REPO"
    exit 1
else
    echo "==> Cloning draw-things-community from $DRAWTHINGS_REPO_URL"
    git clone "$DRAWTHINGS_REPO_URL" "$DRAWTHINGS_REPO"

    if [[ -n "$DRAWTHINGS_REPO_REF" ]]; then
        echo "==> Checking out draw-things-community ref: $DRAWTHINGS_REPO_REF"
        if ! git -C "$DRAWTHINGS_REPO" checkout "$DRAWTHINGS_REPO_REF"; then
            echo "ERROR: Failed to checkout draw-things-community ref: $DRAWTHINGS_REPO_REF"
            exit 1
        fi
    fi
fi

echo "==> Bootstrapping workspace virtual environment..."
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip

echo "==> Installing project Python requirements..."
if [[ -f "$PROJECT_REQ_FILE" ]]; then
    "$VENV_DIR/bin/pip" install -r "$PROJECT_REQ_FILE"
elif [[ -f "$PROJECT_REQ_FALLBACK_FILE" ]]; then
    "$VENV_DIR/bin/pip" install -r "$PROJECT_REQ_FALLBACK_FILE"
else
    echo "WARNING: No project requirements file found at $PROJECT_REQ_FILE or $PROJECT_REQ_FALLBACK_FILE"
fi

if [[ -f "$COMFYUI_REQ_FILE" ]]; then
    "$VENV_DIR/bin/pip" install -r "$COMFYUI_REQ_FILE"
fi

echo "==> Installing Draw Things tool requirements..."
if [[ -f "$TOOLS_REQ_FILE" ]]; then
    "$VENV_DIR/bin/pip" install -r "$TOOLS_REQ_FILE"
else
    "$VENV_DIR/bin/pip" install grpcio grpcio-tools protobuf flatbuffers numpy pillow opencv-python-headless fpzip
fi

echo "==> Ensuring ripgrep (rg) is available..."
if command -v rg >/dev/null 2>&1; then
    echo "==> rg already available: $(command -v rg)"
else
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends ripgrep
fi

if ! command -v rg >/dev/null 2>&1; then
    echo "ERROR: rg not found in PATH after installation."
    exit 1
fi

echo "==> Ensuring grpcurl is available..."
if command -v grpcurl >/dev/null 2>&1; then
    echo "==> grpcurl already available: $(command -v grpcurl)"
else
    if [[ -x "$GRPCURL_INSTALL_SCRIPT" ]]; then
        "$GRPCURL_INSTALL_SCRIPT"
    elif [[ -f "$GRPCURL_INSTALL_SCRIPT" ]]; then
        bash "$GRPCURL_INSTALL_SCRIPT"
    else
        echo "ERROR: grpcurl install script not found: $GRPCURL_INSTALL_SCRIPT"
        exit 1
    fi
fi

if ! command -v grpcurl >/dev/null 2>&1; then
    echo "ERROR: grpcurl not found in PATH after installation."
    exit 1
fi

echo "==> Verifying FlatBuffers tooling availability..."
if ! command -v flatc >/dev/null 2>&1; then
    echo "ERROR: flatc not found in PATH."
    exit 1
fi

echo "==> Creating Draw Things model cache directory..."
mkdir -p "$DRAWTHINGS_MODEL_DIR"

echo "==> Applying Draw Things quantization patch bundle..."
if [[ -x "$QUANT_PATCH_SCRIPT" ]]; then
    "$QUANT_PATCH_SCRIPT" "$WORKSPACE_ROOT"
else
    echo "WARNING: Quantization patch script not found or not executable: $QUANT_PATCH_SCRIPT"
fi

if command -v swift >/dev/null 2>&1 && [[ -d "$DRAWTHINGS_REPO" ]]; then
    if [[ "$DRAWTHINGS_PREBUILD_RELEASE" != "1" ]]; then
        echo "==> Skipping optional Swift prebuilds (set DRAWTHINGS_PREBUILD_RELEASE=1 to enable)."
    elif pgrep -f "swift-(build|package).*$DRAWTHINGS_REPO" >/dev/null 2>&1; then
        echo "WARNING: Another SwiftPM process is active for $DRAWTHINGS_REPO. Skipping optional prebuilds."
    else
        echo "==> Optional prebuild enabled. Compiling release products when missing..."

        if [[ "$DRAWTHINGS_PREBUILD_CLI" == "1" ]] && [[ ! -x "$DRAWTHINGS_REPO/.build/release/draw-things-cli" ]]; then
            if ! swift build --package-path "$DRAWTHINGS_REPO" -c release --product draw-things-cli; then
                echo "WARNING: draw-things-cli prebuild failed; continuing setup."
            fi
        fi

        if [[ ! -x "$DRAWTHINGS_REPO/.build/release/model-quantizer" ]]; then
            if ! swift build --package-path "$DRAWTHINGS_REPO" -c release --product model-quantizer; then
                echo "WARNING: model-quantizer prebuild failed; continuing setup."
            fi
        fi
    fi
fi

echo "==> Verifying Draw Things CLI availability..."
if ! command -v gRPCServerCLI >/dev/null 2>&1; then
    echo "ERROR: gRPCServerCLI not found in PATH."
    exit 1
fi
if ! command -v drawthings-grpc >/dev/null 2>&1; then
    echo "ERROR: drawthings-grpc helper not found in PATH."
    exit 1
fi

echo "==> Checking in-container GPU visibility..."
if [[ -e /dev/nvidia0 || -e /dev/dxg ]]; then
    echo "==> GPU device node visible in container."
else
    echo "WARNING: No GPU device nodes found (/dev/nvidia* or /dev/dxg)."
fi

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "==> GPU check passed (nvidia-smi works inside devcontainer)."
elif [[ -x /usr/lib/wsl/lib/nvidia-smi ]] && /usr/lib/wsl/lib/nvidia-smi >/dev/null 2>&1; then
    echo "==> GPU check passed (WSL nvidia-smi works inside devcontainer)."
elif compgen -G "/usr/lib/wsl/drivers/*/nvidia-smi" >/dev/null 2>&1; then
    wsl_smi="$(ls -1 /usr/lib/wsl/drivers/*/nvidia-smi 2>/dev/null | head -n 1)"
    if [[ -n "$wsl_smi" ]] && "$wsl_smi" >/dev/null 2>&1; then
        echo "==> GPU check passed (WSL driver-store nvidia-smi works inside devcontainer)."
    else
        echo "WARNING: Found WSL driver-store nvidia-smi but invocation failed."
    fi
else
    echo "WARNING: nvidia-smi check failed in this devcontainer session."
    echo "WARNING: Rebuild/reopen may be needed so the runtime GPU device mapping is applied."
fi

DRAWTHINGS_BASHRC_BLOCK_START="# >>> drawthings-cli >>>"
DRAWTHINGS_BASHRC_BLOCK_END="# <<< drawthings-cli <<<"

tmp_bashrc="$(mktemp)"
awk -v start="$DRAWTHINGS_BASHRC_BLOCK_START" -v end="$DRAWTHINGS_BASHRC_BLOCK_END" '
    $0 == start { skip = 1; next }
    $0 == end { skip = 0; next }
    skip != 1 { print }
' "$HOME/.bashrc" > "$tmp_bashrc"

cat >> "$tmp_bashrc" <<EOF

$DRAWTHINGS_BASHRC_BLOCK_START
export DRAWTHINGS_MODEL_DIR="$DRAWTHINGS_MODEL_DIR"
export DRAWTHINGS_WORKSPACE_ROOT="$WORKSPACE_ROOT"
export DRAWTHINGS_ADDRESS="127.0.0.1"
export DRAWTHINGS_PORT="7861"
export DRAWTHINGS_GPU="0"
export PATH="$HOME/.local/bin:$VENV_DIR/bin:\$PATH"
# Starts gRPCServerCLI with stable local defaults and DRAWTHINGS_MODEL_DIR.
alias drawthings-start='drawthings-grpc'
alias dt-venv='source "$VENV_DIR/bin/activate"'
$DRAWTHINGS_BASHRC_BLOCK_END
EOF

mv "$tmp_bashrc" "$HOME/.bashrc"

echo "==> Draw Things devcontainer setup complete."
echo "==> Open a new terminal, then run: drawthings-start"