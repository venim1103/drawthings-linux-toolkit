# Devcontainer Recovery on WSL

## What Failed

The October 23 investigation found two independent issues:

- HTTPS presents a corporate chain containing `GSASartoriusRoot01`. The existing container trusts the mounted Zscaler root, but that alone does not validate this chain. The old bootstrap copied only one hard-coded Zscaler certificate.
- `nvidia-smi` works and `/dev/dxg` is present, but Python cannot load `libcuda.so.1` by name. Explicitly adding `/usr/lib/wsl/lib` makes real PyTorch CUDA matrix multiplication and convolution pass on the RTX A3000, with CUDA 12.8 and cuDNN 9.10.2.

These findings do not indicate a broken Windows NVIDIA driver. Missing `nvcc` and Bazel are a separate native-build-toolchain concern; they are not required for the existing prebuilt server or PyTorch runtime.

## Apply the Fix

In a **WSL terminal outside the devcontainer**, change to the host checkout directory. That may be under your WSL home directory; the container-only path `/workspaces/drawthings-linux-toolkit` is not necessarily the host path.

```bash
bash .devcontainer/prepare-build.sh
bash .devcontainer/install-host-certificates.sh --check .devcontainer/host-ca-certificates.crt
```

On WSL with Windows interoperability enabled, the exporter reads trusted public roots from both the Windows Current User and Local Machine root stores. It does not export private keys. It falls back to the host's Linux CA bundle if Windows export is unavailable, with a warning when the Windows export fails.

If the corporate root is absent from both stores, obtain the approved CA bundle from your IT team and use it explicitly:

```bash
DRAWTHINGS_CA_BUNDLE=/path/to/approved-ca-bundle.pem bash .devcontainer/prepare-build.sh
```

Do not use a certificate captured from an unverified connection as a new trusted root. Do not disable Git, curl, pip, or Python certificate verification.

Then run **Dev Containers: Rebuild Container** in VS Code. The initialize command automatically reruns the exporter, so preserve an explicit bundle override in the host environment used by VS Code if you need one. The generated bundle is ignored by Git.

The Docker build installs the exported roots before `apt` accesses the network. The image and setup script configure Python, Requests, curl, pip, and Node to use the system CA bundle. The image also prepends the mounted WSL driver directory to the inherited library path without discarding the base image's CUDA library paths.

Post-create verifies HTTPS, Git access, and a tiny CUDA computation before applying the existing Draw Things patches. It does not download pretrained models for those checks.

## If Image Pulls Fail

Certificate trust inside a Dockerfile cannot fix TLS errors while the host container engine pulls the `FROM` image. If Git/curl or local Podman also fail in WSL, install the already validated exported roots into the WSL trust store:

```bash
sudo bash .devcontainer/install-host-certificates.sh .devcontainer/host-ca-certificates.crt
git ls-remote https://github.com/drawthingsai/draw-things-community.git HEAD
```

Use your normal WSL terminal for the `sudo` prompt. If your container engine runs in a separate VM or remote machine, its trust store needs the approved roots too; updating this WSL distribution will not update that other machine.

If Windows export was unavailable, check Windows interoperability with `command -v powershell.exe` and `command -v wslpath`. An installation with Windows executables excluded from `PATH` can use the explicit approved-bundle override instead. Do not change unrelated WSL settings or install a Linux display driver to solve certificate trust.

## Current Container Without a Rebuild

After exporting the bundle on the WSL host, the file is visible in the shared checkout. In a container terminal:

```bash
sudo bash .devcontainer/install-host-certificates.sh .devcontainer/host-ca-certificates.crt
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE" CURL_CA_BUNDLE="$SSL_CERT_FILE"
export NODE_EXTRA_CA_CERTS="$SSL_CERT_FILE" PIP_CERT="$SSL_CERT_FILE"
.venv/bin/python .devcontainer/smoke.py --network
```

The exports affect that terminal and its children, not already-running Python language servers or other terminals. Rebuild/reopen the container to apply the image-level environment everywhere. There is no need to rerun the full post-create dependency installation just to import certificates.

## Verification and GPU Troubleshooting

From the workspace inside the rebuilt container:

```bash
nvidia-smi
.venv/bin/python .devcontainer/smoke.py --network
```

The smoke check requires the configured system CA environment, real CUDA matrix multiplication/convolution, verified Python HTTPS to GitHub/Hugging Face/PyPI, and a verified GitHub `git ls-remote`. Omit `--network` for an offline trust-store/compute check; an offline pass does not establish HTTPS connectivity.

If `/dev/dxg` or the GPU disappears in a future session, check GPU visibility in WSL first with `/usr/lib/wsl/lib/nvidia-smi`. Update WSL and the Windows NVIDIA driver only if that host-side check fails or the installed versions are incompatible. Do not install Linux NVIDIA kernel/display drivers into WSL. Native CUDA compiler and Bazel provisioning belongs to the later dedicated Draw Things builder setup.

The container engine/build is not accessible from this editor session, so the revised image build and Windows-root export must be confirmed on the host. The existing `.devcontainer/OTHER` examples are reference files and are not part of the active configuration.