FROM docker.io/drawthingsai/draw-things-grpc-server-cli:latest

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive
ENV DRAWTHINGS_IMAGE=docker.io/drawthingsai/draw-things-grpc-server-cli:latest
ENV DRAWTHINGS_MODEL_DIR=/workspaces/drawthings-linux-toolkit/dt-models
ENV DRAWTHINGS_PORT=7861

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        sudo \
        ca-certificates \
        ca-certificates-java \
        git \
        curl \
        python3 \
        python3-pip \
        python3-venv \
        python-is-python3 \
        binutils \
        build-essential \
        flatbuffers-compiler \
        libxml2 \
        libxml2-dev \
        pkg-config \
        zlib1g-dev \
        libsqlite3-dev \
        sqlite3 \
        libopenblas-dev \
        ripgrep; \
    rm -rf /var/lib/apt/lists/*; \
    existing_user_by_uid="$(getent passwd "${USER_UID}" | cut -d: -f1 || true)"; \
    existing_group_by_gid="$(getent group "${USER_GID}" | cut -d: -f1 || true)"; \
    if id -u "${USERNAME}" >/dev/null 2>&1; then \
        if [ "$(id -u "${USERNAME}")" != "${USER_UID}" ]; then \
            usermod --uid "${USER_UID}" "${USERNAME}"; \
        fi; \
        if [ "$(id -g "${USERNAME}")" != "${USER_GID}" ]; then \
            if [ -n "${existing_group_by_gid}" ]; then \
                usermod --gid "${existing_group_by_gid}" "${USERNAME}"; \
            else \
                if ! getent group "${USERNAME}" >/dev/null 2>&1; then \
                    groupadd --gid "${USER_GID}" "${USERNAME}"; \
                else \
                    groupmod --gid "${USER_GID}" "${USERNAME}"; \
                fi; \
                usermod --gid "${USERNAME}" "${USERNAME}"; \
            fi; \
        fi; \
    elif [ -n "${existing_user_by_uid}" ]; then \
        existing_user_group="$(id -gn "${existing_user_by_uid}")"; \
        if [ "${existing_user_group}" = "${existing_user_by_uid}" ] && [ "${existing_user_group}" != "${USERNAME}" ] && ! getent group "${USERNAME}" >/dev/null 2>&1; then \
            groupmod --new-name "${USERNAME}" "${existing_user_group}"; \
        fi; \
        usermod --login "${USERNAME}" --home "/home/${USERNAME}" --move-home "${existing_user_by_uid}"; \
    else \
        if [ -z "${existing_group_by_gid}" ]; then \
            groupadd --gid "${USER_GID}" "${USERNAME}"; \
        fi; \
        useradd --uid "${USER_UID}" --gid "${USER_GID}" -m -s /bin/bash "${USERNAME}"; \
    fi; \
    user_group="$(id -gn "${USERNAME}")"; \
    mkdir -p "/home/${USERNAME}/.cache" "/home/${USERNAME}/.config" "${DRAWTHINGS_MODEL_DIR}"; \
    chown -R "${USERNAME}:${user_group}" "/home/${USERNAME}/.cache" \
                                         "/home/${USERNAME}/.config" \
                                         "${DRAWTHINGS_MODEL_DIR}"; \
    printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    '' \
    'MODEL_DIR="${DRAWTHINGS_MODEL_DIR:-/workspaces/drawthings-linux-toolkit/dt-models}"' \
    'ADDRESS="${DRAWTHINGS_ADDRESS:-127.0.0.1}"' \
    'PORT="${DRAWTHINGS_PORT:-7861}"' \
    'GPU="${DRAWTHINGS_GPU:-0}"' \
    'EXTRA_LIB_DIRS=(' \
    '  "/usr/local/swift/usr/lib/swift/linux"' \
    '  "/usr/lib/wsl/lib"' \
    '  "/usr/local/cuda/targets/x86_64-linux/lib"' \
    '  "/usr/local/cuda/lib64"' \
    ')' \
    '' \
    'mkdir -p "$MODEL_DIR"' \
    '' \
    'for lib_dir in "${EXTRA_LIB_DIRS[@]}"; do' \
    '    if [ -d "$lib_dir" ]; then' \
    '        export LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"' \
    '    fi' \
    'done' \
    '' \
    'if [ "$#" -eq 0 ]; then' \
    '    exec gRPCServerCLI \' \
    '        --address "$ADDRESS" \' \
    '        --port "$PORT" \' \
    '        --gpu "$GPU" \' \
    '        --no-tls \' \
    '        --model-browser \' \
    '        --no-response-compression \' \
    '        "$MODEL_DIR"' \
    'fi' \
    '' \
    'exec gRPCServerCLI "$@"' \
    > /usr/local/bin/drawthings-grpc; \
    chmod +x /usr/local/bin/drawthings-grpc; \
    printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    '' \
    'for p in /usr/lib/wsl/lib/nvidia-smi /usr/lib/wsl/drivers/*/nvidia-smi /usr/bin/nvidia-smi; do' \
    '  if [ -x "$p" ]; then' \
    '    exec "$p" "$@"' \
    '  fi' \
    'done' \
    '' \
    'echo "nvidia-smi is not available in this container. Check GPU passthrough (/dev/nvidia* or /dev/dxg)." >&2' \
    'exit 127' \
    > /usr/local/bin/nvidia-smi; \
    chmod +x /usr/local/bin/nvidia-smi; \
    echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}"; \
    chmod 0440 "/etc/sudoers.d/${USERNAME}"