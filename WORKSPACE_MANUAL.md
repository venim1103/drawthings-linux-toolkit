# Workspace Manual

Last updated: 2026-06-05

This manual explains how the current workspace is organized, how to run the core workflows, and how to maintain the patching/build system.

## 1) What this workspace contains

Top-level folders and key files:

- .devcontainer/
  - Dockerfile: Base image and native package installs for this workflow.
  - post-create.sh: Environment bootstrap, patch auto-apply, and prebuild steps.
- draw-things-community/
  - Main Draw Things source tree where quantizer and CLI are built.
- DRAW_THINGS_PATCH/
  - Persistent patch bundle for quantization-related changes.
  - Includes both unified patch files and snapshot fallback files.
- tools/
  - Helper scripts for model listing/downloading, patching, config generation, gRPC requests, and tensor conversion.
- dt-models/
  - Local model cache used by Draw Things server and tools.
- requirements-cuda.txt, requirements.txt, requirements-drawthings-tools.txt
  - Python dependency sets.

## 2) First-run setup (devcontainer)

When the container is created, .devcontainer/post-create.sh performs:

- Python virtual environment bootstrap at .venv.
- Python dependency installs from requirements files.
- Draw Things patch application via tools/apply_drawthings_quant_patch.sh.
- Optional Swift prebuilds are skipped by default to keep first-run stable on Linux.
  - Enable with DRAWTHINGS_PREBUILD_RELEASE=1.
  - Optionally include CLI prebuild with DRAWTHINGS_PREBUILD_CLI=1.
- Shell aliases in ~/.bashrc:
  - drawthings-start
  - dt-venv
- drawthings-start defaults:
  - address 127.0.0.1, port 7861, GPU 0
  - no TLS, model browser enabled, no response compression

After opening a new terminal:

    dt-venv
    drawthings-start

If needed, verify GPU visibility:

    nvidia-smi

Quick server health check (after start):

  /workspaces/drawthings-linux-toolkit/.venv/bin/python \
    /workspaces/drawthings-linux-toolkit/tools/dt_api_client.py \
    --host 127.0.0.1:7861 echo --name healthcheck

Legacy port override (for existing probe scripts that use 7859):

  DRAWTHINGS_PORT=7859 drawthings-start

Then validate:

  /workspaces/drawthings-linux-toolkit/.venv/bin/python \
    /workspaces/drawthings-linux-toolkit/tools/dt_api_client.py \
    --host 127.0.0.1:7859 echo --name healthcheck

Telemetry note for probe runs:

- One-shot `nvidia-smi` may report `utilization.gpu=0` even during active inference.
- Prefer interval sampling during active jobs:

    nvidia-smi dmon -s u -d 1

## 3) Patch system

The patch system is designed to survive file drift better than direct file-copy backups.

Patch assets:

- DRAW_THINGS_PATCH/patches/draw-things-community.patch
- DRAW_THINGS_PATCH/patches/s4nnc.patch
- DRAW_THINGS_PATCH/patches/ccv.patch

Snapshot fallback files are also kept under DRAW_THINGS_PATCH/ for hard fallback cases.

### 3.1 Apply patches

    bash /workspaces/drawthings-linux-toolkit/tools/apply_drawthings_quant_patch.sh

Behavior:

- Tries git apply first.
- Tries 3-way apply if needed.
- Falls back to snapshot copy if patch cannot be merged cleanly.

### 3.2 Regenerate patches after new edits

    bash /workspaces/drawthings-linux-toolkit/tools/generate_drawthings_quant_patches.sh

Run this after changing any patched files so the patch bundle stays current.

## 4) Build and use model-quantizer

Build release binary:

    cd /workspaces/drawthings-linux-toolkit/draw-things-community
    swift build -c release --product model-quantizer

Show help:

    ./.build/release/model-quantizer --help

Example quantization command:

    ./.build/release/model-quantizer \
      -i /workspaces/drawthings-linux-toolkit/dt-models/ltx_2.3_22b_distilled_1.1_q6p.ckpt \
      -m ltx2_3 \
      -o /workspaces/drawthings-linux-toolkit/dt-models/ltx_2.3_22b_distilled_1.1_q4p.ckpt \
      --target-codec q4p

Progress + ETA wrapper (recommended for large files):

    cd /workspaces/drawthings-linux-toolkit
    bash tools/dt_quantize_model.sh \
      -i dt-models/ltx_2.3_22b_distilled_1.1_f16.ckpt \
      -m ltx2.3 \
      -o dt-models/ltx_2.3_22b_distilled_1.1_q4p.ckpt \
      --target-codec q4p

Important:

- The -o value must be a file path, not only a directory path.
- For forced codec output, --target-codec supports: auto, q4p, q5p, q6p, q8p, i8x.
- Wrapper env toggles:
  - `DRAWTHINGS_QUANTIZER_MONITOR=0` disables the monitor.
  - `DRAWTHINGS_QUANTIZER_MONITOR_INTERVAL=5` controls refresh interval seconds.

## 4.1) Replay and postpatch safety helpers

Replay helper script:

- tools/run_clipfix2_replay_q6p.sh

What it does:

- Builds model-quantizer and runs forced q6p replay.
- Guards against accidental output clobbering (source/baseline/symlink checks).
- Runs sqlite sanity checks on output by default (`PRAGMA quick_check` and tensors row count).

Example:

    cd /workspaces/drawthings-linux-toolkit
    bash tools/run_clipfix2_replay_q6p.sh \
      -i dt-models/10_e_v1_bf16_regen_0_f16.ckpt \
      -o dt-models/10_e_v1_bf16_regen_0_q6p.ckpt \
      --official-baseline dt-models/ltx_2.3_22b_distilled_1.1_q6p.ckpt

Optional replay safety override flags:

- `--allow-symlink-output`
- `--skip-sqlite-check`

Recovery helper script:

- tools/run_q6p_restore_postpatch_highlimit.sh

What it does:

- Restores target q6p from a chosen source backup.
- Runs clipfix2 postpatch.
- Applies high-limit sqlite one-row fallback for oversized clip tensor row.
- Runs final sqlite `PRAGMA quick_check`.

Example:

    cd /workspaces/drawthings-linux-toolkit
    bash tools/run_q6p_restore_postpatch_highlimit.sh --tag repatch_from_restored

## 5) Model management tools

## 5.1 Fast model CLI wrapper

Script:

- tools/dt_models_cli.sh

Examples:

    cd /workspaces/drawthings-linux-toolkit
    bash tools/dt_models_cli.sh list --models-dir /workspaces/drawthings-linux-toolkit/dt-models
    bash tools/dt_models_cli.sh list --downloaded-only --models-dir /workspaces/drawthings-linux-toolkit/dt-models
    bash tools/dt_models_cli.sh ensure ltx_2.3_22b_distilled_1.1_q6p.ckpt --models-dir /workspaces/drawthings-linux-toolkit/dt-models

Note:

- This wrapper uses prebuilt binaries first to avoid repeated swift run rebuild checks.

## 5.2 Model dependency auditor/downloader

Script:

- tools/dt_model_requirements.py

Examples:

    cd /workspaces/drawthings-linux-toolkit
    python3 tools/dt_model_requirements.py --list --filter ltx
    python3 tools/dt_model_requirements.py --model ltx_2.3_22b_distilled_1.1_q6p.ckpt --models-dir /workspaces/drawthings-linux-toolkit/dt-models
    python3 tools/dt_model_requirements.py --model ltx_2.3_22b_distilled_1.1_q6p.ckpt --models-dir /workspaces/drawthings-linux-toolkit/dt-models --download-missing

## 6) Generation pipeline tools

## 6.1 Build generation config

Script:

- tools/dt_make_config.py

Example:

    python3 /workspaces/drawthings-linux-toolkit/tools/dt_make_config.py \
      --out /workspaces/drawthings-linux-toolkit/output/config.bin \
      --model ltx_2.3_22b_distilled_1.1_q6p.ckpt \
      --width 704 \
      --height 384 \
      --steps 8 \
      --num-frames 81

## 6.2 Call Draw Things gRPC

Script:

- tools/dt_api_client.py

Quick server check:

    python3 /workspaces/drawthings-linux-toolkit/tools/dt_api_client.py --host 127.0.0.1:7861 echo --name test

## 6.3 Convert tensor outputs to playable files

Script:

- tools/dt_tensor_to_playable.py

Example:

    python3 /workspaces/drawthings-linux-toolkit/tools/dt_tensor_to_playable.py \
      --image-bin /workspaces/drawthings-linux-toolkit/output/image_0001.bin \
      --out-dir /workspaces/drawthings-linux-toolkit/output \
      --base-name playable

## 6.4 End-to-end helper

Script:

- tools/dt_generate_video.sh

Example:

    DT_MODEL=ltx_2.3_22b_distilled_1.1_q6p.ckpt \
    DT_HOST=127.0.0.1:7861 \
    bash /workspaces/drawthings-linux-toolkit/tools/dt_generate_video.sh "ocean waves at sunset"

## 7) Why builds seemed to happen repeatedly

Primary reasons:

- Commands using swift run can trigger package graph/build checks often.
- Missing local binaries force rebuild.
- Re-resolved checkouts can invalidate build caches.

Current mitigation in this workspace:

- tools/dt_models_cli.sh resolves to a prebuilt binary first.
- .devcontainer/post-create.sh prebuilds release binaries when missing.

## 8) Common recovery actions

Re-apply all quant patches:

    bash /workspaces/drawthings-linux-toolkit/tools/apply_drawthings_quant_patch.sh

Regenerate patch bundle after edits:

    bash /workspaces/drawthings-linux-toolkit/tools/generate_drawthings_quant_patches.sh

Rebuild quantizer only:

    cd /workspaces/drawthings-linux-toolkit/draw-things-community
    swift build -c release --product model-quantizer

  Run one-command restore/postpatch/high-limit fallback flow:

    cd /workspaces/drawthings-linux-toolkit
    bash tools/run_q6p_restore_postpatch_highlimit.sh --tag repatch_from_restored

## 9) Recommended daily workflow

1. Open devcontainer.
2. Open new terminal and run:

    dt-venv

3. Start server when needed:

    drawthings-start

4. Use tools/dt_models_cli.sh to list/ensure models.
5. Build or run model-quantizer as needed.
6. If upstream changes break build, run patch apply script again.

## 10) Source-of-truth references

- Draw Things patch notes: DRAW_THINGS_PATCH/README.md
- Historic setup notes: DRAWTHINGS_SETUP_LOG.md
- Devcontainer startup logic: .devcontainer/post-create.sh
- Native dependency set: .devcontainer/Dockerfile
