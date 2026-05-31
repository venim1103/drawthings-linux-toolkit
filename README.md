# drawthings-linux-toolkit

Linux-first toolkit for Draw Things workflows: reproducible devcontainer setup, resilient patch management, model utilities, and local quantization automation.

## What is included

- Devcontainer setup and bootstrap scripts.
- Draw Things quantization patch system (unified patches + snapshot fallback).
- Quantization wrapper with live progress and ETA (`tools/dt_quantize_model.sh`).
- Model management and download/audit utilities.
- gRPC generation helpers and tensor-to-playable converters.
- Python utility scripts for Draw Things workflows.

## Start here

- WORKSPACE_MANUAL.md
- REPO_MIGRATION_GUIDE.md
- DRAW_THINGS_PATCH/README.md

## Main directories

- .devcontainer/
- tools/
- DRAW_THINGS_PATCH/
