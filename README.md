# drawthings-linux-toolkit

Linux-first toolkit for Draw Things workflows: reproducible devcontainer setup, resilient patch management, model utilities, and local quantization automation.

## What is included

- Devcontainer setup and bootstrap scripts.
- Draw Things quantization patch system (unified patches + snapshot fallback).
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

## grpcurl in devcontainer

grpcurl is not required for the toolkit workflows and is not committed to the repo, but it is installed automatically by the devcontainer post-create setup.

ripgrep (rg) is also ensured by post-create so search tooling is always available.

To reinstall or pin a specific version manually:

- bash tools/install_grpcurl.sh
