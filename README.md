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

## Recovery and replay helpers (2026-06-05)

- `tools/run_q6p_restore_postpatch_highlimit.sh`
	- One-command recovery flow for the repaired q6p path:
		- restore target from backup/source
		- run clipfix2 postpatch
		- apply high-limit one-row fallback for oversized clip tensor
		- run sqlite `PRAGMA quick_check`
- `tools/run_clipfix2_postpatch_q6p.sh`
	- Hardened to refuse apply-mode symlink targets by default.
	- Refuses `target_real == baseline_real` in apply mode.
- `tools/run_clipfix2_replay_q6p.sh`
	- Hardened to prevent output path clobbering via source/baseline/symlink checks.
	- Runs sqlite quick sanity checks by default:
		- `PRAGMA quick_check`
		- `SELECT COUNT(*) FROM tensors`
	- New options:
		- `--allow-symlink-output`
		- `--skip-sqlite-check`

Quick usage:

		cd /workspaces/drawthings-linux-toolkit

		# Restore -> postpatch -> high-limit fallback -> quick_check
		bash tools/run_q6p_restore_postpatch_highlimit.sh --tag repatch_from_restored

		# Replay quantization with safety checks enabled by default
		bash tools/run_clipfix2_replay_q6p.sh \
			-i dt-models/10_e_v1_bf16_regen_0_f16.ckpt \
			-o dt-models/10_e_v1_bf16_regen_0_q6p.ckpt \
			--official-baseline dt-models/ltx_2.3_22b_distilled_1.1_q6p.ckpt
