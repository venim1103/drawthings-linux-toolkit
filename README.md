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

## In-place q6p dimfix helpers (2026-06-08)

For low-space environments, these scripts patch `10_e_v1` q6p in place (no extra 20G+ copy):

- `tools/run_q6p_inplace_dimfix_from_f16.sh`
- `tools/run_q6p_canary_once.sh`
- `tools/dt_build_q6p_dimfix_names.py`
- `tools/dt_patch_ckpt_metadata_subset.py`
- `tools/patch_sets/10_e_v1_q6p_dimfix770_20260608.txt`

Quick usage (default precomputed names, bounded canary):

		cd /workspaces/drawthings-linux-toolkit
		bash tools/run_q6p_inplace_dimfix_from_f16.sh --canary-timeout-sec 120 --max-responses 10

Optional name rebuild mode (requires baseline + reference q6p files present):

		bash tools/run_q6p_inplace_dimfix_from_f16.sh --rebuild-names

If official q6p baseline is unavailable, clipfix2 can be used as fallback baseline/reference for rebuild mode:

		bash tools/run_q6p_inplace_dimfix_from_f16.sh \
			--rebuild-names \
			--baseline-q6p dt-models/ltx_2.3_22b_distilled_q6p_forcedfix_clipfix2_20260602.ckpt \
			--reference-q6p dt-models/ltx_2.3_22b_distilled_q6p_forcedfix_clipfix2_20260602.ckpt

Handoff docs for continuation:

- `Q6P_HANDOFF_FINDINGS_2026-06-08.md`
- `Q6P_CONTINUATION_RUNBOOK_2026-06-08.md`
- `Q6P_RUN_LOG.md` (append-only iteration record)

## Latest runtime findings (2026-06-05 post-replay A/B)

- Replay regeneration completed with structural checks passing (`PRAGMA quick_check=ok`, `tensors rows=5746`), but runtime stability for custom `10_e_v1` was not restored.
- Official control alias `official_q6p_via_custom` completed end-to-end:
	- `responses=15`, `images written=1`, `audio written=1`, `preview frames seen=5`
	- output directory: `/workspaces/drawthings-linux-toolkit/output/dt_video_20260605_121549`
- Custom alias `10_e_v1` failed at generation start:
	- client error: `gRPC error: UNAVAILABLE: Socket closed`
	- server crash: `SIGSEGV` in loader path `ccv_nnc_tensor_read -> ccv_cnnp_model_read`
- Recorded artifacts:
	- `output/ab_post_replay_20260605_121548.md`
	- `output/ab_post_replay_20260605_121548_official.log`
	- `output/ab_post_replay_20260605_121548_custom.log`
