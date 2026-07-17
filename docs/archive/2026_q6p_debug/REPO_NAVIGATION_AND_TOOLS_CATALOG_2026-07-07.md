# Repository Navigation And Tools Catalog (2026-07-07)

## Canonical Root Document

If you share exactly one document as the entry point for this repository, use this file:

- `REPO_NAVIGATION_AND_TOOLS_CATALOG_2026-07-07.md`

Reason:

- it maps the repository structure,
- explains the purpose of core directories and tools,
- and links to all major deep-dive markdown docs by topic.

## 1) Why This Document Exists

This document answers the practical hand-over questions:

- what each major repository directory is for,
- what the devcontainer setup does,
- what DRAW_THINGS_PATCH is for,
- what is in tools/ and the purpose of each tool,
- what the original plan was,
- and where to find deeper information in markdown docs.

This is a docs-only reference. No execution is performed by this document.

---

## 2) Repository Layout: Where To Find What

### 2.1 Core Workspace Areas

- `.devcontainer/`
  - Container image and startup wiring for reproducible Linux+GPU workflows.
  - Key files: `Dockerfile`, `devcontainer.json`, `post-create.sh`.

- `DRAW_THINGS_PATCH/`
  - Long-lived patch bundle for Draw Things quantization and importer behavior.
  - Stores both unified patch files and snapshot fallback files.

- `draw-things-community/`
  - Working source tree for Draw Things builds and patch application targets.

- `tools/`
  - Operational toolbox: conversion wrappers, probes, patch/repair pipelines, canary runners, alias/schema investigation scripts, and utilities.

- `dt-models/`
  - Local model/checkpoint cache (official controls + custom artifacts).

- `output/`
  - Experiment artifact root. Every run writes logs/summaries/media here.

- Root Q6P docs
  - Investigation history, run chronology, baselines, and continuation guides.

### 2.2 Important Root Docs

- `README.md`: top-level quick start and main workflows.
- `WORKSPACE_MANUAL.md`: operational manual for setup, patching, and core commands.
- `Q6P_RUN_LOG.md`: append-only chronological run ledger.
- `Q6P_WORKING_BASELINES_2026-07-02.md`: latest known-good controls and confounds.
- `Q6P_CONVERT_QUANTIZE_INFERENCE_MASTER_PLAYBOOK_2026-07-06.md`: canonical replay cookbook.
- `REPO_HANDOVER_MASTER_2026-07-07.md`: beginning-to-now narrative hand-over.
- `Q6P_RUN_INDEX_2026-07-07.md`: run-by-run quick index table.

---

## 3) What DRAW_THINGS_PATCH Is For

Purpose:

- preserve critical local modifications to Draw Things behavior,
- make those changes replayable after upstream drift or workspace reset,
- separate patch maintenance from experiment runs.

Patch strategy used in this repo:

- preferred path: unified patches under `DRAW_THINGS_PATCH/patches/*.patch`,
- fallback path: snapshot copies in `DRAW_THINGS_PATCH/` when patch apply is not clean.

Primary scripts:

- `tools/apply_drawthings_quant_patch.sh`
- `tools/generate_drawthings_quant_patches.sh`
- `tools/sync_drawthings_patch_bundle.sh`

Reference:

- `DRAW_THINGS_PATCH/README.md`

---

## 4) What The Devcontainer Setup Does

### 4.1 `devcontainer.json`

Main roles:

- configures container build via `.devcontainer/Dockerfile`,
- enables GPU passthrough runtime args (`nvidia.com/gpu=all`, `/dev/dxg`),
- mounts WSL library path for host GPU stack interop,
- sets Draw Things environment defaults and post-create hook.

### 4.2 `Dockerfile`

Main roles:

- installs core build and diagnostics dependencies (swift/c++ adjacent deps, sqlite3, ripgrep, etc.),
- creates `drawthings-grpc` wrapper that normalizes runtime env,
- prepends runtime library directories into `LD_LIBRARY_PATH` when present,
- adds `nvidia-smi` helper fallback locations.

### 4.3 `post-create.sh`

Main roles:

- bootstraps workspace dependencies,
- runs setup checks and convenience wiring,
- prepares shell aliases and startup helpers.

Key references:

- `.devcontainer/devcontainer.json`
- `.devcontainer/Dockerfile`
- `.devcontainer/post-create.sh`
- `WORKSPACE_MANUAL.md`

---

## 5) What The Initial Plan Was (And How It Evolved)

### 5.1 Original repository plan

Early repo plan was a robust Linux toolkit with:

- reproducible environment setup,
- patch persistence across upstream drift,
- repeatable conversion/quantization workflows,
- script-first experiment discipline,
- and artifactized run evidence.

Foundation context:

- `REPO_MIGRATION_GUIDE.md`
- `WORKSPACE_MANUAL.md`

### 5.2 Q6P technical investigation plan

Root-cause plan (initially formalized in June) targeted a generic custom LTX2.3 pipeline, not one-off patching:

1. lock baselines and acceptance gates,
2. instrument first divergence,
3. isolate causal differences,
4. validate policy deltas predictably,
5. productize safe conversion/quantization path.

Plan source:

- `Q6P_ROOT_CAUSE_PLAN.md`

### 5.3 Current state of that plan

- run history is now deep and highly instrumented,
- tracked row-level mismatch closure has been pushed very far,
- but quality gap remains, implying remaining semantic/config/runtime-path divergence.

Current references:

- `Q6P_WORKING_BASELINES_2026-07-02.md`
- `Q6P_CONVERT_QUANTIZE_INFERENCE_MASTER_PLAYBOOK_2026-07-06.md`
- `REPO_HANDOVER_MASTER_2026-07-07.md`

---

## 6) Tools Catalog (Purpose Of Each Tool In `tools/`)

Notes:

- Purpose lines are derived from script names, code structure, and documented usage.
- Rows marked "(inferred)" are semantically clear from naming/flow but should still be verified against script help when used.

### 6.1 Conversion And Quantization Core

| Path | Type | Purpose | Typical Inputs | Typical Outputs |
|---|---|---|---|---|
| `tools/dt_convert_model.sh` | shell | Wrapper around model conversion with monitoring and optional validation. | source model, profile/model name, output path | converted checkpoint + logs |
| `tools/dt_quantize_model.sh` | shell | Wrapper around model quantizer with progress/monitor behavior and codec controls. | f16 checkpoint + target codec | quantized checkpoint + logs |
| `tools/run_convert_safetensors_to_f16.sh` | shell | End-to-end safetensors -> f16 conversion flow with optional checks. | safetensors input, output file | f16 checkpoint + conversion logs |

### 6.2 Runtime Canary And Matrix Validation

| Path | Type | Purpose | Typical Inputs | Typical Outputs |
|---|---|---|---|---|
| `tools/run_q6p_canary_once.sh` | shell | Single canary inference run with strict/bounded gate options. | model key, timeout, seed/size/steps flags | client/server logs + pass/fail summary |
| `tools/run_10e_v1_model_validity_matrix.sh` | shell | Structured validity matrix for 10_e_v1 f16/q6p branches. | f16/q6p paths, safetensors source | matrix report + canary artifacts |
| `tools/run_ltx23_model_validity_matrix.sh` | shell | Generic LTX2.3 validity matrix with strict gates. | baseline/candidate model paths | matrix summary + per-case logs |
| `tools/run_q6p_strict_stability_matrix.sh` | shell | Multi-seed/multi-size strict stability matrix runner. | model key + seed/size sets | matrix report + case logs |
| `tools/run_ltx23_first_divergence_stage.sh` | shell | Stage-level first-divergence comparison orchestrator. | baseline/candidate checkpoints | divergence report files |
| `tools/run_q6p_restart_per_repeat_textclip_matrix.sh` | shell | Restart-per-repeat text/clip matrix runner. | model + text/clip variants | matrix logs |
| `tools/run_q6p_restart_per_repeat_textgate_boundary_matrix.sh` | shell | Restart matrix focusing text-gate boundary conditions. | model + text-gate variants | matrix logs |
| `tools/run_q6p_restart_per_repeat_textgate_clip_matrix.sh` | shell | Restart matrix combining text-gate and clip dimensions. | model + gate/clip variants | matrix logs |
| `tools/run_q6p_warm_server_mod_auto_repeats.sh` | shell | Warm-server repeat experiment wrapper for modifier/autoencoder toggles. | model + control flags | repeat-run summaries |

### 6.3 Checkpoint Probe Tools (Non-Destructive Diagnostics)

| Path | Type | Purpose | Typical Inputs | Typical Outputs |
|---|---|---|---|---|
| `tools/dt_probe_ckpt_deep_diff.py` | python | Deep shared-tensor metadata/dim/data diff between checkpoints. | file + baseline checkpoints | JSON/markdown diff reports |
| `tools/dt_probe_ckpt_meta_len_rowwise.py` | python | Row-wise metadata/length mismatch probe (blob-safe). | file + baseline checkpoints | row-wise mismatch JSON/TSV |
| `tools/dt_probe_ckpt_equal_len_payload_mismatch.py` | python | Detect equal-length payload signature divergence. | file + baseline checkpoints | mismatch-name list + report |
| `tools/dt_probe_ckpt_targeted_content.py` | python | Targeted family/prefix content parity probe. | file + baseline + name/prefix filters | targeted diff reports |
| `tools/dt_compare_ckpt_tensor_manifests.py` | python | Compare exported manifests and identify first mismatch fields. | two manifest JSONL files | compare JSON + summaries |
| `tools/dt_export_ckpt_tensor_manifest.py` | python | Export deterministic tensor manifest with bounded signatures. | checkpoint file | manifest JSONL |

### 6.4 Checkpoint Alignment And Repair (Mutation)

| Path | Type | Purpose | Typical Inputs | Typical Outputs |
|---|---|---|---|---|
| `tools/dt_align_ckpt_metadata.py` | python | Align metadata columns from baseline into target. | file + baseline + names | patched target checkpoint |
| `tools/dt_patch_ckpt_metadata_subset.py` | python | Patch metadata subset for selected tensor names. | file + baseline + names file | patched target checkpoint |
| `tools/dt_align_ckpt_content_subset.py` | python | Copy selected dim/data payloads from baseline to target. | file + baseline + names file(s) | patched target checkpoint |
| `tools/dt_align_ckpt_payload_subset.py` | python | Payload subset alignment by selected names/groups. | file + baseline + names | patched target checkpoint |
| `tools/dt_reorder_ckpt_readable_rows.py` | python | Reorder readable rows to baseline row order. | file + baseline checkpoints | reordered checkpoint |
| `tools/dt_build_q6p_dimfix_names.py` | python | Build dimfix candidate name sets from mismatch topology. | file + baseline/ref checkpoints | names txt for repair scripts |
| `tools/run_q6p_inplace_dimfix_from_f16.sh` | shell | In-place dimfix orchestration from f16 baseline and optional canary. | target q6p + source f16 + names | patched model + canary artifacts |
| `tools/run_q6p_payloadfix_recursive.sh` | shell | Recursive payloadfix pipeline with split-on-failure strategy. | target + baseline + names | patched model + progress logs |
| `tools/run_q6p_equal_len_payloadfix.sh` | shell | Payloadfix branch focused on equal-length mismatch sets. | target + baseline + probe args | patched model + mismatch logs |
| `tools/run_q6p_window_sig_payloadfix.sh` | shell | (inferred) window-signature-based payloadfix orchestration. | target + baseline + signature args | patched model + signature logs |
| `tools/run_q6p_highlimit_row_patch_canary.sh` | shell | High-limit sqlite single-row patch with canary validation. | target + baseline + row name | patch log + canary logs |
| `tools/run_q6p_highlimit_rows_patch_canary.sh` | shell | High-limit sqlite multi-row patch with canary validation. | target + baseline + rows file | patch log + canary logs |
| `tools/run_q6p_restore_postpatch_highlimit.sh` | shell | Recovery flow: restore, postpatch, high-limit fallback, quick check. | source/target model paths + tag | restored/patched model + logs |
| `tools/run_clipfix2_postpatch_q6p.sh` | shell | Apply clipfix2-style postpatch logic to q6p target safely. | target + baseline | patched checkpoint + logs |
| `tools/run_clipfix2_replay_q6p.sh` | shell | Replay q6p regeneration branch with safety checks and validation. | input f16 + output q6p + baseline | regenerated q6p + sanity logs |

### 6.5 Alias/Schema Investigation Scripts

| Path | Type | Purpose | Typical Inputs | Typical Outputs |
|---|---|---|---|---|
| `tools/run_custom_alias_schema_probe.sh` | shell | Probe custom.json field combinations under strict canary gates. | model entry params + matrix mode | per-case results + summary |
| `tools/run_custom_alias_resolution_probe.sh` | shell | Probe alias resolution/winner behavior under overlapping entries. | model args + alias variants | resolution-context matrix logs |

### 6.6 API, Media, And Utility Tools

| Path | Type | Purpose | Typical Inputs | Typical Outputs |
|---|---|---|---|---|
| `tools/dt_api_client.py` | python | General gRPC API client used by runner scripts and ad-hoc checks. | host + API command args | generated payload binaries + stdout summaries |
| `tools/dt_generate_video.sh` | shell | Prompt-driven video generation helper around API client calls. | prompt/model/run settings | run folder with video/tensor outputs |
| `tools/dt_tensor_to_playable.py` | python | Convert raw tensor binaries to PNG/GIF/MP4/WAV/OGG for inspection. | tensor bin files | playable media files |
| `tools/dt_video_test_run.sh` | shell | Small test-run harness for quick generation checks. | model key + run flags | test run artifacts |
| `tools/dt_make_config.py` | python | Build request config payloads from structured args. | config flags | generated config payload |
| `tools/dt_preflight_safetensors.py` | python | Quick integrity/preflight checks on safetensors source files. | safetensors file | preflight report |
| `tools/dt_model_requirements.py` | python | Resolve/download model dependencies with verification. | model names/specs | downloaded files + audit output |
| `tools/dt_models_cli.sh` | shell | Wrapper around model listing/downloading helper commands. | list/download/ensure operations | model audit/download logs |
| `tools/dt_safe_activate_model.sh` | shell | Safely activate custom model alias and sanity-check it. | model file + alias fields | updated alias config + canary evidence |
| `tools/install_grpcurl.sh` | shell | Install grpcurl helper binary for debugging endpoints. | optional version | local grpcurl binary |

### 6.7 Patch-Bundle Maintenance Scripts

| Path | Type | Purpose | Typical Inputs | Typical Outputs |
|---|---|---|---|---|
| `tools/apply_drawthings_quant_patch.sh` | shell | Apply unified patches (with fallback strategy) into working source tree. | patch bundle + repo tree | patched draw-things-community tree |
| `tools/generate_drawthings_quant_patches.sh` | shell | Regenerate patch files/snapshots from current working tree changes. | patched source tree | updated patch bundle artifacts |
| `tools/sync_drawthings_patch_bundle.sh` | shell | Sync snapshot files and regenerate patch bundle in one step. | DRAW_THINGS_PATCH snapshots | synchronized sources + refreshed patches |

### 6.8 Data Inputs Used By Tooling

| Path | Type | Purpose |
|---|---|---|
| `tools/patch_sets/*.txt` | data | Canonical name lists used by patch/alignment scripts (dimfix, payloadfix, metadata subsets, etc.). |

---

## 7) Markdown Knowledge Map: Where To Read More By Topic

### 7.1 Setup, Environment, And Repo Structure

- `README.md`
- `WORKSPACE_MANUAL.md`
- `REPO_MIGRATION_GUIDE.md`
- `DRAW_THINGS_PATCH/README.md`

### 7.2 End-To-End Investigation Narrative

- `REPO_HANDOVER_MASTER_2026-07-07.md`
- `Q6P_CONVERT_QUANTIZE_INFERENCE_MASTER_PLAYBOOK_2026-07-06.md`
- `Q6P_WORKING_BASELINES_2026-07-02.md`
- `Q6P_RUN_LOG.md`
- `Q6P_RUN_INDEX_2026-07-07.md`

### 7.3 Historical Root-Cause And Milestone Reports

- `CONVERSION_TOOL_FINDINGS_2026-05-28.md`
- `Q6P_HANDOFF_FINDINGS_2026-06-08.md`
- `Q6P_CONTINUATION_RUNBOOK_2026-06-08.md`
- `Q6P_STATUS_UPDATE_2026-06-09.md`
- `Q6P_ROOT_CAUSE_PLAN.md`
- `CLIP_ENCODER_INVESTIGATION_2026-06-02.md`

### 7.4 If You Need The Original Strategic Intent

- `REPO_MIGRATION_GUIDE.md` (repo creation and maintenance strategy)
- `Q6P_ROOT_CAUSE_PLAN.md` (technical root-cause-first strategy)

---

## 8) Recommended Reading Order For A New Contributor

1. `README.md`
2. `WORKSPACE_MANUAL.md`
3. `REPO_NAVIGATION_AND_TOOLS_CATALOG_2026-07-07.md` (this file)
4. `REPO_HANDOVER_MASTER_2026-07-07.md`
5. `Q6P_WORKING_BASELINES_2026-07-02.md`
6. `Q6P_CONVERT_QUANTIZE_INFERENCE_MASTER_PLAYBOOK_2026-07-06.md`
7. `Q6P_RUN_INDEX_2026-07-07.md`
8. `Q6P_RUN_LOG.md`

---

## 9) Practical Note

When using tool scripts in active debugging, prefer script-first reproducible flows and keep all experiment outputs under `output/` with timestamped tags. That practice is what made this repository hand-over possible.

---

## 10) Root-Level Inventory (Detailed)

This section maps the most important top-level items in this repository.

| Path | Kind | Why It Exists | Typical Consumer |
|---|---|---|---|
| `README.md` | markdown | Fast entrypoint and top-level workflows. | Everyone |
| `WORKSPACE_MANUAL.md` | markdown | Operational workspace manual (setup, patch flow, build flow). | Operators |
| `REPO_MIGRATION_GUIDE.md` | markdown | Original repo-splitting and hygiene strategy. | Maintainers |
| `REPO_HANDOVER_MASTER_2026-07-07.md` | markdown | Beginning-to-now strategic and testing hand-over. | New/returning investigators |
| `REPO_NAVIGATION_AND_TOOLS_CATALOG_2026-07-07.md` | markdown | Where-to-find-what + tools purpose reference (this file). | New/returning investigators |
| `Q6P_RUN_LOG.md` | markdown | Append-only run chronology and canonical artifacts. | Investigators |
| `Q6P_RUN_INDEX_2026-07-07.md` | markdown | Quick-linked index for Run 001..117. | Investigators |
| `Q6P_WORKING_BASELINES_2026-07-02.md` | markdown | Active known-good controls, confounds, and current hypothesis. | Investigators |
| `Q6P_CONVERT_QUANTIZE_INFERENCE_MASTER_PLAYBOOK_2026-07-06.md` | markdown | Replay cookbook and branch protocol. | Investigators |
| `Q6P_ROOT_CAUSE_PLAN.md` | markdown | Original root-cause-first strategy and verification gates. | Investigators |
| `Q6P_HANDOFF_FINDINGS_2026-06-08.md` | markdown | Mid-investigation handoff snapshot under low-space constraints. | Investigators |
| `Q6P_CONTINUATION_RUNBOOK_2026-06-08.md` | markdown | Command-first continuation runbook for low-space operations. | Investigators |
| `Q6P_STATUS_UPDATE_2026-06-09.md` | markdown | Point-in-time status and run deltas. | Investigators |
| `CONVERSION_TOOL_FINDINGS_2026-05-28.md` | markdown | Early conversion findings and structural/runtime divergence evidence. | Investigators |
| `CLIP_ENCODER_INVESTIGATION_2026-06-02.md` | markdown | Clip/text path investigation for LTX2.3 runtime behavior. | Investigators |
| `.devcontainer/` | directory | Reproducible environment definition for this repo. | Operators |
| `.vscode/settings.json` | json | Workspace-level VS Code behavior (`podman` path). | Operators |
| `DRAW_THINGS_PATCH/` | directory | Patch bundle source of truth (unified + fallback snapshots). | Maintainers |
| `draw-things-community/` | directory | Build/test target source tree for Draw Things. | Maintainers, investigators |
| `tools/` | directory | All operational scripts and probes. | Investigators |
| `dt-models/` | directory | Local model and checkpoint cache. | Investigators |
| `output/` | directory | Artifact sink for all run outputs. | Investigators |
| `requirements.txt` | text | Python fallback dependency set. | Operators |
| `requirements-cuda.txt` | text | Main project Python requirements for this workflow. | Operators |
| `requirements-drawthings-tools.txt` | text | Python dependencies used by tool scripts. | Operators |
| `example_custom.json` | json | Example custom model schema reference. | Investigators |
| `LICENSE` | text | Repository license. | Everyone |

---

## 11) Devcontainer Deep Reference

### 11.1 Runtime contract (`.devcontainer/devcontainer.json`)

| Setting | Current Value/Behavior | Why It Matters |
|---|---|---|
| Base build | `Dockerfile` in `.devcontainer/` | Keeps dependencies and wrapper behavior reproducible. |
| GPU run args | `--device nvidia.com/gpu=all`, `--device /dev/dxg` | Enables GPU access under host/container bridge scenarios. |
| WSL lib mount | bind mount `/usr/lib/wsl` readonly | Exposes host-side runtime libs needed by certain stacks. |
| NVIDIA env | `NVIDIA_VISIBLE_DEVICES=all`, `NVIDIA_DRIVER_CAPABILITIES=compute,utility` | Aligns container runtime with GPU compute use case. |
| Container env | `DRAWTHINGS_MODEL_DIR`, repo URL/ref, cert env vars | Centralizes model cache location and bootstrap expectations. |
| Post-create | `bash .devcontainer/post-create.sh` | Guarantees scripted setup on container creation. |
| VS Code default shell | `bash -l` | Ensures login shell behavior and aliases are available. |

### 11.2 Image contract (`.devcontainer/Dockerfile`)

Major details:

- Base image is Draw Things gRPC server CLI image.
- Installs core packages used by this repo and workflows:
  - build tooling (`build-essential`, `pkg-config`, `zlib1g-dev`),
  - Python tooling (`python3`, `pip`, `venv`),
  - sqlite tooling (`sqlite3`, `libsqlite3-dev`),
  - search tooling (`ripgrep`),
  - and required runtime/system packages.
- Creates `drawthings-grpc` helper wrapper that:
  - sets stable defaults (`address`, `port`, `gpu`, model dir),
  - prepends runtime libraries into `LD_LIBRARY_PATH` if present,
  - launches `gRPCServerCLI` with consistent flags by default.
- Creates resilient `nvidia-smi` shim with fallback paths.

### 11.3 Bootstrap contract (`.devcontainer/post-create.sh`)

Execution stages:

1. Verifies core runtime/tooling prerequisites (`libxml2`, linker, gcc/g++).
2. Ensures draw-things-community tree exists (clone + optional ref checkout).
3. Creates `.venv` and installs Python requirements.
4. Ensures `rg`, `grpcurl`, and `flatc` availability.
5. Applies patch bundle via `tools/apply_drawthings_quant_patch.sh`.
6. Optionally prebuilds selected Swift products when enabled.
7. Verifies `gRPCServerCLI` and `drawthings-grpc` commands.
8. Performs GPU visibility checks (`/dev/nvidia*` or `/dev/dxg`, `nvidia-smi`).
9. Writes shell aliases (`drawthings-start`, `dt-venv`) into `.bashrc` block.

---

## 12) DRAW_THINGS_PATCH Deep Internals

### 12.1 Patch assets

Current unified patch files in `DRAW_THINGS_PATCH/patches/`:

- `draw-things-community.patch`
- `ccv.patch`
- `s4nnc.patch`

### 12.2 Why both unified and snapshot formats are kept

- Unified patch path is cleaner and tracks drift via patch hunks.
- Snapshot fallback path prevents total blockage when upstream drift makes unified apply fail.

### 12.3 Typical maintenance loop

1. Edit patched source files.
2. Sync snapshots and regenerate patch set:
   - `tools/sync_drawthings_patch_bundle.sh`
3. Re-apply patches in a clean tree to verify replayability:
   - `tools/apply_drawthings_quant_patch.sh`
4. Build verification on required products.

### 12.4 High-risk patching mistakes this repo tries to avoid

- silent drift where snapshots and unified patches disagree,
- patch apply that succeeds but omits critical files,
- mixing large workflow changes and patch maintenance in one untraceable step.

---

## 13) Testing And Evidence Conventions

### 13.1 Acceptance model used in this repo

A branch is treated as working only when all of the following pass:

1. Stream completion and final payload gates.
2. Runtime health (no server crash signatures).
3. Visual coherence (not noise-only output) under matched settings.

### 13.2 Output artifact conventions

Expected run artifacts normally include:

- `summary` file (txt/tsv/md),
- `client.log`,
- `server.log`,
- probe reports (json/md/tsv),
- media outputs when generation succeeded.

### 13.3 Run tag conventions

Common tag style:

- `runNNN_<branch_name>_<YYYYMMDD_HHMMSS>`

Benefits:

- fast grep/searchability,
- deterministic provenance,
- easy side-by-side branch comparison.

### 13.4 Why `Q6P_RUN_LOG.md` is append-only

- preserves chain-of-custody for debugging decisions,
- avoids retroactive interpretation edits,
- supports reproducibility and safe hand-offs.

---

## 14) Troubleshooting Matrix (Where To Look First)

| Symptom | First Checks | Primary Scripts/Docs |
|---|---|---|
| No GPU visibility in container | `/dev/nvidia*` or `/dev/dxg`, `nvidia-smi` | `.devcontainer/devcontainer.json`, `.devcontainer/post-create.sh`, `Q6P_WORKING_BASELINES_2026-07-02.md` |
| Canary socket closed | inspect server crash signature | `tools/run_q6p_canary_once.sh`, `output/<run>/server.log`, `Q6P_RUN_LOG.md` |
| `TextEncoder.encodeLTX2` illegal instruction | alias/schema and text path settings | `tools/run_custom_alias_schema_probe.sh`, `tools/run_custom_alias_resolution_probe.sh`, `CLIP_ENCODER_INVESTIGATION_2026-06-02.md` |
| `ccv_nnc_tensor_read` loader crash | checkpoint content/serialization branch | `tools/dt_probe_ckpt_*`, `tools/dt_align_ckpt_*`, `Q6P_WORKING_BASELINES_2026-07-02.md` |
| Runtime pass but noisy output | quality gate mismatch, semantic/config path divergence | `tools/dt_tensor_to_playable.py`, `Q6P_CONVERT_QUANTIZE_INFERENCE_MASTER_PLAYBOOK_2026-07-06.md`, `REPO_HANDOVER_MASTER_2026-07-07.md` |
| Patch apply drift/fail | patch bundle out-of-sync | `tools/sync_drawthings_patch_bundle.sh`, `tools/apply_drawthings_quant_patch.sh`, `DRAW_THINGS_PATCH/README.md` |
| Difficultly continuing after pause | find last stable branch and artifact roots | `Q6P_RUN_INDEX_2026-07-07.md`, `Q6P_RUN_LOG.md`, `Q6P_WORKING_BASELINES_2026-07-02.md` |

---

## 15) Master Markdown Index (All Topical Docs)

This table is the direct answer to "where to find extra information on all topics".

| Document | Primary Topic | Read This When | Depth |
|---|---|---|---|
| `README.md` | High-level workspace usage | Starting fresh quickly | Short |
| `WORKSPACE_MANUAL.md` | Full operations manual | You need setup/build/patch commands | Medium |
| `REPO_MIGRATION_GUIDE.md` | Repo structure and migration strategy | You are reorganizing or splitting repo content | Medium |
| `DRAW_THINGS_PATCH/README.md` | Patch-bundle mechanics | Patch apply/regenerate/sync questions | Short |
| `CONVERSION_TOOL_FINDINGS_2026-05-28.md` | Early conversion findings and defect history | You need origin story of converter/serialization issues | Long |
| `CLIP_ENCODER_INVESTIGATION_2026-06-02.md` | LTX2.3 clip/text path analysis | You are debugging encodeLTX2 or clip/schema behavior | Medium |
| `Q6P_HANDOFF_FINDINGS_2026-06-08.md` | Midstream low-space findings snapshot | You need historical context for in-place repair era | Medium |
| `Q6P_CONTINUATION_RUNBOOK_2026-06-08.md` | Command-first continuation steps | You need operational replay under constraints | Medium |
| `Q6P_STATUS_UPDATE_2026-06-09.md` | Point-in-time run update | You want delta understanding around that date | Medium |
| `Q6P_ROOT_CAUSE_PLAN.md` | Root-cause-first strategic plan | You need the original intended method and verification gates | Medium |
| `Q6P_RUN_LOG.md` | Canonical run ledger | You need exact chronology and artifacts | Very Long |
| `Q6P_RUN_INDEX_2026-07-07.md` | Fast index into run ledger | You need quick run navigation | Short |
| `Q6P_WORKING_BASELINES_2026-07-02.md` | Current known-good controls and confounds | You are about to run new experiments | Long |
| `Q6P_CONVERT_QUANTIZE_INFERENCE_MASTER_PLAYBOOK_2026-07-06.md` | Replay cookbook and branch protocol | You want exact reproducibility recipes | Long |
| `REPO_HANDOVER_MASTER_2026-07-07.md` | Full beginning-to-now narrative hand-over | You need complete strategic context | Long |
| `REPO_NAVIGATION_AND_TOOLS_CATALOG_2026-07-07.md` | Repo map + tool catalog + topic map | You need findability and tool purpose lookup | Long |

---

## 16) Documentation Maintenance Standards

To keep future handovers strong:

1. Keep `Q6P_RUN_LOG.md` append-only.
2. Add artifact root paths in every run entry.
3. Update baselines doc when interpretation changes.
4. Keep tool docs script-first and command-reproducible.
5. Link new docs from `README.md` and master hand-over docs.
6. Prefer explicit run tags and fixed profile settings in notes.

---

## 17) One-Page Quick Start For Future Sessions

1. Read `Q6P_WORKING_BASELINES_2026-07-02.md` first.
2. Jump through `Q6P_RUN_INDEX_2026-07-07.md` to latest relevant run.
3. Open corresponding section in `Q6P_RUN_LOG.md` for exact artifacts/commands.
4. Use `Q6P_CONVERT_QUANTIZE_INFERENCE_MASTER_PLAYBOOK_2026-07-06.md` for replay.
5. Use this catalog to locate specific tools and operational docs quickly.

This sequence minimizes re-discovery time and reduces repeated investigation branches.
