# Repository Master Hand-Over (2026-07-07)

## 1) Intent Of This Document

This is the durable beginning-to-now hand-over for this repository.

It is designed for a new or returning operator to understand:

- what this repo is for,
- how the workspace is structured,
- how model conversion and quantization are implemented,
- how testing evolved from the first investigations to the latest runs,
- what is proven versus unresolved,
- and exactly how to continue safely.

This update is documentation-only. No new conversion, quantization, or inference execution is performed in this hand-over step.

Companion navigation references:

- `REPO_NAVIGATION_AND_TOOLS_CATALOG_2026-07-07.md` is the canonical single-share root document.
- `REPO_NAVIGATION_AND_TOOLS_CATALOG_2026-07-07.md` (folder map + tools purpose catalog)
- `Q6P_RUN_INDEX_2026-07-07.md` (run-by-run quick index)

---

## 2) Repository Mission

Primary mission:

- provide a Linux-first, reproducible toolkit for Draw Things model workflows,
- support conversion and quantization of custom LTX2.3 checkpoints,
- and establish a repeatable testing contract that distinguishes runtime survival from real output quality.

Current central objective for the active custom model effort:

- achieve coherent first-frame output (and then full video reliability) for custom converted/quantized models, not only runtime pass.

---

## 3) Workspace Architecture

Top-level components and purpose:

- `.devcontainer/`
  - Reproducible Linux devcontainer bootstrap.
  - GPU/runtime prerequisites and wrapper setup.

- `tools/`
  - Canonical script and probe layer.
  - Conversion, quantization, manifest/probe tooling, canary runners, and scripted test matrices.

- `DRAW_THINGS_PATCH/`
  - Persistent patch bundle for Draw Things and dependencies.
  - Unified patch files plus snapshot fallback files.

- `draw-things-community/`
  - Source tree used for local builds and patch application.

- `dt-models/`
  - Local checkpoint storage, including official controls and custom candidates.

- `output/`
  - Canonical artifact root for every experiment branch and run.

- Root-level Q6P docs
  - Investigation history, run chronology, baselines, continuation plans, and now this master hand-over.

Operational guidance docs:

- `README.md`
- `WORKSPACE_MANUAL.md`
- `Q6P_RUN_LOG.md`
- `Q6P_WORKING_BASELINES_2026-07-02.md`
- `Q6P_CONVERT_QUANTIZE_INFERENCE_MASTER_PLAYBOOK_2026-07-06.md`

---

## 4) Historical Timeline From The Beginning

### 4.1 Foundation Period (Migration And Toolkit Setup)

Initial repository hardening and migration guidance were established in:

- `REPO_MIGRATION_GUIDE.md` (2026-05-19)

This period defined:

- what should be versioned,
- what must stay local/generated,
- how patch assets are maintained,
- and how to bootstrap the workspace safely.

### 4.2 Early Conversion Investigation Period

The first deep investigation was captured in:

- `CONVERSION_TOOL_FINDINGS_2026-05-28.md`

Key outcomes from this period:

- importer mapping defects were fixed,
- keyset parity (5746 tensors) with official was achieved,
- metadata parity was later improved,
- but runtime instability remained for custom artifacts,
- proving structural parity alone was not sufficient.

### 4.3 Systematic Run-Log Period (Append-Only)

From Run 001 onward, experiment control moved to append-only run accounting:

- `Q6P_RUN_LOG.md`

Chronology starts at:

- Run 001: `Q6P_RUN_LOG.md#L56`

Current latest documented run:

- Run 117: `Q6P_RUN_LOG.md#L4898`

### 4.4 Current Maturity Period

Latest status and operations are codified across:

- `Q6P_WORKING_BASELINES_2026-07-02.md`
- `Q6P_CONVERT_QUANTIZE_INFERENCE_MASTER_PLAYBOOK_2026-07-06.md`
- and this document (2026-07-07)

---

## 5) Testing Framework: How Validation Is Actually Done

Testing in this repo is not a single pass/fail command. It is a layered gate stack.

### 5.1 Layer A: Environment And Runtime Preconditions

Purpose:

- confirm GPU/runtime path health before model conclusions.

Typical checks:

- device visibility (`/dev/nvidia*` or WSL-equivalent device mapping),
- `nvidia-smi` health,
- runtime library path correctness,
- wrapper binary behavior.

Known confound:

- environment failure can mimic model regression.

### 5.2 Layer B: Structural/Serialization Integrity

Purpose:

- validate checkpoint internals before inference.

Typical checks:

- sqlite quick check,
- tensor row counts,
- profile-specific conversion validation,
- metadata/dim/data parity probes against baseline.

### 5.3 Layer C: Runtime Canary Behavior

Purpose:

- verify stream starts, advances, and completes under bounded or strict final gates.

Canonical script:

- `tools/run_q6p_canary_once.sh`

Signal examples captured repeatedly:

- pass with full completion and final output,
- timeout with no first response,
- socket closed with server crash,
- crash signatures tied to text-path or tensor-read paths.

### 5.4 Layer D: Output Quality Validation

Purpose:

- enforce the true objective: coherent outputs, not only technical completion.

Methods:

- convert tensor payloads to PNG/GIF/MP4,
- visual inspection,
- quantitative grayscale/entropy discriminators,
- matched official-vs-custom A/B comparisons with same settings.

Non-negotiable lesson:

- runtime pass and even deep row-level parity can still produce unusable/noisy outputs.

---

## 6) Testing Evolution By Phase (Runs 001-117)

### Phase 1: Low-Space In-Place Repair Loops (Runs 001-006)

Range:

- `Q6P_RUN_LOG.md#L56` through `Q6P_RUN_LOG.md#L256`

What was tested:

- in-place q6p dim/data/metadata patching under strict storage constraints,
- recursive micro-batching to survive D-state and large-row behavior,
- bounded canary checks after each repair branch.

What was learned:

- row-wise metadata/length parity can be exhausted,
- but pre-stream failure/stall can still persist.

### Phase 2: Scripted Matrix Discipline (Runs 007-018)

Range:

- `Q6P_RUN_LOG.md#L317` through `Q6P_RUN_LOG.md#L736`

What was tested:

- scripted validity matrix adoption,
- equal-length payload mismatch probes,
- family-targeted repair branches,
- high-limit row repair branches for unreadable/oversized rows,
- stricter final-mode timeout policy checks.

What was learned:

- scripted reproducibility improved signal quality,
- but deterministic loader-path failures remained.

### Phase 3: First-Divergence Instrumentation And Policy Tracing (Runs 019-024)

Range:

- `Q6P_RUN_LOG.md#L786` through `Q6P_RUN_LOG.md#L1069`

What was tested:

- stage-localized manifest exports and compares,
- quantizer trace capture,
- old-vs-new q6p divergence localization,
- strict stability matrices over multiple seeds/sizes.

What was learned:

- divergence can be isolated to specific stages/signals,
- traced regeneration branch produced temporary strict pass improvements.

### Phase 4: Alias/Schema/Resolution Failure Surface Mapping (Runs 025-083)

Range:

- `Q6P_RUN_LOG.md#L1108` through `Q6P_RUN_LOG.md#L3448`

What was tested:

- custom alias schema permutations,
- model resolution collisions,
- file-key vs alias-name control paths,
- text-encoder/companion field permutations,
- wrapper-vs-source runtime confounds.

What was learned:

- custom entry schema and resolution semantics are independent failure surfaces,
- multiple deterministic crash classes were isolated,
- passing control paths can fail when specific alias overlap conditions are active.

### Phase 5: Persistent Controls And Quality Discriminators (Runs 084-105)

Range:

- `Q6P_RUN_LOG.md#L3497` through `Q6P_RUN_LOG.md#L4181`

What was tested:

- persistent raw-key controls,
- matched seed sweeps,
- full-gate quality comparisons,
- quantitative image-metric discrimination,
- converter-focused mismatch diagnostics.

What was learned:

- official controls remain coherent under matched settings,
- custom branches can be runtime-stable while still visual-fail,
- broad converter-side divergence exists in targeted families.

### Phase 6: Progressive Content Alignment And Parity Closure (Runs 106-117)

Range:

- `Q6P_RUN_LOG.md#L4229` through `Q6P_RUN_LOG.md#L4898`

What was tested:

- connector + DIT subset alignment expansions,
- semantic targeting sets,
- post-GPUfix strict A/B revalidation,
- remaining-set and metadata-set closure,
- final unreadable-row high-limit repair,
- full tracked shared-set parity closure.

What was learned:

- tracked parity reached 5746/5746 for selected shared names,
- strict runtime can still pass with stable shape,
- output quality remained garbled/noisy,
- therefore the remaining blocker likely sits outside currently instrumented row-level mismatch classes.

---

## 7) Crash And Failure Signature Catalogue

High-frequency signatures seen across investigation history:

1. Tensor-read loader crash
- Typical stack head includes:
  - `ccv_nnc_tensor_read`
  - `ccv_cnnp_model_read`
- Often surfaces as client-side socket close.

2. Text encoder illegal instruction
- Typical stack context includes:
  - `TextEncoder.encodeLTX2`
- Highly correlated with certain alias/schema and text-path configurations.

3. Pre-stream timeout/stall
- No response #1 observed within bounded gate,
- post-echo may still succeed,
- indicates request accepted but generation path not producing stream payloads in time.

Operational rule:

- classify branch by signature before interpreting quality behavior.

---

## 8) Current Ground Truth (As Of 2026-07-07)

What is now established:

- environment can be recovered for strict test profile when runtime library path is explicit,
- official control is healthy in matched strict profiles,
- patched custom candidate can be strict-pass in runtime shape,
- tracked shared-set row-level parity has been pushed to full closure in the targeted set,
- output quality for patched custom branch remains garbled/noisy.

Most likely interpretation:

- unresolved issue is likely semantic/config/runtime-path level,
- not simply remaining row-level metadata/dim/data mismatch in the currently tracked set.

---

## 9) Artifact And Naming Conventions

Conventions used across testing:

- every experiment has a dedicated folder under `output/` with timestamped tag,
- each run records:
  - client log,
  - server log,
  - summary files,
  - probe outputs,
  - generated media (when available),
- canonical chronology references those artifact roots inside `Q6P_RUN_LOG.md`.

Examples of authoritative roots:

- `output/run110_fullgate_patched_vs_official_after_gpufix_20260706_121602`
- `output/run111_remaining_on_run110_retry_20260706_124412`
- `output/run112_meta_len_full_on_run110_20260706_125740`
- `output/run113_metadata_align_on_run110_20260706_125955`
- `output/run117_highlimit_single_row_retry_20260706_131612`

---

## 10) Script Inventory That Matters Most

High-impact scripts by purpose:

Environment/runtime and canaries:

- `tools/run_q6p_canary_once.sh`
- `tools/run_q6p_strict_stability_matrix.sh`
- `tools/run_ltx23_model_validity_matrix.sh`

Conversion and quantization:

- `tools/dt_convert_model.sh`
- `tools/dt_quantize_model.sh`
- `tools/run_convert_safetensors_to_f16.sh`

Checkpoint probes and compares:

- `tools/dt_probe_ckpt_meta_len_rowwise.py`
- `tools/dt_probe_ckpt_targeted_content.py`
- `tools/dt_probe_ckpt_equal_len_payload_mismatch.py`
- `tools/dt_export_ckpt_tensor_manifest.py`
- `tools/dt_compare_ckpt_tensor_manifests.py`

Checkpoint patch/repair flows:

- `tools/dt_align_ckpt_content_subset.py`
- `tools/dt_patch_ckpt_metadata_subset.py`
- `tools/run_q6p_highlimit_row_patch_canary.sh`
- `tools/run_q6p_highlimit_rows_patch_canary.sh`

Alias/resolution investigation:

- `tools/run_custom_alias_schema_probe.sh`
- `tools/run_custom_alias_resolution_probe.sh`

Media conversion and quality checks:

- `tools/dt_tensor_to_playable.py`

---

## 11) Documentation Map And Precedence

Use this order when resuming work:

1. `Q6P_WORKING_BASELINES_2026-07-02.md`
   - known-good controls and active hypotheses.

2. `Q6P_RUN_LOG.md`
   - append-only chronological source of truth.

3. `Q6P_CONVERT_QUANTIZE_INFERENCE_MASTER_PLAYBOOK_2026-07-06.md`
   - canonical replay and branch cookbook.

4. `REPO_HANDOVER_MASTER_2026-07-07.md` (this document)
   - full context and testing architecture overview.

5. Historical context docs:
   - `CONVERSION_TOOL_FINDINGS_2026-05-28.md`
   - `Q6P_HANDOFF_FINDINGS_2026-06-08.md`
   - `Q6P_CONTINUATION_RUNBOOK_2026-06-08.md`
   - `Q6P_ROOT_CAUSE_PLAN.md`

---

## 12) What Has Been Ruled Out Versus Still Open

High-confidence ruled out (within tested conditions):

- seed variation alone as a fix for current garbled branch,
- row-level metadata parity as a sufficient condition,
- tracked selected-set dim/data parity as a sufficient condition,
- fixing the final known unreadable row as a sufficient condition.

Still open and highest priority:

- semantic/configuration-path divergence not represented by current row-level probes,
- model-key/alias resolution interactions in specific schema conditions,
- remaining latent runtime-path confounds not visible in row-level parity metrics.

---

## 13) Safe Continuation Protocol (Planning Only)

When work resumes, continue with this discipline:

1. Confirm environment prerequisites before interpreting model behavior.
2. Reconfirm official control with strict full-gate settings first.
3. Run matched candidate gate with identical settings.
4. Convert outputs and evaluate quality, not only stream completion.
5. Record every branch and artifact in `Q6P_RUN_LOG.md` immediately.

Do not treat any single pass as definitive unless:

- stream completed,
- runtime stayed healthy,
- and visual output is coherent under matched controls.

---

## 14) Final Hand-Over Statement

This repository now contains a mature and highly-instrumented investigation stack.

The team has moved from ad-hoc fixes to deterministic, artifact-driven testing with strong controls. The unresolved problem is no longer basic structure mismatch; it is the deeper semantic/runtime-path gap that persists even after extensive row-level closure.

That is the correct frontier for the next execution cycle.
