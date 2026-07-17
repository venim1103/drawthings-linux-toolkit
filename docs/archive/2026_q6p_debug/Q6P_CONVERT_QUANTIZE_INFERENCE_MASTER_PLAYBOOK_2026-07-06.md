# Q6P Convert/Quantize/Inference Master Playbook (2026-07-06)

## 1) Purpose

This is the exact reproducibility and forensics playbook for this repository.

Companion master hand-over (full repository and testing narrative from beginning):
- `REPO_HANDOVER_MASTER_2026-07-07.md`

Canonical single-share root document:
- `REPO_NAVIGATION_AND_TOOLS_CATALOG_2026-07-07.md`

Goals:
- Re-run all critical environment, conversion, quantization, and inference checks exactly.
- Preserve the full chain of what we tried and what each branch proved.
- Keep a repeatable method for finding and fixing the core issue so custom models can be reliably converted, quantized, and inferred.

Primary run ledger source:
- `Q6P_RUN_LOG.md` (all run sections from Run 001 through Run 117)
- Latest post-Run110 sections are at:
  - Run 110T: `Q6P_RUN_LOG.md#L4695`
  - Run 111A: `Q6P_RUN_LOG.md#L4730`
  - Run 111: `Q6P_RUN_LOG.md#L4745`
  - Run 112: `Q6P_RUN_LOG.md#L4776`
  - Run 113: `Q6P_RUN_LOG.md#L4804`
  - Run 114: `Q6P_RUN_LOG.md#L4837`
  - Run 115: `Q6P_RUN_LOG.md#L4859`
  - Run 116: `Q6P_RUN_LOG.md#L4881`
  - Run 117: `Q6P_RUN_LOG.md#L4898`

Baseline synthesis source:
- `Q6P_WORKING_BASELINES_2026-07-02.md`

---

## 2) Current Ground Truth (As Of Run 117)

What is proven:
- Environment/runtime gate is recovered for the current strict profile when using the known-good library path.
- Official control (`ltx_2.3_22b_distilled_1.1_q6p.ckpt`) is PASS in-session.
- Patched custom branch (`10_e_v1_bf16_regen_0_q6p_run110_semantic_coreav.ckpt`) is also runtime PASS in strict mode.
- Known tracked row-level parity has been exhausted up to full shared set parity in Run 117:
  - `selected_tensors=5746`
  - metadata mismatches = 0
  - dim/data head mismatches = 0
  - small SHA mismatches = 0
  - `full_match=5746`
- Despite full tracked parity and runtime PASS, output remains RGB-noise/garbled for custom branch.

Implication:
- The remaining blocker is likely not in the currently instrumented row-level mismatch space.
- The core issue now likely sits in semantic/config/runtime pathing outside the row-level probes that were used to close 5746/5746 parity.

---

## 3) Locked Environment Configuration

### 3.1 Devcontainer Runtime (Podman + WSL GPU)

File:
- `.devcontainer/devcontainer.json`

Critical runtime args in use:
- `--device nvidia.com/gpu=all`
- `--device /dev/dxg`
- bind mount `/usr/lib/wsl` as readonly
- `NVIDIA_VISIBLE_DEVICES=all`
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility`

VS Code forces Podman via:
- `.vscode/settings.json` (`dev.containers.dockerPath = podman`)

### 3.2 Container Image Tooling

File:
- `.devcontainer/Dockerfile`

Persisted package additions include:
- `sqlite3`
- `libsqlite3-dev`

`drawthings-grpc` wrapper now prepends these library directories to `LD_LIBRARY_PATH` when present:
- `/usr/local/swift/usr/lib/swift/linux`
- `/usr/lib/wsl/lib`
- `/usr/local/cuda/targets/x86_64-linux/lib`
- `/usr/local/cuda/lib64`

### 3.3 Post-Create Checks

File:
- `.devcontainer/post-create.sh`

Preflight checks include:
- GPU device node visibility (`/dev/nvidia*` or `/dev/dxg`)
- `nvidia-smi` availability via multiple WSL fallback paths

---

## 4) Canonical Command Profiles

### 4.1 Strict Full-Gate Canary Profile (Control + Candidate)

Use this profile for all matched A/B runtime tests:

```bash
env LD_LIBRARY_PATH=/usr/local/swift/usr/lib/swift/linux:/usr/lib/wsl/lib:/usr/local/cuda/targets/x86_64-linux/lib \
bash tools/run_q6p_canary_once.sh \
  --model <model_key> \
  --tag <tag> \
  --timeout-sec 600 \
  --max-responses 0 \
  --require-complete-stream \
  --require-final-output \
  --width 256 --height 256 \
  --steps 8 --seed 4242 \
  --soft-fail
```

### 4.2 Convert Tensor Outputs To Playable Media

```bash
python3 tools/dt_tensor_to_playable.py \
  --image-bin <run_dir>/image_rXXXX_01.bin \
  --out-dir <out_dir> \
  --base-name <name>
```

### 4.3 Targeted Content Probe

```bash
python3 tools/dt_probe_ckpt_targeted_content.py \
  --file <candidate.ckpt> \
  --baseline <official.ckpt> \
  --names-file <name_list.txt> \
  --head-bytes 32 \
  --small-hash-limit 4096 \
  --out-json <out.json> \
  --out-md <out.md>
```

### 4.4 Content Subset Alignment

Important: current CLI uses `--file` (not `--target`).

```bash
python3 tools/dt_align_ckpt_content_subset.py \
  --file <candidate.ckpt> \
  --baseline <official.ckpt> \
  --dim-names-file <name_list.txt> \
  --data-names-file <name_list.txt> \
  --mode apply \
  --chunk-size 8 \
  --journal-mode delete \
  --min-free-gb 180 \
  --head-bytes 32
```

### 4.5 Metadata Subset Alignment

```bash
python3 tools/dt_patch_ckpt_metadata_subset.py \
  --file <candidate.ckpt> \
  --baseline <official.ckpt> \
  --names-file <name_list.txt> \
  --journal-mode delete
```

### 4.6 High-Limit Oversized Row Repair

Prerequisites:
- system `sqlite3` installed
- high-limit sqlite binary available at `output/sqlite_highlimit_build/sqlite-autoconf-3530200/sqlite3`

Build high-limit sqlite:

```bash
mkdir -p output/sqlite_highlimit_build
cd output/sqlite_highlimit_build
curl -L --fail -o sqlite-autoconf-3530200.tar.gz https://www.sqlite.org/2026/sqlite-autoconf-3530200.tar.gz
tar -xzf sqlite-autoconf-3530200.tar.gz
cd sqlite-autoconf-3530200
CFLAGS='-O2 -DSQLITE_MAX_LENGTH=2147482624 -DSQLITE_MAX_SQL_LENGTH=2147482624' ./configure --disable-shared --enable-static
make -j"$(nproc)" sqlite3
```

Patch one oversized row:

```bash
bash tools/run_q6p_highlimit_row_patch_canary.sh \
  --target <candidate.ckpt> \
  --baseline <official.ckpt> \
  --highlimit-sqlite output/sqlite_highlimit_build/sqlite-autoconf-3530200/sqlite3 \
  --row-name "__text_feature_extractor__[t-video_aggregate_embed-0-0]" \
  --skip-canary
```

---

## 5) Complete Attempted Strategy Map (Everything Tried)

### 5.1 Early Converter/Quantizer + Crash Isolation Era

Covered in detail by run sections:
- Run 001 through Run 034
- Core themes: conversion diagnostics, payload mismatch mapping, staged patch application, initial crash signatures, canary gates.

Reference:
- `Q6P_RUN_LOG.md#L56` through `Q6P_RUN_LOG.md#L1478`

### 5.2 Alias/Schema + Encoder-Path Matrix Era

Covered by:
- Run 035 through Run 083
- Core themes: alias schema permutations, text/clip pinning, source vs wrapper binary confounds, repeat/restart matrices.

Reference:
- `Q6P_RUN_LOG.md#L1479` through `Q6P_RUN_LOG.md#L3496`

### 5.3 Persistent Controls + Visual Quality Discriminator Era

Covered by:
- Run 084 through Run 105
- Core themes: raw-key controls, official-vs-custom persistent gates, matched seed sweeps, quantitative PNG metrics, converter-side row-wise diagnostics.

Reference:
- `Q6P_RUN_LOG.md#L3497` through `Q6P_RUN_LOG.md#L4228`

### 5.4 Progressive Content Alignment Era

Covered by:
- Run 106 through Run 110T
- Core themes: connector + DIT slice expansion, stratified DIT, semantic CoreAV targeting, environment recovery split, matched post-fix A/B and seed sweep.

Reference:
- `Q6P_RUN_LOG.md#L4229` through `Q6P_RUN_LOG.md#L4729`

### 5.5 Post-110 Exhaustive Closure Attempts (Newest)

Covered by:
- Run 111A through Run 117
- Exact sequence that was attempted:
  - full-copy all-names branch (aborted for runtime cost)
  - remaining-set in-place patch (success, still garbled)
  - full Run105 remaining-set patch (success, still garbled)
  - metadata full-set patch (success/full_match=5745, still garbled)
  - single missing-row patch attempts
  - direct SQL row replace attempts (schema error, then blob limit)
  - high-limit row repair (success/full_match=5746, still garbled)

Reference:
- `Q6P_RUN_LOG.md#L4730` through `Q6P_RUN_LOG.md#L4940`

---

## 6) Latest Branch-by-Branch Results (Run 111A -> Run 117)

1. Run 111A
Result: aborted pre-apply due heavy full-copy overhead.
Artifacts: `output/run111_all5746_20260706_123418/`

2. Run 111
Result: applied remaining 1216 names, strict PASS, output still garbled.
Artifacts: `output/run111_remaining_on_run110_retry_20260706_124412/`

3. Run 112
Result: applied remaining 1956 names from full 5745 set, strict PASS, output still garbled.
Artifacts: `output/run112_meta_len_full_on_run110_20260706_125740/`

4. Run 113
Result: metadata alignment over 5745 names, post-probe `full_match=5745`, strict PASS, output still garbled.
Artifacts: `output/run113_metadata_align_on_run110_20260706_125955/`

5. Run 114
Result: single missing-row content patch failed with `rows_skipped_dataerror=1`.
Artifacts: `output/run114_single_missing_tensor_patch_20260706_130153/`

6. Run 115
Result: direct SQL row replace failed due wrong column (`b.dimensions`).
Artifacts: `output/run115_direct_row_replace_20260706_130351/`

7. Run 116
Result: direct SQL row replace with corrected schema failed with `sqlite3.DataError: string or blob too big`.
Artifacts: `output/run116_direct_row_replace_fixed_20260706_130546/`

8. Run 117
Result: high-limit row patch succeeded (`row_patch_changes=1`), missing row cleared, full set parity reached (`full_match=5746`), strict PASS persisted, output still garbled.
Artifacts: `output/run117_highlimit_single_row_retry_20260706_131612/`

---

## 7) Quantitative Trend Snapshot (Patched vs Official)

Reference control image:
- `output/run110_fullgate_patched_vs_official_after_gpufix_20260706_121602/media/official/official_run110_after_gpufix.png`

Patched branch metric trend:
- patched110: `gray_std=0.124541`, `entropy=7.030824`
- patched111: `gray_std=0.127956`, `entropy=7.067396`
- patched112: `gray_std=0.128240`, `entropy=7.073235`
- patched113: `gray_std=0.120805`, `entropy=6.985534`
- patched117: `gray_std=0.122407`, `entropy=6.996306`

Official baseline:
- official: `gray_std=0.330278`, `entropy=7.488716`

Interpretation:
- Patched images remain clustered in a low-information band far below official dynamic range/entropy.

---

## 8) Core-Issue Analysis Plan From This Point

Now that tracked row-level parity is exhausted, future work should prioritize semantic/config path isolation.

Required next branches:
- Compare request/config payload semantics between official and custom runs at protobuf level.
- Audit model-key mapping and schema path resolution in `custom.json` vs raw-key branch.
- Validate text/video feature path assumptions against baseline model family expectations.
- Introduce minimally different synthetic candidate tests to isolate first semantic divergence point, not row-data divergence.

Branch acceptance rule:
- Runtime PASS alone is insufficient.
- Accept only branches that produce coherent visual output matching official-like structure under matched prompt/seed/profile.

---

## 9) Exact Re-Execution Checklist (If We Need To Repeat Everything)

1. Rebuild devcontainer after Dockerfile changes.
2. Verify GPU and runtime prerequisites.
3. Run official strict canary first.
4. Run candidate strict canary with identical profile.
5. Convert output bins to PNG.
6. Compute metrics and visually inspect.
7. If custom is garbled, run row-level probes and patch branches in order:
   - content subset align
   - metadata subset align
   - missing-row detection
   - high-limit row repair for oversized row
8. Re-run strict canary and compare shape (`responses/images/audio`) plus image quality.
9. Record every branch in `Q6P_RUN_LOG.md` and update `Q6P_WORKING_BASELINES_2026-07-02.md`.

---

## 10) Why This Matters For General "Convert, Quantize, Inference" Reliability

What this campaign proves:
- It is possible to recover runtime and even full tracked row parity while still failing perceptual quality.
- Therefore, a robust pipeline must include both:
  - low-level structural parity checks
  - high-level semantic/visual correctness checks

Required long-term standard:
- Any future custom model should pass both structure and quality gates before being labeled "working".
- This playbook is the reproducible process baseline for that standard.
