# Custom Model — Root-Cause Analysis And Forward Plan (2026-07-07)

## 0) Read This First

This document is a fresh, critical analysis of the *entire* custom-LTX2.3
"convert → quantize → inference" effort in this repository. It is written after:

- reviewing all major markdown docs (README, WORKSPACE_MANUAL, REPO_HANDOVER_MASTER,
  the master playbook, run log/index, baselines, conversion findings, CLIP investigation),
- reading the actual patch bundle in `DRAW_THINGS_PATCH/` (the only sanctioned way we
  modify Draw Things — we never touch or commit to `draw-things-community/`),
- inspecting the importer/converter/quantizer code paths those patches change,
- inspecting `custom.json` entries and the convert/quantize wrappers,
- and a workspace cleanup that freed ~343G of contaminated intermediates.

It deliberately challenges the previous approach. The goal is a *working* custom model,
not more parity closure.

---

## 1) The One-Sentence Problem Statement

We have spent 117+ runs proving we can make a converted+quantized custom checkpoint
**survive runtime**, but we have never once proven that our **converter/quantizer can
turn any safetensors file into a working Draw Things checkpoint** — because we never had
a positive control to test that against.

Everything else follows from that gap.

---

## 2) The Core Flaw In The Previous Approach (Most Important Section)

### 2.1 What we actually did across Runs 001–117

The investigation converged on a single technique: measure row-level tensor parity
between the **custom** ckpt and the **official** ckpt, then repeatedly copy
metadata/dim/data **out of the official ckpt into the custom ckpt** using:

- `tools/dt_align_ckpt_content_subset.py`
- `tools/dt_patch_ckpt_metadata_subset.py`
- `tools/dt_align_ckpt_metadata.py`
- high-limit SQLite row repair scripts

until we reached "5746/5746 parity" (Run 117).

### 2.2 Why that can never produce a working custom model

This method is **circular**:

- The only way to reduce mismatch against the official ckpt is to make the custom ckpt
  *more like the official weights*.
- A finetune's entire value is that its weights are **different** from the base.
- So "perfect parity" and "a working custom finetune" are mutually exclusive goals.

At best, full parity yields a copy of Draw Things' own base model wearing the custom
model's filename. That is exactly consistent with the observed result:
**runtime PASS, output still garbled/low-entropy** (patched `gray_std≈0.12`,
`entropy≈7.0` vs official `gray_std≈0.33`, `entropy≈7.49`).

### 2.3 Why runtime "PASS" was misleading

Runtime survival only proves the checkpoint is *structurally loadable* (right keys,
right shapes, readable rows). It says nothing about whether the **weight values** are
mathematically correct for the LTX2.3 architecture. Parity-copying guarantees loadability
while masking whether the converter ever produced correct values on its own.

Conclusion: the 117-run arc measured and optimized the wrong thing. It is not wasted —
it produced excellent tooling and ruled out many confounds — but it cannot, by
construction, reach the goal.

---

## 3) The Missing Positive Control (The Real Root Gap)

### 3.1 The provenance problem

- Draw Things ships **`.ckpt`** files only. They never published the **safetensors**
  their official `ltx_2.3_22b_distilled_1.1_q6p.ckpt` was built from.
- Therefore every parity check in this repo compares
  *our-converted-custom-ckpt* against *their-already-converted-official-ckpt*.
- We have **never** compared *our safetensors→ckpt output* against a
  *known-good ckpt built from the same safetensors*.

### 3.2 Why this matters more than anything else

Without a positive control, we cannot answer the only question that matters:

> Does `dt_convert_model.sh` + `dt_quantize_model.sh` + the patched `ModelImporter.swift`
> correctly convert **any** LTX2.3 safetensors into a working ckpt?

Every failure so far is ambiguous: it could be a bad custom model, a bad converter, a bad
quantizer, or a bad config — and parity-copying can't disambiguate them.

### 3.3 The control we can now build

Draw Things' `ltx_2.3_22b_distilled_1.1` exists as **both**:

- an official `.ckpt` we already have (`ltx_2.3_22b_distilled_1.1_q6p.ckpt`), and
- a **downloadable safetensors** (`ltx_2.3_22b_distilled_1.1.safetensors`) the user can fetch.

That means we can finally close the loop:

1. Convert the official `1.1.safetensors` ourselves → f16 → q6p.
2. **Run inference on our self-converted official model.**
3. Compare against the official `1.1_q6p.ckpt`.

This is the single highest-value experiment in the whole project, and it was never run.

---

## 4) What The Source/Patch Analysis Shows

We modify Draw Things only through `DRAW_THINGS_PATCH/`. The patch bundle changes,
among others:

- `ModelImporter.swift` (import detection + LTX2.3 weight mapping)
- `Converter.swift`
- `Quantizer.swift`
- `TextEncoder.swift`

### 4.1 LTX2.3 inference is real upstream; LTX2.3 *import* is our custom code

- `.ltx2_3` is a genuine upstream `ModelVersion`. Draw Things officially supports LTX2.3
  for **inference** (Sampler / TextEncoder / ControlModel all handle it), and the full
  model definition lives in upstream `LTX2.swift`.
- However, the **import-time detection and safetensors→DrawThings weight-name mapping**
  for this custom path is **hand-written patch code** (e.g. a custom
  `makeLTX23ConnectorMapping` and manual `ltx23EmbedderAliases`).

### 4.2 Why that is the prime suspect

Garbled-but-loadable output is the classic signature of a **weight-mapping error**:

- If some safetensors tensors are mapped to the wrong Draw Things keys (or skipped,
  transposed, or left at default), the model still *loads and runs* (shapes line up)
  but computes **semantically wrong** activations → RGB noise.
- Our hand-written connector/embedder mapping is exactly where such an error would live,
  and it has never been validated against a known-good conversion.

This is consistent with the fact that **correctly-configured** `ltx2.3` entries
(`10_e_v1_custom_main_official_clip`, `trace021`, `main_official_clip_test`,
`f16_fullschema_test`) *all* failed. Config was ruled out; mapping was not.

---

## 5) What We Can Rule Out vs. What Remains Open

### 5.1 Ruled out (well-supported by prior runs)

- Config/alias schema alone (later correctly-configured `ltx2.3` entries still failed).
- Seed variation as a fix.
- Row-level metadata/dim/data parity as a *sufficient* condition (Run 117 proved this).
- Fixing the final unreadable row as sufficient (Run 117).
- Environment/GPU as the persistent cause (official control passes end-to-end).

### 5.2 Still open (ordered by likelihood)

1. **Converter/importer weight-mapping correctness** — never positively controlled.
2. **Quantizer correctness for LTX2.3** — q6p path never isolated from conversion.
3. **Custom finetune format assumptions** — key naming/layout of `10_e_v1_bf16.safetensors`
   may differ from what our mapping expects.
4. Latent runtime/config path interactions not visible to row-level probes.

---

## 6) Forward Plan (Control-First, Bisecting Strategy)

The plan is designed around slow (~15+ min) generations and tight Windows/WSL disk limits.
It replaces "chase parity" with "prove each stage against a control, one at a time."

### Phase 0 — Reset and reclaim disk (DONE / in progress)

- [x] Delete contaminated parity-copy experiments and stale intermediates
      (freed ~343G *inside the container*).
- [ ] **Compact the WSL VHDX from Windows** so the host actually reclaims space
      (deleting inside the container does not shrink the Windows disk by itself):
  ```powershell
  wsl --shutdown
  wsl --manage <distro> --set-sparse true
  # or Optimize-VHD / diskpart compact on the distro's ext4.vhdx
  ```
- [x] Keep only irreplaceable/expensive assets:
  `10_e_v1_bf16.safetensors` (custom source),
  `ltx_2.3_22b_distilled_1.1_q6p.ckpt` (+tensordata, control target),
  companions (Gemma text encoder, LTX2.3 VAE, CLIP, upscalers), `custom.json`.

### Phase 1 — Build the missing POSITIVE CONTROL (highest priority)

Goal: prove the converter/quantizer can produce a **working** ckpt from a **known** source.

1. User downloads `ltx_2.3_22b_distilled_1.1.safetensors` (the official base's own source).
2. Convert it with our tooling:
   ```bash
   bash tools/run_convert_safetensors_to_f16.sh \
     <path>/ltx_2.3_22b_distilled_1.1.safetensors \
     dt-models/ltx23_control_f16.ckpt
   ```
3. Run inference on the **self-converted f16** (skip quantization first) using the
   *same working config* proven for the official model (`version: ltx2.3`,
   proper autoencoder/clip_encoder/modifier=kontext).
4. **Decision gate:**
   - If the self-converted **official f16** generates a coherent first frame →
     the converter is proven correct. Proceed to Phase 2.
   - If it produces garbled output → the **converter/importer mapping is the bug**.
     Fix the mapping (Phase 4) before touching the custom model again.

This single test disambiguates converter vs. model — something 117 runs could not do.

### Phase 2 — Isolate the QUANTIZER

Only if Phase 1 f16 works:

1. Quantize the self-converted control f16 → q6p:
   ```bash
   bash tools/dt_quantize_model.sh dt-models/ltx23_control_f16.ckpt q6p
   ```
2. Run inference on the self-converted **q6p** control.
3. Decision gate:
   - Works → quantizer proven. The pipeline is fully validated end-to-end on a control.
   - Fails → quantizer is the bug (isolate independently of conversion).

### Phase 3 — Apply the PROVEN pipeline to the custom model

Only after Phases 1–2 pass on the control:

1. Convert `10_e_v1_bf16.safetensors` → f16 with the now-proven converter.
2. Inference on custom **f16 first** (no quantization) with the working `ltx2.3` config.
3. If f16 works, quantize → q6p and re-test.
4. Accept only if output is **coherent**, not merely runtime-PASS.

If the custom f16 fails while the control f16 passed, the divergence is in the
**custom finetune's key layout**, and we compare its safetensors key names/shapes
directly against the control safetensors (a real, legitimate comparison — unlike the
old parity-copy).

### Phase 4 — Fix the mapping (only if Phase 1 fails)

If the converter is the bug:

1. Diff our patched `makeLTX23ConnectorMapping` / `ltx23EmbedderAliases` against the
   upstream `LTX2.swift` expected key set.
2. Export our converted control ckpt's key set and compare, name-by-name, to the official
   `1.1_q6p.ckpt` key set — but this time to **fix the mapping code**, not to copy weights.
3. Regenerate the patch via `tools/sync_drawthings_patch_bundle.sh`, re-apply, rebuild,
   and re-run Phase 1.

---

## 7) New Acceptance Gate (Replaces "Runtime PASS")

A branch is "working" only if **all** hold under matched prompt/seed/profile:

1. Stream completes with final output.
2. No server crash signature.
3. Output is **visually coherent** (not RGB noise) — the objective, not a proxy.
4. For the control: it is comparably coherent to the official `1.1_q6p` reference.

Row-level parity is demoted to a *diagnostic*, never an acceptance criterion.

---

## 8) Disk Discipline For The Slow/Constrained Environment

- Prefer **f16-first** testing; only quantize once f16 is proven (avoids duplicate 20–44G artifacts).
- Keep at most: 1 control safetensors, 1 control f16, 1 control q6p, 1 custom safetensors,
  1 custom f16, 1 custom q6p at any time.
- Delete each intermediate as soon as its decision gate is recorded.
- After large deletes, **compact the WSL VHDX from Windows** or the host disk won't shrink.

---

## 9) Why This Plan Is Different (And Should Finally Work)

| Previous approach | This plan |
|---|---|
| Chased parity against official ckpt | Builds a positive control from official *safetensors* |
| Copied official weights into custom | Never copies weights; fixes the *pipeline* |
| "Runtime PASS" accepted | Only "coherent output" accepted |
| Converter never isolated | Converter proven/refuted in Phase 1 |
| Quantizer never isolated | Quantizer proven/refuted in Phase 2 |
| Couldn't tell converter-bug from model-bug | Bisects converter vs. quantizer vs. model |

The core realization: **you cannot debug a custom model until you have proven the pipeline
works on a non-custom model.** Build that control first; everything else becomes tractable.

---

## 10) Immediate Next Actions

1. (User) Finish compacting the WSL VHDX from Windows to reclaim host space.
2. (User) Download `ltx_2.3_22b_distilled_1.1.safetensors`.
3. (Us) Run Phase 1: convert it → f16 → inference with the proven `ltx2.3` config.
4. (Us) Record the Phase 1 decision gate in `Q6P_RUN_LOG.md` and branch to Phase 2 or Phase 4.

Do not resume any parity-copy or row-repair work. That path is closed.
