# LTX2.3 clip_encoder Investigation (2026-06-02)

## Executive Summary

The `clip_encoder` issue is not a UI alias problem. For `ltx2_3`, Draw Things uses the `clip_encoder` checkpoint as the source for `text_feature_extractor` (and related projection path), not as a cosmetic optional encoder.

Current evidence shows:

- `file = forcedfix` + `clip_encoder = official 1.1 q6p` can run.
- `file = forcedfix` + `clip_encoder = forcedfix` fails/behaves badly.
- fresh targeted diff on clip-side families shows `261/262` selected tensors mismatch vs official baseline, with `260` data-length mismatches.

Most likely root cause: forced-q6p quantization policy still over-quantizes / re-encodes fragile LTX families used by the clip-side feature extractor and connectors.

## Scope and Method

I investigated in this order:

1. Existing local documentation and prior findings.
2. Toolkit/runtime source paths (`custom.json` -> ModelZoo -> LocalImageGenerator -> TextEncoder).
3. Patch archaeology (`DRAW_THINGS_PATCH`, especially quantizer changes).
4. Online code research (ComfyUI-GGUF PR #399, `draw-things-comfyui`).
5. New validation probes run today against your latest forcedfix artifact.

## Key Runtime Facts (Local Code Evidence)

### 1) Your active custom entry mixes forcedfix main file with official clip file

From `dt-models/custom.json`:

- `10_e_v1` uses:
  - `file: ltx_2.3_22b_distilled_q6p_forcedfix_20260601.ckpt`
  - `clip_encoder: ltx_2.3_22b_distilled_1.1_q6p.ckpt`
  - `text_encoder: gemma_3_12b_it_qat_q8p.ckpt`

Reference: `dt-models/custom.json:3-14`.

### 2) `clip_encoder` is appended into text encoder file list

`LocalImageGenerator` builds:

- first: `textEncoderForModel(file)`
- then: `CLIPEncodersForModel(file)`
- then optional T5

Reference: `draw-things-community/Libraries/LocalImageGenerator/Sources/LocalImageGenerator.swift:3616-3622`.

`TextEncoder` is initialized with that exact ordered array.

Reference: `draw-things-community/Libraries/LocalImageGenerator/Sources/LocalImageGenerator.swift:4056-4065`.

### 3) For `ltx2_3`, `encodeLTX2` uses `filePaths[1]` for `text_feature_extractor`

Inside `encodeLTX2`:

- `filePaths[0]` loads `text_model`.
- `filePaths[1]` is opened and `text_feature_extractor` is read from it.

References:

- `draw-things-community/Libraries/SwiftDiffusion/Sources/TextEncoder.swift:3128` (function entry)
- `draw-things-community/Libraries/SwiftDiffusion/Sources/TextEncoder.swift:3285-3286` (`openStore(filePaths[1])`)
- `draw-things-community/Libraries/SwiftDiffusion/Sources/TextEncoder.swift:3296-3322` (`store.read("text_feature_extractor", ...)`)

This is the core reason `clip_encoder` is decisive for LTX2.3 runtime stability.

### 4) Built-in LTX2.3 specs set `clipEncoder` to LTX checkpoint itself

ModelZoo defaults for LTX2.3 commonly set `clipEncoder` equal to the LTX model file variant (q8/q6/i8), including distilled 1.1.

Reference examples: `draw-things-community/Libraries/ModelZoo/Sources/ModelZoo.swift:841-874`.

### 5) Custom specs from `custom.json` are merged and can override built-ins

`custom.json` is decoded with snake_case conversion and appended into `availableSpecifications`.

References:

- `draw-things-community/Libraries/ModelZoo/Sources/ModelZoo.swift:2127-2149`
- `draw-things-community/Libraries/ModelZoo/Sources/ModelZoo.swift:2518-2522` (`specificationForModel` lookup by file mapping)
- `draw-things-community/Libraries/ModelZoo/Sources/ModelZoo.swift:2732-2735` (`CLIPEncodersForModel`)

## Patch Archaeology Findings

### Importer-side LTX2.3 patching is substantial, but mostly structural

`DRAW_THINGS_PATCH/patches/draw-things-community.patch` includes major LTX importer work in `ModelImporter.swift`, including:

- dedicated `ltx2` vs `ltx2_3` handling instead of lumped branch,
- explicit mapping for `text_video_connector` / `text_audio_connector`,
- explicit alias mapping for `text_feature_extractor` (`video_aggregate_embed` / `audio_aggregate_embed`),
- explicit write path for connector learnable registers (`text_video_connector_learnable_registers`, `text_audio_connector_learnable_registers`).

Representative hunk: `DRAW_THINGS_PATCH/patches/draw-things-community.patch:692-873`.

Interpretation:

- this patch explains why keyset/mapping parity improved compared with early failures,
- but it does not by itself guarantee payload-level equivalence after quantization.

### Quantizer forced-codec branch remains high risk for LTX

In `DRAW_THINGS_PATCH/Apps/ModelQuantizer/Quantizer.swift`, forced mode (`--target-codec`) has special LTX handling but still funnels many multidimensional tensors to forced codec unless key patterns match its preserve list.

Key branch:

- forced branch entry: `:94`
- LTX forced sub-branch: `:95`
- preserve list includes (`embedder`, `pos_embed`, `-linear-`, `scale_shift_table`, `caption_projection`, `patchify_proj`, `proj_out`): `:98-100`
- else path uses `codec: [forcedCodec, .ezm7]`: `:109`

This aligns with prior findings that forced mode is still more aggressive than model-specific mixed policy.

Prior documented conclusion already pointed here:

- `CONVERSION_TOOL_FINDINGS_2026-05-28.md:1428-1515`

## Fresh Validation (Run 2026-06-02)

I ran fresh probes against:

- candidate: `dt-models/ltx_2.3_22b_distilled_q6p_forcedfix_20260601.ckpt`
- baseline: `dt-models/ltx_2.3_22b_distilled_1.1_q6p.ckpt`

### A) Targeted clip-path probe (highest relevance)

Command class:

- `tools/dt_probe_ckpt_targeted_content.py` with prefixes:
  - `__text_feature_extractor__`
  - `__text_video_connector__`
  - `__text_audio_connector__`
  - `text_video_connector_learnable_registers`
  - `text_audio_connector_learnable_registers`

Artifact:

- `output/probe_forcedfix_vs_official_clip_path_20260602.md`

Critical stats:

- `selected_tensors=262`
- `readable_selected=261`
- `mismatch_any=261`
- `full_match=0`
- `metadata_mismatch_type=261`
- `metadata_mismatch_datatype=5`
- `data_len_mismatch=260`
- `data_head_mismatch=261`

Family map (same artifact):

- `__text_video_connector__`: `128/128` mismatched
- `__text_audio_connector__`: `128/128` mismatched
- `__text_feature_extractor__`: `3/4` mismatched (`1` baseline unreadable row persists)
- connector learnable registers: both mismatched

Interpretation:

- The clip-side tensors required by `encodeLTX2` are almost entirely non-equivalent to official q6p.
- This strongly explains why forcedfix cannot safely serve as `clip_encoder` in current state.

### B) Full deep diff (whole-checkpoint context)

Artifact:

- `output/probe_deep_diff_forcedfix_vs_official_q6p_20260602.md`

Critical stats:

- `shared_tensors=5746` (keyset parity)
- `metadata_mismatch_type=1837`
- `metadata_mismatch_datatype=1545`
- `data_len_mismatch=3621`
- `data_head_mismatch=5641`
- `full_signature_match=0`

Top mismatch families include `__dit__`, plus the same clip-side connector/feature-extractor families.

Interpretation:

- Structural parity exists, but payload/encoding semantics remain far from official.

## Online Research

### 1) ComfyUI-GGUF PR #399

Patch reviewed: `https://patch-diff.githubusercontent.com/raw/city96/ComfyUI-GGUF/pull/399.patch`

What it does:

- adds GGUF scalar metadata extraction (`get_gguf_metadata`)
- passes metadata into Comfy diffusion model loader
- refactors `gguf_sd_loader` return shape to include extra metadata

Relevance to this incident:

- low direct relevance.
- This is Comfy GGUF metadata plumbing, not Draw Things CKPT quantization or `text_feature_extractor` serialization semantics.

### 2) draw-things-comfyui bridge

Sources reviewed:

- `https://raw.githubusercontent.com/drawthingsai/draw-things-comfyui/main/src/config.py`
- `https://raw.githubusercontent.com/drawthingsai/draw-things-comfyui/main/src/data_types.py`
- `https://raw.githubusercontent.com/drawthingsai/draw-things-comfyui/main/src/nodes.py`
- `https://raw.githubusercontent.com/drawthingsai/draw-things-comfyui/main/src/draw_things.py`

Observed behavior:

- generation config sets `configT.model` from selected model file.
- node pipeline passes `model.file` and `model.version` as primary selection.
- no dedicated `clip_encoder` field is independently set in flatbuffer generation config path.

Implication:

- clip encoder resolution still primarily depends on server-side model specification (built-in/custom spec), not an independent front-end clip override path.

## Ranked Hypotheses

1. **Clip-side quantization incompatibility in forcedfix artifact** (High confidence)
   - supported by direct `encodeLTX2` file path usage and targeted mismatch profile (`261/262`).

2. **Forced codec policy mismatch vs official mixed policy** (High confidence)
   - consistent with quantizer branch behavior and repeated historical findings.

3. **Model selection/alias mismatch confound** (Low-to-medium confidence)
   - possible in general because runtime falls back to `ModelZoo.defaultSpecification.file` when selected model is unavailable (`LocalImageGenerator.swift:3598-3601`), and clip entries are filtered by `isModelDownloaded` (`LocalImageGenerator.swift:3620`).
   - still does not best explain your current A/B where changing only `clip_encoder` flips behavior while main model stays fixed.

4. **Comfy-side metadata plumbing regression (GGUF PR #399 class)** (Low confidence)
   - architecture mismatch to this failure domain.

## Direct Answer: Why `clip_encoder` Cannot Be Forcedfix (Yet)

Because for LTX2.3, `clip_encoder` is effectively the source for `text_feature_extractor` and related connector path consumed in `TextEncoder.encodeLTX2`.

Your forcedfix artifact currently has broad non-equivalence in exactly those families versus the official q6p baseline. When used as `clip_encoder`, that path receives incompatible tensor payloads/encodings. Using the official clip file restores a known-good feature-extractor path, which is why generation can proceed.

## Recommended Next Actions

### Immediate mitigation

- Keep `clip_encoder` pinned to official `ltx_2.3_22b_distilled_1.1_q6p.ckpt` for production runs.

### Short-term engineering fix

- Patch forced quantization policy for LTX so connector + feature-extractor families are preserved at safer precision (or match official mixed policy behavior), instead of falling through to blanket forced codec.

Candidate preserve/safer families to treat specially:

- `__text_feature_extractor__`
- `__text_video_connector__`
- `__text_audio_connector__`
- `text_video_connector_learnable_registers`
- `text_audio_connector_learnable_registers`

## Validation matrix after patch

1. Quantize new candidate from same FP16 source.
2. Run targeted clip-path probe; success gate:
   - drastic reduction in `data_len_mismatch` and `metadata_mismatch_type/datatype` for the five families above.
3. A/B runtime:
   - `clip_encoder=official` vs `clip_encoder=new candidate` should both run.
4. If runtime passes, then evaluate output quality parity and stability across prompts.

## Evidence Index

- `CONVERSION_TOOL_FINDINGS_2026-05-28.md`
- `dt-models/custom.json`
- `draw-things-community/Libraries/LocalImageGenerator/Sources/LocalImageGenerator.swift`
- `draw-things-community/Libraries/SwiftDiffusion/Sources/TextEncoder.swift`
- `draw-things-community/Libraries/ModelZoo/Sources/ModelZoo.swift`
- `DRAW_THINGS_PATCH/Apps/ModelQuantizer/Quantizer.swift`
- `DRAW_THINGS_PATCH/patches/draw-things-community.patch`
- `output/probe_forcedfix_vs_official_clip_path_20260602.md`
- `output/probe_deep_diff_forcedfix_vs_official_q6p_20260602.md`
