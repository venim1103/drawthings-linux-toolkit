# Conversion Findings Report (2026-05-28)

## Scope

This report captures the full investigation and patch work for converting custom LTX-2.3 model weights into Draw Things checkpoints, including structural fixes in the importer, quantization outcomes, and runtime comparison against the official LTX-2.3 q6p baseline.

## Current Status (Latest Verified)

- Converter-side structural defects for this custom LTX-2.3 source were fixed.
- The fixed F16 and derived q6p checkpoints now match the official LTX-2.3 q6p keyset exactly (`5746` tensors, no missing/extra keys).
- Controlled one-response probes show official q6/f16 controls reach `textEncoded`, while custom F16/q6 checkpoints still reproduce runtime failure (`ccv_nnc_tensor_read` -> `ccv_cnnp_model_read`, server exit 139).
- Metadata-column alignment against official F16 now reaches strict guard parity (`type/format/datatype` mismatches reduced to `0` on readable shared tensors), but runtime crash persists on aligned custom F16.
- Deep row-level probe after metadata alignment shows `1542` tensor `data` blob length mismatches vs official F16 (`1540` under `__dit__`), while metadata columns remain at parity on readable shared tensors.
- Targeted payload alignment for the `80%` `__dit__` subgroup subset (`1248` tensors) reduced global `data` length mismatches from `1542` to `294`, but one-response runtime still crashed in the same reader path.
- A second targeted payload pass on the remaining `294` tensor names reduced readable shared-tensor `data` length mismatches to `0` (`5745/5745` readable shared rows), but one-response runtime still crashed in the same read path.
- Targeted non-`__dit__` content probe after length parity found `115/261` rows with `data` head-signature mismatch (zero metadata/length mismatches), concentrated in connector and feature-extractor families.
- Micro-batched `__dit__` subgroup content probe over the prior 80% subset (`26` subgroups, `1248` rows) completed successfully and found `dim` head-signature mismatch on `1248/1248` rows, with zero metadata/length/data-head mismatches.
- Runtime with custom q4p (quantized from fixed F16) caused Draw Things server crashes (`Bad pointer dereference` in ccv read path), so q4p is currently unsafe for this model.
- Official exact F16 baseline file (`ltx_2.3_22b_distilled_f16.ckpt`) was downloaded and checksum-verified against the published value in-session.
- Custom model metadata now points to q6p in `dt-models/custom.json`.

## Executive Summary

- Initial failures were caused by importer mapping gaps (missing embedder/connector/feature tensors and placeholder keys).
- After importer patching, converted outputs passed strict structural validation and reached key parity with official q6p.
- Runtime behavior now diverges by artifact provenance, not by keyset parity:
   - official q6/f16 controls: stable to first signpost in one-response probes.
   - custom q4p/q6/f16: current runtime reader path crashes (`ccv_nnc_tensor_read` / `ccv_cnnp_model_read`).
- Remaining blocker is serialization compatibility beyond keyset parity and beyond `type/format/datatype` column parity.
- New evidence narrows the remaining divergence to tensor payload content semantics (byte-level mismatch at equal lengths), not column metadata fields.

## Applied Patch Summary

### 1) Importer Mapping Fixes (Community + Toolkit Mirror)

Files:

- `draw-things-community/Libraries/ModelOp/Sources/ModelImporter.swift`
- `DRAW_THINGS_PATCH/Libraries/ModelOp/Sources/ModelImporter.swift`

Key changes:

1. Expanded LTX diffusers detection and key lookup fallback to include `diffusion_model.*` prefixed layouts.
2. Added normalization for `diffusion_model.` prefix stripping in LTX branches.
3. Added explicit LTX2.3 connector mapping builder (`makeLTX23ConnectorMapping`) for deterministic source-to-target mapping.
4. Added explicit alias mappings for:
   - `patchify_proj.* -> __dit__[t-x_embedder-*]`
   - `audio_patchify_proj.* -> __dit__[t-a_embedder-*]`
5. Added explicit text feature extractor aggregate aliases:
   - `text_embedding_projection.video_aggregate_embed.{weight,bias}`
   - `text_embedding_projection.audio_aggregate_embed.{weight,bias}`
6. Skipped empty destination names during mapping writes to prevent placeholder keys.
7. Relaxed additional mapping consumption guard to ensure aliases are applied when source keys exist.

### 2) Conversion Wrapper Validation

Files:

- `tools/dt_convert_model.sh`
- `tools/dt_validate_converted_ckpt.py`

Key changes:

1. Added automatic post-conversion validation controls (`DRAWTHINGS_CONVERTER_VALIDATE*`).
2. Added profile-aware structural checks for LTX2.3:
   - required prefix families,
   - placeholder key rejection,
   - source safetensors-assisted profile detection.

### 3) Runtime Wrapper Guardrails

File:

- `tools/dt_generate_video.sh`

Key changes:

1. Added LTX-2.3 upscaler presence checks.
2. Avoided implicit hires-fix enable when required upscalers are absent.
3. Added warnings for manually enabled hires-fix without required upscalers.

### 4) Model Metadata Update

File:

- `dt-models/custom.json`

Key changes:

1. Main custom entry switched from q4p to q6p:
   - `file: 10_e_v1_bf16_q6p.ckpt`
   - `clip_encoder: 10_e_v1_bf16_q6p.ckpt`

## Reproduction Results

### A) Historical pre-fix failure (custom q4p)

Command:

```bash
./tools/dt_video_test_run.sh 10_e_v1_bf16_q4p.ckpt
```

Output folder:

- `output/dt_video_20260528_095514`

Observed stream summary:

- `responses: 11`
- `images written: 0`
- `audio written: 0`
- `preview frames seen: 5`
- stream ended without final `generatedImages` payloads

### B) Post-fix structural pass (custom fix2 f16)

Command:

```bash
python tools/dt_validate_converted_ckpt.py \
  --file dt-models/10_e_v1_bf16_fix2_f16.ckpt \
  --source-safetensors dt-models/10_e_v1_bf16.safetensors
```

Observed:

- `tensor_count=5746`
- `RESULT=PASS`

### C) q4p from fixed F16 (post-fix quantized) crash

Command:

```bash
./tools/dt_video_test_run.sh 10_e_v1_bf16_q4p.ckpt
```

Observed:

- gRPC server crash with segmentation fault:
  - `Bad pointer dereference`
  - stack includes `ccv_nnc_tensor_read`, `ccv_cnnp_model_read`
- client error: `UNAVAILABLE: Socket closed`

### D) q6p from fixed F16 (post-fix quantized) runtime

Command:

```bash
./tools/dt_video_test_run.sh 10_e_v1_bf16_q6p.ckpt
```

Output folder:

- `output/dt_video_20260528_171805`

Observed stream summary:

- `responses: 14`
- `images written: 1`
- `audio written: 0`
- `preview frames seen: 5`

Result:

- final image payload returned, but output remains visually noisy.

### E) Official LTX-2.3 q6p control

Command:

```bash
./tools/dt_video_test_run.sh ltx_2.3_22b_distilled_1.1_q6p.ckpt
```

Output folder:

- `output/dt_video_20260528_171821`

Observed stream summary:

- `responses: 15`
- `images written: 1`
- `audio written: 1`
- `preview frames seen: 5`

Result:

- coherent final image and audio payloads returned.

## Structural Comparison (Final State)

Compared files:

- `dt-models/10_e_v1_bf16_q6p.ckpt` (custom, post-fix)
- `dt-models/ltx_2.3_22b_distilled_1.1_q6p.ckpt` (official)

Counts:

- `custom_count = 5746`
- `official_count = 5746`
- `official_minus_custom = 0`
- `custom_minus_official = 0`

Interpretation:

- Structural/key-family compatibility has been achieved.
- Structural key parity is necessary but not sufficient for runtime compatibility.

## Detailed Metadata / Serialization Diff (2026-05-29)

### Reproduction update (post-structural-fix)

Control (official q6) with one-response probe:

- Reached first signpost (`textEncoded`) and remained stable.

Custom probes:

- `10_e_v1_bf16_q6p.ckpt`: server crashed in `ccv_nnc_tensor_read` / `ccv_cnnp_model_read`.
- `10_e_v1_bf16_fix2_f16.ckpt`: same crash path.

Implication:

- The crash is not q6-only; current custom F16 and derived q6 are both incompatible with the active runtime reader path.

### Controlled A/B update with official exact F16 (2026-05-30)

Probe setup (single-frame, low-cost):

- width/height: `384x704`
- steps: `8`
- sampler: `19`
- shift: `5.0`
- `--max-responses 1`

Official exact F16 control (`ltx_2.3_22b_distilled_f16.ckpt`):

- Probe returned `response #1` with signpost `textEncoded`.
- Client exit: `PROBE_OFFICIAL_F16_EXIT=0`.

Custom fixed F16 under same probe style (`10_e_v1_bf16_fix2_f16.ckpt`):

- Request started, then client failed with `gRPC error: UNAVAILABLE: Socket closed`.
- Client exit: `PROBE_CUSTOM_F16_AGAIN_EXIT=1`.
- Server crashed with `Bad pointer dereference` and backtrace rooted at:
   - `ccv_nnc_tensor_read`
   - `ccv_cnnp_model_read`
   - process exit: `139` (`Segmentation fault`).

Guidance confound check (custom F16, guidance forced to `1.0`):

- Re-ran one-response probe with guidance matching official control.
- Client again failed with `gRPC error: UNAVAILABLE: Socket closed`.
- Client exit: `PROBE_CUSTOM_F16_G1_EXIT=1`.
- Server again crashed in `ccv_nnc_tensor_read` / `ccv_cnnp_model_read`.

Interpretation:

- This session's official-vs-custom exact-F16 A/B confirms runtime instability is specific to custom serialization, not just q6 quantization or guidance-scale mismatch.

### Metadata-column alignment apply test (2026-05-31)

Alignment apply command:

```bash
python tools/dt_align_ckpt_metadata.py \
   --file dt-models/10_e_v1_bf16_fix2_f16.ckpt \
   --baseline dt-models/ltx_2.3_22b_distilled_f16.ckpt \
   --mode apply \
   --sample-limit 3
```

Observed:

- `planned_updates=5745`
- post-apply summary:
   - `type_mismatch=0`
   - `format_mismatch=0`
   - `datatype_mismatch=0`
   - `unreadable_metadata_only_target=0`
   - `unreadable_metadata_only_baseline=0`
- aligner exit: `RESULT=PASS`

Strict guard re-validation command:

```bash
python tools/dt_validate_converted_ckpt.py \
   --file dt-models/10_e_v1_bf16_fix2_f16.ckpt \
   --profile auto \
   --serialization-baseline dt-models/ltx_2.3_22b_distilled_f16.ckpt \
   --serialization-guard-profile ltx2_3 \
   --serialization-report-limit 5
```

Observed:

- `type_mismatch_count=0`
- `format_mismatch_count=0`
- `datatype_mismatch_count=0`
- `RESULT=PASS`

Runtime probe after alignment (one-response, same low-cost settings):

```bash
python tools/dt_api_client.py \
   --host 127.0.0.1:7861 \
   --max-recv-bytes 134217728 \
   generate-raw \
   --config-bin output/probe_custom_f16_aligned_config.bin \
   --prompt "a cinematic shot of a red sports car driving on a mountain road at sunset, detailed, realistic" \
   --negative-prompt "blurry, distorted, low quality, artifacts" \
   --output-dir output/probe_custom_f16_aligned \
   --chunked \
   --save-preview \
   --max-responses 1
```

Observed:

- server accepted the request (`Received image processing request, begin.`)
- server still crashed with the same backtrace root:
   - `ccv_nnc_tensor_read`
   - `ccv_cnnp_model_read`
- no successful `response #1` was emitted before crash.

Interpretation:

- Aligning the metadata columns (`type/format/datatype`) is necessary for diagnostics but not sufficient to make custom F16 runtime-loadable in the current server build.
- Remaining incompatibility is now narrowed to deeper serialization/read-path differences outside these three metadata columns.

### Deep row-length probe (2026-05-31)

Report artifacts:

- `output/probe_deep_diff_lengths_20260531.json`
- `output/probe_deep_diff_lengths_20260531.md`

Probe mode:

- metadata and blob length parity checks only (`type/format/datatype`, `length(dim)`, `length(data)`)
- per-key DataError-safe reads

Observed:

- `shared_tensors=5746`
- `unreadable_both=1` (`__text_feature_extractor__[t-video_aggregate_embed-0-0]`)
- readable shared tensors: `5745`
- metadata mismatches on readable shared tensors:
   - `metadata_mismatch_type=0`
   - `metadata_mismatch_format=0`
   - `metadata_mismatch_datatype=0`
- dimensionality blob size mismatches:
   - `dim_len_mismatch=0`
- payload blob size mismatches:
   - `data_len_mismatch=1542`

Top affected families by `data_len_mismatch`:

- `__dit__`: `1540`
- `text_audio_connector_learnable_registers`: `1`
- `text_video_connector_learnable_registers`: `1`

Representative examples:

- `__dit__[t-attn1_ada_ln_0-0-0]`: custom `8192`, official `16384`
- `__dit__[t-attn1_ada_ln_0-1-0]`: custom `8192`, official `16384`

Interpretation:

- After metadata parity is enforced, large residual `data` blob length divergence remains across `1542` tensors.
- This is strong evidence that crash persistence is tied to deeper tensor payload serialization/read-path expectations, not keyset or metadata-column mismatch.

### DIT family-by-family mismatch map (2026-05-31)

Report artifacts:

- `output/probe_dit_family_map_20260531.json`
- `output/probe_dit_family_map_20260531.md`
- `output/probe_dit_subset_80pct_20260531.txt`
- `output/probe_dit_subset_80pct_20260531.md`

Key map findings:

- `data_len_mismatch_total=1542`
- `data_len_mismatch_dit=1540`
- `data_len_mismatch_non_dit=2`
- `dit_subgroup_count=36`

Coverage summary over `__dit__` mismatches:

- `50%` target: `17` subgroups cover `816/1540` (`52.987%`)
- `80%` target: `26` subgroups cover `1248/1540` (`81.039%`)
- `90%` target: `29` subgroups cover `1392/1540` (`90.3896%`)
- `95%` target: `31` subgroups cover `1488/1540` (`96.6234%`)

Dominant subgroup pattern:

- Most top subgroups contribute exactly `48` mismatches each.
- Repeating length-pair classes dominate:
   - `8192 -> 16384`
   - `4096 -> 8192`

Representative top subgroups:

- `t-attn1_ada_ln_0` through `t-attn1_ada_ln_8` (each `48`, mostly `8192 -> 16384`)
- `t-audio_attn1_ada_ln_0` through `t-audio_attn1_ada_ln_8` (each `48`, mostly `4096 -> 8192`)
- `t-audio_prompt_scale_shift_ada_ln_*`, `t-prompt_scale_shift_ada_ln_*`, and cross-attn `*_to_*_attn_ada_ln_*` families also dominate early coverage.

Interpretation:

- The mismatch distribution is broad but highly regular, with repeated subgroup motifs rather than isolated outliers.
- The `80%` subset artifact (`probe_dit_subset_80pct_20260531.*`) provides a practical minimal candidate set for targeted payload-alignment experiments before attempting full-table rewrites.

### Targeted payload alignment experiment (subset80, 2026-05-31)

Tooling and artifacts:

- `tools/dt_align_ckpt_payload_subset.py`
- `output/probe_dit_subset_80pct_tensor_names_20260531.txt`
- `output/probe_deep_diff_lengths_after_subset80_20260531.json`
- `output/probe_deep_diff_lengths_after_subset80_20260531.md`

Subset apply summary:

- parsed subgroups: `26`
- resolved tensor names: `1248`
- pre-selected mismatch count: `1248`
- rows updated: `1248`
- post-selected mismatch count: `0`

Post-apply global parity probe (`length_and_metadata_only`):

- before subset80 apply: `data_len_mismatch=1542`
- after subset80 apply: `data_len_mismatch=294`
- metadata columns stayed aligned:
   - `metadata_mismatch_type=0`
   - `metadata_mismatch_format=0`
   - `metadata_mismatch_datatype=0`

Runtime probe after subset80 apply (same one-response settings):

- request was accepted (`Received image processing request, begin.`)
- server still crashed with:
   - `ccv_nnc_tensor_read`
   - `ccv_cnnp_model_read`
- crash signature remained unchanged from prior aligned-custom probes.

Interpretation:

- Large, targeted payload correction materially reduced mismatch volume, proving the subset selection/update path works as intended.
- Crash persistence despite reducing mismatches from `1542` to `294` indicates either:
   - the remaining `294` tensors include critical blockers, and/or
   - at least one additional incompatibility dimension (beyond current metadata/length parity checks) is still present.

### Targeted payload alignment experiment (remaining294, 2026-05-31)

Tooling and artifacts:

- `tools/dt_align_ckpt_payload_subset.py` (extended with `--tensor-list-file` mode)
- `output/probe_remaining_mismatch_tensor_names_after_subset80_20260531.txt`
- `output/probe_remaining_mismatch_selected_names_20260531.txt`
- `output/probe_data_len_recount_after_remaining294_20260531.json`
- `output/probe_data_len_recount_after_remaining294_20260531.md`

Stage-2 apply summary:

- parsed tensor names: `294`
- selected tensor names: `294`
- pre-selected mismatch count: `294`
- rows updated: `294`
- post-selected mismatch count: `0`

Post-apply recount (`length(data)` row-wise, DataError-safe):

- shared_tensors: `5746`
- unreadable_both: `1`
- readable_shared_tensors: `5745`
- `data_len_mismatch=0` on readable shared tensors

Runtime probe after remaining294 apply (same one-response settings):

- request reached runtime (`Received image processing request, begin.`)
- server still crashed with the same signature:
   - `ccv_nnc_tensor_read`
   - `ccv_cnnp_model_read`

Interpretation:

- `data` blob length parity is now complete for every readable shared tensor row.
- Crash persistence after length parity indicates the blocker is deeper than metadata columns and blob length parity, likely in one or more of:
   - unreadable-row handling (`__text_feature_extractor__[t-video_aggregate_embed-0-0]` remains unreadable on both files),
   - payload byte-content semantics at equal length,
   - tensor layout/reader expectations not captured by current probes.

### Targeted content parity map (non-`__dit__`, 2026-05-31)

Tooling and artifacts:

- `tools/dt_probe_ckpt_targeted_content.py`
- `output/probe_targeted_content_family_map_non_dit_after_lenparity_20260531.json`
- `output/probe_targeted_content_family_map_non_dit_after_lenparity_20260531.md`

Scope and setup:

- families included:
   - `__text_audio_connector__`
   - `__text_video_connector__`
   - `__text_feature_extractor__`
   - `text_audio_connector_learnable_registers`
   - `text_video_connector_learnable_registers`
- excluded known problematic row:
   - `__text_feature_extractor__[t-video_aggregate_embed-0-0]`

Probe result summary:

- `selected_tensors=261`
- `metadata_mismatch_type=0`
- `metadata_mismatch_format=0`
- `metadata_mismatch_datatype=0`
- `dim_len_mismatch=0`
- `data_len_mismatch=0`
- `dim_head_mismatch=64`
- `data_head_mismatch=115`
- `data_small_sha256_compared=73`
- `data_small_sha256_mismatch=41`
- `mismatch_any=115`

Top families by `mismatch_any`:

- `__text_video_connector__`: `56/128`
- `__text_audio_connector__`: `56/128`
- `__text_feature_extractor__`: `3/3` (excluding unreadable row)

Interpretation:

- Even with metadata parity and `data_len` parity, connector/feature-extractor payload bytes remain divergent.
- This provides a concrete, family-scoped next target for content-level parity fixes.

### Targeted content parity map (`__dit__` subset80 micro-batch, 2026-05-31)

Tooling and artifacts:

- `tools/dt_probe_ckpt_targeted_content.py` (subgroup-window mode)
- `output/probe_targeted_content_dit_subgroup_batch01_20260531.json`
- `output/probe_targeted_content_dit_subgroup_batch02_20260531.json`
- `output/probe_targeted_content_dit_subgroup_batch03_20260531.json`
- `output/probe_targeted_content_dit_subgroup_batch04_20260531.json`
- `output/probe_targeted_content_dit_subgroup_batch05_20260531.json`
- `output/probe_targeted_content_dit_subgroup_batch06_20260531.json`
- `output/probe_targeted_content_dit_subgroup_batch07_20260531.json`
- `output/probe_targeted_content_dit_subgroup_80pct_aggregate_20260531.json`
- `output/probe_targeted_content_dit_subgroup_80pct_aggregate_20260531.md`

Scope and setup:

- source subgroup list: `output/probe_dit_subset_80pct_20260531.txt`
- subgroup windows: `1-4`, `5-8`, `9-12`, `13-16`, `17-20`, `21-24`, `25-26`
- total covered: `26` subgroups (`1248` tensor rows)

Aggregate result summary:

- `selected_tensors=1248`
- `readable_selected=1248`
- `metadata_mismatch_type=0`
- `metadata_mismatch_format=0`
- `metadata_mismatch_datatype=0`
- `dim_len_mismatch=0`
- `data_len_mismatch=0`
- `dim_head_mismatch=1248`
- `data_head_mismatch=0`
- `mismatch_any=1248`
- `full_match=0`

Interpretation:

- For the dominant `__dit__` subset already used in payload-length remediation, divergence now appears concentrated in `dim` content bytes (head signatures), not in metadata columns, not in `data` length, and not in `data` head signatures.
- Combined with non-`__dit__` results, this indicates multi-family content-semantic divergence at equal lengths, split by tensor domain:
   - connectors/feature-extractor: strong `data` content mismatch,
   - `__dit__` subset: strong `dim` content mismatch.

### Targeted content-copy apply test (2026-05-31)

Tooling and input lists:

- `tools/dt_align_ckpt_content_subset.py`
- `output/probe_targeted_dim_copy_union_20260531.txt` (`1312` names; `__dit__` subset80 + non-`__dit__` dim-head mismatch names)
- `output/probe_targeted_non_dit_data_head_mismatch_names_20260531.txt` (`115` names)

Apply command mode and summary:

- dry-run:
   - `dim_names_selected=1312`
   - `data_names_selected=115`
   - `union_selected=1363`
   - `pre_dim_head_mismatch=1312`
   - `pre_data_head_mismatch=115`
- apply:
   - `rows_updated=1363`
   - `dim_rows_updated=1312`
   - `data_rows_updated=115`
   - `rows_skipped_dataerror=0`
   - `post_dim_head_mismatch=0`
   - `post_data_head_mismatch=0`
   - `RESULT=PASS`

Post-apply targeted re-probes:

- non-`__dit__` full selected set (`261` names, explicit names-file mode):
   - artifacts:
      - `output/probe_targeted_non_dit_names_all_20260531.txt`
      - `output/probe_targeted_content_family_map_non_dit_fullsamples_after_contentcopy_20260531.json`
      - `output/probe_targeted_content_family_map_non_dit_fullsamples_after_contentcopy_20260531.md`
   - result:
      - `selected_tensors=261`
      - `dim_head_mismatch=0`
      - `data_head_mismatch=0`
      - `mismatch_any=0`
      - `full_match=261`

- `__dit__` subset80 names-file re-probe (`1248` names):
   - artifacts:
      - `output/probe_targeted_content_dit_subset80_after_contentcopy_20260531.json`
      - `output/probe_targeted_content_dit_subset80_after_contentcopy_20260531.md`
   - result:
      - `selected_tensors=1248`
      - `dim_head_mismatch=0`
      - `data_head_mismatch=0`
      - `mismatch_any=0`
      - `full_match=1248`

Strict guard re-validation after content-copy apply:

- `python tools/dt_validate_converted_ckpt.py --file dt-models/10_e_v1_bf16_fix2_f16.ckpt --profile auto --source-safetensors dt-models/10_e_v1_bf16.safetensors --serialization-baseline dt-models/ltx_2.3_22b_distilled_f16.ckpt --serialization-guard-profile ltx2_3 --serialization-report-limit 5`
- observed:
   - `type_mismatch_count=0`
   - `format_mismatch_count=0`
   - `datatype_mismatch_count=0`
   - `RESULT=PASS`

Runtime probe after content-copy apply (one-response, same low-cost settings):

- server start:
   - `drawthings-grpc --address 127.0.0.1 --port 7861 --gpu 0 --no-tls --model-browser --no-response-compression dt-models`
- probe:
   - `python tools/dt_api_client.py --host 127.0.0.1:7861 --max-recv-bytes 134217728 generate-raw --config-bin output/probe_custom_f16_aligned_config.bin --prompt "a cinematic shot of a red sports car driving on a mountain road at sunset, detailed, realistic" --negative-prompt "blurry, distorted, low quality, artifacts" --output-dir output/probe_custom_f16_after_contentcopy_20260531 --chunked --save-preview --max-responses 1`
- observed server outcome:
   - request accepted (`Received image processing request, begin.`)
   - same crash root remained:
      - `ccv_nnc_tensor_read`
      - `ccv_cnnp_model_read`
   - crash: `Bad pointer dereference`
   - no emitted payload files in `output/probe_custom_f16_after_contentcopy_20260531`

Interpretation:

- Targeted content parity remediation on the previously divergent domains succeeded at the measured signature level (`0` mismatch across `261 + 1248` selected rows), but runtime loading still crashes in the same reader path.
- The remaining incompatibility is therefore deeper than metadata policy, length parity, and these targeted dim/data blob substitutions.

### Tensor row-order parity probe (2026-05-31)

Report artifacts:

- `output/probe_row_order_diff_20260531.json`
- `output/probe_row_order_diff_20260531.txt`

Method:

- row-order comparison by `rowid` between:
   - custom: `dt-models/10_e_v1_bf16_fix2_f16.ckpt`
   - baseline: `dt-models/ltx_2.3_22b_distilled_f16.ckpt`
- robust per-rowid reads with DataError handling to avoid bulk-read failure on unreadable rows.

Observed:

- `custom_count=5746`
- `baseline_count=5746`
- `position_mismatch_count=5746`
- `position_match_count=0`
- `mismatch_ratio=1.000000`
- unreadable row position differs:
   - custom unreadable rowid: `5746`
   - baseline unreadable rowid: `3`

Representative ordering deltas:

- position `1`:
   - custom: `text_video_connector_learnable_registers`
   - baseline: `__text_feature_extractor__[t-audio_aggregate_embed-0-0]`
- position `2`:
   - custom: `text_audio_connector_learnable_registers`
   - baseline: `__text_feature_extractor__[t-audio_aggregate_embed-0-1]`
- large absolute deltas exceed `5k` rows for multiple connector/feature-extractor tensors.

Interpretation:

- Beyond content parity metrics, table iteration order itself is fully divergent from baseline.
- If runtime loader behavior depends on row order (even partially), this is a plausible crash vector independent of keyset/metadata/length checks.

### Readable-row reorder experiment status (2026-05-31)

Tooling:

- `tools/dt_reorder_ckpt_readable_rows.py`

Dry-run summary:

- baseline readable names (`rowid`-scan excluding DataError rows): `5745`
- baseline unreadable rowids: `[3]`
- target missing names from baseline readable set: `0`
- pre-reorder readable position mismatch: `5745`

Apply attempt status:

- Native/sqlite and Python single-shot apply attempts did hit repeated `D+` behavior earlier in this container.
- A resumed micro-batch apply path was then executed successfully using:
   - `python tools/dt_reorder_ckpt_readable_rows.py --file dt-models/10_e_v1_bf16_fix2_f16.ckpt --baseline dt-models/ltx_2.3_22b_distilled_f16.ckpt --mode apply --chunk-size 96 --sample-limit 5`
- completed outcome:
   - `post_position_mismatch=0`
   - `RESULT=PASS`
- unreadable-row constraint remains:
   - baseline unreadable row at `rowid=3` and custom unreadable row at `rowid=5746` cannot be moved (`sqlite3` rowid-update attempts for those rows return `string or blob too big (18)`).

Runtime probe after successful readable-row reorder (one-response, same low-cost settings):

- server start:
   - `drawthings-grpc --address 127.0.0.1 --port 7861 --gpu 0 --no-tls --model-browser --no-response-compression dt-models`
- probe:
   - `python tools/dt_api_client.py --host 127.0.0.1:7861 --max-recv-bytes 134217728 generate-raw --config-bin output/probe_custom_f16_aligned_config.bin --prompt "a cinematic shot of a red sports car driving on a mountain road at sunset, detailed, realistic" --negative-prompt "blurry, distorted, low quality, artifacts" --output-dir output/probe_custom_f16_after_roworder_20260531 --chunked --save-preview --max-responses 1`
- observed server outcome:
   - request accepted (`Received image processing request, begin.`)
   - same crash root remained:
      - `ccv_nnc_tensor_read`
      - `ccv_cnnp_model_read`
   - crash: `Bad pointer dereference` (process exit `139`)
   - no emitted payload files (`output/probe_custom_f16_after_roworder_20260531` not created)

Operational implication:

- Readable-row order divergence was real and is now remediated (`5745` readable rows aligned to compressed baseline order).
- Runtime crash persistence after reorder means row-order mismatch was not sufficient to explain the loader failure.
- Remaining high-signal differentiator is the persistent unreadable tensor row path / deeper serialization invariants inside the loader.

### Metadata diff evidence (no regeneration)

Report artifacts:

- `output/probe_metadata_diff_20260529.json`
- `output/probe_metadata_diff_20260529.md`

Key counts from `custom_q6` vs `official_q6`:

- `format_diff_cq6_vs_oq6 = 5743`
- `type_diff_cq6_vs_oq6 = 1837`
- `datatype_diff_cq6_vs_oq6 = 1545`
- `large_data_len_delta_cq6_vs_oq6 = 1689`
- `quantized_only_in_custom_q6_vs_cf16 = 260`

Key counts from `custom_f16` vs `official_q6`:

- `format_diff_cf16_vs_oq6 = 5743`
- `type_diff_cf16_vs_oq6 = 3904`
- `datatype_diff_cf16_vs_oq6 = 1545`

Top mismatch families:

- `__dit__`
- `__text_video_connector__`
- `__text_audio_connector__`
- `__text_feature_extractor__`

Representative mismatches:

- Many custom tensors are stored with `format=1` where official q6 uses `format=2`.
- Many custom tensors keep `datatype=131072` where official q6 uses `datatype=16384` or `datatype=524288`.
- `custom_q6` quantizes 260 tensors that official q6 effectively keeps at F16-style metadata behavior (notably connector `down_proj` blocks and `proj_out` blocks).

### Toolchain compatibility note

Attempt to run a source-matched local `gRPCServerCLI` build was blocked in this Linux container:

- Swift build for `gRPCServerCLI` failed with `no such module 'CoreML'` while resolving `LocalImageGenerator` dependencies.

This prevented direct same-source runtime validation in the current environment.

### Probe scalability note

Attempts to run full-table content signature probes (head/tail byte extraction across all shared rows) in this container hit practical limits:

- intermittent `sqlite3.DataError: string or blob too big` on broad SQL/head-signature passes,
- process termination (`exit 137`) on heavier full-table signature loops.

Operational implication:

- Keep content-level analysis targeted (family-scoped or explicit tensor-name lists) with strict sampling, rather than all-row deep signature sweeps in one pass.
- Micro-batched subgroup windows are currently the reliable execution pattern for `__dit__` content probes in this environment.

## Runtime Numeric Snapshot (Image Tensor Distribution)

Custom q6 image tensor (`output/dt_video_20260528_171805/image_r0014_01.bin`):

- min/max: `-3.275391 / 4.406250`
- p1/p50/p99: `-1.636719 / 0.154053 / 1.933594`

Official q6 image tensor (`output/dt_video_20260528_171821/image_r0014_01.bin`):

- min/max: `-1.233398 / 1.217773`
- p1/p50/p99: `-0.958496 / -0.616211 / 1.006836`

Interpretation:

- Custom output distribution remains wider and inconsistent with official behavior, matching observed noisy decode.

## Fresh Regenerated F16 Attempt (2026-05-31)

Command:

```bash
tools/dt_convert_model.sh \
   --file dt-models/10_e_v1_bf16.safetensors \
   --name 10_e_v1_bf16_regen_20260531 \
   --output-directory dt-models
```

Observed conversion status:

- converter import/encode completed successfully in-session (`elapsed=264s`)
- generated file: `dt-models/10_e_v1_bf16_regen_20260531_f16.ckpt` (about `42G`)

Post-conversion validation status (serialization parity guard vs official exact F16 baseline):

- `missing_from_converted=0`
- `extra_in_converted=0`
- `unreadable_metadata_both=1` (`__text_feature_extractor__[t-video_aggregate_embed-0-0]`)
- `type_mismatch_count=5745`
- `format_mismatch_count=5743`
- `datatype_mismatch_count=1545`
- `RESULT=FAIL`
- wrapper exit status: `1` (`error: converted checkpoint failed integrity validation`)

Interpretation:

- This regenerated checkpoint does not preserve the previously achieved serialization metadata parity and is not a runtime candidate.
- The run confirms that a direct fresh conversion from safetensors (with current converter state) currently regresses to broad serialization-policy mismatch against the official exact F16 baseline.

## Conclusion

1. Converter/importer structural issues were real and are now fixed.
2. Current runtime still crashes when loading custom checkpoints (both fixed F16 and derived q6) in `ccv_nnc_tensor_read` / `ccv_cnnp_model_read`.
3. The blocker is deeper read-path compatibility beyond the fixes applied so far. Matching `type/format/datatype` policy and achieving `data` blob length parity (`0` mismatches on `5745` readable shared tensors) did not resolve the crash. Additional targeted content-copy remediation also reached measured parity on the previously divergent domains (`261/261` non-`__dit__` rows and `1248/1248` `__dit__` subset80 rows now full-match on the probe metrics), and readable-row reorder was completed to `post_position_mismatch=0`, but runtime still crashed in the same loader path.
4. q4p remains unstable and should be treated as non-usable.
5. No new long quantization/conversion run should be started until serialization compatibility checks are enforced.

## Immediate Operational Guidance

- Keep using official `ltx_2.3_22b_distilled_1.1_q6p.ckpt` as the runtime baseline for day-to-day generation.
- Use official `ltx_2.3_22b_distilled_f16.ckpt` as a control when validating loader stability/signpost progression.
- Treat custom `10_e_v1_bf16_fix2_f16.ckpt`, `10_e_v1_bf16_q6p.ckpt`, and `10_e_v1_bf16_q4p.ckpt` as non-runnable in the current runtime environment.
- Before any new 4+ hour regeneration, require metadata diff checks against the official q6 baseline (see `output/probe_metadata_diff_20260529.*`) and keep targeted content-signature probes in the acceptance gate.
- Since targeted content-copy parity and readable-row reorder did not clear runtime crash, prioritize deeper reader-path diagnostics next (row-encoding invariants, tensor record decoding assumptions, and focused investigation of the persistent unreadable row path).

## Detailed Analysis and Next Working Plan (2026-05-31)

### Current snapshot

- Fresh regenerated artifact (`dt-models/10_e_v1_bf16_regen_20260531_f16.ckpt`) completes converter import/encode, but fails integrity guard with broad serialization mismatches:
   - `type_mismatch_count=5745`
   - `format_mismatch_count=5743`
   - `datatype_mismatch_count=1545`
- Existing remediated artifact (`dt-models/10_e_v1_bf16_fix2_f16.ckpt`) still validates with strict metadata parity vs official exact F16 baseline:
   - `type_mismatch_count=0`
   - `format_mismatch_count=0`
   - `datatype_mismatch_count=0`
   - `unreadable_metadata_both=1` (`__text_feature_extractor__[t-video_aggregate_embed-0-0]`)
- Runtime crash signature remains unchanged on custom artifacts in one-response probes:
   - `ccv_nnc_tensor_read`
   - `ccv_cnnp_model_read`
   - server exit `139`

### What prior experiments proved

1. Importer/mapping structural defects were real and were fixed (keyset parity reached `5746` with no missing/extra keys).
2. Metadata-column parity (`type/format/datatype`) can be enforced to zero mismatch on readable shared tensors.
3. Blob-length parity can be forced to zero mismatch on readable shared tensors.
4. Targeted content-copy parity can be achieved for measured non-`__dit__` and dominant `__dit__` subset domains.
5. Readable-row order can be aligned to baseline (`post_position_mismatch=0`).
6. Despite 2-5, runtime still crashes in the same loader path, narrowing blocker to deeper read-path invariants.
7. Fresh direct conversion regresses immediately to broad metadata mismatch profile, so converter output is still not directly runtime-compatible with current strict baseline policy.

### Most likely remaining blockers (ranked)

1. Residual full-scope content semantic divergence outside previously targeted subsets.
2. Unreadable-row handling/ordering edge case around `__text_feature_extractor__[t-video_aggregate_embed-0-0]`.
3. Converter write-path serialization invariants not yet matching official baseline conventions (hence repeated fresh-output metadata mismatch profile).

### Immediate plan before any new conversion run

1. Do not start another long conversion yet.
2. Use current regenerated file (`10_e_v1_bf16_regen_20260531_f16.ckpt`) as remediation substrate.
3. Apply deterministic metadata alignment to baseline (`dt_align_ckpt_metadata.py --mode apply`), then re-validate.
4. Apply broader content-alignment pass (not only prior subset windows), then re-probe mismatch families.
5. Rebuild tensors table in exact baseline order via copy/reinsert strategy (instead of rowid-only moves) to include the unreadable-row position case.
6. Re-run the same low-cost one-response runtime probe harness.
7. Only if this stabilized path still crashes, prioritize deeper reader-path diagnostics over launching another fresh conversion.

### WAL incident root cause and prevention (2026-05-31)

Observed root cause from tooling audit:

- `tools/dt_reorder_ckpt_readable_rows.py` previously forced `PRAGMA journal_mode=WAL` on target checkpoints.
- This journal mode persists per SQLite database file, so later large write operations on the same `.ckpt` can append very large `-wal` sidecars.
- The earlier disk spike (about `40G` WAL sidecar on `10_e_v1_bf16_fix2_f16.ckpt`) is consistent with this behavior under large write transactions.

Confirmed state after cleanup:

- `dt-models/10_e_v1_bf16_fix2_f16.ckpt` was checkpoint-truncated and reset to `journal_mode=delete`.
- No active `-wal` sidecar remains for current model files.

Preventive controls now added to mutation tools:

- `tools/dt_reorder_ckpt_readable_rows.py`
- `tools/dt_align_ckpt_metadata.py`
- `tools/dt_align_ckpt_content_subset.py`
- `tools/dt_align_ckpt_payload_subset.py`

All now support and default to:

- `--journal-mode delete` (safe default; no persistent WAL requirement)
- chunked write transactions (`--chunk-size ...`) instead of single huge transactions
- disk guard rail (`--min-free-gb ...`) with abort on low free space
- WAL checkpoint+truncate per chunk only when WAL mode is explicitly active

Operational policy going forward (space-constrained runs):

1. Before any large apply step, verify `PRAGMA journal_mode` is `delete` for the target `.ckpt`.
2. Run mutation scripts with conservative chunk sizes and `--min-free-gb 180` (or higher) in this low-host-space setup.
3. If WAL mode must be used for a special case, checkpoint-truncate immediately after each chunk and reset to `delete` at the end.
4. Never run long single-transaction blob rewrite passes on low headroom.

## Incremental Remediation Update (2026-05-31, pre-final-validation checkpoint)

### Confirmed successful steps in this pass

- Metadata alignment apply on regenerated artifact completed under `journal_mode=delete` and free-space guard rails.
- Strict validator rerun after metadata alignment reached full serialization parity against official exact F16 baseline:
   - `type_mismatch_count=0`
   - `format_mismatch_count=0`
   - `datatype_mismatch_count=0`
   - `RESULT=PASS`
- Payload subset alignment apply succeeded:
   - `selected_tensor_names=1542`
   - `rows_updated=1542`
   - `post_selected_data_len_mismatch=0`
   - `RESULT=PASS`
- Content subset alignment apply succeeded:
   - `dim_names_selected=1312`
   - `data_names_selected=115`
   - `union_selected=1363`
   - `post_dim_head_mismatch=0`
   - `post_data_head_mismatch=0`
   - `RESULT=PASS`
- Disk safety remained within guard rails across apply steps (observed free space about `320 GiB` with `--min-free-gb 180`).

### Reorder status (completed)

- Readable-row reorder apply completed on `dt-models/10_e_v1_bf16_regen_20260531_f16.ckpt`.
- Confirmed preconditions:
   - `baseline_readable_names=5745`
   - `missing_names_in_target=0`
- Recovery note:
   - A fast restart with default `--rowid-offset 10000000` hit `sqlite3.IntegrityError: UNIQUE constraint failed: tensors.rowid` because partial phase2 state already occupied `>= 10000000` rowids.
   - Resume strategy switched to `--rowid-offset 20000000` to avoid collisions while preserving data.
- Final completion metrics:
   - `phase1_chunk=23/23`
   - `phase2_chunk=23/23`
   - `post_position_mismatch=0`
   - `RESULT=PASS`
   - command exit status: `0`

### Post-reorder validation and runtime retry (completed)

- Strict validator rerun after reorder remained clean vs official exact F16 baseline:
   - `missing_from_converted=0`
   - `extra_in_converted=0`
   - `type_mismatch_count=0`
   - `format_mismatch_count=0`
   - `datatype_mismatch_count=0`
   - `RESULT=PASS`
- Runtime retry sequence on regenerated F16 model (`10_e_v1_bf16_regen_20260531_f16.ckpt`):
   - server relaunched cleanly on `127.0.0.1:7861` via `gRPCServerCLI`
   - probe config regenerated to explicitly target regen artifact:
      - `output/probe_regen_f16_aligned_config.bin`
      - embedded model string confirmed: `10_e_v1_bf16_regen_20260531_f16.ckpt`
   - first retry with `--max-responses 1` intentionally cancelled stream early after `textEncoded` (server stayed healthy)
   - full-stream retry with `--max-responses 0` completed successfully:
      - responses: `15`
      - preview frames: `5`
      - images written: `1`
      - output dir: `output/probe_regen_f16_after_reorder_20260531_retry2`
   - server log outcome after full-stream retry:
      - `Image processed`
      - `Image processed successfully, should send in chunks? true`

Operational implication at this checkpoint:

- Regenerated/reordered F16 path now passes strict serialization guard and completes at least one end-to-end runtime generation request without reproducing the earlier immediate loader crash.
- q6p fallback quantization is not required for this checkpoint validation gate.

## No-Upscaling Differential Retest (2026-05-31, late)

### Test A: custom entry, explicit no-upscaling, low guidance

Command:

```bash
DT_HIRES_FIX=false \
DT_HIRES_FIX_WIDTH=0 \
DT_HIRES_FIX_HEIGHT=0 \
DT_GUIDANCE=1.0 \
DT_SHIFT=5.0 \
DT_MODEL="10_e_v1_bf16_regen (LTX-2.3 custom)" \
DT_WIDTH=384 DT_HEIGHT=704 DT_STEPS=8 DT_TEST_ONE_FRAME=1 DT_FPS_ID=5 \
bash tools/dt_generate_video.sh \
  "a cinematic shot of a red sports car driving on a mountain road at sunset, detailed, realistic" \
  "blurry, distorted, low quality, artifacts"
```

Observed:

- config confirmed `hires_fix: False`
- server crashed with `Illegal instruction` in:
   - `TextEncoder.encodeLTX2(...)`
- client failed with `gRPC error: UNAVAILABLE: Socket closed`
- output dir contained config only:
   - `output/dt_video_20260531_221156/config.bin`

### Test B: direct file model name, same settings (custom-entry bypass)

Command:

```bash
DT_HIRES_FIX=false \
DT_HIRES_FIX_WIDTH=0 \
DT_HIRES_FIX_HEIGHT=0 \
DT_GUIDANCE=1.0 \
DT_SHIFT=5.0 \
DT_MODEL="10_e_v1_bf16_regen_20260531_f16.ckpt" \
DT_WIDTH=384 DT_HEIGHT=704 DT_STEPS=8 DT_TEST_ONE_FRAME=1 DT_FPS_ID=5 \
bash tools/dt_generate_video.sh \
  "a cinematic shot of a red sports car driving on a mountain road at sunset, detailed, realistic" \
  "blurry, distorted, low quality, artifacts"
```

Observed:

- config confirmed `hires_fix: False`
- server crashed with `Bad pointer dereference` in:
   - `ccv_nnc_tensor_read`
   - `ccv_cnnp_model_read`
- run produced config only:
   - `output/dt_video_20260531_221515/config.bin`
- no image/audio tensor payloads were emitted before termination.

### Interpretation

- Downloading LTX-2.3 hires-fix upscaler checkpoints is not required to explain this failure path.
- Failures reproduce with hires-fix explicitly disabled and with `latents_upscalers: []` in custom metadata.
- The active blocker is upstream of final decode/upscaling and remains in runtime model/tensor read or text-encoding paths.

## Custom Entry Remap To Official Original F16 (2026-06-01)

Goal:

- test whether the custom-entry crash path persists when the custom entry points to the official original F16 checkpoint.

Metadata change applied:

- in `dt-models/custom.json`, entry `10_e_v1_bf16_regen (LTX-2.3 custom)` was remapped to:
   - `file: ltx_2.3_22b_distilled_f16.ckpt`
   - `clip_encoder: ltx_2.3_22b_distilled_f16.ckpt`

Retest command (same no-upscaling settings as prior custom-entry repro):

```bash
DT_HIRES_FIX=false \
DT_HIRES_FIX_WIDTH=0 \
DT_HIRES_FIX_HEIGHT=0 \
DT_GUIDANCE=1.0 \
DT_SHIFT=5.0 \
DT_MODEL="10_e_v1_bf16_regen (LTX-2.3 custom)" \
DT_WIDTH=384 DT_HEIGHT=704 DT_STEPS=8 DT_TEST_ONE_FRAME=1 DT_FPS_ID=5 \
bash tools/dt_generate_video.sh \
  "a cinematic shot of a red sports car driving on a mountain road at sunset, detailed, realistic" \
  "blurry, distorted, low quality, artifacts"
```

Observed:

- config confirmed `hires_fix: False`
- server still crashed with `Illegal instruction` at:
   - `TextEncoder.encodeLTX2(...)`
- process exit status: `132`
- run output was config-only:
   - `output/dt_video_20260601_034218/config.bin`

Interpretation:

- Switching the custom entry to the official original F16 checkpoint does not clear the custom-entry crash path.
- This strongly suggests the blocker is in the custom-entry runtime branch / text-encoding path for this request shape, not in hires-fix assets and not solely in the regenerated checkpoint payload.

## Custom Model Resolution Audit + q6p Rebaseline (2026-06-01, follow-up)

### New root-cause findings from model-resolution checks

- `dt-models/custom.json` entry `10_e_v1_bf16 (LTX-2.3 custom)` referenced a missing file:
   - `10_e_v1_bf16_q6p.ckpt` (not present on disk)
- Requested custom model strings are only used directly when `ModelZoo.isModelDownloaded(model)` is true; otherwise generation falls back to `ModelZoo.defaultSpecification.file`.
- Current default specification in this codebase is:
   - `ltx_2.3_22b_distilled_1.1_q8p.ckpt`
- That default q8p checkpoint was also missing locally in this session.

Operational implication:

- Prior custom-entry crash runs were confounded by model-resolution fallback (requested custom alias did not guarantee loading the intended checkpoint).

### Corrective config changes applied

`dt-models/custom.json` was corrected to avoid missing-file references and accidental file-key collision with official exact F16:

- entry `10_e_v1_bf16 (LTX-2.3 custom)`:
   - `file: ltx_2.3_22b_distilled_1.1_q6p.ckpt`
   - `clip_encoder: ltx_2.3_22b_distilled_1.1_q6p.ckpt`
- entry `10_e_v1_bf16_regen (LTX-2.3 custom)`:
   - `file: 10_e_v1_bf16_regen_20260531_f16.ckpt`
   - `clip_encoder: 10_e_v1_bf16_regen_20260531_f16.ckpt`

### q6p test-path bootstrap status

Goal for immediate testing was to restore a known-good q6p control path.

Download command:

```bash
bash tools/dt_models_cli.sh ensure ltx_2.3_22b_distilled_1.1_q6p.ckpt \
  --no-include-dependencies \
  --models-dir /workspaces/drawthings-linux-toolkit/dt-models
```

Observed:

- model resolver recognizes this q6p file and provides the direct static URL.
- expected file size is about `19.98GB`.
- background/resumable download is active, writing:
   - `dt-models/ltx_2.3_22b_distilled_1.1_q6p.ckpt.part`

Pending gate to run q6p validation:

- wait for `ltx_2.3_22b_distilled_1.1_q6p.ckpt` to finish and pass checksum,
- then rerun no-upscaling one-frame probe using q6p first as the baseline control.

## Local q6p Quantization Sanity + Runtime Retest (2026-06-01)

Question addressed:

- whether the locally produced q6p artifact was corrupted due interrupted/retried quantization control flow.

Sanity checks on local q6p artifact:

- file under test:
   - `dt-models/10_e_v1_bf16_regen_20260531_q6p.ckpt`
- SQLite quick integrity check:
   - `PRAGMA quick_check` -> `ok`
- tensor table count:
   - `SELECT COUNT(*) FROM tensors` -> `5746`
- observed file size in-session:
   - about `17G`

Conclusion from sanity checks:

- no evidence of file corruption from the quantization run.

Custom metadata used for q6p retest:

- `dt-models/custom.json` entry `10_e_v1_bf16 (LTX-2.3 custom)` remapped to:
   - `file: 10_e_v1_bf16_regen_20260531_q6p.ckpt`
   - `clip_encoder: 10_e_v1_bf16_regen_20260531_q6p.ckpt`

Retest command (same no-upscaling control profile):

```bash
DT_HIRES_FIX=false \
DT_HIRES_FIX_WIDTH=0 \
DT_HIRES_FIX_HEIGHT=0 \
DT_GUIDANCE=1.0 \
DT_SHIFT=5.0 \
DT_MODEL="10_e_v1_bf16 (LTX-2.3 custom)" \
DT_WIDTH=384 DT_HEIGHT=704 DT_STEPS=8 DT_TEST_ONE_FRAME=1 DT_FPS_ID=5 \
bash tools/dt_generate_video.sh \
  "a cinematic shot of a red sports car driving on a mountain road at sunset, detailed, realistic" \
  "blurry, distorted, low quality, artifacts"
```

Observed:

- server crash persisted with same signature:
   - `Illegal instruction`
   - `TextEncoder.encodeLTX2(...)`
- gRPC client result:
   - `UNAVAILABLE: Socket closed`
- server process exit:
   - `132`

Interpretation:

- the q6p artifact itself is structurally readable and not obviously corrupted,
- crash persistence indicates the blocker remains in runtime text-encoding/read-path behavior, not a trivial incomplete-write corruption case.

## Official Downloaded q6p Differential Check (2026-06-01)

Goal:

- verify whether the same immediate crash reproduces when using the real downloaded official q6p checkpoint.

Artifact verification:

- file:
   - `dt-models/ltx_2.3_22b_distilled_1.1_q6p.ckpt`
- local size:
   - about `20G`
- SQLite sanity:
   - `PRAGMA quick_check` -> `ok`
   - `SELECT COUNT(*) FROM tensors` -> `5746`

Control run command (same no-upscaling profile):

```bash
DT_HIRES_FIX=false \
DT_HIRES_FIX_WIDTH=0 \
DT_HIRES_FIX_HEIGHT=0 \
DT_GUIDANCE=1.0 \
DT_SHIFT=5.0 \
DT_MODEL="ltx_2.3_22b_distilled_1.1_q6p.ckpt" \
DT_WIDTH=384 DT_HEIGHT=704 DT_STEPS=8 DT_TEST_ONE_FRAME=1 DT_FPS_ID=5 \
bash tools/dt_generate_video.sh \
  "a cinematic shot of a red sports car driving on a mountain road at sunset, detailed, realistic" \
  "blurry, distorted, low quality, artifacts"
```

Observed:

- stream reached:
   - `response #1` with signpost `textEncoded`
   - `response #2` with signpost `imageEncoded`
- no immediate `Illegal instruction` / `TextEncoder.encodeLTX2` crash was observed on this official q6p path during that run window.
- run termination in this session was operator-induced (`pkill`, exit `143`) during follow-up process cleanup, not a server crash signature.

Interpretation:

- the real official q6p control does **not** reproduce the same immediate failure pattern seen on custom q6p/f16 aliases in this environment.
- this reinforces that the blocker is custom-path/model-path specific, not a universal q6p runtime incompatibility.

## Official q6p Through custom.json Alias (2026-06-01)

Goal:

- test the same official q6p checkpoint, but resolved via `custom.json`, to isolate whether custom-spec resolution itself triggers the failure.

Custom alias added:

- in `dt-models/custom.json`:
   - `name: official_q6p_via_custom (LTX-2.3 custom)`
   - `file: ltx_2.3_22b_distilled_1.1_q6p.ckpt`
   - `clip_encoder: ltx_2.3_22b_distilled_1.1_q6p.ckpt`
   - `text_encoder: gemma_3_12b_it_qat_q8p.ckpt`

Retest command (same no-upscaling control profile):

```bash
DT_HIRES_FIX=false \
DT_HIRES_FIX_WIDTH=0 \
DT_HIRES_FIX_HEIGHT=0 \
DT_GUIDANCE=1.0 \
DT_SHIFT=5.0 \
DT_MODEL="official_q6p_via_custom (LTX-2.3 custom)" \
DT_WIDTH=384 DT_HEIGHT=704 DT_STEPS=8 DT_TEST_ONE_FRAME=1 DT_FPS_ID=5 \
bash tools/dt_generate_video.sh \
  "a cinematic shot of a red sports car driving on a mountain road at sunset, detailed, realistic" \
  "blurry, distorted, low quality, artifacts"
```

Observed:

- server crashed with the same signature seen on prior custom-path runs:
   - `Illegal instruction`
   - `TextEncoder.encodeLTX2(...)`
- gRPC client result:
   - `UNAVAILABLE: Socket closed`
- server process exit:
   - `132`

Interpretation:

- with identical checkpoint bytes (`ltx_2.3_22b_distilled_1.1_q6p.ckpt`), direct-file run and custom-alias run diverge.
- this strongly localizes the failure to custom-spec resolution/runtime-path behavior rather than the official q6p file payload itself.

## custom.json Alias Root Cause + Tooling Fix (2026-06-01)

Correction from additional control:

- A direct-file probe using the same official q6p file while the custom override entry existed still reached:
   - `response #1` with signpost `textEncoded`
   - client exit `0` with `--max-responses 1`
- This indicates the immediate crash was not caused by merely having a custom spec row for that file.

Root cause identified:

- `tools/dt_make_config.py` previously wrote `--model` into config verbatim.
- when `--model` was a custom alias name (for example, `official_q6p_via_custom (LTX-2.3 custom)`), runtime did not treat it as a downloaded file key.
- in `LocalImageGenerator.generateTextOnly`, model selection uses:
   - `ModelZoo.isModelDownloaded(configuration.model)`
   - if false, fallback to `ModelZoo.defaultSpecification.file`
- therefore alias strings could silently fall back to default model selection instead of the intended file path.

Fix implemented:

- `tools/dt_make_config.py` now resolves custom aliases from `dt-models/custom.json` (`name -> file`) before writing config.
- logs explicit resolution when applied, for example:
   - `resolved model alias: official_q6p_via_custom (LTX-2.3 custom) -> ltx_2.3_22b_distilled_1.1_q6p.ckpt (custom.json:name)`

Post-fix validation:

- config generated from alias now records:
   - `model: ltx_2.3_22b_distilled_1.1_q6p.ckpt`
- one-response probe using the resolved config reached:
   - `response #1` with signpost `textEncoded`
   - stream closed cleanly at `--max-responses 1` (exit `0`)

Operational conclusion:

- The custom alias path in toolkit config generation is now normalized to file keys.
- This removes alias-string fallback as a confounder for further custom.json investigations.

## Post-fix Non-Truncated Alias Stream Check (2026-06-01)

Goal:

- run alias-based generation without `--max-responses 1` to confirm behavior beyond one-response probe.

Runs executed:

- primary profile:
   - config: `output/probe_alias_resolve_config_20260601.bin`
   - shape: `384x704`, `steps=8`
   - output dir: `output/probe_alias_resolve_full_after_fix_20260601`
- fast profile:
   - config: `output/probe_alias_full_fast_config_20260601.bin`
   - shape: `256x256`, `steps=4`, `seed=424242`
   - output dir: `output/probe_alias_resolve_full_fast_20260601`

Common observed stream signals:

- `response #1` with signpost `textEncoded`
- `response #2` with signpost `imageEncoded`
- each response showed `chunk state: LAST_CHUNK`
- no immediate `Illegal instruction` / `TextEncoder.encodeLTX2` crash in these post-fix runs.

Observed lifecycle state during this session:

- both non-truncated probes remained active after `response #2` until operator cleanup.
- output directories above remained empty before cleanup.
- termination was operator-induced (`pkill`, exit `143`), not a server crash signature.

Interpretation:

- alias normalization fix is sufficient to clear the prior immediate custom-alias text-encoder crash path.
- in this session window, full-stream closure/file emission was not observed before manual cleanup, so completion behavior remains a follow-up check.

## Custom Regen Alias Recheck: Preview-Only Stream End (2026-06-01, latest)

Goal:

- verify whether the custom regen alias can complete a one-frame run with a final `generatedImages` payload after recent metadata adjustments.

Metadata adjustment used for this recheck:

- in `dt-models/custom.json`, the two regen-related custom entries were updated to use the official clip encoder while keeping the regen q6p checkpoint as the primary `file`:
   - `name: 10_e_v1_bf16 (LTX-2.3 custom)`
   - `name: 10_e_v1_bf16_regen (LTX-2.3 custom)`
   - `clip_encoder: ltx_2.3_22b_distilled_1.1_q6p.ckpt`
   - `file: 10_e_v1_bf16_regen_20260531_q6p.ckpt`

Retest command:

```bash
DT_HIRES_FIX=false \
DT_GUIDANCE=1.0 \
DT_SHIFT=5.0 \
DT_MODEL="10_e_v1_bf16_regen (LTX-2.3 custom)" \
DT_WIDTH=384 DT_HEIGHT=704 DT_STEPS=8 DT_TEST_ONE_FRAME=1 \
bash tools/dt_generate_video.sh \
  "a cinematic shot of a red sports car driving on a mountain road at sunset, detailed, realistic" \
  "blurry, distorted, low quality, artifacts"
```

Observed:

- no immediate server crash during early phases; stream progressed through:
   - `response #1` -> `textEncoded`
   - `response #2` -> `imageEncoded`
   - later responses -> `sampling`
- stream ended normally (client exit `0`) with summary:
   - `responses: 11`
   - `images written: 0`
   - `preview frames seen: 5`
   - warning emitted: `stream ended without final generatedImages payloads; only preview frames were received`
- wrapper produced playable fallback from preview frames:
   - `output/dt_video_20260601_081915/playable.png`
   - `output/dt_video_20260601_081915/playable.gif`
   - `output/dt_video_20260601_081915/playable.mp4`

Interpretation:

- this run is **not** a successful final-image generation for validation purposes.
- although the fast-crash path improved, the model request still failed the required completion criterion because no final `generatedImages` payload was returned.
- preview fallback artifacts should be treated as diagnostic output only (often low-data/blank) and not as proof of correct generation completion.

## Deep Analysis: Why custom `ltx_2.3_22b_distilled_q6p.ckpt` fails while official `1.1_q6p` works (2026-06-01)

Question addressed:

- user reported that `ltx_2.3_22b_distilled_1.1_q6p.ckpt` runs, while newly quantized `ltx_2.3_22b_distilled_q6p.ckpt` crashes/fails similarly to prior custom models.

### Control integrity checks (structure only)

All three checkpoints pass fast structural validation and LTX2.3 required-key checks:

- `dt-models/ltx_2.3_22b_distilled_1.1_q6p.ckpt`
- `dt-models/ltx_2.3_22b_distilled_q6p.ckpt`
- `dt-models/ltx_2.3_22b_distilled_f16.ckpt`

So this is not a simple "bad SQLite file" issue.

### Official q6p vs custom q6p deep diff (high signal)

`output/probe_deep_diff_ltx_q6p_vs_official_20260601.md` summary:

- tensor sets match (`5746` shared; no missing/extra)
- `metadata_mismatch_type=1837`
- `metadata_mismatch_datatype=1545`
- `data_len_mismatch=3621`
- `data_head_mismatch=5641`
- `full_signature_match=0`
- mismatch concentration is dominated by `__dit__` family.

Representative mismatch examples:

- many `__dit__[t-attn1_ada_ln_*]` rows changed from baseline datatype `16384` to file datatype `131072`.
- corresponding type values also shift (baseline `1` -> file `5570572582913`).

### Source f16 vs custom q6p targeted comparison

Fast row-wise comparison against the exact source file used for quantization (`ltx_2.3_22b_distilled_f16.ckpt`) shows:

- same tensor keyset (`5746` / `5746`)
- `metadata_mismatch_type=5745`
- `metadata_mismatch_datatype=1545`
- `data_len_mismatch=5678`
- mismatch concentration again dominated by `__dit__`.

Critical observation:

- source FP16 rows (`datatype=16384`) count: `1542`
- source FP16 preserved in custom q6p: `0`
- source FP16 changed in custom q6p: `1542`
    - mostly `__dit__` (`1540`), plus connector register tensors.

### Quantizer-path root cause hypothesis (most likely)

Quantizer behavior in `DRAW_THINGS_PATCH/Apps/ModelQuantizer/Quantizer.swift`:

- forced codec mode (`--target-codec q6p`) enters the `if let forcedCodec` branch.
- that branch quantizes almost all multi-dimensional tensors to the forced codec, with only a limited FP16 allowlist (`embedder`, `pos_embed`, `visual_proj`, `encoder_hid_proj`, `register_tokens`, `refiner_`).

In contrast, LTX2.3 model-specific path (`case .ltx2, .ltx2_3`) uses mixed policy:

- keeps some tensors FP16 (e.g., `embedder`, `pos_embed`, `-linear-`)
- keeps some higher precision via `q8p` route (e.g., `ada_ln`, convs)
- uses `q6p` for other tensors.

Interpretation:

- custom file likely over-quantized tensors that runtime expects in higher precision/mixed codecs.
- this is consistent with observed runtime failure patterns and with loss of the full FP16 subset (`1542 -> 0`).

### Practical conclusion

Most probable issue is **quantization policy mismatch**, not alias resolution and not basic DB corruption:

- official working q6p appears mixed-precision-compatible for runtime,
- custom forced-q6p output is globally more aggressive and likely invalid for LTX2.3 runtime expectations.

### Recommended next test

Re-quantize from `ltx_2.3_22b_distilled_f16.ckpt` using model-specific policy (no forced q6p), then retest generation:

```bash
bash tools/dt_quantize_model.sh \
   -i dt-models/ltx_2.3_22b_distilled_f16.ckpt \
   -m ltx2.3 \
   -o dt-models/ltx_2.3_22b_distilled_auto_mix.ckpt
```

If a forced codec is still required, quantizer code should be patched so LTX2.3 forced mode preserves the same fragile tensor families as the model-specific mixed policy.

## Incremental Update (2026-06-02): Probe Pack + Runtime/Server Checks

This addendum records the latest probe-heavy validation cycle, including clipfix2 parity checks, bounded runtime A/B behavior, GPU telemetry, and server startup verification.

### New artifacts from this cycle

- `output/probe_clipfix2_vs_official_clip_path_20260602.json`
- `output/probe_clipfix2_vs_official_clip_path_20260602.md`
- `output/probe_clipfix2_vs_sourcef16_clip_path_20260602.json`
- `output/probe_clipfix2_vs_sourcef16_clip_path_20260602.md`
- `output/probe_deep_diff_clipfix2_vs_official_q6p_20260602.json`
- `output/probe_deep_diff_clipfix2_vs_official_q6p_20260602.md`
- `output/probe_ab_official_cfg_20260602.bin`
- `output/probe_ab_clipfix2_cfg_20260602.bin`
- `output/gpu_probe_official_debug_20260602/client.log`
- `output/gpu_probe_official_debug_20260602/gpu_dmon.log`

### Probe result: clipfix2 vs official q6p (clip-path targeted)

Source report:

- `output/probe_clipfix2_vs_official_clip_path_20260602.md`

Summary stats:

- `selected_tensors=262`
- `readable_selected=261`
- `mismatch_any=16`
- `full_match=245`
- `metadata_mismatch_type=0`
- `metadata_mismatch_format=0`
- `metadata_mismatch_datatype=0`
- `dim_len_mismatch=0`
- `data_len_mismatch=0`
- `dim_head_mismatch=16`
- `data_head_mismatch=0`
- `data_small_sha256_mismatch=0`

Family map highlights:

- `__text_audio_connector__`: `16/128` mismatched (all `dim_head_mismatch`)
- `__text_video_connector__`: `0/128` mismatched
- `__text_feature_extractor__`: `0/4` mismatched with `1` unreadable-on-both row retained (`t-video_aggregate_embed-0-0`)
- connector learnable registers: `0/2` mismatched

Interpretation:

- Clip-path parity improved materially versus earlier forcedfix variants; remaining drift is narrow and concentrated in audio connector dim-head encoding.

### Probe result: clipfix2 vs source f16 (clip-path targeted)

Source report:

- `output/probe_clipfix2_vs_sourcef16_clip_path_20260602.md`

Summary stats:

- `selected_tensors=262`
- `readable_selected=261`
- `mismatch_any=261`
- `full_match=0`
- `metadata_mismatch_type=261`
- `metadata_mismatch_format=0`
- `metadata_mismatch_datatype=0`
- `dim_len_mismatch=0`
- `data_len_mismatch=0`
- `dim_head_mismatch=0`
- `data_head_mismatch=0`

Interpretation:

- Type-policy mismatch versus source f16 is expected for quantized output and should not be used alone as a runtime-fail signal.

### Probe result: clipfix2 vs official q6p (full deep diff)

Source report:

- `output/probe_deep_diff_clipfix2_vs_official_q6p_20260602.md`

Summary stats:

- `shared_tensors=5746`
- `full_signature_match=245`
- `metadata_mismatch_type=1540`
- `metadata_mismatch_datatype=1540`
- `data_len_mismatch=3325`
- `data_head_mismatch=5380`
- `data_small_sha256_mismatch=4088`
- top mismatch prefix remains `__dit__`

Interpretation:

- Even with strong clip-path improvement, whole-model divergence is still dominated by `__dit__` tensor payload/metadata differences.

### Runtime A/B probe update (bounded, response-capped)

Both official and clipfix2 were run through identical bounded streaming probes (`--max-responses 3`) to stabilize comparisons under intermittent long-run backend instability.

Observed parity for both paths:

- signpost sequence reached: `textEncoded -> imageEncoded -> sampling`
- clean exit status (`0`)
- `responses=3`
- `images written=0`, `audio written=0`, `preview frames seen=0` (expected due early stop)

Operational implication:

- clipfix2 no longer fails earlier than official in the same bounded runtime path.

### GPU telemetry caveat and fix

One-shot `nvidia-smi` snapshots in this devcontainer can misleadingly show `utilization.gpu=0` and process name `[Not Found]` during active inference.

Reliable method:

- run `nvidia-smi dmon -s u -d 1` during an active probe.

Observed on official debug probe:

- `samples=115`
- `max_sm=100`
- `avg_sm=31.4`

Conclusion:

- GPU kernels were executing; use interval telemetry instead of one-shot snapshots for runtime diagnosis.

### drawthings-start startup validation

`drawthings-start` was validated end-to-end in this container.

Resolution and defaults:

- `drawthings-start` is an alias to `drawthings-grpc`
- launcher defaults:
   - address `127.0.0.1`
   - port `7861`
   - GPU `0`
   - `--no-tls`
   - `--model-browser`
   - `--no-response-compression`
   - model dir from `DRAWTHINGS_MODEL_DIR` or `dt-models`

Health checks executed:

- default start on `127.0.0.1:7861` + successful Echo RPC
- port-override start (`DRAWTHINGS_PORT=7859`) + successful Echo RPC

Operational implication:

- Existing probe tooling can use `drawthings-start` directly (default 7861) or with `DRAWTHINGS_PORT=7859` for legacy host/port alignment.

### Next-run guidance

- Keep clip-path targeted probe and full deep diff together in the acceptance bundle.
- Use bounded A/B probes first to compare progression parity before launching full unbounded streams.
- For GPU diagnosis, always collect `dmon` telemetry in parallel with client logs.
