# Community Upgrade and MOSS Image Plan

Assessment date: 2026-10-23. This is an analysis and implementation plan, not an executed upgrade.

## 1. Recommendation

Upgrade the CUDA inference runtime in an isolated candidate checkout, preserve the working CPU conversion toolchain, and validate existing LTX checkpoints before introducing MOSS. Start MOSS with the official Draw Things conversions of both Base and SFT. Custom MOSS conversion and quantization should be a subsequent milestone.

The main findings change the scope substantially:

- Current upstream already implements MOSS inference, Base/SFT registrations, mixed-precision model builders, and LoRA import. We do not need to implement the architecture from scratch.
- The latest published server release explicitly adds CUDA 4-bit S support, but predates MOSS integration. Pulling the current Docker `latest` is not sufficient to obtain MOSS support.
- Our `q4p` workflow is not the new `i4x` / 4-bit S workflow.
- The generic upstream model importer still does not implement MOSS import. Running an official conversion and converting arbitrary MOSS weights are different deliverables.
- Our local SwiftPM build is a CPU utility build. Updating it does not update the CUDA server installed in the container.
- A blanket replay of the old patch bundle would restore obsolete dependency pins and omit newly required LLM sources. Rebase by purpose, not by whole-file snapshot.

The first useful outcome is a reproducibly pinned CUDA server that preserves LTX behavior and renders official MOSS Base and SFT checkpoints. It should not depend on completing a custom MOSS converter first.

## 2. Verified Baseline

| Surface | Observed state |
| --- | --- |
| Toolkit worktree before this report | Clean, on `main` |
| Community source HEAD | `9f3f04b7a0729a50384caf58179bed592044d64d`, 2026-05-18 |
| Community local changes | 16 modified tracked files and one new activation helper |
| Active s4nnc checkout | `c91108d94ebadc53b7e782e90c51e48c1162c4de`, 2026-04-30, locally patched |
| Active CCV checkout | `0b3bb15153c19f120a9e41f842c26e6ab6c09ece`, 2026-04-30, locally patched |
| Managed patch files | 17 community files, 2 s4nnc files, 6 CCV files |
| Snapshot consistency check | All 25 snapshot/source pairs are byte-identical |
| Active shell syntax check | All 20 active shell/manifest files pass `bash -n` |
| Local conversion binaries | Converter built July 13; quantizer July 23; LoRA converter July 16 |
| Inference executable | `/usr/local/bin/gRPCServerCLI`, invoked through the container launcher by default |
| Installed server SHA-256 | `5241dff6220e4f6fbcc7be7cc500b5e0df5f5331ffe30203d778e9c6f1f176ea` |
| Installed Swift | 6.0.3, Linux x86_64 |
| GPU | NVIDIA RTX A3000 Laptop, 12,288 MiB, SM86, driver 595.95 |
| Container-visible RAM | About 29 GiB total, 24 GiB available at inspection; 8 GiB swap |
| Container-visible free disk | About 524 GiB; Windows host physical free space was not established |
| Native CUDA build prerequisites | Neither `bazel` nor `nvcc` is on `PATH` |
| Secure GitHub access from Git | Certificate verification fails, including with the system CA bundle explicitly selected |

The installed server has a reproducible-build timestamp of 2000-01-01, so its modification time cannot identify its release. A string scan found a Qwen3.5 encoder filename but no MOSS model registrations. This is a clue, not a substitute for a runtime capability test or build provenance.

The source checkout and installed runtime must be tracked independently. The current [launcher definition](../.devcontainer/Dockerfile#L1) resolves the server from `PATH`, while conversion wrappers resolve local SwiftPM products.

## 3. Upstream Target and Release Gap

The assessed upstream candidate is [0621d8198f3ed505bb7cefc71b84f8f0e1ce5936][upstream-head], committed 2026-10-22. GitHub's comparison reports it is 320 commits ahead of our community HEAD, with our HEAD as the merge base.

Its relevant dependency pins are:

| Dependency | Candidate revision |
| --- | --- |
| s4nnc | `b5476a77e7f67f12729872e4b9f1a649b3cd3df0` |
| CCV | `69b1ed1f4e281520c29cd4b45b93c6fcdde8d2e8` |
| Bazel | 7.4.1 |

The Linux build still uses WORKSPACE dependencies: the candidate disables Bzlmod and loads s4nnc from [WORKSPACE.linux][upstream-workspace]. Updating only the SwiftPM lockfile would not update a Bazel-built runtime.

Important chronology:

1. [v26.1014.0][server-release] was published October 15. Its notes explicitly say "Support 4-bit S variant on CUDA."
2. Docker Hub's public `latest` was also last pushed October 15. The observed Linux/amd64 image digest is `sha256:71cb9882ab60deb8b5e63f1c6b0fb14deb71f21bb4d15c416b77f7b0b3352873`. This is the registry observation, not a verified digest of the running devcontainer.
3. [a32aa0bb][moss-initial] added MOSS inference on October 20.
4. [6c1dd093][moss-followup] added the related MOSS model registrations and LoRA import later on October 20.

Therefore, use a source-built, pinned post-MOSS candidate now, or a later packaged release whose provenance actually includes those commits. Do not equate `latest`, a current source tree, and the active executable.

The candidate's Linux WORKSPACE references a CUDA 12.8.1 / Swift 6.0.3 / Ubuntu 22.04 builder image. Its README still contains older CUDA 12.4 examples. Follow the pinned build configuration and verify the actual CUDA/cuDNN/compiler versions rather than copying the old example blindly. A Swift major-version upgrade is not established as necessary by this assessment.

## 4. What 4-bit S Means Here

| Name | Meaning in this stack | Consequence |
| --- | --- | --- |
| `q4p` | 4-bit palettization | Existing toolkit option; not interchangeable with 4-bit S |
| `q6p`, `q8p` | Palettized formats | Preserve as regression controls |
| `i8x` | Row-wise integer format, exposed as 8-bit S | Distinct storage and compute path |
| `i4x` | Published filename convention for packed 4-bit S weights | s4nnc represents the Q4_K format through `.i8x(.q4k)`, not a new `.i4x` codec constant |
| `i5x` encoder artifact | Published Qwen3.5 companion checkpoint | Use the official encoder artifact initially; do not apply denoiser quantization rules to it |

The new [s4nnc store implementation][upstream-store] supports packed row-wise formats, optional Hadamard rotation, importance-matrix inputs, external data, and trailer stores. Its universal JIT decoder dispatches on encoded identifiers. Consequently, the absence of an explicit `i4x` spelling in a model's codec list is not proof that the model cannot load 4-bit S weights.

The pinned [CCV CUDA GEMM implementation][upstream-cuda-gemm] has a packed row-wise path. It can decode packed weights into row-wise int8 workspace, apply the appropriate activation rotation, and use a supported GEMM path; a more general dense decode/cuBLAS path also exists. This is concrete CUDA implementation evidence, but not a benchmark of our GPU.

Do not promise native INT4 arithmetic everywhere, a fourfold peak-memory reduction, or a speedup based on the filename alone. Validate loading, numerical correctness, temporary workspace, and latency on SM86.

The stock candidate [standalone quantizer][upstream-quantizer] still lacks our `--target-codec` and LTX trace options. Its LTX cases still terminate with `fatalError()`. Its MOSS policy preserves original FP32 tensors and otherwise produces a model-specific q8p output; it is not a general CLI for producing the published `i4x` variants.

## 5. Patch Disposition

The [manifest](../DRAW_THINGS_PATCH/patches/manifest.sh#L1) is an accurate inventory of today's working sources, but not a declaration that every patch remains appropriate upstream.

| Patch purpose | Recommended treatment |
| --- | --- |
| April CCV/s4nnc pins and old lockfile | Do not carry into the candidate runtime. Use its matched dependency revisions. |
| SwiftPM converter/quantizer/LoRA products | Retain or replace with the existing Bazel targets. They are not declared as products in the candidate SwiftPM manifest. |
| LLM target restricted to two source files | Remove this restriction in a modernized build. MOSS needs Qwen3.5 sources and their dependencies; preserve upstream's expanded module graph. |
| LTX model detection and raw import path | Preserve until replaced by a demonstrated equivalent. Generic upstream LTX import is still incomplete. |
| CPU rotary permutation and QK-norm mapping fixes | Rebase narrowly against the new mapping API, then prove value/rank parity. Do not assume a clean patch application proves correctness. |
| BF16 aggregate embeddings and F32/rank repairs | Preserve the established LTX serialization contract. Retire repairs only when converter-native output passes the same checks. |
| Forced codec options and LTX quantization policy | Preserve the LTX CLI contract and precision safeguards, but keep them model-specific. A generic forced FP16 path must not bypass MOSS's FP32 preservation. |
| Linux CoreML/ImageIO/NetService guards and ZIP integration | Port only the guards still needed by the selected build route. Do not replace newer files with old platform stubs. |
| Activation helper and HiDream type disambiguations | Candidates for retirement after compilation against the new s4nnc API. Check for duplicate or changed upstream APIs first. |
| LTX tracing | Optional diagnostic layer, disabled by default. It is not a MOSS prerequisite. |
| CCV reshape/view/parameter/backend assertion suppression | Do not replay into the CUDA server. Reassess each CPU-conversion workaround with a minimal reproducer and correct the caller where possible. |
| Global SQLite `synchronous=OFF` | Keep out of the runtime baseline. Any retained fast-write mode should be explicit, limited to regenerable conversion outputs, and paired with validation and atomic promotion. |

Two implementation hazards deserve explicit gates:

- [dt_quantize.sh](../tools/dt_quantize.sh#L1) accepts a model-version option but still writes LTX companions, LTX metadata, and invokes LTX norm repairs. It must not become the MOSS entrypoint merely by passing a different version.
- [dt_convert.sh](../tools/dt_convert.sh#L1) treats preflight and checkpoint validation failures as warnings. A new custom MOSS path should fail closed on unsupported architecture, missing tensors, or precision violations.

Historical findings contain retractions. Preserve the final LTX lessons, especially the frame-count correction in [section 30](../CUSTOM_MODEL_CONVERTER_FINDINGS_2026-07-09.md#L1764), rather than reviving earlier precision diagnoses. Older iOS compatibility conclusions are version-specific and are not evidence about the new CUDA 4-bit S path.

## 6. Upgrade Guard Improvements

The existing strict update workflow is useful, but insufficient as the sole migration acceptance test:

- [Patch verification](../tools/apply_drawthings_quant_patch.sh#L1) compares `git status --porcelain` before and after the second application. File contents can change while the same paths remain marked modified. Add content-based comparison, including new files, and distinguish an already-applied patch from a conflicting one.
- The 3-way fallback can modify the index/worktree before returning failure. Use an isolated disposable candidate; check the already-applied case before attempting a mutating merge and explicitly detect conflict state.
- [Patch regeneration](../tools/generate_drawthings_quant_patches.sh#L1) uses ordinary `git diff`, which omits staged changes. A successful 3-way apply may stage changes. Regeneration must account for both index and worktree changes relative to the recorded base, plus untracked manifest files.
- The safety script's default build is only the CPU quantizer. That does not prove the CUDA server, protocol, model loading, or MOSS works.
- Record the community/dependency revisions, build route, binary hashes, container digest, schema hashes, and model identities together. The current revision-only baseline is not enough to reproduce a mixed runtime/tool installation.

Do not run the default snapshot-fallback apply mode against a newly updated tree. Do not regenerate the production bundle until the candidate changes have been reviewed and validated. Keep the existing bundle usable for rollback while producing the replacement.

## 7. Staged Upgrade

### U0. Freeze the working installation

Preserve the current source diffs, dependency diffs, untracked activation helper, patch snapshots, conversion binaries, runtime binary/image identity, model catalogs, and exact known-good request configurations. Keep original checkpoints and any sidecars protected from in-place conversion or repair experiments.

Use separate build/cache/output directories for the candidate. Back up catalogs rather than testing alias registration against the production copies. Record model size, storage layout, critical tensor metadata, and checksums from trusted manifests; hash large files only where necessary.

Current cheap gates already passed: active shell syntax, all 25 snapshot comparisons, installed-server help, and the local quantizer's CLI contract. A fresh GPU baseline has not been run in this assessment.

Exit gate: a recoverable old installation and an unambiguous baseline manifest. No requirement to overwrite it or create a new Git branch.

### U1. Build one pinned CUDA candidate

Fix the trusted CA chain first. The devcontainer already has a corporate-root-certificate mounting/update mechanism; verify the actual certificate and trust configuration. Do not disable TLS verification to make fetches succeed.

Create a separate checkout at the assessed upstream SHA. Provision Bazelisk/Bazel 7.4.1, a matching CUDA development toolchain, cuDNN headers/libraries, and the Linux build prerequisites there or in a dedicated builder image. Keep the current runtime container available for rollback. Limit build concurrency for the approximately 29 GiB RAM budget.

Prefer upstream's Bazel route for the GPU runtime. After configuring the candidate's Linux setup, the documented build target is:

```bash
bazel build //Apps:gRPCServerCLI --spawn_strategy=local --compilation_mode=opt
```

Include SM86 in the CUDA architecture configuration and verify the effective configuration. Do not assume a successful CPU SwiftPM build contains CUDA kernels. Do not apply the CPU CCV bundle to this build.

Keep upstream's newer DiffusionMappings/LLM/tokenizer/generated-source structure. Pin source, dependencies, and builder identity. Confirm `--help`, dynamic-library resolution, CUDA device discovery, and explicit executable selection before loading weights.

Exit gate: a launchable, identified CUDA executable built from the intended revision. A successful compile alone does not certify inference.

### U2. Match the wire format and preserve LTX

Use the candidate's protobuf and FlatBuffers schemas for candidate requests. Regenerate Python bindings into a revision-specific cache, and retain the old bindings for the old server. The client already accepts a protobuf path and the configuration builder accepts a bindings directory; use those boundaries before adding new machinery.

Run the candidate on loopback and a separate port, such as 7862, with an explicit binary path. The updated [canary runner](../tools/run_q6p_canary_once.sh#L1) owns its PID and stops only that process; older revisions included broad process kills tied to port 7861. Keep insecure/no-TLS operation limited to the intended local test boundary. Use `--models-dir` with private writable copies of the selected checkpoints, sidecars, and catalogs: runtime offload preparation can rewrite model storage in place.

Run the smallest meaningful LTX regression set under identical emitted configurations:

1. Known-good official LTX q6p, without LoRA.
2. Known-good custom LTX q6p, without LoRA.
3. Known-good base plus the validated distilled LoRA, using its established TCDTrailing schedule and 9 frames.
4. A previously validated low-bit LoRA pairing if that remains part of the supported workflow.

LTX video tests must preserve the established valid temporal geometry: start with 9 frames, not the historical failing 5-frame case. Compare emitted configurations before interpreting model differences. Distillation schedules belong to their specific LoRAs, not to MOSS SFT.

Require a completed stream and finite final decoded output, not merely a preview, signpost, or zero exit status. [The raw client](../tools/dt_api_client.py#L150) can return success after preview-only output, and response caps can end a stream early. An acceptance runner must enforce final payload, complete chunk assembly, expected dimensions, and finite values itself.

Exit gate: existing LTX workflows remain coherent on the candidate, with measured memory/latency and a tested binary-selection rollback.

### U3. Rebase and consolidate CPU utilities

Use the patch disposition table to port the remaining necessary conversion/quantization changes. Choose deliberately between preserving a minimal Linux CPU SwiftPM layer and moving utility wrappers to the existing Bazel targets. Do not attempt both build-system migrations at once merely for completeness.

Keep old utility binaries pinned and accessible until replacements pass:

- CLI argument compatibility, including forced codecs and LoRA conversion.
- Small CPU permutation fixtures, QK-norm rank/dtype checks, and BF16 embedding preservation.
- Strict tensor coverage and round-trip checks.
- Conversion of the official LTX source as a positive control, followed by custom conversion through the same path.
- A matched LTX canary using the new artifacts on the validated new runtime.

Regenerate the reviewed replacement patch bundle against its actual base. Run strict application and content-based idempotence on a fresh candidate before making it the default.

Exit gate: a reproducible replacement utility stack. MOSS official-checkpoint testing can proceed after U2 and need not wait for custom MOSS conversion.

## 8. MOSS Architecture and Existing Integration

Base and SFT use the same architecture and inference pipeline. SFT is a separate full checkpoint, not an acceleration LoRA or a different model version.

| Component | Verified contract |
| --- | --- |
| Denoiser | Approximately 5B parameters; 42 layers; width 3072; 24 heads of width 128; FFN width 8192 |
| Text features | Qwen3.5-9B, hidden width 4096; hybrid linear/full attention; MOSS has a two-block text refiner |
| Image latent | 32 VAE channels; 2x2 packing produces 128 values per image token |
| Rotary positions | Four axes, with centered spatial coordinates and prompt-length-dependent positions |
| Precision | FP32 residual/modulation computation and an FP32-sensitive second timestep projection in the Draw Things implementation |
| Guidance | Separate conditional/unconditional forwards; the unconditional branch skips layer 22 |
| Variant identity | `moss_image` for both Base and SFT; separate checkpoint filenames |

The [MOSS model builders][upstream-moss] and [runtime dispatch][upstream-unet] already implement the main architecture, LoRA variants, and layer-22 skipping. Skipping is hardcoded in the runtime using zero-length rotary inputs, not a missing client configuration field. Exercise that zero-length CUDA path explicitly during validation rather than replacing it with old assertion-suppression patches.

The candidate server initializes the Qwen3.5 tokenizer from its own vocabulary and merge resources. Reusing the old Gemma or Qwen3 tokenizer is not valid. Model registration alone cannot repair missing tokenizer resources or an LLM target pruned to the old source subset.

The [generic importer][upstream-importer] has no MOSS architecture detection and still has MOSS `fatalError()` branches in import/specification construction. That is the principal custom-conversion gap, not missing inference architecture.

## 9. Official Checkpoints and Resource Budget

Both [Draw Things Base][dt-moss-base] and [Draw Things SFT][dt-moss-sft] repositories publish the following. Sizes are rounded decimal GB from the file listings, not measured VRAM allocations.

| Artifact | Approximate size |
| --- | --- |
| Each Base/SFT `f16` denoiser | 10.2 GB |
| Each Base/SFT `q8p` denoiser | 5.28 GB |
| Each Base/SFT `i8x` denoiser | 5.13 GB |
| Each Base/SFT `i4x` denoiser | 2.95 GB |
| Shared `qwen_3.5_9b_i5x.ckpt` | 8.75 GB |
| Shared `flux_2_vae_f16.ckpt` | 0.169 GB |

Approximate download/storage sets:

- One i4x variant plus companions: 11.9 GB.
- Both i4x variants with companions downloaded once: 14.8 GB.
- Both i4x variants plus one q8p control and shared companions: 20.1 GB.
- Both q8p and both i4x variants with shared companions: 25.4 GB.

Pin the Hugging Face revisions and validate each file against its published hash. The observed conversion revisions are `add235e9063a9fb0eec06f61cba6a970337abf1e` for Base and `55d7bdb21604cba9ff35795b6b0e163f0d51802a` for SFT. Verify shared-component identity before deduplicating; identical names alone are insufficient.

Do not download complete repository snapshots containing all formats and duplicated companions by default. Reserve additional space for partial downloads, extraction/repacking, build caches, reference environments, and later conversion outputs. The free-space reading inside WSL does not establish the host's physical headroom.

Even the i4x set is close to the GPU's capacity before activations, temporary decodes, attention workspace, and driver/display usage. The q8p set exceeds 12 GiB if treated as simultaneously resident stored weights. These are warning-level storage comparisons, not precise VRAM predictions.

Start with batch size 1, weights cache 0, verified component/partial offload, and modest smoke-test resolution. Measure encoder, denoiser, and VAE peaks separately. Keep FlashAttention enabled on this SM86 GPU initially; indiscriminately disabling it can worsen memory use. Run inference on AC power, consistent with the recorded laptop/WSL GPU-reset history.

## 10. MOSS Functional Milestones

### M1. Official checkpoint inference

Start with Base q8p as the first correctness control, then test SFT, as selected for the experimental M1 work below. Use their built-in specifications rather than manufacturing LTX-shaped aliases.

The [upstream model zoo][upstream-zoo] already specifies:

- `moss_image_sft_q8p.ckpt`, `moss_image_sft_i8x.ckpt`, and `moss_image_sft_i4x.ckpt`.
- Equivalent `moss_image_base_*` registrations.
- `qwen_3.5_9b_i5x.ckpt` and `flux_2_vae_f16.ckpt` as companions.
- Flow/velocity objective with condition scale 1000 and default scale 16.
- No LTX connector-as-CLIP dependency or `kontext` modifier.

Recommended request profile, grounded in the [official model card][official-moss]:

| Setting | Initial correctness profile |
| --- | --- |
| Output | One image; batch size 1; no audio/video workflow |
| Resolution | Small smoke test, then 1024x1024 and a portrait/landscape pair |
| Steps | 50 for correctness/quality acceptance |
| Guidance | CFG 3 |
| Shift | Fixed 5; resolution-dependent shift disabled for controlled comparisons |
| Negative prompt | Empty initially |
| Layer skipping | Upstream's existing negative-branch layer 22 behavior |
| Sampler | Validate the engine's deterministic flow-Euler-equivalent schedule; do not inherit UniPC/TCD defaults |
| Seed | Fixed and recorded, with compatible noise generation verified before cross-engine numerical comparison |
| Other acceleration | Hires fix, TeaCache, CFG shortcuts, and approximate attention disabled initially |

Shorter runs establish loading/finiteness only, not the model's final quality. Compare q8p and i4x using the same variant, prompt, emitted configuration, and initial noise where possible. Use offload for the q8p control; an OOM is not evidence of a model-mapping defect. Add i8x as a diagnostic control only if needed to isolate codec-specific behavior.

Keep the image workflow separate from LTX's frame guards and distilled schedules. The current [configuration builder](../tools/dt_make_config.py#L177) defaults to 8 steps, CFG 1, sampler 17, and video-oriented fields. Its start dimensions use 64-pixel units. Initially choose dimensions it can represent exactly; do not silently round the model card's arbitrary multiple-of-16 resolutions.

The reference pipeline requests BF16 transformer/encoder, FP32 initial noise and VAE, and describes a small FLUX.2 decoder. Draw Things publishes its own converted VAE under the generic FLUX.2 filename and uses its own mixed-precision implementation. Validate latent normalization, packing, decoder compatibility, and finite output before asserting reference parity or substituting components by name.

Acceptance: both variants produce finite, coherent final images; all chunks are complete; memory fits the chosen offload policy; repeated requests and Base/SFT switches do not crash or reuse the wrong weights.

### M2. Image workflow and quality validation

Reuse the existing gRPC transport, configuration generation, requirements checks, and tensor decoding where they fit. Add one small image-oriented test entrypoint or explicit image mode, not a second parallel toolkit. Do not loosen LTX safety checks globally to accommodate images.

Require final image presence and successful PNG decoding, finite values before clipping, exact dimensions, and saved run metadata. Include tests for preview-only responses, interrupted chunk streams, missing companions, unknown models, and wrong schema selection. Prefer named sampler selection resolved from generated enums to new hardcoded integer literals.

Use a small fixed prompt suite covering natural-language scenes, English and Chinese text, tag-style prompts, and a structured JSON layout. Pass layout JSON as a prompt string with the model's documented centered coordinates; it is not a new ControlNet interface. Inspect text placement and object layout, not only image standard deviation. Investigate vertical banding against the reference before attributing it to quantization.

Validate full 50-step generation, a second request on the same server, variant switching, and at least one non-square resolution. Characterize i4x quality drift against the same variant's q8p control. Do not require pixel identity between Base and SFT, or between independently seeded CUDA and PyTorch runs.

Acceptance: reproducible image commands and per-variant baseline artifacts, while LTX requests still behave as before.

### M3. Custom MOSS conversion

Only after official conversions work, extend the generic importer and its specification inference for MOSS. Reuse the existing MOSS main/fixed builders and weight mappers. Keep official text encoder and VAE companions initially, so custom denoiser conversion is the only new variable.

Implementation requirements:

- Detect and validate the actual architecture from config and tensor shapes, not just filename or a version override.
- Support the actual source packaging: a selected supported single-file format first, or an explicit sharded safetensors/index reader. Reject unsupported directories/shards clearly rather than silently importing a subset.
- Cover both the 42-layer denoiser and fixed text-refiner/timestep/modulation parameters.
- Use the new mapper's Q/K interleaving metadata, including QK norms. Preserve packed FFN up/gate order and head-modulation order.
- Preserve original FP32 timestep projection values before any FP16 cast. Widening already-downcast values cannot recover underflowed weights.
- Derive tensor dtype/rank contracts from the runtime parameter schema and validate key coverage. Do not import LTX tensor-count constants or run LTX post-processors over MOSS.
- Use strict read/write results, component-aware metadata validation, and atomic promotion of completed output. Recognize inline, sidecar, and trailer storage where applicable.
- Keep variants distinct in model identity, outputs, aliases, hashes, and caches while sharing architecture code.

Use the official source checkpoint as the positive control. Compare converted metadata and representative values to the official Draw Things f16 conversion, including every transformed parameter category. Compile and strictly load the converted model. Compare text features, selected denoiser outputs, and one sampling update before an expensive full render.

Then run the identical converter on SFT and on a custom finetune. Do not use official model weights as a repair source for a custom model. The transferable lesson from LTX is to validate transformations and serialization policy, not to copy its repair scripts.

Acceptance: source-to-checkpoint conversion preserves the intended values, metadata, and inference behavior for both variants and a custom checkpoint.

### M4. Custom 4-bit S and LoRA

Treat custom i4x generation as a separate policy surface. Use the appropriate packed s4nnc format, preserve sensitive tensors, identify any calibration/Hadamard requirements of the selected recipe, and trace actual codec identifiers rather than trusting an output suffix. Do not requantize an already-low-bit official checkpoint as the reference experiment.

Start from the validated mixed-precision conversion, establish q8p parity, and then evaluate i8x/i4x only as needed. Keep text-encoder quantization separate from denoiser quantization. Do not invent an `.i4x` codec constant or relabel a q4p checkpoint.

Upstream already includes MOSS LoRA main/fixed builders and import support. Integrate those after the no-LoRA baseline. Check the main transformer and text refiner/modulation projections, split/packed projection mappings, ranks, alpha scaling, and zero-weight behavior. Our current LoRA quantization wrapper is explicitly LTX2.3-specific and needs a model-aware path before reuse.

MOSS SFT does not imply a distilled sampling schedule. Community acceleration LoRAs, full training, LyCORIS variants, image editing, and further encoder quantization are follow-up work with their own acceptance tests, not prerequisites for Base/SFT image inference.

## 11. Reference Environment and Test Gates

For a PyTorch reference, pin the official MOSS source/weights and the reviewed custom pipeline code. Its model card requires Python 3.10+, PyTorch 2.9+, Diffusers 0.38+, Transformers 5.3+, Accelerate 1.10+, Safetensors 0.6+, and Pillow 10+.

The toolkit's CUDA requirements already specify PyTorch 2.10/CUDA 12.8, but several other requirements are unpinned. Installed compatibility was not measured. Prefer an isolated reference environment rather than broadly upgrading the working toolkit environment. Review remote pipeline code before enabling remote-code execution, and record its exact revision.

| Gate | Required evidence |
| --- | --- |
| Reproducible build | Source/dependency/builder pins, executable hash, GPU-enabled build, library resolution |
| Wire compatibility | Schemas and generators tied to candidate revision; serialized config round-trip |
| LTX preservation | Matched official/custom controls and known-good LoRA configuration; final finite decoded video |
| Packed codec support | Small CUDA decode/GEMM correctness checks for relevant shapes/dtypes/rotation; measured temporary memory |
| MOSS runtime | Encoder, fixed model, denoiser and VAE load; CFG path with layer-22 skip; final finite image |
| Variant support | Base and SFT tested independently; shared companion hashes verified; cache-switch test |
| Quality | Fixed prompt suite and same-variant higher-precision control, not just nonzero pixel variance |
| Custom import | Strict tensor coverage, dtype/rank/value checks, positive-control conversion and sampling parity |
| Patch safety | Strict application on a fresh base; content-based second-pass no-op; reviewed regenerated bundle |

Use existing nearby test homes and small fixtures where possible. Resolve tokens, sigma/timestep sequence, CFG formula, initial noise, and VAE latent normalization before setting numerical parity tolerances. Do not make cross-framework same-seed pixel equality a release gate.

## 12. Scope, Rollback, and Next Action

Suggested implementation units are:

1. Freeze/provenance manifest and trustworthy candidate build setup.
2. Explicit runtime/schema selection and LTX regression gates.
3. Reviewed CPU patch rebase and improved bundle verification.
4. Official MOSS image profile and complete-output acceptance runner.
5. Custom MOSS import with serialization/value tests.
6. Custom packed quantization and optional LoRA workflow.

Runtime selection, CPU utility selection, and model registration should be independently reversible. Promote defaults only after their gates pass. Rollback should select the saved binary/image and catalogs, not reset user work or rewrite existing checkpoints. Keep the old and new bundles clearly associated with their respective base revisions.

The next implementation action should be U0/U1: establish provenance and a separate pinned CUDA build environment, then validate LTX before downloading and testing the official MOSS set. This avoids making the relatively large custom-conversion task the critical path for initial MOSS support.

At the time of the initial assessment, work comprised source/patch inspection, upstream source/release/model metadata research, binary help/fingerprint inspection, hardware/resource queries, shell syntax checks, and snapshot comparisons. No source rebase, dependency resolution, build, patch application, model download, quantization, or GPU inference had been performed. Implementation progress is recorded separately below; throughput, VRAM peaks, numerical parity, and image quality remain unverified until their gates pass.

### Implementation Progress: October 23

The [update guard](../tools/drawthings_update_guard.sh) now provides a read-only `snapshot` command. The first baseline is under `output/upgrades/baselines/20261023T092901_649458Z`, including repository heads/status, separate staged and working binary diffs, untracked source, the patch bundle, catalogs, CPU tools, installed server/launcher, schemas, and canary configurations. Model files are inventoried by size/mtime, not copied or fully hashed.

Independent verification passed for all 89 saved-file hashes, all three original dependency/community source states, and the seven active binary/catalog rollback copies. The 18 inventoried model files retain their recorded size/mtime. The actual running container image digest remains unknown; the manifest records the declared image and an unset digest rather than inventing provenance.

The [native provisioner](../.devcontainer/install-build-tools.sh) and [candidate build helper](../tools/drawthings_build_candidate.sh) implement the isolated U1 route. Bazel 7.4.1, Swift 6.0.3, CUDA 12.8, and LLVM/Clang 14 are in use. Existing CUDA/cuDNN runtime package versions are preserved. The candidate remains detached at the pinned community SHA with no legacy CPU patches applied.

Build setup resolved Java CA trust, missing CUDA development components, the CUDA wrapper's host-compiler selection, canonical Clang header paths, and CUTLASS template matching. The CCV and NNC CUDA compatibility targets compile successfully, including the packed row-wise GEMM/dequantization code. The full pinned server build also succeeded, with executable SHA-256 `fd9f2d82119e6d79172a70b4230f79c75933958b78a36b2cd8e4799cfb265b5a`. Compilation alone does not establish packed codec numerical correctness. Subsequent builds default to 16 parallel jobs, overridable with `DT_BUILD_JOBS`.

The candidate's CLI help and runtime library resolution pass. A temporary server on `127.0.0.1:7862` passes Echo and FilesExist RPCs using candidate-only protobuf bindings. Its owned process was stopped afterward and the existing model catalogs remain unchanged; results are saved under `output/upgrades/runtime-0621d819`. Keep the Bazel build tree available for its runtime library/resource layout rather than assuming that copying the executable alone produces a standalone installation.

The rebuilt-container smoke check also passes real PyTorch CUDA matrix multiplication/convolution and verified HTTPS/Git access. It downloads no model weights.

Candidate-only Python bindings are generated under `output/upgrades/schemas/0621d819`, with source hashes and generator versions recorded in its manifest. Stock `flatc` accepts the configuration schema after declaring the SQLite-only `primary` and `indexed` attributes; all fields and defaults remain unchanged. FlatBuffer field and protobuf request round-trips pass, including the 9-frame field and a sampler resolved by name from the generated enum. This fixture checks serialization only, not an inference schedule or image/video quality. Existing `--bindings-dir`, `--proto`, and `--cache-dir` options allow explicit candidate selection without modifying the current client caches.

The [existing LTX canary](../tools/run_q6p_canary_once.sh) now forwards those schema options, binds the requested host/port, rejects occupied endpoints, and cleans up only its own child PID. Broad process-name kills are removed. `--dry-run` writes the configuration and prints the launch command without starting a server or GPU generation. Tests pass for occupied endpoints, failed-start cleanup, unrelated-process survival, IPv6 address parsing, and invalid ports.

```bash
bash tools/drawthings_update_guard.sh snapshot
sudo bash .devcontainer/install-build-tools.sh
bash tools/drawthings_build_candidate.sh --check
bash tools/drawthings_build_candidate.sh build
```

Once the candidate binary exists, inspect its isolated canary wiring without rendering:

```bash
CANDIDATE=output/upgrades/community-0621d819
bash tools/run_q6p_canary_once.sh \
	--grpc-bin "$CANDIDATE/bazel-bin/Apps/gRPCServerCLI" \
	--host 127.0.0.1:7862 --model 10_e_v1.5_q6p.ckpt \
	--proto "$CANDIDATE/Libraries/GRPC/Models/Sources/imageService/imageService.proto" \
	--cache-dir output/upgrades/schemas/0621d819 \
	--bindings-dir output/upgrades/schemas/0621d819/flatbuffers \
	--server-cpu-offload --server-weights-cache 0 \
	--require-complete-stream --require-final-output --final-mode --dry-run
```

This only verifies wiring. Select the matched known-good LTX configuration and enforce finite, fully decoded final frames before treating a real run as an acceptance result.

No default runtime has been replaced. The new native prerequisites are opt-in and must be provisioned again after rebuilding a runtime-only devcontainer; see the [builder instructions](../.devcontainer/README.md#optional-native-cuda-builder).

### U2 Render Results: October 23

Matched requests used the available custom `10_e_v1.5_q6p.ckpt`, its registered Gemma and LTX VAE companions, 256x256 pixels, 9 frames, seed 4242, TCDTrailing (19), 2 steps, guidance 1, shift 3, CPU offload, and weights cache 0. The prompt was the canary's default red sports car on a mountain road at sunset. Old-server requests used the original schema/cache on port 7863; candidate requests used revision-specific bindings on port 7862. Both successful pairs also used `--fread-preferred`.

| Profile | Old Request Time | Candidate Request Time | Observed Result |
| --- | --- | --- | --- |
| No LoRA | 192 s | 220 s | All 9 final image tensors byte-identical; finite audio, maximum audio sample difference `8.35e-7` |
| Locally converted f16 acceleration LoRA, weight 1 | 396 s | 406 s | Both return 9 finite frames and finite audio; frames are not byte-identical |

Times are whole seconds between the server's request-begin and image-processed log messages, including loading and decoding. These sequential cold-process runs are not controlled throughput benchmarks; GPU peak memory was not recorded.

The no-LoRA pair has identical 276-byte configurations, SHA-256 `a3667c366a477174776873b1fea9fff996fac33993ecb92596726a7aa065f573`. At only two steps the image is visibly under-denoised, so this establishes numerical parity, not a no-LoRA quality gate. The LoRA pair has identical 392-byte configurations, SHA-256 `6569f3eeda53651b2b5faeb87b4d33823ebef8e25ab31d2a80bde3db4ff6e472`. Its first frames are visually coherent red cars matching the prompt. Across the 9 frames, raw mean absolute differences range from `0.01593` to `0.01714`, with maximum absolute difference `0.64783`. Audio's maximum sample difference is `0.12509`. These differences are measured, not certified numerical equivalence; within-runtime repeatability and acceptance tolerances remain unresolved.

All final image tensors decode to `[1, 256, 256, 3]`, and all raw image/audio values are finite before clipping or normalization. Streams finish with 9 image payloads and 1 audio payload, and post-generation Echo succeeds. The candidate LoRA output differs from its no-LoRA output. The [media converter](../tools/dt_tensor_to_playable.py) now offers opt-in `--strict` raw finiteness checks and `--expected-frames`; it validates audio before writing any media. Six CPU-only tests cover finite and non-finite raw/zipped tensors, truncated payloads, frame-count rejection, and strict failures before artifact creation:

```bash
.venv/bin/python -m unittest tools.test_dt_tensor_to_playable -v
```

Successful raw run directories under `output/`:

- `q6p_canary_upgrade_old_base_fread_20261023T115014Z`
- `q6p_canary_upgrade_candidate_base_fread_20261023T115502Z`
- `q6p_canary_upgrade_old_lora_fread_retry_20261023T121121Z`
- `q6p_canary_upgrade_candidate_lora_fread_20261023T120258Z`

Strictly validated PNG/GIF/MP4/WAV exports are under `output/upgrades/regression-0621d819/{candidate_base,candidate_lora,old_lora}`. These generated artifacts are not committed.

**Storage-layout change:** the first old-server generation run automatically externalized the active custom q6p store before the candidate was run. The original 21,456,027,648-byte SQLite checkpoint is now a 36,331,520-byte SQLite file plus `10_e_v1.5_q6p.ckpt-tensordata` at 21,369,192,448 bytes. This is the offload preparation implemented by upstream `TensorData.makeExternalData`, not deletion of the tensor payload. The resulting SQLite `quick_check` passes, all 5,746 tensor names match the existing f16 source, and both runtimes render from the pair. The rollback snapshot did not copy or hash model payloads, so original payload byte identity is not established and the original single-file layout has not been restored. Keep both files together; compatibility of this layout with the existing CPU utilities is not yet established. The other 17 inventoried model files retain their recorded size/mtime. Catalog hashes and all five original server/launcher/CPU-tool hashes still match the snapshot.

The initial old-server no-LoRA request and the first old-server LoRA request crashed before streamed output, with cuFile errors and a null dereference in `_ccv_nnc_gemm_forw`. Successful runs also emit cuFile warnings. The opt-in canary `--server-fread-preferred` flag is a diagnostic control, not a proven fix or a new default. The old LoRA run succeeded on one retry without changing generation settings. No source patch or CUDA-driver change has been made to conceal this failure.

Further renders were stopped after discovering the active-store migration. `--models-dir` now selects the same directory for server loading, model/LoRA file checks, and config-builder catalog lookup. A dry-run fixture verifies independent staged catalogs, including aliases containing spaces; it starts no server. This option does not copy models or make the legacy default directory read-only. Prepare real private copies, not hardlinks or symlinks, before the next GPU test.

### U2 Private-Store Low-Bit Follow-Up: October 23

GPU tests resumed using `output/upgrades/models-ltx-0621d819-20261023`, containing private writable copies of both catalogs, the split q6p model, the split Gemma encoder, the LTX VAE, and the locally converted f16/q4p LoRAs. The nine files were copied with `cp --reflink=auto --preserve=mode,timestamps`, not hardlinked or symlinked. Full source/copy SHA-256 matches and independent file identities were verified before inference. The manifest is `source.sha256` in the private directory. After the old/candidate q4p runs, all nine active files and all nine staged copies still match that manifest. This protects the already externalized layout; it does not establish the original single-file model's byte identity.

The canary's opt-in `--gpu-metrics` records timestamp, device index, memory used, GPU utilization, temperature, and power in `gpu.csv` every 500 ms. These are sampled **whole-device** readings, not exact allocator/process peaks, and can include unrelated workloads. The monitor is owned and reaped with the server; dry-run starts no monitor. Missing/failed telemetry setup aborts before server launch. Failed-start cleanup and successful-run cleanup were checked.

The matched profile is unchanged from the earlier LoRA comparison except for selecting the locally converted **q4p** LoRA at weight 1 and the private model directory. Both successful configurations are 392 bytes, SHA-256 `4496c0882c9140ccf7c1f0f4198a380b7842f0d9454a1241792c46ad60721492`.

| Runtime | Request Time | Sampled Device Peak | Samples | Maximum GPU Temperature | Result |
| --- | --- | --- | --- | --- | --- |
| Installed | 437 s | 7,007 MiB | 868 | 60 C | 9 finite frames and finite audio; complete stream and post-Echo pass |
| Candidate, unchanged retry | 394 s | 7,007 MiB | 785 | 61 C | 9 finite frames and finite audio; complete stream and post-Echo pass |

All nine final image tensors are byte-identical between these two runs and decode to `[1, 256, 256, 3]`. Audio's mean absolute difference is `3.36e-7`, with maximum absolute difference `2.24e-6`. Raw finiteness was checked before normalization. These sequential runs are not controlled performance benchmarks, and this result concerns palettized `q4p`, not packed 4-bit S / `i4x`.

The **first candidate attempt failed** before any streamed response, with `UNAVAILABLE: Socket closed`, signal 11, and a null dereference at `_ccv_nnc_gemm_forw + 54`. Its post-Echo failed. An unchanged retry succeeded; this does not resolve the startup reliability gate. cuFile warnings occur in both attempts, and the warning path has an mmap/copy fallback, so those warnings alone do not establish the crash cause. All old/q4p candidate server and monitor PIDs were confirmed reaped after these runs.

Disassembly of the pinned candidate identifies the fault at ELF address `0x11591b6`: `mov (%r12),%ebp`, with `r12 = 0`. The function first loads `inputs[0]` into `r12`; this is a null input-tensor object, before matrix setup or the packed-weight branch, not evidence of a packed arithmetic-kernel failure. The owning source is CCV's `lib/nnc/cmd/blas/gpu/ccv_nnc_gemm_gpu_cublas.cu`. The graph/caller that supplied the null input still needs identification; no crash-suppression patch has been applied.

Strict PNG/GIF/MP4/WAV exports are under `output/upgrades/regression-0621d819/{old_q4plora,candidate_q4plora}`. The inspected first frames are coherent, identical red-car images matching the prompt. This is a small canary, not a general temporal-quality assessment.

Raw run directories under `output/`:

- `q6p_canary_upgrade_staged_old_q4plora_20261023T131742Z`
- `q6p_canary_upgrade_staged_candidate_q4plora_20261023T132616Z` (failed)
- `q6p_canary_upgrade_staged_candidate_q4plora_retry_20261023T133003Z`

The registered official control `ltx_2.3_22b_distilled_1.1_q6p.ckpt` is still absent. The pinned model zoo records SHA-256 `f7c804dd0e4ae14af2ff200fc96b2848b2967f7595bac830bf270ea39b47a611`. Metadata-only HEAD probes returned 404 from `models.drawthings.ai` and 403 from `static.libnnc.org`; a public Hugging Face model search for author `drawthingsai` and search term `ltx` returned no results. The pinned `MediaGenerationEnvironment+Ensure.swift` confirms `https://static.libnnc.org/<filename>` as the model download route. A ranged GET there also returned 403; no model response body was read. These probes do not prove that every mirror is unavailable. No weights were downloaded or substituted for the official control.

An attempted candidate f16 repeat under `q6p_canary_upgrade_staged_candidate_f16lora_repeat_20261023T134622Z` was invalidated by editing the executing shell harness: Bash later reported a syntax error and exited 2, despite the on-disk script passing `bash -n`. Both owned processes were reaped. Exclude this interrupted attempt from runtime reliability and numerical comparisons. Freeze shell-script contents until their processes exit; documentation edits do not affect the running harness.

### U2 Candidate f16 Repeatability Follow-Up: October 23

The uninterrupted repeat is `output/q6p_canary_upgrade_staged_candidate_f16lora_repeat_retry_20261023T135528Z`. It uses the private stores and the same 392-byte configuration as the earlier candidate f16 run, SHA-256 `6569f3eeda53651b2b5faeb87b4d33823ebef8e25ab31d2a80bde3db4ff6e472`. It returned 9 finite `[1, 256, 256, 3]` image tensors and finite audio, completed the stream, and passed post-generation Echo. Request time was 460 s; 916 telemetry samples recorded a 7,007 MiB whole-device peak and maximum temperature 61 C. This is not a controlled throughput benchmark.

All nine image tensors are byte-identical to the earlier candidate f16 run. Within-candidate audio mean absolute difference is `5.11e-7`, maximum `6.85e-6`. Against the earlier installed-server f16 run, image mean absolute differences remain `0.01593` to `0.01714`, maximum `0.64783`; audio maximum difference is `0.12509`. These two candidate runs therefore establish exact image repeatability for this one seed/profile, but do not explain or accept the cross-runtime f16 difference. Installed-server repeatability and numerical acceptance tolerances remain open.

After the final repeat, all nine active files and all nine staged copies again matched the pre-render SHA-256 manifest, and the final server/monitor PIDs were confirmed reaped. Six tensor-validator tests, shell syntax/help checks, and the documentation diff check pass.

U2 remains incomplete: full no-LoRA quality, the official control, cross-runtime f16 LoRA numerical acceptance, and early-load reliability remain open. No default binary or catalog was promoted. U3 CPU patch rebase and all MOSS download/inference/custom-conversion gates remain pending. The next reliability investigation should identify the graph/caller supplying GEMM's null first input; the successful retry is not a fix.

### M1 Base q8p Preparation and First Attempts: October 23

The user authorized experimental MOSS work while deferring the GEMM investigation. This does not waive U2 or permit default-runtime promotion. The pinned candidate remains unmodified, and SFT/i4x/custom-conversion work has not started.

Downloaded only three files from `drawthingsai/MOSS-Image-Base` at revision `add235e9063a9fb0eec06f61cba6a970337abf1e`, totaling 14,196,940,800 bytes:

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `moss_image_base_q8p.ckpt` | 5,283,102,720 | `0361102e9a21f05d63a686ce3ff2c8f9d6e59f86b7aa4e5fb116dd8f1d5eb745` |
| `qwen_3.5_9b_i5x.ckpt` | 8,745,304,064 | `559d41f5e6721b4edb5a661c42b6f328a243876c2637cad36cf9937fb0d453c1` |
| `flux_2_vae_f16.ckpt` | 168,534,016 | `48369d4b1495aec3df0579591f4bf9c199a73cc288130df2359c4a173c5ae864` |

The denoiser hash also matches the pinned model zoo. The three verified downloads were initially under `output/upgrades/moss-image-base-add235e9/source` and were subsequently moved, at the user's request, into `dt-models`. All three hashes were checked immediately before and after relocation; no existing files were overwritten. The old source directory retains download-cache metadata only. Private writable inference copies remain under `output/upgrades/moss-image-base-add235e9/models`, with independent file identities and no symlinks/hardlinks to `dt-models`. Both sets matched the manifest before inference and after the initial tests. Runtime copies retained their original sizes without new sidecars. Trailer stores can remain inline/read-only; sidecar absence alone does not prove that loading failed. Inference tests continue to select the private directory, not the active model store. The common basename/hash manifest remains `output/upgrades/moss-image-base-add235e9/source.sha256`.

The existing canary now has an explicit `--image-mode`, preserving legacy LTX defaults. Image mode selects one frame, disables resolution-dependent shift, requires a complete stream without leftover partial chunks, exactly one finite final image at the requested dimensions, no audio, and successful post-Echo. Its success path saves `image.png` and `image_acceptance.json`; no success artifact is produced for these failed MOSS attempts. Repeatable `--require-model-file` preflights companions. Guidance, shift, prompt, and negative prompt are explicit CLI options. The config builder accepts sampler names from the selected generated enum as well as legacy integers.

Selected `DDIMTrailing` (candidate enum 16): the local sampler factory uses trailing linear discretization, and DDIM's `.u` branch performs the deterministic velocity/Euler update without added sampling noise for text-to-image. Euler A was rejected because its flow branch injects ancestral noise. This source-level mapping is not a claim of cross-framework numerical parity.

The profile is 512x512, batch 1, 50 steps, CFG 3, fixed shift 5, seed 4242, empty negative prompt, no LoRA/hires/TeaCache, CPU offload, fread-preferred, and weights cache 0. Prompt: `An orange ceramic teapot on a white table beside a green plant, bright window light, product photograph.` Its 288-byte config has SHA-256 `f4a1fac323101d86d63fb2650483a9baa4d3cb8c1ef7edb3d64e81f16d5facf9`. Schema round-trip verifies these fields, including one frame and fixed shift. Legacy LTX serialization remains byte-identical; a sampler name and its numeric ID also emit identical configs.

| Attempt | Observed Result | Sampled Device Peak | Samples |
| --- | --- | --- | --- |
| CFG 3, empty negative | SIGFPE in scaled dot-product attention after `sampling`; no final output | 983 MiB | 199 |
| Same request, `--server-no-flash-attention` | Same SIGFPE and instruction; the flag does not bypass the failing operation | 983 MiB | 112 |
| CFG 3, negative prompt `blur`, FlashAttention enabled | Same SIGFPE; changing the negative prompt did not fix it | 983 MiB | 108 |
| CFG 1, empty negative, FlashAttention enabled, diagnostic only | Stream and post-Echo succeed, but only 4 previews and zero final images; acceptance correctly fails | 979 MiB | 71 |

These are partial failed-attempt whole-device measurements, not full-inference VRAM requirements or proof of successful offload. Maximum sampled temperatures were 59-61 C. The guided failures are distinct from the deferred GEMM null-input crash: `_ccv_nnc_scaled_dot_product_attention_forw + 3614`, ELF `0x11be4ee`, faults at `idiv %ebx` while preparing the attention head ratio. The call site/tensor dimensions still need runtime identification. A zero-length text/refiner or skipped-layer descriptor remains a hypothesis, not an established cause. The CLI embeds Qwen3.5 resources and applies a chat template even to empty negative text; no missing-tokenizer or empty-negative-only cause has been proven. No assertion-suppression or upstream source patch was applied.

Raw directories under `output/`:

- `q6p_canary_moss_base_q8p_512_20261023T143626Z`
- `q6p_canary_moss_base_q8p_512_noflash_20261023T144006Z`
- `q6p_canary_moss_base_q8p_512_nonempty_negative_diagnostic_20261023T144305Z`
- `q6p_canary_moss_base_q8p_512_cfg1_diagnostic_20261023T144616Z`

All eight owned server/monitor PIDs were confirmed reaped. Ten CPU tests cover existing tensor validation, single-image finite/geometry/batch rejection, and sampler resolution. Dry-run starts no monitor; missing companions and unknown sampler names fail before launch. The real CFG-1 run verifies preview-only rejection. M1 remains blocked: there is no accepted MOSS image, full CFG-3 sampling/decoding and memory behavior are unverified, and SFT/packed-quality tests must not be treated as passed. Next, capture the failing attention tensor descriptors/caller before choosing a narrowly scoped candidate fix.

Post-test preservation checks also passed for all three staged checkpoint hashes. The GPU was idle with 0 MiB allocated after cleanup; shell syntax, all ten CPU tests, editor diagnostics, and `git diff --check` passed.

### M1 Native Attention and Backend Diagnosis: October 23

The candidate binary and candidate source tree remain unchanged. Its SHA-256 still equals `fd9f2d82119e6d79172a70b4230f79c75933958b78a36b2cd8e4799cfb265b5a`. Diagnosis used ignored files under `output/upgrades/moss-image-base-add235e9`, not runtime patches. The existing Swift LLDB required the additional Ubuntu runtime package `libpython3.10`; installing it enabled debugger launch without rebuilding the server.

The original 50-step CFG-3 request was repeated under LLDB in `output/q6p_canary_moss_base_q8p_512_lldb_descriptor_20261023`. At the same `idiv` fault, both saved Q and K tensor headers have all-zero dimension arrays (rank zero), with GPU memory type, NHWC format, and FP16 datatype. The native backtrace reaches `UNetFromNNC.callAsFunction` through its `run(... skipsLayers:)` path and the model builder, rather than text encoding or the fixed text refiner. MOSS explicitly preserves its block-22 negative-branch weights by building the skipped block on zero-length tensors. This identifies the empty-operation path leading to the head-ratio divide by zero.

A reversible LLDB-only probe returns success at the attention entry point only when Q, K, V, and output all have first dimension zero. It changes neither the executable nor nonempty attention operations. With a two-step CFG-3 request in `output/q6p_canary_moss_base_q8p_512_lldb_empty_noop_2steps_20261023`, this bypass is hit twice, avoids SIGFPE, and permits stream completion and post-Echo. However, it yields one latent preview and zero final images, so the canary still fails. The existing strict raw-tensor validator rejects that preview for **131,072 non-finite values before conversion**. The generator rejects NaN decoded output with a nil result. Avoiding the crash alone is not an inference fix or a quality result.

Qwen3.5's `Qwen3_5LinearAttention` directly uses both `GatedDelta` and `RMSNormGated`. The pinned CCV registrations list CPU and MPS implementations, but no CUDA backend for either. A small native probe compiled against the exact pinned headers calls the unchanged executable's retained `ccv_nnc_cmd_find_backend` after `ccv_nnc_init`; this independently confirms the compiled registry, not just source-file absence:

| Operation | Command ID | NHWC CUDA FP16 Selection | NHWC CPU FP32 Selection |
| --- | --- | --- | --- |
| Gated delta | `0x6de198ac` | `0` (no backend) | `0x3d9883e5` (CPU reference) |
| Gated RMS normalization | `0x9b8af346` | `0` (no backend) | `0x3d9883e5` (CPU reference) |
| Standard scaled dot-product attention, positive control | `0x284ed926` | `0x5f19790a` (CUDA reference) | Not queried |

Probe source, shared object, and registry log are `backend-probe.c`, `backend-probe.so`, and `backend-probe.log` under the ignored MOSS output directory. LLDB resolves stripped functions through ASLR-aware loaded symbol addresses and passes them through the header-grounded probe, avoiding guessed by-value struct layouts. The missing CUDA operations are a confirmed prerequisite gap; locating the first non-finite intermediate still requires further evidence. Do not claim that the attention bypass or a successful backend lookup establishes valid numerical output.

Before implementing a runtime fix, the existing `DRAW_THINGS_PATCH/README.md`, manifest, applier, and generator were reviewed as requested. Preserve their unified-diff/snapshot provenance and strict application/idempotence verification approach. Keep native-CUDA changes explicitly separate from the legacy CPU quantization bundle: its Package and assertion patches must not be replayed into the CUDA candidate. The next implementation needs validated CUDA support or an explicit tested fallback for both Qwen3.5 operations, plus safe empty-attention handling. Record any such patch in `DRAW_THINGS_PATCH` against the pinned community/s4nnc/CCV revisions, build an isolated candidate, and require finite encoder/latent/final-image tests before accepting M1. No upstream runtime patch, rebuild, installation, catalog edit, or promotion was performed during this diagnosis.

### M1 CUDA Integration And Empty SDPA: October 26

The subsequent [hybrid hardening and native integration](WEELLM_HYBRID_TOOLING_PLAN_2026-10-24.md) implements the two Qwen CUDA backends in a separate pinned [CUDA profile](../DRAW_THINGS_PATCH/cuda/README.md). The legacy CPU bundle, installed runtime, and pristine candidate are preserved. The original operator-integrated binary is retained at `output/upgrades/gRPCServerCLI-0621d819-qwen35-cuda-before-empty-sdpa`; the empty-tensor probe reproduces its SIGFPE at SDPA `+3614`.

The profile now adds an early all-empty SDPA no-op and rejects partially empty descriptors or nonempty auxiliary outputs. No assertion suppression or generated-table hand edits are used. The new isolated binary has SHA-256 `5cca983e5883b5e9091f24d34475af99bc6054e67aa51da1f7f6fa8e7441febd`, built with 16 jobs against a fresh private CCV copy. Native acceptance passes all six Qwen precision selections, actual Qwen GPU/CPU comparisons, eight SDPA empty/mixed/projected contracts, and a nonempty grouped FP16 attention control. The full CUDA-enabled suite passes 56 tests.

The short guided MOSS profile (512x512, two steps, CFG 3, fixed shift 5, seed 4242, the same prompt/settings as M1) was rerun using independent private copies after all six canonical/staged model hashes passed. The crash is resolved and stream/post-Echo finish, but there are zero final images and one preview containing 131,072 non-finite values. The strict canary correctly fails. Output: `output/q6p_canary_moss_base_q8p_512_qwen35_empty_sdpa_2steps_20261026`. Observed whole-device peak is 979 MiB over 323 samples, maximum 57 C; four utilization readings are unavailable. These failed-run measurements do not establish full-model memory requirements. The 50-step run was withheld.

Separate bounded live-call diagnostics confirm the first actual FP16 Qwen gated-delta and gated-RMSNorm calls return success with finite inputs and outputs after synchronization. They do not prove all later layers remain finite. The next gate is finding the first subsequent non-finite activation across encoder/denoiser/decode boundaries. `GEMM_EX` parameter-12 warnings are recorded but not established as the cause; the old LTX GEMM investigation remains deferred. No image acceptance, SFT/packed-quality result, default installation, or promotion is claimed.

Post-test hashes match for all three canonical downloads, all three private model copies, the installed server/launcher, all three CPU converters, both active catalogs, and the pristine candidate. Diagnostic inferiors were explicitly killed and GPU memory returned to zero.

### M1 Packed Embedding Fix And Finite Output: October 26

The remaining non-finite values were traced to Qwen's first nonempty token lookup, not to the new recurrent operators: the packed Q5_K embedding table was interpreted as dense float storage by CUDA index-select. The output already contained 1,848 non-finite values before the first decoder normalization. The pinned CUDA profile now uses CCV's existing packed decoder before typed lookup, adds QX registration and explicit validation, and retains the pre-fix binary as a negative control. The new isolated binary is SHA-256 `920bf727c5a27a7d1ad90735735c74b7edfbbc1459598cea018f71e2d713dd96`; it was rebuilt with 16 jobs without changing installed or legacy CPU runtimes.

The native packed lookup gate checks Q5_K/Q6_K, F32/F16/BF16, and repeated/reordered indices against the pinned native CPU encoder/decoder. F32 is exact; reduced precision permits two ULPs with strict finite/sign checks. The combined suite now passes 57 tests with CUDA enabled. The original candidate and pre-fix controls remain independently retained.

Both the two-step smoke profile and original 50-step M1 profile now pass strict final-image checks: one 512x512 finite final image, zero audio, complete stream/chunks, successful post-Echo. Acceptance reports and images are under `output/q6p_canary_moss_base_q8p_512_packed_embedding_2steps_20261026` and `output/q6p_canary_moss_base_q8p_512_packed_embedding_50steps_20261026`. The full config hash still matches the original `f4a1fac323101d86d63fb2650483a9baa4d3cb8c1ef7edb3d64e81f16d5facf9`. Telemetry spans about 120/475 seconds including startup; both runs sample a 2,305 MiB whole-device peak, with maximum temperatures 56/59 C. One full-run utilization sample is unavailable.

**M1 quality remains unaccepted.** The 50-step image is a repeated green structure, not the requested orange teapot, so finite numerical output is not evidence of correct conditioning or model parity. Next, compare reference text conditioning, denoiser, sampler, and decode boundaries. SFT, packed-quality comparisons, and default promotion remain gated. The remaining cuBLAS warnings were traced to zero-sized skipped GEMMs with matching FP16 operands, not a mixed-datatype call; the deferred LTX crash work was not reopened.

### M1 Quality Boundary Checks: October 26

No runtime source change is made in this follow-up; the candidate remains SHA-256 `920bf727c5a27a7d1ad90735735c74b7edfbbc1459598cea018f71e2d713dd96`. Bounded actual activation captures independently pass CPU checks for MOSS text projection, first text attention and log-sum-exp, rotary multiplication/positions, and FP32 timestep projection. The timestep weights match the checkpoint exactly; text conditioning is finite, nonzero, and distinct between prompts. These are operation-level checks, not proof of full text-encoder or denoiser correctness.

New `compare-capture` tooling records hashes, shapes, datatypes, tolerances, and errors and refuses malformed/non-finite/oversized or unsupported captures. The packed-GEMM gate now checks nonuniform weights/activations plus a dense GPU control; its CPU model accounts for the documented int8 activation quantization in the packed fast path. The full CUDA-enabled suite passes 60 tests.

The 50-step CFG-1 isolation test passes finite-output checks but still depicts an unrelated structured scene, not the teapot. Removing the special negative branch is insufficient. A debugger-only bypass of packed fast GEMM instead causes non-finite encoder conditioning and was rejected. The normal CFG-3 profile, source files, models, and installed/default runtime were not changed by these experiments. M1 still needs full reference-boundary and prompt-fidelity acceptance.

### M1 First Prompt-Following Base Result: October 27

A complete Qwen recurrent trace identified the remaining quality blocker: native CUDA RMSNorm rejected Qwen's non-unit key/query normalization scales with `CCV_NNC_EXEC_INVALID`. The pinned profile now supports finite non-unit forward output scales by running its established scale-one path and scaling only normalized output, preserving saved inverse-standard-deviation values. The previous binary remains a negative control; 36 scaled affine/non-affine precision cases and the complete 29-operation recurrent layer now pass. The combined CUDA-enabled suite passes 67 tests.

The isolated candidate is SHA-256 `1d27f8d444d6211322496d3ce20871173b5d7d5942401bbe07fba9fa216a91b5`, built with 16 jobs against a fresh private CCV copy. The original 50-step CFG-3, shift-5, seed-4242, 512x512 Base profile now produces one finite image visibly depicting the requested orange teapot on a white table beside a plant. Stream/chunks/post-Echo pass, and config SHA-256 remains `f4a1fac323101d86d63fb2650483a9baa4d3cb8c1ef7edb3d64e81f16d5facf9`. Raw/PNG/acceptance and GPU evidence are under `output/q6p_canary_moss_base_q8p_scaled_rmsnorm_50steps_20261027`. Sampled whole-device peak is 2,305 MiB; telemetry spans about 129 seconds including startup, maximum 65 C, with no missing readings.

This is the first prompt-following M1 Base bring-up observation, not production/default promotion or universal image-quality certification. Repeatability, another prompt, SFT, quantization-quality comparisons, and broader framework parity remain separate gates. The isolated reference packages did not change normal toolkit dependencies or download new model weights. Existing installed binaries, CPU tools, and canonical payloads/catalogs remain preserved; the old LTX GEMM crash investigation remains deferred.

### M1 Base Repeatability And Second Subject: October 27

The accepted teapot profile was repeated in a fresh server process on candidate `1d27f8d...`. Config, raw final tensor payload, decoded tensor values, and PNG are all byte-identical; mean/max absolute tensor error are both zero. Raw final SHA-256 is `c307642f2b15eff92552b64f077455663bb3dc7fe7575289a510c00bc11c5b56`. Evidence: `output/q6p_canary_moss_base_q8p_scaled_rmsnorm_50steps_repeat_20261027` and `output/upgrades/moss-image-base-add235e9/base-repeatability-summary.json`.

A second prompt and seed (`4243`) requested a blue glass vase with three white daisies on a red book. The 512x512, 50-step CFG-3/shift-5 result is finite and visibly depicts those objects/colors, with complete stream/chunks/post-Echo and zero audio. Evidence: `output/q6p_canary_moss_base_q8p_scaled_rmsnorm_second_prompt_20261027`; its config SHA-256 is `ac4856ce713a3a67dfa63c24fa9aba99f7aea09cc3041c6198c164e29b19c6b2`.

Repeat/second-prompt telemetry spans about 135/127 seconds including startup, both with sampled whole-device peak 2,305 MiB and maximum 65 C, no missing readings. This establishes repeatability for one fixed profile and prompt-following observations on two subjects/seeds, not universal model quality, exact cross-framework parity, higher-resolution coverage, SFT, or promotion. Runtime and canonical payload/catalog hashes remain protected. Next model gate is SFT q8p at the separately pinned official artifact.

The already-prepared isolated setup can reproduce the Base quality canary without selecting the live model directory:

```bash
models="$PWD/output/upgrades/moss-image-base-add235e9/models"
schema="$PWD/output/upgrades/schemas/0621d819"
candidate="$PWD/output/upgrades/community-0621d819-qwen35-cuda"
LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}" \
bash tools/run_q6p_canary_once.sh \
	--grpc-bin "$candidate/bazel-bin/Apps/gRPCServerCLI" \
	--proto "$candidate/Libraries/GRPC/Models/Sources/imageService/imageService.proto" \
	--cache-dir "$schema" --bindings-dir "$schema/flatbuffers" \
	--models-dir "$models" --model moss_image_base_q8p.ckpt \
	--require-model-file qwen_3.5_9b_i5x.ckpt --require-model-file flux_2_vae_f16.ckpt \
	--host 127.0.0.1:7862 --image-mode --width 512 --height 512 --steps 50 \
	--sampler DDIMTrailing --guidance-scale 3 --shift 5 --seed 4242 \
	--prompt 'An orange ceramic teapot on a white table beside a green plant, bright window light, product photograph.' \
	--negative-prompt '' --server-cpu-offload --server-fread-preferred \
	--server-weights-cache 0 --gpu-metrics --final-mode --timeout-sec 3600
```

The canary's automatic timestamp tag avoids overwriting the verification runs above. It starts an explicitly selected local candidate only for the test and cleans it up afterward; it does not install or promote a default server.

## Sources

- [Pinned community source][upstream-head] and [320-commit comparison][upstream-compare].
- [Published server release][server-release] and [Docker Hub tag metadata][docker-latest].
- [Initial MOSS integration][moss-initial] and [model/LoRA follow-up][moss-followup].
- [Candidate model zoo][upstream-zoo], [MOSS builders][upstream-moss], [runtime dispatch][upstream-unet], [importer][upstream-importer], [quantizer][upstream-quantizer], and [SwiftPM manifest][upstream-package].
- [Linux WORKSPACE][upstream-workspace], [Bazel version][upstream-bazel-version], and [configuration schema][upstream-config].
- [Pinned s4nnc codecs/store][upstream-store] and [pinned CCV CUDA GEMM][upstream-cuda-gemm].
- [Official MOSS Base card][official-moss], [SFT card][official-moss-sft], and [reference project][moss-project].
- [Draw Things Base artifacts][dt-moss-base] and [SFT artifacts][dt-moss-sft].
- Local [pipeline](../PIPELINE.md#L1), [findings history](../CUSTOM_MODEL_CONVERTER_FINDINGS_2026-07-09.md#L1), and [patch maintenance instructions](../DRAW_THINGS_PATCH/README.md#L1).

[upstream-head]: https://github.com/drawthingsai/draw-things-community/commit/0621d8198f3ed505bb7cefc71b84f8f0e1ce5936
[upstream-compare]: https://api.github.com/repos/drawthingsai/draw-things-community/compare/9f3f04b7...0621d8198f3ed505bb7cefc71b84f8f0e1ce5936?per_page=1&page=2
[server-release]: https://github.com/drawthingsai/draw-things-community/releases/tag/v26.1014.0
[docker-latest]: https://hub.docker.com/v2/repositories/drawthingsai/draw-things-grpc-server-cli/tags/latest
[moss-initial]: https://github.com/drawthingsai/draw-things-community/commit/a32aa0bbfd82ab37418acf58578eb164f076396c
[moss-followup]: https://github.com/drawthingsai/draw-things-community/commit/6c1dd09303cde8e23b18d1f1e57894a559a498db
[upstream-zoo]: https://github.com/drawthingsai/draw-things-community/blob/0621d8198f3ed505bb7cefc71b84f8f0e1ce5936/Libraries/ModelZoo/Sources/ModelZoo.swift
[upstream-moss]: https://github.com/drawthingsai/draw-things-community/blob/0621d8198f3ed505bb7cefc71b84f8f0e1ce5936/Libraries/SwiftDiffusion/Sources/Models/MOSSImage.swift
[upstream-unet]: https://github.com/drawthingsai/draw-things-community/blob/0621d8198f3ed505bb7cefc71b84f8f0e1ce5936/Libraries/SwiftDiffusion/Sources/Models/UNetProtocol.swift
[upstream-importer]: https://github.com/drawthingsai/draw-things-community/blob/0621d8198f3ed505bb7cefc71b84f8f0e1ce5936/Libraries/ModelOp/Sources/ModelImporter.swift
[upstream-quantizer]: https://github.com/drawthingsai/draw-things-community/blob/0621d8198f3ed505bb7cefc71b84f8f0e1ce5936/Apps/ModelQuantizer/Quantizer.swift
[upstream-package]: https://github.com/drawthingsai/draw-things-community/blob/0621d8198f3ed505bb7cefc71b84f8f0e1ce5936/Package.swift
[upstream-workspace]: https://github.com/drawthingsai/draw-things-community/blob/0621d8198f3ed505bb7cefc71b84f8f0e1ce5936/WORKSPACE.linux
[upstream-bazel-version]: https://github.com/drawthingsai/draw-things-community/blob/0621d8198f3ed505bb7cefc71b84f8f0e1ce5936/.bazelversion
[upstream-config]: https://github.com/drawthingsai/draw-things-community/blob/0621d8198f3ed505bb7cefc71b84f8f0e1ce5936/Libraries/DataModels/Sources/config.fbs
[upstream-store]: https://github.com/liuliu/s4nnc/blob/b5476a77e7f67f12729872e4b9f1a649b3cd3df0/nnc/Store.swift
[upstream-cuda-gemm]: https://github.com/liuliu/ccv/blob/69b1ed1f4e281520c29cd4b45b93c6fcdde8d2e8/lib/nnc/cmd/blas/gpu/ccv_nnc_gemm_gpu_cublas.cu
[official-moss]: https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/blob/85975793d8b3d432f1b6e27f352abaa4c2a4c0d3/README.md
[official-moss-sft]: https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT
[moss-project]: https://github.com/OpenMOSS/MOSS-Image
[dt-moss-base]: https://huggingface.co/drawthingsai/MOSS-Image-Base/tree/add235e9063a9fb0eec06f61cba6a970337abf1e
[dt-moss-sft]: https://huggingface.co/drawthingsai/MOSS-Image-SFT/tree/55d7bdb21604cba9ff35795b6b0e163f0d51802a