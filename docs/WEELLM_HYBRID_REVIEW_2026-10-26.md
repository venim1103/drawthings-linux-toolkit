# WeeLLM Hybrid Tooling Review

Review date: 2026-10-26. Local baseline: `cf8baab`. Reviewed range: `ef88d54..cf8baab`, seven commits, 21 changed files, approximately 5,000 added lines. External reference: WeeLLM `88233af976568bfb134f082b62434c1b3f849b28`, the exact revision cited by the weekend plan, not its moving `main`.

This is a review, not an implementation change. Findings distinguish executed reproductions from source inspection and outstanding validation. Tests used Python 3.10.12, tiny temporary fixtures, and read-only metadata queries on existing models. No full model conversion, quantization, GPU inference, native rebuild, active-catalog write, or installed-runtime change was performed.

## Findings

### F01. The MOSS Contract Does Not Describe The Actual MOSS Model [High]

Locations: [MOSS contract generator](../tools/dt_keymap_compiler.py#L432), [checked-in contract](../contracts/moss_image.json), [native MOSS mapper][native-moss], [WeeLLM MOSS implementation][weellm-moss].

The generated rules use `transformer_blocks.*`, per-block `scale_shift_table`, projection biases, and `text_refiner_blocks.*`. The native mapper and the pinned WeeLLM model instead use `layers.*.proj.*`, packed `ffn_up_gate`, shared block/head modulation, `text_embedder.proj`, and `text_embedder.blocks.*`. Main Q/K/V/O projections have no bias. MOSS QK-norm vectors are per-head width 128; the new rules assume a 24-head, 3,072-element permutation and an LTX-style rank policy.

Executed a covering-index keyset comparison against the official downloaded MOSS q8p checkpoint:

| Measure | Result |
| --- | ---: |
| Official checkpoint keys | 524 |
| Generated target keys | 862 |
| Intersection | 170 |
| Official keys missing from contract | 354 |
| Generated keys absent from official checkpoint | 692 |

Missing targets include context embedding/refiner weights and shared modulation projections. The second timestep projection is actually `time_embedding.2` / native `time_embedder_2`, not the contract's `timestep_embedder.linear_2` / `t_embedder_1`. Therefore adding aliases or relaxing missing-key checks cannot fix this contract. The direct MOSS conversion advertised in the playbook is not usable as implemented.

**Required correction:** derive the complete mapping from the pinned native mapper, including fixed and main models, shared modulation split order, fused FFN split, bias presence, per-head norm shape, and per-parameter storage precision. Freeze an independently checked 524-key target manifest before attempting full conversion. Do not transplant LTX norm/rank conventions into MOSS.

### F02. External-Store Detection Reads The Wrong Bits [High]

Locations: [metadata reader](../tools/dt_weellm/sqlite_store.py#L65), [payload reader](../tools/dt_weellm/sqlite_store.py#L92), [native SQLite writer][native-tensor-io].

Native storage packs `type = (codec_identifier << 32) | tensor_memory_type`. The external-data marker `0x10000000` belongs to the **identifier**, but the Python reader tests that bit directly in the packed 64-bit `type`.

Executed a native-format synthetic row with `type = (0x10000000 << 32) | 1`, a valid 24-byte external descriptor, and a matching sidecar. The reader reports `is_external=False` and then mistakes the descriptor for tensor values, failing to read the valid tensor. Missing/truncated sidecars also lack a fail-closed descriptor validation path.

The reader does not implement Draw Things palettized/packed codecs, even when they originate from GGUF. GGUF Q4_K decoding does not decode a Draw Things q4p or i4x store. Metadata must separate the memory type, codec identifier, datatype, and reserved upper bits before deciding whether bytes are raw floats.

**Required correction:** implement or bridge the actual native codec/storage contract, validate offsets and exact read lengths, and test raw, compressed, external, and supported trailer variants independently. Until then, restrict supported formats explicitly rather than advertising transparent store compatibility.

### F03. The Direct Writer Cannot Store A Real LTX Aggregate Tensor [High]

Locations: [single-BLOB writer](../tools/dt_weellm/sqlite_store.py#L151), [conversion loop](../tools/dt_direct_ckpt_builder.py#L64).

Source-header inspection, without loading values, found `text_embedding_projection.video_aggregate_embed.weight` has BF16 shape `[4096, 188160]`: **1,541,406,720 bytes**. The active Python SQLite reports `MAX_LENGTH=1000000000`. `write_tensor` always binds the entire payload into one BLOB and does not implement native large-tensor splitting or externalized output.

The scalar/small-matrix unit tests cannot expose this. The existing LTX source resolves all 4,448 generated source rules, but real conversion will still hit the storage limit. Catching `sqlite3.DataError` while reading a large tensor and returning `None` also obscures unsupported-size failures.

**Required correction:** support native-compatible `tensor_splits` or validated external storage, or fail before materialization with an actionable size check. Add a low-limit boundary test plus a native reader round-trip. Merely changing the in-memory array representation does not remove SQLite's limit.

### F04. Comparison And Health Gates Can Pass Corrupt Or Unchecked Data [High]

Locations: [checkpoint diff](../tools/dt_direct_ckpt_builder.py#L262), [health inspection](../tools/dt_layer_parity_oracle.py#L237), [contract validation](../tools/dt_validate_converted_ckpt.py#L329).

Executed temporary-fixture results:

| Case | Observed result |
| --- | --- |
| Matching metadata, malformed one-byte payloads | `is_equivalent=True`, zero compared values |
| Candidate contains NaNs, reference is finite | `is_equivalent=True`, reported as a value match |
| Candidate contains an extra tensor | `is_equivalent=True` |
| Unsupported datatype in both files | `is_equivalent=True`, zero compared values |
| Health audit of unsupported datatype | Pass, `finite_checked=0`, `total_tensors=1` |
| One arbitrary `__dit__[totally_wrong]` row with MOSS contract | Validator prints `RESULT=PASS` |

The diff silently catches comparison exceptions. `NaN > atol` is false, and equivalence excludes neither extra keys nor incomplete coverage. Health inspection skips unknown datatypes. The checked-in MOSS contract only requires one `__dit__[` prefix and has no expected total, so it cannot establish model completeness.

The documented F16-vs-official-q8p gate is also not byte equivalence: it requires explicit dequantization and a quantization-aware error policy. The optional `contract_path` argument to the diff is currently unused.

**Required correction:** report pass/fail/inconclusive separately; require complete intended key and value coverage, exact shape checks, finite values, and explicit supported codecs. Make zero checks an error. Test each of the cases above. Use an unquantized positive control for serialization parity and separate tolerances for quantization error.

### F05. Strict Conversion Does Not Enforce Its Stated Safety Guarantees [High]

Locations: [contract loader](../tools/dt_direct_ckpt_builder.py#L45), [conversion and promotion](../tools/dt_direct_ckpt_builder.py#L64), [contract schema](../contracts/schema.json), [float serialization](../tools/dt_weellm/transforms.py#L308).

`strict=True` only makes missing source keys fatal. It does not reject unmapped input tensors, enforce expected output count, validate ordinary tensor shapes, detect multiple rules resolving to one source/target, or validate the JSON Schema. `INSERT OR REPLACE` can hide duplicate targets. Neither contract supplies per-rule shape expectations.

Executed a tiny strict conversion with an expected count of 99, one mapped finite F32 tensor containing `1e10`, and an extra source tensor. It promoted a one-tensor checkpoint containing FP16 infinity. Input finiteness before a narrowing cast does not guarantee output finiteness.

There is also no SQLite integrity/native-reader acceptance gate before `os.replace`. Atomic replacement makes publication indivisible; it does not establish correctness or durability. Existing destinations and their sidecars require an explicit overwrite/storage-pair policy.

**Required correction:** validate a resolved conversion plan before reading payloads, account for every required/ignored source and unique target, enforce shapes and precision, check after transformations and after casts, validate the staged database, and promote only after all gates pass. Keep failures away from canonical model files.

### F06. Contract Norm Restoration Can Corrupt Rank And Codec Metadata [High]

Location: [in-place restorer](../tools/dt_direct_ckpt_builder.py#L148).

Executed a rule declaring `dim_rank_policy="preserve"` for a `[2,3]` F32 timestep matrix. Restoration changes it to `[1,1,6]`. The function applies the LTX norm rank policy indiscriminately, including MOSS timestep matrices selected by broad regexes.

Executed additional cases:

- A `bf16_from_source` rule is ignored; a zeroed F16 target remains zeroed F16.
- Missing required source values increment `skipped_missing` without making the operation fail.
- Replacing a row marked with q8p codec identifier `0x8a1e8b` leaves that identifier in the packed `type`, even though its payload is now raw F32. The resulting data and codec metadata disagree.

Already-F32/rank-3 rows are skipped without checking that their values match the source. The implementation bypasses its own store abstraction, does not read externalized/compressed source values, and commits in-place batches rather than validating/staging the whole change.

**Required correction:** honor the exact per-tensor datatype/rank/storage policy, rewrite codec metadata consistently, compare own-source values, and fail on required skips or unsupported storage. Stage the database/storage pair or use a rollback-capable transaction after complete preflight. Retain known-working legacy restorers until independent parity tests establish a replacement.

### F07. LoRA Normalization And Verification Are Not Fail-Closed [High]

Locations: [Kohya unmangling](../tools/dt_weellm/keymaps.py#L185), [normalizer](../tools/dt_lora_universal_bridge.py#L65), [verifier](../tools/dt_lora_universal_bridge.py#L118).

The placeholders used to protect compound names contain underscores. The subsequent underscore-to-dot replacement changes the placeholders themselves, preventing restoration. Executed:

```text
input:    lora_unet_transformer_blocks_0_attn1_to_q.lora_down.weight
parsed:   ..TOK0...0.attn1...TOK15...weight
expected: transformer_blocks.0.attn1.to_q.weight
```

Executed verifier failures in the opposite direction: a fused A/B pair whose contract has three target names returns **pass with zero verified pairs against an empty checkpoint**; completely unrecognized input does the same. Multi-target rules are silently skipped. Alpha/rank scaling, fused split rules, collision detection, complete target coverage, and non-finite/shape checks are not implemented as advertised.

The native `__up__`/`__down__` suffix convention does match an existing trusted converted LoRA, which is a useful starting point; the failures are in normalization and proof coverage, not that suffix convention.

**Required correction:** use collision-safe tokenization, require at least one recognized pair and full mapped coverage, reject unsupported LoRA families instead of claiming universal support, and compare the effective delta including alpha/rank and fused transforms. Test malformed names, duplicate aliases, missing pairs, empty inputs, NaNs, and scaled adapters.

### F08. Qwen3.5 Fixtures Do Not Match The Actual Native Command Contract [High]

Locations: [gated-delta reference](../tools/dt_layer_parity_oracle.py#L66), [fixture generation](../tools/dt_layer_parity_oracle.py#L184), [native gated-delta implementation][native-gated-delta], [Qwen3.5 configuration][native-qwen35].

The 9B model uses **16 key heads and 32 value heads**. The NumPy recurrence assumes equal head counts. Executing a smaller 2-key/4-value-head case raises a broadcasting error. It returns state `[B,H,Dk,Dv]`, while the native command expects `[B,Hv,Dv,Dk]`. A non-square 3-by-5 fixture exposes the transposition hidden by the generated 16-by-16 case.

The native path also accepts an initial state, uses `logDecay: true` for Qwen3.5, and supports state checkpoints. The fixture always starts at zero, stores exponentiated decay, and records neither a log-decay mode flag nor a state-layout conversion. The recurrence is useful for its restricted equal-head, exp-decay case, but it is not yet a drop-in golden vector for the real command. RMSNorm fixtures likewise do not encode the real per-value-head grouping.

**Required correction:** specify layout, head expansion, gate convention, initial/final state, epsilon, dtype, and command flags in the fixture manifest. Validate against the existing native CPU backend and a pinned Transformers/official reference. Include grouped heads, unequal Dk/Dv, nonzero state, multiple chunks, and log-decay tests before using fixtures to approve a CUDA implementation.

### F09. The Parity Oracle Does Not Execute Either Runtime [High]

Locations: [weight verifier](../tools/dt_layer_parity_oracle.py#L115), [actual CLI modes](../tools/dt_layer_parity_oracle.py#L362), [claimed oracle stages](WEELLM_HYBRID_TOOLING_PLAN_2026-10-24.md#L247).

There is no WeeLLM model loading, text-encoder hook, native activation capture, single-block forward comparison, timestep comparison, or conditional/unconditional step bisection in the delivered tool. Its three modes are a NumPy fixture generator, weight comparison, and checkpoint inspection. Consequently W2 does not yet isolate the first MOSS non-finite intermediate or validate missing CUDA kernels.

The weight verifier also implements a different transform subset from the builder: it ignores mixed rotary slices, offsets, scale, and split-policy details. A valid three-way fused QKV conversion with rotary slices `[true,true,false]` is rejected with max difference 2.0. Conversely NaN-aware comparisons are missing. Shared mapping assumptions cannot independently prove the mapping itself is correct.

**Required correction:** distinguish serialization checks from execution parity. Implement reference and native adapters that consume identical tokens, weights, activations, and sigma arrays and emit compatible stage fixtures. Use one declared transform execution path plus independent native golden vectors. Do not advertise forward or five-second runtime parity until it has actually been measured.

### F10. Per-Output-Channel FP8 Scaling Is Applied To The Wrong Axis [High]

Location: [FP8 scaling](../tools/dt_weellm/seek_io.py#L242), [upstream scale handling][weellm-comfy].

The reader multiplies `[out,in]` weights directly by a one-dimensional `[out]` scale. NumPy broadcasts across the last axis, not the output-channel axis. On a square matrix this silently produces wrong values; on many rectangular matrices it raises a broadcasting error.

Executed a 2-by-2 FP8 matrix of ones with scales `[2,3]`: actual `[[2,3],[2,3]]`, required `[[2,2],[3,3]]`. The pinned upstream implementation explicitly reshapes row scales to `[out,1,...]`.

**Required correction:** validate and distinguish scalar, per-output-channel, and supported block-scale layouts. Reject unsupported ComfyUI/NVFP4/SVDQ variants explicitly. Add square and rectangular matrix/conv tests; scalar-only FP8 examples are insufficient.

### F11. Three Documented CLI Entrypoints Fail Before Argument Parsing [High]

Locations: [builder imports](../tools/dt_direct_ckpt_builder.py#L28), [oracle imports](../tools/dt_layer_parity_oracle.py#L27), [LoRA bridge imports](../tools/dt_lora_universal_bridge.py#L24), shell call sites.

Executed exactly the documented form, from the workspace root:

```bash
.venv/bin/python tools/dt_direct_ckpt_builder.py --help
.venv/bin/python tools/dt_layer_parity_oracle.py --help
.venv/bin/python tools/dt_lora_universal_bridge.py --help
```

All three exit 1 with `ModuleNotFoundError: No module named 'tools'`. Running unittest as a module does not exercise direct-script imports. The keymap compiler's help does work, but that does not establish its optional imports on other commands.

**Required correction:** choose supported package/module entrypoints and use them consistently in documentation and shell wrappers, including execution from outside the repository. Add CLI help and tiny end-to-end subprocess smoke tests without relying on a developer's `PYTHONPATH`.

### F12. Shell Workflows Can Publish Or Continue Before Verification [High]

Locations: [direct conversion](../tools/dt_convert.sh#L112), [quantize/restore](../tools/dt_quantize.sh#L126), [LoRA catalog write before verification](../tools/dt_lora_convert.sh#L204).

The new opt-in paths inherit warning-and-continue validation behavior that is incompatible with the plan's fail-closed claims. Direct conversion does not forward the contract to final validation. An explicitly missing contract silently falls back to LTX. The wrapper rejects a sharded input directory with `-f`, although the underlying seeker supports directories; preflight is still single-file-oriented.

LoRA registration happens before `--verify-post`. A missing contract silently disables verification, and verification reads the original rather than the normalized input. Thus a later failure can leave an active alias for an unapproved artifact. Quantization selects its model version before reading the contract; contract metadata only affects the later alias write. Catalog updates are not atomic.

The playbook also mentions i4x while this shell interface accepts only q4p/q6p/q8p. Existing CPU binaries and their version support are not upgraded merely by introducing contracts. No MOSS-capable quantization path was demonstrated.

**Required correction:** validate all arguments/contracts before expensive work; make profile/version agreement explicit; stage outputs and perform all required checks before atomic artifact/catalog publication. Keep legacy behavior unchanged by default, but make the new strict path genuinely strict. Validate missing-contract and failed-verification scenarios in an isolated copied workspace, not against live catalogs.

### F13. Sharded Safetensors Loading Can Read A Key From The Wrong Shard [Medium]

Location: [index ingestion](../tools/dt_weellm/seek_io.py#L110).

The loader merges all tensor headers and only verifies that every index key appears somewhere. A duplicate silently overwrites the previously loaded entry, ignoring the index's key-to-shard assignment.

Executed an index assigning `w` to shard A with value 1, while shard B contains another `w` with value 9 plus its own indexed tensor. Reading `w` returns 9. This is a silent source-data selection error, not just a missing-shard problem.

**Required correction:** treat the index as authoritative, reject duplicate/misplaced keys and path traversal, enforce supported header/shape/offset limits, and check overlapping ranges/duplicate JSON keys. Preserve legitimate ignored tensors through an explicit policy, not implicit overwriting.

### F14. Architecture-Specific Alias Resolution Is Not Wired Into Conversion [Medium]

Locations: [builder resolution call](../tools/dt_direct_ckpt_builder.py#L95), [seeker default profile](../tools/dt_weellm/seek_io.py#L185), [GGUF resolver](../tools/dt_weellm/seek_io.py#L403).

Both seekers default to `ltx2_3`. The builder and weight oracle never pass the selected contract profile or per-rule aliases. Executed a canonical MOSS Q key against its implemented `layers.0.attention.wq.weight` alias: default lookup returns `None`, explicit `profile="moss_image"` resolves it.

The external WeeLLM keymap loader is exported but not consumed by the compiler/converter; the compiler's actual CLI has no `--weellm-keymap` or `--swift-model` options shown in the proposed recipe. This is a hand-maintained registry prototype, not an automatic integration with WeeLLM keymaps.

**Required correction:** carry the profile/source dialect through a resolved mapping plan, detect ambiguous aliases, and test complete representative headers for every advertised dialect. Keep source naming adaptation separate from target serialization.

### F15. Swift Auditing Is A Heuristic, Not A Complete Static Gate [Medium]

Location: [auditor](../tools/dt_keymap_compiler.py#L750).

The function-declaration regex does not handle generic functions such as `MOSSImageBlock<T: TensorNumeric>`. Detection is centered on `to_q`/`to_k` and literal `interleaved: true`, while MOSS uses `proj.q` and `interleavedIndices`. It computes expected head parameters but does not compare them. The promised generic-importer/fatalError rule is absent; the precision rule only covers aggregate embeddings.

Executed a wrong-head norm mapping with `interleaved: true`: no findings. Executed a `ModelImporter.swift` sample containing an unconditional `fatalError`: no findings. Therefore the tool's PASS is not evidence of MOSS importer support or exact precision/rotary correctness.

**Required correction:** describe the supported lint patterns honestly, add negative cases for the real MOSS syntax, and prefer an exported native mapping manifest to regex parsing for correctness gates. Full source/destination-layout semantics in `TensorDescriptor.swift` also need independent fixtures before claiming every transform is replicated.

### F16. Completion Claims Exceed The Test Coverage [Medium]

Locations: [status table](WEELLM_HYBRID_TOOLING_PLAN_2026-10-24.md#L333), [unit tests](../tools/test_dt_weellm.py#L446).

Executed `.venv/bin/python -m unittest tools.test_dt_weellm -v`: **13 tests, 12 passed, 1 failed**. The failing test expects 1,252 LTX source rules while hydration now yields 4,448, so it exits before reaching its Kohya name assertions. Fixing only the count will expose the actual normalization defect in F07.

The tests mostly use a reader, writer, generated mapping, and verifier from the same implementation. They do not prove native storage compatibility, real MOSS conversion, full forward parity, sidecar correctness, CLI invocation, fused LoRA scaling, or safe publication. GGUF coverage in the committed suite is principally Q8_0, despite the ten-format implementation claim.

**Required correction:** mark W0-W4 as partial/prototype with explicit evidence levels. Add independent positive controls and failure-path tests before updating the completion table. Do not change an expected count merely to make CI green without checking the native manifest.

### F17. Reference Reproducibility, Precision, And Attribution Need Explicit Boundaries [Medium]

Sources: [pinned WeeLLM packaging][weellm-package], [Qwen streamer][weellm-qwen35], [GGUF decoder attribution][weellm-gguf].

WeeLLM uses a Git-based moving Diffusers dependency, broad version constraints, and `trust_remote_code=True` for the Qwen skeleton. The Qwen streamer also monkey-patches its model class's `.to` method. This is useful bring-up machinery but should run in a pinned isolated process/environment, not be imported opportunistically into converter/runtime processes.

The MOSS streamer loads weights using its selected floating dtype. Native MOSS deliberately preserves the second time projection in F32. A BF16-streamed implementation is not automatically a bit-exact oracle for an FP16/F32 native graph. The reference scheduler snaps to a nearest training grid; native DDIMTrailing must be compared on exported sigma arrays, not just matching sampler descriptions. Identical seeds do not establish identical random latent tensors across engines.

Package metadata declares MIT, while the GGUF decoder explicitly attributes Apache-2.0 ComfyUI-GGUF code. The local adapted modules need a provenance/notice review appropriate to each reused component; a repository badge is not a substitute for that review. This is a compliance prerequisite, not a conclusion that the adaptation is unlawful.

**Required correction:** lock code and model/tokenizer revisions and dependencies; record input hashes, token IDs, latent arrays, dtypes, sigma arrays, branch/skip settings, and hardware. Preserve applicable third-party attribution. Treat memory/performance claims as measurements still to be made on the A3000, not inherited guarantees.

## What Is Worth Keeping

**Recommendation: retain the approach as an isolated engineering workbench, but do not use the new artifact-mutation or approval paths on canonical models yet.** There is no need to discard the whole implementation.

- The architectural separation is sound: PyTorch/NumPy for fast experiments and diagnostics; pinned Draw Things/s4nnc/CCV for the production target. Python can remove repeated full-converter/full-server rebuilds from many investigative loops.
- Independent differential tests in this review compared all ten implemented GGML block formats with **`gguf==0.13.0`**: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K. All matched exactly on three deterministic blocks per format with finite scales. This is useful positive evidence, not exhaustive format certification.
- Rotary permutation and its inverse match the explicit indexed formula tested in the suite. BF16 conversion and basic raw SQLite/safetensors round-trips provide useful primitives.
- The LTX source rules resolve all 4,448 source keys, with no rotary-axis size mismatches in metadata checks. Generated/native target sets both contain 5,746 entries; 5,744 match. The two wrong names are the audio/video learnable registers: native names are `text_audio_connector_learnable_registers` and `text_video_connector_learnable_registers`, not wrapped `__text_*_connector__[t-learnable_registers-0-0]` names. This is substantially closer than MOSS, although storage/value/rank acceptance remains open.
- Per-tensor seeking avoids loading an entire checkpoint and is useful even without a full reference runtime. It is not literally zero-copy: `pread`, dtype widening, contiguous transforms, serialization, and SQLite binding allocate buffers. One 1.54 GB BF16 tensor alone expands to about 3.08 GB in F32 before other copies.
- The existing native CPU gated-delta implementation is already an independent oracle. It can make the first useful CUDA microtests much smaller than running a complete 9B text encoder.

The target should be **faster validated iterations**, not a second production inference engine or a universal format bridge before one real model has passed end to end. The plan's unconditional compute-bound/native-faster and exact VRAM/seconds claims remain unbenchmarked. Native external-on-demand inference can also be I/O-bound.

## Recommended Implementation Order

1. **Contain and make the tooling executable.** Preserve the proven LTX/default runtime and catalogs. Repair invocation, add subprocess/negative-path tests, and make unsupported formats or zero-coverage checks fail. Keep the new paths opt-in and stop publishing before validation.
2. **Finish a real MOSS kernel-fixture workbench.** Specify GatedDelta/RMSNormGated's actual grouped-head/state/log-decay ABI, compare NumPy against native CPU and a pinned official/Transformers reference, then use the fixtures for standalone CUDA tests. Package native changes through `DRAW_THINGS_PATCH` with explicit CUDA scope, pinned revisions, strict application, and idempotence checks. Never replay the legacy CPU CUDA-exclusion/assertion patches into that candidate.
3. **Add an actual reference/native activation adapter.** Capture identical tokens/embeddings and the first denoiser step at 512x512, including the unconditional block-22 skip. Compare the first divergent intermediate and FP32-sensitive projections. Keep exact serialization comparison separate from quantization-aware activation tolerances. This is the most direct route to the existing M1 blocker.
4. **Repair storage and prove one unquantized artifact.** Use a native-validated minimal store, correct external/large-tensor handling, unique key/shape manifests, post-cast finite checks, and atomic staging. Prefer the closest known LTX control for serializer conformance; rebuild the MOSS contract from its actual 524-key native layout before MOSS full conversion.
5. **Only then generalize.** Add GGUF/FP8 source dialects with independent goldens, per-architecture restoration, and scaled/fused LoRA verification. Require native load, finite final output, private-copy preservation checks, and recorded memory/time measurements before changing aliases or declaring M1/M3/M4 complete.

An acceptable first milestone is not "all tools exist". It is: **one CLI works from a clean environment, one native microtest agrees with an independent reference, one staged checkpoint round-trips through the native reader, and negative cases cannot report PASS.**

## Open Questions And Validation Limits

- Full native acceptance of Python-written stores has not been demonstrated. SQLite schema similarity alone is insufficient.
- No WeeLLM model, full raw MOSS source, or Qwen text-encoder forward was run in this review. Existing Draw Things q8p/i5x artifacts are not drop-in raw WeeLLM model sources. Model/config/tokenizer provenance must be planned before downloading more weights.
- The missing CUDA operations are known prerequisites; the review does not prove they are the first cause of every non-finite intermediate. The empty-attention operation and successful finite generation remain separate gates.
- It is not established that any particular activation tolerance is appropriate across BF16, FP16, palettization, packed codecs, and different attention implementations. Determine tolerances with positive controls, not a universal `1e-4` threshold.
- The ten-format differential result does not certify every malformed GGUF, tensor dimension order, FP8 scale convention, or unsupported IQ/NVFP4/SVDQ type.
- Existing shell defaults were inspected and syntax-checked, but expensive conversion/quantization and catalog-mutating workflows were deliberately not executed. New integration issues above are source-grounded unless explicitly listed as reproduced.

## Validation Record

| Check | Result |
| --- | --- |
| Worktree before review | Clean at `cf8baab` |
| Weekend CPU suite | 12 passed, 1 failed |
| Four direct-script help commands | Compiler passes; builder/oracle/LoRA bridge fail imports |
| Three modified shell scripts | Syntax checks pass |
| Independent GGML decoder fixtures | 10/10 formats exact on three blocks each |
| Native LTX keyset vs generated | 5,744/5,746 overlap; two register-name mismatches |
| LTX source metadata | 4,448/4,448 rules resolve; zero rotary-axis mismatches |
| Native MOSS keyset vs generated | 170/524 native keys overlap; 692 invented targets |
| External descriptor, false PASS, restoration, FP8, shard, LoRA fixtures | Defects reproduced as detailed above |
| Full model conversion, GPU/native runtime parity, performance | Not run; no acceptance claimed |
| Installed runtime, native patch bundle, canonical model data/catalogs | Not modified |

The detached upstream checkout and the 75 KB reference wheel are under ignored `output/reviews`; the wheel was downloaded without installing it or its dependencies. Small reproduction payloads were created in automatically cleaned temporary directories. Real model queries used read-only SQLite connections and covering indexes or safetensors headers, not full tensor reads.

## Commit And Material Coverage

| Commit | Scope reviewed | Review outcome |
| --- | --- | --- |
| `6403f7e` | Initial WeeLLM architecture and recipe | Useful separation of roles; performance/coverage estimates need evidence |
| `7da03ca` | Core primitives, compiler, direct builder, tests | Retain primitives; fix native storage and fail-closed contracts |
| `c182c8d` | Oracle, MOSS contract, validator hooks | Oracle scaffold, not runtime parity; MOSS contract incompatible |
| `da73cc6` | Transforms, GGML dequantizers, keymap registry | Ten-format decoder positive control; alias/profile and scaling gaps |
| `ce6fe7f` | Remaining compiler/LoRA/shell work | Several incomplete or false-PASS paths, not W1-W4 completion |
| `950406a` | Contract alias registration and prefix checks | Metadata plumbing exists; publication order and validation too weak |
| `cf8baab` | Completion status and practical playbook | Must distinguish implemented helpers, synthetic tests, and native acceptance |

Reviewed all new library, compiler, builder, oracle, and bridge modules, public exports, the full test file, both profile contracts and schema, all weekend diffs to existing preflight/validator/shell/README files, and the complete hybrid plan. Cross-checked against the existing MOSS blocker record, native mapper/store/gated-delta sources, the patch-bundle workflow, real LTX/MOSS metadata, and relevant pinned WeeLLM seeker/decoder/model/streamer/scheduler/packaging sources.

## Reference Pins

- Native community: `0621d8198f3ed505bb7cefc71b84f8f0e1ce5936`.
- Native s4nnc: `b5476a77e7f67f12729872e4b9f1a649b3cd3df0`.
- Native CCV: `69b1ed1f4e281520c29cd4b45b93c6fcdde8d2e8`.
- WeeLLM: `88233af976568bfb134f082b62434c1b3f849b28`.
- Independent decoder package: `gguf==0.13.0`, pure-Python wheel used only in the probe process.
- Reference wheel SHA-256: `4e92a73b827c4618b55a615547b5120a904c2b34d02d5179dad5093c680623f6`.
- Prior native findings: [upgrade and MOSS record](UPGRADE_AND_MOSS_IMAGE_PLAN_2026-10-23.md#L519).
- Existing patch policy: [patch bundle](../DRAW_THINGS_PATCH/README.md).

[native-moss]: https://github.com/drawthingsai/draw-things-community/blob/0621d8198f3ed505bb7cefc71b84f8f0e1ce5936/Libraries/SwiftDiffusion/Sources/Models/MOSSImage.swift
[native-qwen35]: https://github.com/drawthingsai/draw-things-community/blob/0621d8198f3ed505bb7cefc71b84f8f0e1ce5936/Libraries/SwiftLLM/Sources/Models/Qwen3_5.swift
[native-tensor-io]: https://github.com/liuliu/ccv/blob/69b1ed1f4e281520c29cd4b45b93c6fcdde8d2e8/lib/nnc/ccv_nnc_tensor_io.c
[native-gated-delta]: https://github.com/liuliu/ccv/blob/69b1ed1f4e281520c29cd4b45b93c6fcdde8d2e8/lib/nnc/cmd/gated_delta/ccv_nnc_gated_delta_cpu_ref.c
[weellm-moss]: https://github.com/Jit-Roy/WeeLLM/blob/88233af976568bfb134f082b62434c1b3f849b28/weellm/models/transformers/moss_image_transformer_2d_model.py
[weellm-qwen35]: https://github.com/Jit-Roy/WeeLLM/blob/88233af976568bfb134f082b62434c1b3f849b28/weellm/models/text_encoders/qwen3_5_model.py
[weellm-comfy]: https://github.com/Jit-Roy/WeeLLM/blob/88233af976568bfb134f082b62434c1b3f849b28/weellm/io/safetensors/comfy_dequant.py
[weellm-gguf]: https://github.com/Jit-Roy/WeeLLM/blob/88233af976568bfb134f082b62434c1b3f849b28/weellm/io/ggufs/gguf_dequant.py
[weellm-package]: https://github.com/Jit-Roy/WeeLLM/blob/88233af976568bfb134f082b62434c1b3f849b28/pyproject.toml