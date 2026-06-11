# Q6P Root-Cause Plan for Custom LTX2.3

Last updated: 2026-06-10
Status: Active
Owner: Copilot + user

## Objective
Deliver a generic, repeatable pipeline that converts custom LTX2.3 safetensors models into stable q6p-usable Draw Things checkpoints.

Success means full generation completes (not only first streamed response) and no deterministic ccv_nnc_tensor_read / ccv_cnnp_model_read crash occurs.

## Agreed Constraints
- Mixed precision is acceptable inside the output artifact when needed for stability.
- Priority is generic custom LTX2.3 pipeline support, not a one-off fix for 10_e_v1.
- Post-hoc row patching is not the primary strategy.

## Root-Cause-First Strategy
1. Lock Baselines and Acceptance Gates
- Freeze one known-good control path and one failing custom path.
- Use one acceptance contract for all runs: structure validation + runtime canary + full output completion gate.
- Save all artifacts to comparable output folders.

2. Add First-Divergence Instrumentation
- Emit stage manifests for conversion and quantization outputs.
- Include tensor name, shape, type/format/datatype, data length, and bounded content signatures.
- Add quantizer decision tracing for LTX2.3 so each tensor codec choice is auditable.

3. Isolate First Divergence
- Run deterministic A/B comparisons between known-good and failing paths.
- Identify first stage with stable, reproducible divergence.
- Partition differences by tensor family and codec class.

4. Prove Causality
- Change only one suspect policy slice at a time.
- Re-run full acceptance contract.
- Keep only changes that produce predictable crash/pass behavior shifts.

5. Redesign Policy for Generic Models
- Replace blanket forced-q6p behavior with explicit adaptive rules for LTX2.3.
- Preserve or elevate precision where crash correlation is observed.
- Keep metadata alignment as a guardrail only where validated safe.

6. Productize the Pipeline
- Build one entrypoint workflow for custom LTX2.3 safetensors -> q6p.
- Embed manifests and acceptance gates by default.
- Validate on 10_e_v1 and at least one additional custom LTX2.3 model.

## Verification Gates
1. Stage manifests are deterministic and complete.
2. Comparator reports a stable first-divergence stage.
3. Minimal policy deltas produce predictable runtime behavior changes.
4. Both validation models complete full generation under final-mode acceptance.
5. Known-good distilled control remains healthy after policy changes.

## Initial Implementation Targets
- tools/run_q6p_canary_once.sh
- tools/run_10e_v1_model_validity_matrix.sh (reference)
- tools/new generic LTX2.3 validity matrix wrapper
- DRAW_THINGS_PATCH/Apps/ModelQuantizer/Quantizer.swift
- DRAW_THINGS_PATCH/Libraries/ModelOp/Sources/ModelImporter.swift

## Session Change Log
- 2026-06-09: Plan created as persistent root-level document.
- 2026-06-09: Added strict completion options to tools/run_q6p_canary_once.sh
	(--require-complete-stream, --require-final-output).
- 2026-06-09: Added tools/run_ltx23_model_validity_matrix.sh for generic
	custom LTX2.3 f16/q6p validation with full-completion canary gates.
- 2026-06-09: Launched Phase 1 baseline run with generic matrix.
	- Initial launch failed due stale filename assumptions.
	- Relaunch tag: phase1_retry_20260609_143001.
	- f16 validation and strict f16 canary passed.
	- q6p strict canary crashed with SIGSEGV (`ccv_nnc_tensor_read` -> `ccv_cnnp_model_read`).
- 2026-06-10: Added Phase 2 first-divergence instrumentation tooling.
	- `tools/dt_export_ckpt_tensor_manifest.py`
	- `tools/dt_compare_ckpt_tensor_manifests.py`
	- `tools/run_ltx23_first_divergence_stage.sh`
	- Smoke run (`phase2_smoke_20260610_073716`) passed with `first_divergent_stage=none` when baseline=candidate.
- 2026-06-10: Run 019 micro divergence localization (`phase2_run019_micro_20260610_074323`) reported `first_divergent_stage=q6p` with first mismatch at `__dit__[t-a2v_adaln_single_0-0-0]` on `metadata.type`.
- 2026-06-10: Added optional LTX quantizer decision trace output in `DRAW_THINGS_PATCH/Apps/ModelQuantizer/Quantizer.swift` via `--ltx-trace-output` JSONL.
- 2026-06-10: Run 020 full-row stage-localization (`phase2_run020_full_20260610`) completed.
	- Using available official-vs-custom baseline/candidate pair, summary reported `first_divergent_stage=f16` (payload/signature divergence on `__dit__[t-a2v_adaln_single_0-0-0]`).
	- q6p compare still added a distinct metadata/codec divergence signal (`metadata.type`, `codec_key`) with `mismatch_rows=5745`.
	- Derived set delta isolated q6p-only rows to 144 names, all in connector families:
		- `__text_audio_connector__`: 72
		- `__text_video_connector__`: 72
	- Next causal branch: regenerate custom q6p with `--ltx-trace-output` and test whether connector-policy decisions explain q6p-only mismatch rows and runtime behavior.
- 2026-06-10: Run 021 traced q6p regeneration (`10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt`) completed.
	- SQLite sanity passed (`quick_check=ok`, `tensors=5746`).
	- Quantizer decision trace captured 5746 rows (`output/quant_trace_run021_20260610/ltx_trace.jsonl`).
	- Strict q6p canary with complete-stream + final-output gates passed (`canary_rc=0`, `images written: 1`, no loader crash).
- 2026-06-10: Run 022 old-vs-new same-model q6p compare (`phase2_run021_old_vs_new_q6p_full_20260610`) completed.
	- f16 parity held exactly (`mismatch_rows=0`).
	- divergence remained strictly q6p-stage (`first_divergent_stage=q6p`).
	- q6p compare reported broad delta (`mismatch_rows=5745`, first field `metadata.type`).
- 2026-06-10: Run 022 reproducibility rerun (`phase2_run021_old_vs_new_q6p_20260610`) matched prior results exactly.
	- `first_divergent_stage=q6p`, `f16 mismatch_rows=0`, `q6p mismatch_rows=5745`.
- 2026-06-10: Run 023 strict matrix retry (`phase1_trace021_retry_20260610`) passed all gates.
	- f16 strict canary: pass.
	- traced q6p strict canary: pass with final image output.
	- matrix final result: `RESULT=PASS`.
- 2026-06-10: Added canary parameterization and strict stability matrix tooling.
	- `tools/run_q6p_canary_once.sh` now supports `--width`, `--height`, `--steps`, `--seed`.
	- Added `tools/run_q6p_strict_stability_matrix.sh` for repeatable multi-case strict canaries.
- 2026-06-10: Run 024 strict stability matrix (`run024_trace021_stability_20260610`) passed.
	- coverage: 3 seeds x 2 sizes (6 cases total)
	- all cases passed (`canary_rc=0`, `post_echo_rc=0`, final image output present)
	- Current causal branch: derive minimal policy slice from run021 trace/compare deltas, then productize traced-q6p generation as default path for custom LTX2.3.
- 2026-06-10: Run 025 second-model strict matrix (`run025_clipfix2_stability_20260610`) failed 6/6.
	- model: `ltx_2.3_22b_distilled_q6p_forcedfix_clipfix2_20260602.ckpt`
	- failure signature: `Illegal instruction` in `TextEncoder.encodeLTX2` (no streamed payloads).
- 2026-06-10: Run 026 final-mode single-canary checks on clipfix2 and official-1.1 q6p both failed.
	- clipfix2 and `ltx_2.3_22b_distilled_1.1_q6p.ckpt` reproduced the same `TextEncoder.encodeLTX2` illegal-instruction crash.
	- indicates failure is not explained by non-final-mode-only execution.
- 2026-06-10: Run 027 traced-key final-mode control failed after local alias experiment.
	- `10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt` hit loader-path crash (`ccv_nnc_tensor_read` -> `ccv_cnnp_model_read`).
- 2026-06-10: Run 028 tmp-key hardlink control passed.
	- same traced content under non-custom key (`..._tmpkey.ckpt`) passed strict final-mode canary.
	- isolates instability to model-key/custom-entry path rather than traced checkpoint content.
- 2026-06-10: Run 029 local alias variant (`file=traced`, `clip_encoder=old_q6p`) remained unstable.
	- produced stalled/no-stream behavior; short-timeout confirm run (`run029b...`) failed with timeout.
- 2026-06-10: Run 030 post-revert control passed.
	- after reverting local custom entry file mapping away from traced key, strict final-mode canary for traced key passed again.
	- Current causal branch: treat custom-entry/key-resolution path as independent failure surface; keep traced q6p validation on non-custom key path while deriving a safe custom.json alias schema.
- 2026-06-10: Run 031 controlled custom alias schema probe (`run031_alias_schema_probe_20260610`) completed.
	- Added `tools/run_custom_alias_schema_probe.sh` to sweep controlled entry variants and auto-restore `dt-models/custom.json`.
	- Probe matrix result: 5 cases total, pass=1, fail=4.
	- Only unmatched-file control passed (`file=10_e_v1_bf16_regen_0_q6p.ckpt`).
	- All tested variants where entry `file` matched traced key failed (loader-path crash/timeout or `TextEncoder.encodeLTX2` illegal instruction).
	- `default_scale` variation (1 vs 12) did not recover a pass when `file=traced`.
	- Current causal branch: custom-entry match on traced file key is a high-confidence trigger under tested schema space; next isolation should target other entry fields (`version`, `modifier`, `objective`, `text_encoder`, `autoencoder`) and model-browser resolution semantics.
- 2026-06-10: Run 032b extended-field custom alias schema probe (`run032b_alias_schema_extended_fields_20260610`) completed.
	- Extended `tools/run_custom_alias_schema_probe.sh` with `--matrix core|extended-fields` and per-case mutation support for `version`, `modifier`, `objective`, `text_encoder`, `autoencoder`.
	- Added periodic heartbeat output per case for unattended long runs.
	- Extended probe matrix result: 10 cases total, pass=1, fail=9.
	- Only unmatched-file control passed (same as run031).
	- All nine variants with entry `file=10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt` failed with timeout signature (`canary_rc=124`, no streamed payloads), including mutations of:
		- `modifier` (`none`, `kontext_kv`)
		- `version` (`ltx2`)
		- `objective` (set `u.condition_scale=1`, removed)
		- `text_encoder` (removed)
		- `autoencoder` (removed)
	- Refined causal branch: in tested custom schema space, the file-key match/resolution axis remains the dominant trigger; secondary entry-field mutations above are not sufficient to restore stability.
	- Next isolation target: model-resolution semantics and mapping collisions around `specificationForModel(file)` keying (custom entry shadowing by `file`) rather than additional per-entry parameter tweaks.
- 2026-06-11: Run 033b alias-resolution semantics probe rerun (`run033b_alias_resolution_probe_20260611`) completed.
	- Executed `tools/run_custom_alias_resolution_probe.sh` to test model argument path (file key vs alias), duplicate alias ordering, and tmpkey hardlink behavior under strict final-mode canary.
	- Matrix result: 9 cases total, pass=2, fail=7.
	- Passing controls:
		- traced key with no probe aliases (`control_trace_noncustom`)
		- tmpkey hardlink with no probe aliases (`control_tmpkey_noncustom`)
	- Failing probes:
		- all scenarios that introduced custom alias entries for traced or tmpkey file content failed (mostly `loader_crash`, one `timeout`), independent of request via file key or alias name.
		- duplicate alias order (`ab` vs `ba`) did not change failure outcome.
	- Refined causal branch: failure is strongly coupled to custom-entry resolution/shadowing semantics for model file lookup, not to traced tensor payload validity and not to key-string identity alone.
	- Next isolation target: instrument and inspect `specificationForModel(file)`/resolution call path behavior under duplicate and probe alias entries in ModelZoo-facing code paths.
- 2026-06-11: Run 034 cross-file alias-resolution matrix (`run034_alias_resolution_crossfile_20260611`) completed.
	- Extended `tools/run_custom_alias_resolution_probe.sh` with `--matrix core|cross-file` and `alias_both` mode to test simultaneous probe aliases.
	- Matrix result: 9 cases total, pass=5, fail=4.
	- Both baseline controls passed (trace key and tmpkey).
	- Cross-file single-alias probes passed (alias for one key active while requesting the other key).
	- Failures occurred only when both probe aliases were active simultaneously (`alias_both`), with loader-crash signature and `canary_rc/post_echo_rc=124`.
	- Refined causal branch: trigger is not global alias contamination from one key; failure emerges under multi-entry shadowing/collision conditions, pointing to lookup/selection behavior when overlapping custom entries are active together.
	- Next isolation target: add source-level instrumentation around ModelZoo resolution map construction/lookup order (`availableSpecifications` -> `specificationMapping[file]`) and validate winner selection under overlapping alias sets.
- 2026-06-11: Run 035 core alias-resolution matrix with winner-context instrumentation (`run035_alias_resolution_core_ctx_20260611`) completed.
	- Added resolution-context export in `tools/run_custom_alias_resolution_probe.sh` (`arg_source`, resolved file, per-key match counts, winner name/modifier).
	- Matrix result: 9 cases total, pass=2, fail=7 (same pass/fail shape as run033b).
	- Passing controls had `arg_match_count=0` (no custom winner for requested file key).
	- Every failing case had `arg_match_count>=1` with an active winner entry for the resolved file key.
	- Duplicate alias ordering changed winner identity (`ab` vs `ba`) but did not change fail outcome.
	- Refined causal branch: winner identity differences are secondary; the dominant trigger is presence of an active custom winner over the requested file key path.
- 2026-06-11: Run 036 cross-file alias-resolution matrix with winner-context instrumentation (`run036_alias_resolution_crossfile_ctx_20260611`) completed.
	- Matrix result: 9 cases total, pass=5, fail=4 (reproduces run034).
	- Cross-file single-alias requests passed while requested key still had `arg_match_count=0` (no active winner on requested key), even when the other key had active matches.
	- All `alias_both` failures had simultaneous overlap (`trace_match_count=1` and `tmpkey_match_count=1`) plus active arg winner.
	- In `alias_both`, file-key requests failed as timeouts while alias-name requests failed as loader-crash; both remain terminal FAIL signatures.
	- Next isolation target (unchanged, now stronger): instrument `ModelZoo.specificationForModel(file)` / `specificationMapping[file]` map construction and winner replacement behavior under overlapping entries to identify first control-flow divergence.
- 2026-06-11: Run 037 minimal-v1 custom-winner isolation matrix (`run037_alias_resolution_minv1_ctx_20260611`) completed.
	- Extended `tools/run_custom_alias_resolution_probe.sh` with `--matrix minimal-v1` and minimal-entry modes (`alias_trace_min_v1`, `alias_tmpkey_min_v1`, `alias_both_min_v1`).
	- Minimal entry schema intentionally removed LTX-specific fields and kept only required/default-like keys: `name`, `file`, `prefix=""`, `version="v1"`, `upcast_attention=false`, `default_scale=8`.
	- Matrix result: 8 cases total, pass=8, fail=0.
	- Crucial discriminator: even with active winners (`arg_match_count=1`) and simultaneous overlap (`trace_match_count=1`, `tmpkey_match_count=1`), all cases stayed PASS.
	- Revised causal branch: custom winner/overlap state alone is insufficient; failure requires one or more LTX-style custom-entry fields from earlier failing modes (most likely among `version=ltx2.3`, `clip_encoder=file`, `modifier`, `default_scale`, and associated LTX config fields).
	- Next isolation target: run additive field-ladder from minimal-v1 baseline to prior failing alias schema to identify first failing field transition.
- 2026-06-11: Run 038 additive field-ladder matrix (`run038_alias_resolution_field_ladder_20260611`) completed.
	- Extended `tools/run_custom_alias_resolution_probe.sh` with `--matrix field-ladder` and additive trace-entry modes.
	- Matrix result: 7 cases total, pass=2, fail=5.
	- First-fail transition identified: switching only `version` from `v1` to `ltx2.3` (`alias_trace_ladder_ltx23_min`) already fails with `textencoder_illegal`.
	- Subsequent additions (`modifier=kontext`, `default_scale=1`) remained `textencoder_illegal`; adding `clip_encoder=file` shifted signature to timeout, matching full-base alias behavior.
	- Updated causal branch: dominant trigger is LTX2.3 custom-version path activation itself; clip/default-scale/modifier modulate signature severity but are not required for failure onset.
	- Next isolation target: instrument or emulate the LTX2.3 version-driven text-encoder selection path under custom entries (especially TextEncoder.encodeLTX2 file list construction and fallback rules).
- 2026-06-11: Run 039 LTX2.3 encoder-path variant matrix (`run039_alias_resolution_ltx23_encoders_20260611`) completed.
	- Extended `tools/run_custom_alias_resolution_probe.sh` with `--matrix ltx23-encoders` and ltx2.3-fixed variant modes (`min_text`, `min_auto`, `min_text_auto`, `min_clip`).
	- Matrix result: 7 cases total, pass=1, fail=6.
	- Control remained PASS only without custom ltx2.3 entry.
	- All ltx2.3 custom variants failed; signature split observed:
		- `textencoder_illegal`: minimal ltx2.3, ltx2.3+autoencoder
		- `timeout`: ltx2.3+text_encoder, ltx2.3+text_encoder+autoencoder, ltx2.3+clip_encoder, full-base
	- Refined branch: ltx2.3 custom-version path is necessary and sufficient for failure onset in this harness; encoder-related fields change failure manifestation but not pass/fail outcome.
	- Next isolation target: source-level instrumentation around LTX2.3 text-encoder file-path assembly and encoder loading in LocalImageGenerator/TextEncoder to explain illegal-instruction vs timeout bifurcation.
- 2026-06-11: Run 040 LTX2.3 encoder matrix with resolved file-list context (`run040_alias_resolution_ltx23_encoders_ctx2_20260611`) completed.
	- Extended probe context columns to emit resolved encoder file list and ltx2.3 safety flag (`arg_text_files_count`, `arg_text_file0`, `arg_text_file1`, `arg_ltx23_textfiles_ok`).
	- Matrix result: 7 cases total, pass=1, fail=6 (same outcome shape as run039).
	- Strong source-level correlation established:
		- all immediate `textencoder_illegal` cases had `arg_winner_version=ltx2.3`, `arg_text_files_count=1`, `arg_ltx23_textfiles_ok=0`
		- server backtraces point to `TextEncoder.encodeLTX2(...)+12255`
	- Cases with `arg_text_files_count=2` (`arg_ltx23_textfiles_ok=1`) no longer showed immediate illegal-instruction, instead failing later via timeout/loader-crash.
	- Updated causal branch: missing second text-encoder file in ltx2.3 custom specs is a high-confidence immediate-crash trigger; downstream timeout/loader-crash branch remains after satisfying this precondition.
	- Next isolation target: construct a controlled two-file ltx2.3 custom entry that avoids clip-in-model coupling path (or uses known-good companion file) to separate post-precondition loader failures from core path validity.
- 2026-06-11: Run 041 controlled two-file companion matrix (`run041_ltx23_companion`) completed.
	- Extended probe harness with `--matrix ltx23-companion` and configurable companion clip candidates.
	- Matrix result: 7 cases total, pass=1, fail=6.
	- One-file ltx2.3 custom case remained immediate `textencoder_illegal` (`canary_rc=1`, `TextEncoder.encodeLTX2`).
	- Two-file ltx2.3 cases avoided immediate illegal-instruction but bifurcated into timeout (`canary_rc=124`, `post_echo_rc=0`) and loader-crash (`canary_rc=124`, `post_echo_rc=124`, `ccv_nnc_tensor_read`) depending on companion clip choice.
	- Updated causal branch: second-file precondition is necessary to avoid immediate encode crash but not sufficient for end-to-end pass; a downstream load/inference branch remains unresolved.
	- Next isolation target: add source-level instrumentation in ltx2.3 load path to capture first divergence after two-file list construction (file ordering, model-read path, and connector loading interactions).
- 2026-06-11: Run 042 ltx2.3 ordering matrix (`run042_ltx23_order`) completed.
	- Extended probe harness with `--matrix ltx23-order` and explicit text/clip swap modes for companion files.
	- Matrix result: 8 cases total, pass=1, fail=7.
	- Under current local base-entry state, all failing variants converged to timeout (`canary_rc=124`, `post_echo_rc=0`), including one-file ltx2.3 case.
	- Drift note: local `10_e_v1` now resolves `text_encoder=gemma_3_12b_it_qat_q8p.ckpt`; one-file case did not reproduce prior immediate `TextEncoder.encodeLTX2` illegal-instruction.
	- Updated causal branch: companion ordering alone did not explain downstream split in this state; active text-encoder source/value is a strong state-sensitive confound.
	- Next isolation target: pin text-encoder field explicitly in probe modes (legacy clip-vs-gemma controls) and rerun one-file/two-file matrix to separate true ordering effects from base-entry drift.
