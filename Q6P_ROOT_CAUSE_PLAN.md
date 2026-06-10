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
