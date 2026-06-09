# Q6P Root-Cause Plan for Custom LTX2.3

Last updated: 2026-06-09
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
	- f16 validation and strict f16 canary passed; q6p strict canary running.
