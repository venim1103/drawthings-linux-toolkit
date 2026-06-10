# Q6P Status Update (2026-06-09)

This handoff summarizes the latest state after Runs 015-017 and timeout-policy updates.

## 1) What changed today

- Added final-mode timeout policy to canary flows:
  - default final validation timeout = 900s (15 min)
  - optional no-timeout mode for long validation
- Updated scripts:
  - `tools/run_q6p_canary_once.sh`
  - `tools/run_q6p_payloadfix_recursive.sh`
  - `tools/run_q6p_window_sig_payloadfix.sh`
  - `tools/run_q6p_highlimit_row_patch_canary.sh`
  - `tools/run_q6p_equal_len_payloadfix.sh`
- Added new script for deterministic family patching:
  - `tools/run_q6p_highlimit_rows_patch_canary.sh`

## 2) Latest run outcomes

- Run 015 (`20260609_run015_dit_sig_hash64k_batch1`):
  - DIT probe branch was manually interrupted (exit 130).
  - Processed chunks up to 1801-2100 with no mismatches so far.
  - Result not treated as complete coverage.

- Run 016 (`20260609_run016_highlimit_textfeat_leaf_finalmode`):
  - high-limit patch on `__text_feature_extractor__[t-video_aggregate_embed-0-0]` succeeded (`row_patch_changes=1`, quick_check ok)
  - final-mode canary still failed by SIGSEGV
  - stack head unchanged: `ccv_nnc_tensor_read` -> `ccv_cnnp_model_read`

- Run 017 (`20260609_run017_highlimit_textfeat_family_finalmode`):
  - high-limit patch applied to all 4 `__text_feature_extractor__` rows
  - all row updates reported changes=1; quick_check ok
  - final-mode canary still failed by SIGSEGV with same stack head

- Run 018 (`phase1_retry_20260609_143001`):
  - first full execution of generic strict matrix (`tools/run_ltx23_model_validity_matrix.sh`)
  - f16 path passed all gates:
    - SQLite sanity ok
    - profile-resolved LTX2.3 validation passed
    - strict completion canary passed with final image payload
  - q6p path failed strict canary:
    - client: `UNAVAILABLE: Socket closed`
    - server: deterministic SIGSEGV
      - `ccv_nnc_tensor_read`
      - `ccv_cnnp_model_read`
  - confirms baseline remains crash-driven and reproducible under strict completion criteria

- Run 019 (`phase2_run019_micro_20260610_074323`):
  - first deterministic phase-2 stage-localization run using manifest export + compare tooling
  - f16 stage baseline/candidate parity held on sampled window (`mismatch_rows=0`)
  - q6p stage diverged immediately (`mismatch_rows=128` on first 128 rows)
  - first q6p divergence:
    - tensor: `__dit__[t-a2v_adaln_single_0-0-0]`
    - field: `metadata.type`
    - baseline=`258`, candidate=`38877129135357953`
  - reported `first_divergent_stage=q6p`

## 3) Current conclusion

- Timeout policy issue is solved for final validation (900s available and wired through wrappers).
- Current blocker is not timeout-only behavior.
- Current blocker is deterministic server crash during model load/request handling:
  - `ccv_nnc_tensor_read`
  - `ccv_cnnp_model_read`

## 4) Key artifacts

- Main run history:
  - `Q6P_RUN_LOG.md`
- Run 016 canary logs:
  - `output/q6p_canary_20260609_run016_highlimit_textfeat_leaf_finalmode_canary/client.log`
  - `output/q6p_canary_20260609_run016_highlimit_textfeat_leaf_finalmode_canary/server.log`
- Run 017 patch/canary logs:
  - `output/20260609_run017_highlimit_textfeat_family_finalmode/highlimit_patch.log`
  - `output/20260609_run017_highlimit_textfeat_family_finalmode/patch_rows.sql`
  - `output/20260609_run017_highlimit_textfeat_family_finalmode/rows_norm.txt`
  - `output/q6p_canary_20260609_run017_highlimit_textfeat_family_finalmode_canary/client.log`
  - `output/q6p_canary_20260609_run017_highlimit_textfeat_family_finalmode_canary/server.log`
- Run 018 strict matrix artifacts:
  - `output/model_validity_ltx23_phase1_retry_20260609_143001/summary.md`
  - `output/model_validity_ltx23_phase1_retry_20260609_143001/f16_validator.log`
  - `output/model_validity_ltx23_phase1_retry_20260609_143001/f16_canary.log`
  - `output/model_validity_ltx23_phase1_retry_20260609_143001/q6p_canary.log`
  - `output/q6p_canary_phase1_retry_20260609_143001_q6p/client.log`
  - `output/q6p_canary_phase1_retry_20260609_143001_q6p/server.log`
- Run 019 phase-2 micro artifacts:
  - `output/first_divergence_phase2_run019_micro_20260610_074323/summary.json`
  - `output/first_divergence_phase2_run019_micro_20260610_074323/summary.md`
  - `output/first_divergence_phase2_run019_micro_20260610_074323/compare_f16.json`
  - `output/first_divergence_phase2_run019_micro_20260610_074323/compare_q6p.json`
  - `output/first_divergence_phase2_run019_micro_20260610_074323/compare_q6p_mismatch_names.txt`

## 5) Suggested next branch (when resumed)

- Keep final-mode enabled for long validations.
- Pivot from timeout tuning to crash-localization around tensor read path:
  - isolate whether crash is tied to a specific serialization pattern beyond current row-level content substitutions
  - focus on reproducible, script-first probes near model-load read path assumptions

## 6) Phase-2 instrumentation added (2026-06-10)

- Added deterministic tensor manifest exporter:
  - `tools/dt_export_ckpt_tensor_manifest.py`
- Added deterministic manifest comparator with first-divergence reporting:
  - `tools/dt_compare_ckpt_tensor_manifests.py`
- Added staged first-divergence runner for baseline vs candidate paths:
  - `tools/run_ltx23_first_divergence_stage.sh`
- Smoke validation:
  - run tag `phase2_smoke_20260610_073716`
  - baseline=candidate at both stages correctly produced `first_divergent_stage=none`
