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

## 5) Suggested next branch (when resumed)

- Keep final-mode enabled for long validations.
- Pivot from timeout tuning to crash-localization around tensor read path:
  - isolate whether crash is tied to a specific serialization pattern beyond current row-level content substitutions
  - focus on reproducible, script-first probes near model-load read path assumptions
