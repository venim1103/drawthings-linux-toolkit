# Q6P Runtime Investigation Handoff (2026-06-08)

This document captures the current state of the `10_e_v1` q6p remediation effort so work can resume quickly without re-deriving context.

## 1) Current operating constraints

- Disk headroom is limited; avoid new 20G+ full checkpoint copies.
- Preferred workflow is in-place mutation on the existing target q6p.
- Official baseline `dt-models/ltx_2.3_22b_distilled_1.1_q6p.ckpt` was intentionally deleted to save space.
- `dt-models/ltx_2.3_22b_distilled_q6p_forcedfix_clipfix2_20260602.ckpt` can be used as fallback comparison baseline when needed.

## 2) File state snapshot (2026-06-08)

- Present:
  - `dt-models/10_e_v1_bf16_regen_0_f16.ckpt` (~43G)
  - `dt-models/10_e_v1_bf16_regen_0_q6p.ckpt` (~22G)
  - `dt-models/ltx_2.3_22b_distilled_q6p_forcedfix_clipfix2_20260602.ckpt` (~20G)
- Missing:
  - `dt-models/ltx_2.3_22b_distilled_1.1_q6p.ckpt`

## 3) Reusable automation now available

- `tools/run_q6p_inplace_dimfix_from_f16.sh`
  - End-to-end in-place patch flow: dim/data subset copy + metadata subset copy + structural validate + bounded canary.
- `tools/dt_build_q6p_dimfix_names.py`
  - Builds candidate tensor names from mismatch topology.
  - Supports selector modes (`meta_and_extra_dim`, `meta_and_dim`, `extra_dim_only`, `dim_only`, `meta_only`).
- `tools/dt_patch_ckpt_metadata_subset.py`
  - Copies `type/format/datatype` for selected tensor names.
- `tools/dt_probe_ckpt_meta_len_rowwise.py`
  - Row-wise safe metadata/length probe that avoids bulk blob reads and emits mismatch-name sets.
- `tools/patch_sets/10_e_v1_q6p_dimfix770_20260608.txt`
  - Precomputed 770-row selection (no baseline file required).
- `tools/run_q6p_canary_once.sh`
  - Single bounded canary runner for post-patch runtime checks.

## 4) Confirmed findings (high signal)

1. Structural parity by itself is not sufficient.
   - Multiple rounds reached healthy SQLite checks (`PRAGMA quick_check=ok`, expected tensor counts) while runtime still crashed.
2. Metadata parity (`type/format/datatype`) and broad data-length parity were previously achieved but did not remove the loader crash.
3. Targeted content substitutions (non-dit + selected dit sets) also failed to remove the crash.
4. Readable-row order alignment to baseline was achieved in earlier work and still did not remove the crash.
5. Official control q6p has repeatedly worked under equivalent bounded profiles, localizing failure to custom artifact content/path rather than universal q6p runtime failure.
6. 2026-06-08 low-space in-place dimfix was implemented and executed:
   - Selected rows patched in place: `770`
   - Structural validation: pass
   - Runtime canary: fail (timeout followed by server crash)
7. Expanded candidate attempt (2026-06-08):
  - clipfix2-based `extra_dim_only` set: `2756`
  - staged apply set (`770 + extra 512`): `1282`
  - dim/data + metadata updates: pass
  - no first streamed response observed
8. Post-expanded direct canary still crashed with the same loader stack head:
  - `ccv_nnc_tensor_read`
  - `ccv_cnnp_model_read`
9. Full extra-dim branch (2756 rows, 2026-06-08 10:39) also failed:
  - dim/data apply pass (`rows_updated=2756`, selected post mismatch zero)
  - metadata apply pass (`metadata_rows_updated=2756`)
  - structural validation pass (`tensor_count=5746`, `ltx2_3` pass)
  - bounded canary still ended in kill/hang before first streamed response
10. Post-Run-004 row-wise meta/len probe + micro-batched metadata branch (2026-06-08 12:25) reached global metadata parity:
  - `metadata_mismatch_type=0`, `metadata_mismatch_format=0`, `metadata_mismatch_datatype=0`
  - `dim_len_mismatch=0` retained
  - residual `data_len_mismatch=2728` remained across readable shared tensors
  - single large metadata apply repeatedly entered D-state / exit 137; chunking to 900/300/100/25 was required
  - post-fix bounded canary timed out (`canary_rc=124`) with no streamed response

## 5) Latest run evidence (2026-06-08)

Run artifacts:

- `output/q6p_inplace_dimfix_20260608_080006/client.log`
- `output/q6p_inplace_dimfix_20260608_080006/server.log`

Observed behavior:

- Client emitted request start and then no streamed responses before timeout.
- Server crashed with `SIGSEGV` / bad pointer dereference.
- Backtrace head remained:
  - `ccv_nnc_tensor_read`
  - `ccv_cnnp_model_read`

Interpretation:

- In-place 770-row dequantization+metadata restore does not resolve the runtime loader fault.
- Remaining blocker is likely a deeper serialization/encoding invariant not covered by this subset.

Additional evidence from expanded branch:

- `output/q6p_inplace_dimfix_20260608_clipfix2_plus_extra512/client.log`
- `output/q6p_inplace_dimfix_20260608_clipfix2_plus_extra512/server.log`
- `output/q6p_canary_20260608_post_extra512_canary/client.log`
- `output/q6p_canary_20260608_post_extra512_canary/server.log`
- `output/q6p_inplace_dimfix_20260608_clipfix2_extra2756/client.log`
- `output/q6p_inplace_dimfix_20260608_clipfix2_extra2756/server.log`
- `output/probe_meta_len_all5746_post_metachunkfix_20260608.json`
- `output/probe_meta_len_all5746_post_metachunkfix_20260608.tsv`
- `output/q6p_canary_20260608_post_metachunkfix/client.log`
- `output/q6p_canary_20260608_post_metachunkfix/server.log`

Observed behavior on expanded branch:

- 1282-row in-place apply finished with structural checks passing.
- In-place-run canary did not reach first response and required manual cleanup.
- Direct post-patch canary reproduced `SIGSEGV` with unchanged backtrace head (`ccv_nnc_tensor_read -> ccv_cnnp_model_read`).
- Metadata-only parity branch required chunked mutation to avoid D-state/kill during apply.
- After metadata parity reached zero mismatches, canary still produced no `response #1` and timed out at 120s.
- In post-metachunkfix canary logs, server reached request begin and later responded to post-echo, but no streamed generation payload arrived.

## 6) Working baseline strategy going forward

- Primary low-space path:
  - keep using precomputed names file (`tools/patch_sets/10_e_v1_q6p_dimfix770_20260608.txt`) to avoid baseline dependency.
- Fallback comparison baseline:
  - use `dt-models/ltx_2.3_22b_distilled_q6p_forcedfix_clipfix2_20260602.ckpt` as `--baseline-q6p` when rebuilding names.
- Heuristic caution:
  - if clipfix2 is used for both baseline and reference in the name-builder formula, candidate selection becomes broader (`meta_mismatch intersect dim_mismatch`) because the extra-dim subtraction term collapses.

## 7) Open questions / next branches

1. Residual payload branch:
  - Apply source-f16 content to the residual `data_len_mismatch=2728` set using strict micro-batching (small windows, immediate checks).
2. Stall isolation:
  - Identify whether specific name windows consistently trigger D-state during content-copy apply and quarantine those names for separate handling.
3. Crash localization granularity:
  - Determine whether failure remains pre-`response #1` after each residual batch window.
4. Baseline restoration decision:
  - Re-download official q6p only if needed for stronger comparator fidelity and if storage budget allows.

## 8) Quick continuation pointer

- Command runbook: `Q6P_CONTINUATION_RUNBOOK_2026-06-08.md`
- Existing long-form history: `CONVERSION_TOOL_FINDINGS_2026-05-28.md`

## 9) Run 006 update (2026-06-08 13:21 UTC)

Objective:

- Eliminate the residual `data_len_mismatch=2728` branch from Run 005 using strict recursive micro-batching, then re-measure and canary.

Execution summary:

- Run-006 payload remediation traversed recursive splits down to leaf windows under repeated D-state/exit-137 pressure.
- Initial run completion reported one irreducible leaf in:
  - `output/20260608_run006_payloadfix/failed_leaf_names.txt`
  - name: `__dit__[t-x_out_proj-17-0]`
- Immediate one-name retry (`chunk-size=1`) succeeded:
  - `rows_updated=1`
  - `post_data_head_mismatch=0`
  - `RESULT=PASS`

Post-retry row-wise verification:

- Probe artifact:
  - `output/20260608_run006_payloadfix/probe_meta_len_rowwise_after_leafretry_20260608.json`
- Measured state:
  - `metadata_mismatch_type=0`
  - `metadata_mismatch_format=0`
  - `metadata_mismatch_datatype=0`
  - `dim_len_mismatch=0`
  - `data_len_mismatch=0`
  - `mismatch_any=0`
  - `full_match=5745`
  - `unreadable_both=1`
- Mismatch-name list is empty (`0` lines):
  - `output/20260608_run006_payloadfix/probe_mismatch_names_after_leafretry_20260608.txt`

Post-parity canary:

- Command tag: `20260608_run006_post_leafretry`
- Artifacts:
  - `output/q6p_canary_20260608_run006_post_leafretry/client.log`
  - `output/q6p_canary_20260608_run006_post_leafretry/server.log`
- Outcome:
  - `canary_rc=124`
  - `post_echo_rc=0`
  - no `response #1` observed
  - server log shows request-begin and post-echo, without the earlier SIGSEGV stack in this run folder

Interpretation:

- Metadata parity and row-wise length parity are now fully exhausted as explanatory causes on readable shared tensors.
- Runtime failure mode persists as pre-stream stall/timeout even at `mismatch_any=0` by row-wise metadata/length criteria.
- Next iterations should target byte-level payload semantics for equal-length rows (family-level signatures/content probes) rather than additional meta/len remediation.
