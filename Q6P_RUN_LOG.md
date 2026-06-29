# Q6P Iteration Run Log (Append-Only)

Purpose:

- Keep a compact, reproducible record of each remediation attempt.
- Avoid losing context between sessions.
- Make it easy to compare candidate sets, canary behavior, and crash signatures.

Rules:

- Append only. Do not rewrite previous entries.
- One entry per run attempt.
- Always include command, names file, and artifact paths.
- Keep interpretation short and actionable.

## Entry Template

Copy this block and append it at the bottom for each run:

```md
## Run NNN - YYYY-MM-DD HH:MM (UTC)

- Goal:
- Target q6p:
- Source f16:
- Baseline q6p:
- Reference q6p:
- Names file:
- Names count:
- Command:

```bash
# paste exact command here
```

- Structural checks:
  - quick_check:
  - tensors count:
  - validator profile:
- Canary:
  - timeout sec:
  - max responses:
  - canary rc:
  - post echo rc:
  - saw response #1:
- Crash signature:
  - stack head:
- Artifacts:
  - work dir:
  - client log:
  - server log:
- Outcome:
- Next branch:
```

## Run 001 - 2026-06-08 08:00 (UTC)

- Goal: Low-space in-place dimfix using precomputed 770-row set.
- Target q6p: `dt-models/10_e_v1_bf16_regen_0_q6p.ckpt`
- Source f16: `dt-models/10_e_v1_bf16_regen_0_f16.ckpt`
- Baseline q6p: `dt-models/ltx_2.3_22b_distilled_1.1_q6p.ckpt` (missing at present)
- Reference q6p: `dt-models/ltx_2.3_22b_distilled_q6p_forcedfix_clipfix2_20260602.ckpt`
- Names file: `tools/patch_sets/10_e_v1_q6p_dimfix770_20260608.txt`
- Names count: `770`
- Command:

```bash
bash tools/run_q6p_inplace_dimfix_from_f16.sh --canary-timeout-sec 120 --max-responses 10
```

- Structural checks:
  - quick_check: pass (via workflow)
  - tensors count: 5746
  - validator profile: ltx2_3 pass
- Canary:
  - timeout sec: 120
  - max responses: 10
  - canary rc: 124
  - post echo rc: 1
  - saw response #1: no
- Crash signature:
  - stack head: ccv_nnc_tensor_read -> ccv_cnnp_model_read
- Artifacts:
  - work dir: output/q6p_inplace_dimfix_20260608_080006
  - client log: output/q6p_inplace_dimfix_20260608_080006/client.log
  - server log: output/q6p_inplace_dimfix_20260608_080006/server.log
- Outcome: In-place 770-row patch insufficient.
- Next branch: Rebuild candidate set with clipfix2 fallback baseline and retest under bounded canary.

## Run 002 - 2026-06-08 09:26 (UTC)

- Goal: Expand patch scope beyond original 770 rows without full-copy artifacts.
- Target q6p: `dt-models/10_e_v1_bf16_regen_0_q6p.ckpt`
- Source f16: `dt-models/10_e_v1_bf16_regen_0_f16.ckpt`
- Baseline q6p: `dt-models/ltx_2.3_22b_distilled_q6p_forcedfix_clipfix2_20260602.ckpt`
- Reference q6p: `dt-models/ltx_2.3_22b_distilled_q6p_forcedfix_clipfix2_20260602.ckpt`
- Names file: `tools/patch_sets/10_e_v1_q6p_dimfix770_plus_extra512_clipfix2base_20260608.txt`
- Names count: `1282`
- Command:

```bash
bash tools/run_q6p_inplace_dimfix_from_f16.sh \
  --names-file tools/patch_sets/10_e_v1_q6p_dimfix770_plus_extra512_clipfix2base_20260608.txt \
  --tag 20260608_clipfix2_plus_extra512 \
  --canary-timeout-sec 120 \
  --max-responses 10 \
  --min-free-gb 50
```

- Structural checks:
  - quick_check: pass (via workflow)
  - tensors count: 5746
  - validator profile: ltx2_3 pass
- Canary:
  - timeout sec: 120
  - max responses: 10
  - canary rc: terminal exited 137 after manual cleanup of hung run
  - post echo rc: not captured by script due kill
  - saw response #1: no
- Crash signature:
  - stack head: none captured in this run folder (server log stops at request begin)
- Artifacts:
  - work dir: output/q6p_inplace_dimfix_20260608_clipfix2_plus_extra512
  - client log: output/q6p_inplace_dimfix_20260608_clipfix2_plus_extra512/client.log
  - server log: output/q6p_inplace_dimfix_20260608_clipfix2_plus_extra512/server.log
- Outcome: Expanded 1282-row in-place patch still did not reach first streamed response.
- Next branch: Run post-patch canary directly and capture crash stack outside the long patch run.

## Run 003 - 2026-06-08 09:31 (UTC)

- Goal: Canary-only retest after Run 002 expanded patch.
- Target q6p: `dt-models/10_e_v1_bf16_regen_0_q6p.ckpt`
- Source f16: n/a (canary-only)
- Baseline q6p: n/a (canary-only)
- Reference q6p: n/a (canary-only)
- Names file: n/a
- Names count: n/a
- Command:

```bash
bash tools/run_q6p_canary_once.sh \
  --model 10_e_v1_bf16_regen_0_q6p.ckpt \
  --timeout-sec 90 \
  --max-responses 10 \
  --tag 20260608_post_extra512_canary
```

- Structural checks:
  - quick_check: not part of canary-only command
  - tensors count: not part of canary-only command
  - validator profile: not part of canary-only command
- Canary:
  - timeout sec: 90
  - max responses: 10
  - canary rc: terminal exit observed as 137
  - post echo rc: unavailable due crash/termination
  - saw response #1: no
- Crash signature:
  - stack head: ccv_nnc_tensor_read -> ccv_cnnp_model_read
- Artifacts:
  - work dir: output/q6p_canary_20260608_post_extra512_canary
  - client log: output/q6p_canary_20260608_post_extra512_canary/client.log
  - server log: output/q6p_canary_20260608_post_extra512_canary/server.log
- Outcome: Crash signature unchanged after expanded 1282-row patch.
- Next branch: Move selection strategy away from current dim-head heuristics; target deeper serialization invariant differences.

## Run 004 - 2026-06-08 10:39 (UTC)

- Goal: Full extra-dim patch branch (all 2756 rows) under low-space in-place flow.
- Target q6p: `dt-models/10_e_v1_bf16_regen_0_q6p.ckpt`
- Source f16: `dt-models/10_e_v1_bf16_regen_0_f16.ckpt`
- Baseline q6p: `dt-models/ltx_2.3_22b_distilled_q6p_forcedfix_clipfix2_20260602.ckpt`
- Reference q6p: `dt-models/ltx_2.3_22b_distilled_q6p_forcedfix_clipfix2_20260602.ckpt`
- Names file: `tools/patch_sets/10_e_v1_q6p_dimfix_extra_dim_2756_clipfix2base_20260608.txt`
- Names count: `2756`
- Command:

```bash
bash tools/run_q6p_inplace_dimfix_from_f16.sh \
  --names-file tools/patch_sets/10_e_v1_q6p_dimfix_extra_dim_2756_clipfix2base_20260608.txt \
  --tag 20260608_clipfix2_extra2756 \
  --canary-timeout-sec 120 \
  --max-responses 10 \
  --min-free-gb 50
```

- Structural checks:
  - quick_check: pass (via workflow)
  - tensors count: 5746
  - validator profile: ltx2_3 pass
- Canary:
  - timeout sec: 120
  - max responses: 10
  - canary rc: terminal ended with `Killed` / exit 137 notification
  - post echo rc: not emitted in run output before kill
  - saw response #1: no
- Crash signature:
  - stack head: not printed in this run folder logs (server log stops at request begin)
- Artifacts:
  - work dir: output/q6p_inplace_dimfix_20260608_clipfix2_extra2756
  - client log: output/q6p_inplace_dimfix_20260608_clipfix2_extra2756/client.log
  - server log: output/q6p_inplace_dimfix_20260608_clipfix2_extra2756/server.log
- Outcome: Full 2756-row patch still failed to produce first streamed response.
- Next branch: Run a canary with explicit stack capture immediately after patch and/or pivot to serialization-invariant probes beyond current dim/data subset heuristics.

## Run 005 - 2026-06-08 12:25 (UTC)

- Goal: Probe residual mismatches after Run 004 and apply metadata-only parity fix in micro-batches.
- Target q6p: `dt-models/10_e_v1_bf16_regen_0_q6p.ckpt`
- Source f16: `dt-models/10_e_v1_bf16_regen_0_f16.ckpt`
- Baseline q6p: n/a (source-f16 comparison branch)
- Reference q6p: n/a (source-f16 comparison branch)
- Names file: `tools/patch_sets/10_e_v1_q6p_run005_mismatch_meta_len_all5746_20260608.txt`
- Names count: `2989`
- Command:

```bash
# 1) Build residual mismatch names with row-wise safe probe
/workspaces/drawthings-linux-toolkit/.venv/bin/python tools/dt_probe_ckpt_meta_len_rowwise.py \
  --file dt-models/10_e_v1_bf16_regen_0_q6p.ckpt \
  --baseline dt-models/10_e_v1_bf16_regen_0_f16.ckpt \
  --out-mismatch-names tools/patch_sets/10_e_v1_q6p_run005_mismatch_meta_len_all5746_20260608.txt

# 2) Apply metadata in chunks (single large pass repeatedly entered D-state/exit 137)
split -d -l 900 tools/patch_sets/10_e_v1_q6p_run005_mismatch_meta_len_all5746_20260608.txt tools/patch_sets/run005_meta_chunks_20260608/chunk_
# ...then sub-split failing chunks to 300/100/25 and apply with tools/dt_patch_ckpt_metadata_subset.py

# 3) Post-fix canary
bash tools/run_q6p_canary_once.sh \
  --model 10_e_v1_bf16_regen_0_q6p.ckpt \
  --timeout-sec 120 \
  --max-responses 10 \
  --tag 20260608_post_metachunkfix
```

- Structural checks:
  - quick_check: not run in this branch
  - tensors count: 5746 (from row-wise probe)
  - validator profile: not run in this branch
- Canary:
  - timeout sec: 120
  - max responses: 10
  - canary rc: 124
  - post echo rc: 0
  - saw response #1: no
- Crash signature:
  - stack head: none printed in server log for this run; server stayed alive through post-echo
- Artifacts:
  - work dir: output/q6p_canary_20260608_post_metachunkfix
  - client log: output/q6p_canary_20260608_post_metachunkfix/client.log
  - server log: output/q6p_canary_20260608_post_metachunkfix/server.log
  - probe report: output/probe_meta_len_all5746_post_metachunkfix_20260608.json
- Outcome: metadata parity reached (`metadata_mismatch_type/format/datatype=0`) but runtime still produced no streamed response; residual `data_len_mismatch=2728` remains.
- Next branch: Target residual 2728 data-length mismatches with even stricter micro-batching and immediate canary after each batch window.

## Run 006 - 2026-06-08 13:21 (UTC)

- Goal: Drain residual payload mismatches (`data_len_mismatch=2728`) via recursive micro-batching, then re-verify structural parity and runtime.
- Target q6p: `dt-models/10_e_v1_bf16_regen_0_q6p.ckpt`
- Source f16: `dt-models/10_e_v1_bf16_regen_0_f16.ckpt`
- Baseline q6p: n/a (source-f16 comparison branch)
- Reference q6p: n/a (source-f16 comparison branch)
- Names file: `output/20260608_run006_payloadfix/chunks/init_*` (generated recursively from run-005 residuals)
- Names count: `2728` (initial residual set)
- Command:

```bash
# 1) Recursive payload apply in micro-batches (chunk-size 4; split on failure)
#    tools/dt_align_ckpt_content_subset.py --mode apply --chunk-size 4 --journal-mode delete --min-free-gb 50 --head-bytes 32
# 2) One-name retry for final failing leaf
/workspaces/drawthings-linux-toolkit/.venv/bin/python tools/dt_align_ckpt_content_subset.py \
  --file dt-models/10_e_v1_bf16_regen_0_q6p.ckpt \
  --baseline dt-models/10_e_v1_bf16_regen_0_f16.ckpt \
  --data-names-file output/20260608_run006_payloadfix/failed_leaf_names.txt \
  --mode apply --chunk-size 1 --journal-mode delete --min-free-gb 50 --head-bytes 32

# 3) Post-retry row-wise verification
/workspaces/drawthings-linux-toolkit/.venv/bin/python tools/dt_probe_ckpt_meta_len_rowwise.py \
  --file dt-models/10_e_v1_bf16_regen_0_q6p.ckpt \
  --baseline dt-models/10_e_v1_bf16_regen_0_f16.ckpt \
  --progress-every 400 \
  --out-json output/20260608_run006_payloadfix/probe_meta_len_rowwise_after_leafretry_20260608.json \
  --out-tsv output/20260608_run006_payloadfix/probe_meta_len_rowwise_after_leafretry_20260608.tsv \
  --out-mismatch-names output/20260608_run006_payloadfix/probe_mismatch_names_after_leafretry_20260608.txt

# 4) Post-parity bounded canary
bash tools/run_q6p_canary_once.sh \
  --model 10_e_v1_bf16_regen_0_q6p.ckpt \
  --timeout-sec 120 \
  --max-responses 10 \
  --tag 20260608_run006_post_leafretry
```

- Structural checks:
  - quick_check: not run in this branch
  - tensors count: 5746 (row-wise probe)
  - validator profile: not run in this branch
- Canary:
  - timeout sec: 120
  - max responses: 10
  - canary rc: 124
  - post echo rc: 0
  - saw response #1: no
- Crash signature:
  - stack head: none in run-006 canary logs (server stayed alive and answered post-echo)
- Artifacts:
  - payload work dir: `output/20260608_run006_payloadfix`
  - failed leaf list: `output/20260608_run006_payloadfix/failed_leaf_names.txt`
  - post-retry probe: `output/20260608_run006_payloadfix/probe_meta_len_rowwise_after_leafretry_20260608.json`
  - mismatch names: `output/20260608_run006_payloadfix/probe_mismatch_names_after_leafretry_20260608.txt`
  - canary work dir: `output/q6p_canary_20260608_run006_post_leafretry`
  - canary client log: `output/q6p_canary_20260608_run006_post_leafretry/client.log`
  - canary server log: `output/q6p_canary_20260608_run006_post_leafretry/server.log`
- Outcome: Row-wise metadata/length parity reached full-match state (`mismatch_any=0`, `full_match=5745`, `unreadable_both=1`), but runtime still timed out pre-stream (`canary_rc=124`, no `response #1`).
- Next branch: Focus on byte-level payload semantics for equal-length rows (family-targeted content probes and/or constrained content-copy experiments on stable micro-windows).

## Run 007 - 2026-06-08 14:11 (UTC)

- Goal: Move from ad-hoc commands to a repeatable scripted validity matrix and confirm whether `10_e_v1_bf16_regen_0_f16.ckpt` is valid independently of q6p runtime failures.
- Script added: `tools/run_10e_v1_model_validity_matrix.sh`
- Matrix command:

```bash
bash tools/run_10e_v1_model_validity_matrix.sh --tag 20260608_matrix_check_01
```

- f16 structural checks (scripted):
  - tensor count: 5746
  - null-name rows: 0
  - converter validator (`dt_validate_converted_ckpt.py --source-safetensors dt-models/10_e_v1_bf16.safetensors`): `RESULT=PASS`, profile `ltx2_3`
- f16 runtime canary (scripted):
  - timeout sec: 240
  - max responses: 30
  - canary rc: 0
  - post echo rc: 0
  - reached streamed signposts through `imageDecoded`
  - wrote generated image payload (`image_r0010_01.bin`)
- q6p runtime canary (scripted comparison branch):
  - timeout sec: 120
  - max responses: 10
  - canary rc: 137
  - post echo rc: 1
  - saw response #1: no
  - client log stopped at request-start; server terminated during request
- Artifacts:
  - matrix summary: `output/model_validity_20260608_matrix_check_01/summary.md`
  - f16 validator log: `output/model_validity_20260608_matrix_check_01/f16_validator.log`
  - f16 canary log: `output/model_validity_20260608_matrix_check_01/f16_canary.log`
  - q6p canary log: `output/model_validity_20260608_matrix_check_01/q6p_canary.log`
  - f16 canary work dir: `output/q6p_canary_20260608_matrix_check_01_f16`
  - q6p canary work dir: `output/q6p_canary_20260608_matrix_check_01_q6p`
- Outcome: Scripted evidence indicates the `10_e_v1_bf16_regen_0_f16.ckpt` conversion is structurally valid and runtime-usable; active blocker remains q6p runtime behavior.
- Next branch: Keep triage scripted and q6p-focused (payload semantics / serialization invariants), using the new matrix script as a baseline gate before each q6p experiment.

## Run 008 - 2026-06-09 10:55 (UTC)

- Goal: Execute a no-skip equal-length payload-semantics pass, apply the resulting mismatch set, and re-run bounded canary.
- Probe workflow (scripted, full coverage):
  - `tools/dt_probe_ckpt_equal_len_payload_mismatch.py` was run in deterministic 500-row chunks over the full shared set (5746 rows), then merged.
  - This avoided long single-process D-state stalls while preserving full row coverage.
- Probe aggregated results:
  - selected_tensors: 5746
  - unreadable_both: 1
  - readable_selected: 5745
  - metadata_mismatch: 0
  - data_len_equal: 5745
  - data_small_sha256_compared: 2310
  - data_small_sha256_mismatch: 26
  - data_len_equal_sig_mismatch: 26
  - merged mismatch names: 26 (all `__dit__` family)
- Equal-length mismatch patch run:
  - command: `bash tools/run_q6p_payloadfix_recursive.sh --names-file output/run008_equal_len_payloadfix_01/equal_len_payload_mismatch_names.txt --tag 20260609_run008_payloadfix_26 --head-bytes 32 --chunk-size 1 --initial-split-lines 26 --canary-timeout-sec 120 --max-responses 10`
  - apply summary:
    - pre_selected_data_head_mismatch: 26
    - rows_updated: 26
    - rows_skipped_dataerror: 0
    - post_selected_data_head_mismatch: 0
- Post-apply row-wise parity check:
  - metadata_mismatch_type/format/datatype: 0/0/0
  - dim_len_mismatch: 0
  - data_len_mismatch: 0
  - mismatch_any: 0
  - full_match: 5745
  - unreadable_both: 1
- Post-apply bounded canary:
  - timeout sec: 120
  - max responses: 10
  - canary rc: 124 (timed out)
  - post echo rc: 0
  - saw response #1: no
  - client tail: only request-start metadata (no streamed response)
  - server tail: request begin + config-steps log; no streamed generation signposts before timeout
- Artifacts:
  - chunked probe dir: `output/run008_equal_len_payloadfix_01/chunks`
  - merged mismatch names: `output/run008_equal_len_payloadfix_01/equal_len_payload_mismatch_names.txt`
  - payloadfix work dir: `output/20260609_run008_payloadfix_26`
  - post-fix row-wise probe json: `output/20260609_run008_payloadfix_26/probe_meta_len_rowwise_after_leafretry.json`
  - canary work dir: `output/q6p_canary_20260609_run008_payloadfix_26_post_leafretry`
  - canary client log: `output/q6p_canary_20260609_run008_payloadfix_26_post_leafretry/client.log`
  - canary server log: `output/q6p_canary_20260609_run008_payloadfix_26_post_leafretry/server.log`
- Outcome: Repairing all currently detected equal-length data-head/small-hash mismatches (26 rows) did not restore runtime streaming; q6p still times out before first response.
- Next branch: Expand byte-level serialization checks beyond current head+small-hash signatures (chunked tail/mid sampling and/or focused full-payload comparisons on suspect DIT families) while keeping all runs scripted and artifactized.

## Run 009 - 2026-06-09 11:02 (UTC)

- Goal: Re-run the equal-length branch with a higher hash threshold (`small_hash_limit=16384`) and chunked full coverage to capture additional 8KB-scale payload mismatches.
- Script path: `tools/run_q6p_equal_len_payloadfix.sh` (updated with chunked probe mode).
- Command:

```bash
bash tools/run_q6p_equal_len_payloadfix.sh \
  --tag 20260609_run009_hash16k_chunked \
  --probe-small-hash-limit 16384 \
  --probe-chunk-rows 500 \
  --head-bytes 32 \
  --probe-progress-every 50 \
  --skip-matrix \
  --chunk-size 1 \
  --initial-split-lines 200 \
  --canary-timeout-sec 120 \
  --max-responses 10
```

- Result:
  - chunk `1-500`: `RESULT=PASS` with `mismatch_names_count=0`
  - chunk `501-1000`: probe process entered D-state (`folio_wait_bit_common`) and did not progress.
  - Run was aborted before recursive apply/canary to avoid indefinite hang.
- Artifacts:
  - run dir: `output/20260609_run009_hash16k_chunked`
  - partial probe chunk output: `output/20260609_run009_hash16k_chunked/probe_chunks/report_1_500.json`
  - stalled target chunk path: `output/20260609_run009_hash16k_chunked/probe_chunks/report_501_1000.json` (not completed)
- Outcome: Wider small-hash probing remains unstable in this environment at current chunk shape; need a lighter targeted branch.

## Run 010 - 2026-06-09 11:09 (UTC)

- Goal: Execute a family-targeted fallback by copying all DIT gate `*-1` payload rows from baseline and retesting runtime.
- Names build (scripted snippet) produced:
  - `output/run010_gate_family_all_minus1_names.txt`
  - names_count: 192
  - families: `__dit__[t-a_gate|t-ax_gate|t-x_gate|t-xa_gate]-*-1`
- Apply + probe + canary command:

```bash
bash tools/run_q6p_payloadfix_recursive.sh \
  --names-file output/run010_gate_family_all_minus1_names.txt \
  --tag 20260609_run010_gate_family_minus1 \
  --head-bytes 32 \
  --chunk-size 1 \
  --initial-split-lines 192 \
  --canary-timeout-sec 120 \
  --max-responses 10
```

- Apply summary:
  - union_selected: 192
  - rows_updated: 192
  - rows_skipped_dataerror: 0
  - pre_selected_data_head_mismatch: 0
  - post_selected_data_head_mismatch: 0
- Post-apply row-wise probe:
  - metadata_mismatch_type/format/datatype: 0/0/0
  - dim_len_mismatch: 0
  - data_len_mismatch: 0
  - mismatch_any: 0
  - full_match: 5745
  - unreadable_both: 1
- Canary outcome:
  - timeout sec: 120
  - max responses: 10
  - canary rc: 124 (timed out)
  - post echo rc: 0
  - saw response #1: no
  - server log: request begin/config-steps only; no streamed response signposts
- Artifacts:
  - names file: `output/run010_gate_family_all_minus1_names.txt`
  - payloadfix work dir: `output/20260609_run010_gate_family_minus1`
  - post-fix row-wise probe json: `output/20260609_run010_gate_family_minus1/probe_meta_len_rowwise_after_leafretry.json`
  - canary work dir: `output/q6p_canary_20260609_run010_gate_family_minus1_post_leafretry`
  - canary client log: `output/q6p_canary_20260609_run010_gate_family_minus1_post_leafretry/client.log`
  - canary server log: `output/q6p_canary_20260609_run010_gate_family_minus1_post_leafretry/server.log`
- Outcome: Even copying all 192 DIT gate `*-1` payloads did not restore runtime streaming; q6p still times out before first response.
- Next branch: Move beyond gate/data-head classes to targeted tail/mid or full-payload checks on a smaller suspect set (scripted micro-batches with hard per-batch time bounds), then canary after each batch window.

## Run 011 - 2026-06-09 11:33 (UTC)

- Goal: Run a full DIT-only window-signature probe (head+mid+tail) with chunking and bounded workflow to detect payload drift beyond head-only checks.
- Script path: `tools/run_q6p_window_sig_payloadfix.sh`
- Command:

```bash
bash tools/run_q6p_window_sig_payloadfix.sh \
  --tag 20260609_run011_window_sig_batch1 \
  --head-bytes 32 \
  --mid-bytes 64 \
  --tail-bytes 64 \
  --probe-small-hash-limit 0 \
  --probe-chunk-rows 400 \
  --probe-progress-every 100 \
  --prefix __dit__ \
  --batch-size 24 \
  --max-batches 1 \
  --initial-split-lines 24 \
  --chunk-size 1 \
  --canary-timeout-sec 120 \
  --max-responses 10
```

- Probe summary:
  - filtered shared rows (`__dit__`): 5484
  - `data_len_equal_sig_mismatch`: 0
  - `data_mid_hex_mismatch`: 0
  - `data_tail_hex_mismatch`: 0
  - `mismatch_names_count`: 0
- Apply/canary:
  - no remediation batches executed (`batches_executed=0`).
- Artifacts:
  - summary: `output/20260609_run011_window_sig_batch1/summary.md`
  - probe log/json: `output/20260609_run011_window_sig_batch1/window_sig_probe.log`, `output/20260609_run011_window_sig_batch1/window_sig_probe.json`
  - mismatch names: `output/20260609_run011_window_sig_batch1/window_sig_mismatch_names.txt`
- Outcome: DIT head/mid/tail signature probe did not surface actionable equal-length payload mismatches.

## Run 012 - 2026-06-09 11:47 (UTC)

- Goal: Target non-DIT family (`__text_feature_extractor__`) with window signatures plus small full-hash compare (`<=64KB`) to probe unresolved unreadable region.
- Script path: `tools/run_q6p_window_sig_payloadfix.sh`
- Command:

```bash
bash tools/run_q6p_window_sig_payloadfix.sh \
  --tag 20260609_run012_textfeat_sig_hash64k_batch1 \
  --head-bytes 32 \
  --mid-bytes 64 \
  --tail-bytes 64 \
  --probe-small-hash-limit 65536 \
  --probe-chunk-rows 200 \
  --probe-progress-every 50 \
  --prefix __text_feature_extractor__ \
  --batch-size 8 \
  --max-batches 1 \
  --initial-split-lines 8 \
  --chunk-size 1 \
  --canary-timeout-sec 120 \
  --max-responses 10
```

- Probe summary:
  - filtered shared rows: 4
  - `readable_selected`: 3
  - `unreadable_both`: 1
  - signature mismatches: 0
  - `mismatch_names_count`: 0
- Apply/canary:
  - no remediation batches executed (`batches_executed=0`).
- Artifacts:
  - summary: `output/20260609_run012_textfeat_sig_hash64k_batch1/summary.md`
  - probe log/json: `output/20260609_run012_textfeat_sig_hash64k_batch1/window_sig_probe.log`, `output/20260609_run012_textfeat_sig_hash64k_batch1/window_sig_probe.json`
- Outcome: No signature mismatch in readable text-feature rows; one unreadable row remained unresolved.

## Run 013 - 2026-06-09 11:52 (UTC)

- Goal: Force-copy full `__text_feature_extractor__` family payloads (4 rows) via recursive split-on-failure and observe runtime behavior.
- Names file build:
  - `output/run013_text_feature_family_names.txt`
  - names_count: 4
  - includes `__text_feature_extractor__[t-video_aggregate_embed-0-0]`
- Command:

```bash
bash tools/run_q6p_payloadfix_recursive.sh \
  --names-file output/run013_text_feature_family_names.txt \
  --tag 20260609_run013_textfeat_family_all \
  --head-bytes 32 \
  --chunk-size 1 \
  --initial-split-lines 4 \
  --canary-timeout-sec 120 \
  --max-responses 10
```

- Apply summary:
  - 3 rows updated successfully.
  - 1 row persisted as DataError leaf even after retry:
    - `__text_feature_extractor__[t-video_aggregate_embed-0-0]`
  - failed leaf list: `output/20260609_run013_textfeat_family_all/failed_leaf_names.txt`
- Post-apply row-wise probe:
  - metadata mismatches: 0
  - dim/data_len mismatches: 0
  - `full_match=5745`, `unreadable_both=1`
- Canary outcome:
  - server crashed with SIGSEGV while reading tensor payload during model load:
    - `ccv_nnc_tensor_read`
    - `ccv_cnnp_model_read`
  - server log: `output/q6p_canary_20260609_run013_textfeat_family_all_post_leafretry/server.log`
- Outcome: Unreadable leaf row correlates with deterministic model-load crash path.

## Run 014 - 2026-06-09 12:18 (UTC)

- Goal: Patch the exact failed leaf row using high-limit sqlite and retest canary.
- Script path: `tools/run_q6p_highlimit_row_patch_canary.sh`
- Command:

```bash
bash tools/run_q6p_highlimit_row_patch_canary.sh \
  --tag 20260609_run014_highlimit_textfeat_leaf \
  --row-name "__text_feature_extractor__[t-video_aggregate_embed-0-0]" \
  --canary-timeout-sec 120 \
  --max-responses 10
```

- Patch summary:
  - pre `quick_check`: `ok`
  - high-limit row patch changes: `row_patch_changes=1`
  - post `quick_check`: `ok`
- Canary outcome:
  - `canary_rc=124` (timeout)
  - `post_echo_rc=0`
  - no SIGSEGV stack in server log; request begin/config-steps only
  - logs:
    - `output/q6p_canary_20260609_run014_highlimit_textfeat_leaf_canary/client.log`
    - `output/q6p_canary_20260609_run014_highlimit_textfeat_leaf_canary/server.log`
- Outcome: High-limit patch removes the crash path for the unreadable row but does not restore streaming response; runtime still stalls before first response.
- Next branch: Run DIT-focused small-blob full-hash pass (`<=64KB`) with chunked scripted probe to search for non-window payload drift not visible in head/mid/tail signatures.

## Run 015 - 2026-06-09 12:33 (UTC)

- Goal: Execute DIT-focused window+small-hash (`<=64KB`) full coverage branch in chunked scripted mode.
- Script path: `tools/run_q6p_window_sig_payloadfix.sh`
- Command:

```bash
bash tools/run_q6p_window_sig_payloadfix.sh \
  --tag 20260609_run015_dit_sig_hash64k_batch1 \
  --head-bytes 32 \
  --mid-bytes 64 \
  --tail-bytes 64 \
  --probe-small-hash-limit 65536 \
  --probe-chunk-rows 300 \
  --probe-progress-every 100 \
  --prefix __dit__ \
  --batch-size 24 \
  --max-batches 1 \
  --initial-split-lines 24 \
  --chunk-size 1 \
  --canary-timeout-sec 120 \
  --max-responses 10
```

- Observed partial results before interruption:
  - processed chunks through range `1801-2100` with zero mismatches so far
  - repeated chunk stats: `data_len_equal_sig_mismatch=0`, `data_small_sha256_mismatch=0`
- Termination:
  - run interrupted manually (`KeyboardInterrupt`) during subsequent probe chunk
  - exit code: 130
- Outcome: No evidence of mismatches in processed ranges, but run incomplete; cannot treat as final coverage result.

## Run 016 - 2026-06-09 12:52 (UTC)

- Goal: Re-run high-limit leaf patch branch with final-mode timeout policy (15 minutes) to ensure timeout is not the primary blocker.
- Script path: `tools/run_q6p_highlimit_row_patch_canary.sh`
- Command:

```bash
bash tools/run_q6p_highlimit_row_patch_canary.sh \
  --tag 20260609_run016_highlimit_textfeat_leaf_finalmode \
  --row-name "__text_feature_extractor__[t-video_aggregate_embed-0-0]" \
  --final-mode \
  --max-responses 10
```

- Patch summary:
  - pre `quick_check`: `ok`
  - high-limit patch changes: `row_patch_changes=1`
  - post `quick_check`: `ok`
- Canary summary:
  - configured timeout: 900s (final-mode policy)
  - `canary_rc=1`
  - `post_echo_rc=1`
  - client observed: `gRPC error: UNAVAILABLE: Socket closed`
  - server crashed with SIGSEGV during request handling:
    - `ccv_nnc_tensor_read`
    - `ccv_cnnp_model_read`
- Artifacts:
  - canary dir: `output/q6p_canary_20260609_run016_highlimit_textfeat_leaf_finalmode_canary`
  - client log: `output/q6p_canary_20260609_run016_highlimit_textfeat_leaf_finalmode_canary/client.log`
  - server log: `output/q6p_canary_20260609_run016_highlimit_textfeat_leaf_finalmode_canary/server.log`
- Outcome: Extending timeout to final-mode (15 min) did not address the failure; this branch fails due deterministic server crash, not timeout.
- Next branch: Keep final-mode for long tests, but prioritize root-cause isolation of model-load crash path (row-level/binary-serialization integrity around `ccv_nnc_tensor_read`) before additional timeout-focused experiments.

## Run 017 - 2026-06-09 13:28 (UTC)

- Goal: Patch the full `__text_feature_extractor__` family (4 rows) with high-limit sqlite and rerun canary in final-mode timeout policy.
- Script path: `tools/run_q6p_highlimit_rows_patch_canary.sh`
- Command:

```bash
bash tools/run_q6p_highlimit_rows_patch_canary.sh \
  --tag 20260609_run017_highlimit_textfeat_family_finalmode \
  --rows-file output/run013_text_feature_family_names.txt \
  --final-mode \
  --max-responses 10
```

- Patch summary:
  - pre `quick_check`: `ok`
  - all 4 row updates reported `changes=1`:
    - `__text_feature_extractor__[t-audio_aggregate_embed-0-0]`
    - `__text_feature_extractor__[t-audio_aggregate_embed-0-1]`
    - `__text_feature_extractor__[t-video_aggregate_embed-0-0]`
    - `__text_feature_extractor__[t-video_aggregate_embed-0-1]`
  - post `quick_check`: `ok`
- Canary summary (final-mode):
  - `timeout_sec=900`, `final_mode=1`
  - `canary_rc=1`
  - `post_echo_rc=1`
  - client observed: `gRPC error: UNAVAILABLE: Socket closed`
  - server crashed with SIGSEGV stack head:
    - `ccv_nnc_tensor_read`
    - `ccv_cnnp_model_read`
  - `crash_detected=1`
- Artifacts:
  - patch work dir: `output/20260609_run017_highlimit_textfeat_family_finalmode`
  - normalized rows list: `output/20260609_run017_highlimit_textfeat_family_finalmode/rows_norm.txt`
  - applied SQL: `output/20260609_run017_highlimit_textfeat_family_finalmode/patch_rows.sql`
  - patch log: `output/20260609_run017_highlimit_textfeat_family_finalmode/highlimit_patch.log`
  - canary dir: `output/q6p_canary_20260609_run017_highlimit_textfeat_family_finalmode_canary`
  - client log: `output/q6p_canary_20260609_run017_highlimit_textfeat_family_finalmode_canary/client.log`
  - server log: `output/q6p_canary_20260609_run017_highlimit_textfeat_family_finalmode_canary/server.log`
- Outcome: Even full text-feature family high-limit patch does not prevent the runtime crash; failure remains pre-stream with the same loader crash signature.

## Pause Note - 2026-06-09

- Execution paused by request for markdown handoff updates.
- Current highest-signal conclusion:
  - Increasing timeout to final-mode (900s) is necessary for long validations but not sufficient for this failure path.
  - Latest branches fail by deterministic server SIGSEGV (`ccv_nnc_tensor_read` -> `ccv_cnnp_model_read`) rather than timeout-only stall.

## Run 018 - 2026-06-09 14:29 (UTC)

- Goal: Execute the new generic LTX2.3 validity matrix with strict completion gates to lock a reproducible Phase 1 baseline.
- Script path: `tools/run_ltx23_model_validity_matrix.sh`
- Primary command:

```bash
tools/run_ltx23_model_validity_matrix.sh \
  --f16-ckpt dt-models/10_e_v1_bf16_regen_0_f16.ckpt \
  --q6p-ckpt dt-models/10_e_v1_bf16_regen_0_q6p.ckpt \
  --source-safetensors dt-models/10_e_v1_bf16.safetensors \
  --tag phase1_retry_20260609_143001
```

- Attempt note:
  - initial tag `phase1_20260609_142934` failed early due pre-fix canary guard (`--max-responses must be >= 1` when strict mode set unlimited responses)
  - rerun with patched canary completed end-to-end and produced definitive baseline outcome
- Matrix summary (`phase1_retry_20260609_143001`):
  - SQLite sanity:
    - f16 tensor_count=5746, null_name_count=0
    - q6p tensor_count=5746, null_name_count=0
  - f16 validation:
    - profile auto resolved to `ltx2_3`
    - required LTX2.3 prefix checks passed
    - `f16_validate_rc=0`
  - strict f16 canary:
    - `f16_canary_rc=0`
    - stream finished with final output (`images written: 1`)
  - strict q6p canary:
    - `q6p_canary_rc=1`
    - client observed `gRPC error: UNAVAILABLE: Socket closed`
    - server SIGSEGV stack head:
      - `ccv_nnc_tensor_read`
      - `ccv_cnnp_model_read`
  - final matrix result:
    - `final_rc=1`
    - `RESULT=FAIL`
- Artifacts:
  - matrix summary: `output/model_validity_ltx23_phase1_retry_20260609_143001/summary.md`
  - matrix logs:
    - `output/model_validity_ltx23_phase1_retry_20260609_143001/f16_validator.log`
    - `output/model_validity_ltx23_phase1_retry_20260609_143001/f16_canary.log`
    - `output/model_validity_ltx23_phase1_retry_20260609_143001/q6p_canary.log`
  - canary logs:
    - `output/q6p_canary_phase1_retry_20260609_143001_f16/client.log`
    - `output/q6p_canary_phase1_retry_20260609_143001_f16/server.log`
    - `output/q6p_canary_phase1_retry_20260609_143001_q6p/client.log`
    - `output/q6p_canary_phase1_retry_20260609_143001_q6p/server.log`
- Outcome: Phase 1 baseline is now locked with strict completion gating and still reproduces the deterministic q6p crash path.

## Run 019 - 2026-06-10 07:43 (UTC)

- Goal: Execute first-divergence stage instrumentation (Phase 2) to localize where baseline/candidate divergence first appears.
- New tooling used:
  - `tools/dt_export_ckpt_tensor_manifest.py`
  - `tools/dt_compare_ckpt_tensor_manifests.py`
  - `tools/run_ltx23_first_divergence_stage.sh`
- Primary micro-run command:

```bash
tools/run_ltx23_first_divergence_stage.sh \
  --baseline-f16 dt-models/10_e_v1_bf16_regen_0_f16.ckpt \
  --candidate-f16 dt-models/10_e_v1_bf16_regen_0_f16.ckpt \
  --baseline-q6p dt-models/10_e_v1_bf16_regen_0_q6p.ckpt \
  --candidate-q6p dt-models/10_e_v1_bf16_regen_0_q6p.pre_replay_replay_after_ab_20260605.ckpt \
  --max-rows 128 \
  --progress-every 0 \
  --tag phase2_run019_micro_20260610_074323
```

- Results (`phase2_run019_micro_20260610_074323`):
  - f16 compare:
    - `mismatch_rows=0`
    - `first_divergence=none`
  - q6p compare:
    - `mismatch_rows=128` (on first 128 rows)
    - `field_mismatch_total=847`
    - first divergence:
      - `name=__dit__[t-a2v_adaln_single_0-0-0]`
      - `field=metadata.type`
      - baseline value: `258`
      - candidate value: `38877129135357953`
  - stage summary:
    - `first_divergent_stage=q6p`
    - `RESULT=PASS` (instrumentation runner succeeded)

- Attempt note:
  - full-row run tags `phase2_run019_20260610_073734` and `phase2_run019_quick_20260610_074155` were started but intentionally terminated to avoid long blocking exports after confirming throughput bottlenecks.

- Artifacts:
  - summary: `output/first_divergence_phase2_run019_micro_20260610_074323/summary.json`
  - summary markdown: `output/first_divergence_phase2_run019_micro_20260610_074323/summary.md`
  - f16 compare: `output/first_divergence_phase2_run019_micro_20260610_074323/compare_f16.json`
  - q6p compare: `output/first_divergence_phase2_run019_micro_20260610_074323/compare_q6p.json`
  - q6p mismatch names: `output/first_divergence_phase2_run019_micro_20260610_074323/compare_q6p_mismatch_names.txt`

- Outcome: first-divergence instrumentation now provides deterministic stage localization; in this controlled run divergence is isolated to q6p stage while f16 remains parity-clean.

## Run 020 - 2026-06-10 08:20 (UTC)

- Goal: Complete full-row Phase 2 stage-localization run using currently available baseline/candidate model pairs and extract q6p-only divergence set.
- Script path: `tools/run_ltx23_first_divergence_stage.sh`
- Primary command:

```bash
bash tools/run_ltx23_first_divergence_stage.sh \
  --baseline-f16 dt-models/ltx_2.3_22b_distilled_f16.ckpt \
  --candidate-f16 dt-models/10_e_v1_bf16_regen_0_f16.ckpt \
  --baseline-q6p dt-models/ltx_2.3_22b_distilled_q6p_forcedfix_clipfix2_20260602.ckpt \
  --candidate-q6p dt-models/10_e_v1_bf16_regen_0_q6p.ckpt \
  --tag phase2_run020_full_20260610
```

- Full-row stage compare result (`phase2_run020_full_20260610`):
  - f16 compare:
    - `mismatch_rows=5601`
    - first divergence:
      - `name=__dit__[t-a2v_adaln_single_0-0-0]`
      - `field=signatures.data_head_hex`
  - q6p compare:
    - `mismatch_rows=5745`
    - first divergence:
      - `name=__dit__[t-a2v_adaln_single_0-0-0]`
      - `field=metadata.type`
  - stage summary:
    - `first_divergent_stage=f16`

- Derived mismatch-set delta analysis:
  - `q6p_only_count=144`
  - `f16_only_count=0`
  - `shared_count=5601`
  - q6p-only prefix counts:
    - `__text_audio_connector__`: 72
    - `__text_video_connector__`: 72
  - q6p-only subfamily counts:
    - `t-up_proj`: 32
    - `t-to_v`: 32
    - `t-to_o`: 32
    - `t-to_gate`: 32
    - `t-down_proj`: 16

- Artifacts:
  - summary: `output/first_divergence_phase2_run020_full_20260610/summary.json`
  - f16 compare: `output/first_divergence_phase2_run020_full_20260610/compare_f16.json`
  - q6p compare: `output/first_divergence_phase2_run020_full_20260610/compare_q6p.json`
  - q6p-only names: `output/first_divergence_phase2_run020_full_20260610/q6p_only_mismatch_names.txt`
  - q6p-only summary: `output/first_divergence_phase2_run020_full_20260610/q6p_only_summary.txt`

- Outcome: full official-vs-custom comparison is model-identity dominated at f16 payload/signature level, but q6p still introduces a narrow additional mismatch slice isolated to text connector families.
- Next branch: regenerate a fresh custom q6p with quantizer trace enabled and compare old-vs-new custom q6p plus canary behavior to test whether connector policy differences drive runtime outcomes.

## Run 021 - 2026-06-10 11:21 (UTC)

- Goal: Regenerate custom q6p from current custom f16 with LTX quantizer decision trace enabled, then retest runtime under strict completion gate.
- Quantization command:

```bash
bash tools/dt_quantize_model.sh \
  -i dt-models/10_e_v1_bf16_regen_0_f16.ckpt \
  -m ltx2.3 \
  -o dt-models/10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt \
  --target-codec q6p \
  --ltx-trace-output output/quant_trace_run021_20260610/ltx_trace.jsonl
```

- Quantization output:
  - output ckpt: `dt-models/10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt` (about 20G)
  - trace jsonl: `output/quant_trace_run021_20260610/ltx_trace.jsonl`
  - quantize log: `output/quant_trace_run021_20260610/quantize.log`

- Structural checks:
  - `PRAGMA quick_check`: `ok`
  - `tensors` row count: `5746`
  - trace row count: `5746`

- Trace decision distribution (reason):
  - `forced_ltx_scalar_path`: 3780
  - `forced_ltx_default_quant_path`: 1632
  - `forced_ltx_text_feature_path`: 262
  - `forced_ltx_sensitive_projection`: 40
  - `forced_ltx_high_precision_path`: 32

- Trace decision distribution (decision):
  - `ezm7`: 3780
  - `[q6p,ezm7]`: 1632
  - `preserve_original`: 262
  - `fp16`: 40
  - `[q8p,ezm7]`: 32

- Strict q6p canary command:

```bash
bash tools/run_q6p_canary_once.sh \
  --model 10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt \
  --timeout-sec 240 \
  --max-responses 0 \
  --require-complete-stream \
  --require-final-output \
  --tag run021_trace_q6p_strict_20260610
```

- Canary outcome:
  - `canary_rc=0`
  - `post_echo_rc=0`
  - streamed signposts reached: `textEncoded` -> `imageEncoded` -> `sampling` -> `imageDecoded`
  - stream finished with final output (`images written: 1`)
  - no deterministic `ccv_nnc_tensor_read` / `ccv_cnnp_model_read` crash in this run

- Artifacts:
  - quant trace dir: `output/quant_trace_run021_20260610`
  - strict canary dir: `output/q6p_canary_run021_trace_q6p_strict_20260610`
  - canary client log: `output/q6p_canary_run021_trace_q6p_strict_20260610/client.log`
  - canary server log: `output/q6p_canary_run021_trace_q6p_strict_20260610/server.log`

- Trace decision histogram (`ltx_trace.jsonl`, 5746 rows):
  - decisions:
    - `ezm7`: 3780
    - `[q6p,ezm7]`: 1632
    - `preserve_original`: 262
    - `fp16`: 40
    - `[q8p,ezm7]`: 32
  - reasons:
    - `forced_ltx_scalar_path`: 3780
    - `forced_ltx_default_quant_path`: 1632
    - `forced_ltx_text_feature_path`: 262
    - `forced_ltx_sensitive_projection`: 40
    - `forced_ltx_high_precision_path`: 32

- Outcome: traced regenerated q6p candidate passed strict runtime gate end-to-end in this controlled run; this is the first clear pass shift against the previously deterministic crash branch.
- Next branch: complete old-vs-new q6p manifest compare to map which tensor-policy deltas align with the runtime pass shift.

## Run 022 - 2026-06-10 11:38 (UTC)

- Goal: Complete same-model old-vs-new q6p stage localization to isolate what changed between the prior failing q6p and the traced regenerated q6p candidate.
- Script path: `tools/run_ltx23_first_divergence_stage.sh`
- Command:

```bash
bash tools/run_ltx23_first_divergence_stage.sh \
  --baseline-f16 dt-models/10_e_v1_bf16_regen_0_f16.ckpt \
  --candidate-f16 dt-models/10_e_v1_bf16_regen_0_f16.ckpt \
  --baseline-q6p dt-models/10_e_v1_bf16_regen_0_q6p.ckpt \
  --candidate-q6p dt-models/10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt \
  --progress-every 200 \
  --tag phase2_run021_old_vs_new_q6p_full_20260610
```

- Results (`phase2_run021_old_vs_new_q6p_full_20260610`):
  - f16 compare:
    - `mismatch_rows=0`
    - `first_divergence=none`
  - q6p compare:
    - `mismatch_rows=5745`
    - `field_mismatch_total=38853`
    - first divergence:
      - `name=__dit__[t-a2v_adaln_single_0-0-0]`
      - `field=metadata.type`
      - baseline value: `258`
      - candidate value: `38877129135357953`
  - stage summary:
    - `first_divergent_stage=q6p`

- q6p compare shape (top prefixes):
  - `__dit__`: 5484
  - `__text_audio_connector__`: 128
  - `__text_video_connector__`: 128
  - `__text_feature_extractor__`: 3
  - `text_audio_connector_learnable_registers`: 1
  - `text_video_connector_learnable_registers`: 1

- Artifacts:
  - summary: `output/first_divergence_phase2_run021_old_vs_new_q6p_full_20260610/summary.json`
  - summary md: `output/first_divergence_phase2_run021_old_vs_new_q6p_full_20260610/summary.md`
  - q6p compare: `output/first_divergence_phase2_run021_old_vs_new_q6p_full_20260610/compare_q6p.json`
  - q6p mismatch names: `output/first_divergence_phase2_run021_old_vs_new_q6p_full_20260610/compare_q6p_mismatch_names.txt`

- Reproducibility rerun (`phase2_run021_old_vs_new_q6p_20260610`):
  - repeated same-model compare without `--progress-every` override
  - reproduced stage decision and counts exactly:
    - `first_divergent_stage=q6p`
    - `f16 mismatch_rows=0`
    - `q6p mismatch_rows=5745`
    - `field_mismatch_total=38853`
  - artifact summary: `output/first_divergence_phase2_run021_old_vs_new_q6p_20260610/summary.json`

- Outcome: with identical f16 inputs, first divergence isolates strictly to q6p stage and indicates broad q6p-row rewrite between old and traced candidates.

## Run 023 - 2026-06-10 11:32 (UTC)

- Goal: Re-run the strict Phase 1 acceptance matrix using the traced q6p candidate to confirm the pass shift under the established gate contract.
- Script path: `tools/run_ltx23_model_validity_matrix.sh`
- Command:

```bash
bash tools/run_ltx23_model_validity_matrix.sh \
  --f16-ckpt dt-models/10_e_v1_bf16_regen_0_f16.ckpt \
  --q6p-ckpt dt-models/10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt \
  --source-safetensors dt-models/10_e_v1_bf16.safetensors \
  --tag phase1_trace021_retry_20260610
```

- Matrix summary (`phase1_trace021_retry_20260610`):
  - SQLite sanity:
    - f16 tensor_count=5746, null_name_count=0
    - q6p tensor_count=5746, null_name_count=0
  - f16 validation:
    - profile auto resolved to `ltx2_3`
    - required LTX2.3 prefix checks passed
    - `f16_validate_rc=0`
  - strict f16 canary:
    - `f16_canary_rc=0`
    - complete stream with final image output (`images written: 1`)
  - strict q6p canary (traced candidate):
    - `q6p_canary_rc=0`
    - complete stream with final image output (`images written: 1`)
  - final matrix result:
    - `final_rc=0`
    - `RESULT=PASS`

- Artifacts:
  - matrix summary: `output/model_validity_ltx23_phase1_trace021_retry_20260610/summary.md`
  - matrix logs:
    - `output/model_validity_ltx23_phase1_trace021_retry_20260610/f16_validator.log`
    - `output/model_validity_ltx23_phase1_trace021_retry_20260610/f16_canary.log`
    - `output/model_validity_ltx23_phase1_trace021_retry_20260610/q6p_canary.log`
  - canary logs:
    - `output/q6p_canary_phase1_trace021_retry_20260610_f16/client.log`
    - `output/q6p_canary_phase1_trace021_retry_20260610_f16/server.log`
    - `output/q6p_canary_phase1_trace021_retry_20260610_q6p/client.log`
    - `output/q6p_canary_phase1_trace021_retry_20260610_q6p/server.log`

- Outcome: traced q6p candidate now satisfies the same strict matrix gates that previously failed, including final-output completion.

## Run 024 - 2026-06-10 12:02 (UTC)

- Goal: Verify traced q6p stability beyond a single seed/size using strict completion gates.
- Harness changes:
  - `tools/run_q6p_canary_once.sh` extended with:
    - `--width`
    - `--height`
    - `--steps`
    - `--seed`
  - Added `tools/run_q6p_strict_stability_matrix.sh` to run repeatable multi-case strict canaries.

- Command:

```bash
bash tools/run_q6p_strict_stability_matrix.sh \
  --model 10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt \
  --tag run024_trace021_stability_20260610 \
  --seeds 4242,7777,1337 \
  --sizes 256x256,384x704 \
  --steps 4 \
  --timeout-sec 300
```

- Matrix summary (`run024_trace021_stability_20260610`):
  - cases: 6
  - pass: 6
  - fail: 0
  - per-case strict signals:
    - all cases `canary_rc=0`
    - all cases `post_echo_rc=0`
    - all cases produced final image output (`images=1`)

- Artifacts:
  - summary: `output/q6p_strict_stability_run024_trace021_stability_20260610/summary.md`
  - case table: `output/q6p_strict_stability_run024_trace021_stability_20260610/results.tsv`
  - per-case logs: `output/q6p_strict_stability_run024_trace021_stability_20260610/cases/*.log`

- Outcome: traced q6p candidate passes strict runtime gates across multi-seed and multi-size coverage in this matrix.

## Run 025 - 2026-06-10 13:10 (UTC)

- Goal: Validate whether a second LTX2.3 q6p artifact is stable under the same strict multi-case matrix used for traced custom q6p.
- Script path: `tools/run_q6p_strict_stability_matrix.sh`
- Command:

```bash
bash tools/run_q6p_strict_stability_matrix.sh \
  --model ltx_2.3_22b_distilled_q6p_forcedfix_clipfix2_20260602.ckpt \
  --tag run025_clipfix2_stability_20260610 \
  --seeds 4242,7777,1337 \
  --sizes 256x256,384x704 \
  --steps 4 \
  --timeout-sec 240
```

- Matrix summary (`run025_clipfix2_stability_20260610`):
  - cases: 6
  - pass: 0
  - fail: 6
  - all cases failed before any stream payload (`responses=0`, `images=0`)
- Primary failure signature:
  - server crash in `TextEncoder.encodeLTX2`
  - signal: `Illegal instruction`
  - client observed: `gRPC error: UNAVAILABLE: Socket closed`

- Artifacts:
  - summary: `output/q6p_strict_stability_run025_clipfix2_stability_20260610/summary.md`
  - case table: `output/q6p_strict_stability_run025_clipfix2_stability_20260610/results.tsv`
  - per-case logs: `output/q6p_strict_stability_run025_clipfix2_stability_20260610/cases/*.log`

- Outcome: second-model strict matrix failed 6/6 with deterministic early crash pattern.

## Run 026 - 2026-06-10 13:20 (UTC)

- Goal: Determine whether run025 failures were caused by non-final-mode settings or by model family behavior.
- Script path: `tools/run_q6p_canary_once.sh`
- Commands:

```bash
bash tools/run_q6p_canary_once.sh \
  --model ltx_2.3_22b_distilled_q6p_forcedfix_clipfix2_20260602.ckpt \
  --timeout-sec 240 --max-responses 0 \
  --require-complete-stream --require-final-output --final-mode \
  --tag run026_clipfix2_finalmode_canary_20260610

bash tools/run_q6p_canary_once.sh \
  --model ltx_2.3_22b_distilled_1.1_q6p.ckpt \
  --timeout-sec 240 --max-responses 0 \
  --require-complete-stream --require-final-output --final-mode \
  --tag run026_official11_finalmode_canary_20260610
```

- Results:
  - clipfix2: `canary_rc=1`, `post_echo_rc=1`, `RESULT=FAIL`
  - official 1.1 q6p: `canary_rc=1`, `post_echo_rc=1`, `RESULT=FAIL`
  - both failed with the same `Illegal instruction` crash in `TextEncoder.encodeLTX2`

- Artifacts:
  - `output/q6p_canary_run026_clipfix2_finalmode_canary_20260610/client.log`
  - `output/q6p_canary_run026_clipfix2_finalmode_canary_20260610/server.log`
  - `output/q6p_canary_run026_official11_finalmode_canary_20260610/client.log`
  - `output/q6p_canary_run026_official11_finalmode_canary_20260610/server.log`

- Outcome: failure is not explained by non-final-mode-only behavior; it persists with `--final-mode` and reproduces across two non-traced q6p artifacts.

## Run 027 - 2026-06-10 13:22 (UTC)

- Goal: Re-check traced q6p candidate under strict final-mode after local custom alias promotion experiment.
- Script path: `tools/run_q6p_canary_once.sh`
- Command:

```bash
bash tools/run_q6p_canary_once.sh \
  --model 10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt \
  --timeout-sec 240 --max-responses 0 \
  --require-complete-stream --require-final-output --final-mode \
  --tag run027_trace021_finalmode_control_20260610
```

- Result:
  - `canary_rc=1`, `post_echo_rc=1`, `RESULT=FAIL`
  - crash signature shifted to loader path:
    - `ccv_nnc_tensor_read`
    - `ccv_cnnp_model_read`

- Artifacts:
  - `output/q6p_canary_run027_trace021_finalmode_control_20260610/client.log`
  - `output/q6p_canary_run027_trace021_finalmode_control_20260610/server.log`

- Outcome: traced q6p stability became sensitive after local alias experiment; this prompted an isolation control.

## Run 028 - 2026-06-10 13:25 (UTC)

- Goal: Isolate whether the traced checkpoint itself regressed or whether key-path/custom-entry resolution was the destabilizer.
- Method:
  - temporary hardlink key (same inode/content):
    - `10_e_v1_bf16_regen_0_q6p_trace021_20260610_tmpkey.ckpt`
  - run strict final-mode canary with the temporary key.
- Script path: `tools/run_q6p_canary_once.sh`
- Command:

```bash
ln dt-models/10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt \
  dt-models/10_e_v1_bf16_regen_0_q6p_trace021_20260610_tmpkey.ckpt

bash tools/run_q6p_canary_once.sh \
  --model 10_e_v1_bf16_regen_0_q6p_trace021_20260610_tmpkey.ckpt \
  --timeout-sec 240 --max-responses 0 \
  --require-complete-stream --require-final-output --final-mode \
  --tag run028_trace021_tmpkey_control_20260610
```

- Result:
  - `canary_rc=0`, `post_echo_rc=0`, `RESULT=PASS`
  - complete stream with final image output (`responses=10`, `images written=1`)

- Artifacts:
  - `output/q6p_canary_run028_trace021_tmpkey_control_20260610/client.log`
  - `output/q6p_canary_run028_trace021_tmpkey_control_20260610/server.log`

- Outcome: traced checkpoint content remains viable; instability is tied to the model-key/custom-entry path, not raw tensor content.

## Run 029 - 2026-06-10 13:25 (UTC)

- Goal: Test whether changing `clip_encoder` in local `dt-models/custom.json` can stabilize traced-file key path.
- Local alias variant tested:
  - entry file: `10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt`
  - entry clip_encoder: `10_e_v1_bf16_regen_0_q6p.ckpt`
- Script path: `tools/run_q6p_canary_once.sh`

- Observed behavior:
  - long stall with no streamed responses from client
  - server showed loader-path crash stack in log during one run attempt
  - short-timeout confirmation run (`run029b_trace021_alias_clipoldq6p_shorttimeout_20260610`) returned:
    - `canary_rc=124` (timeout)
    - `post_echo_rc=0`
    - `RESULT=FAIL canary timed out (60s)`

- Artifacts:
  - `output/q6p_canary_run029_trace021_alias_clipoldq6p_20260610/client.log`
  - `output/q6p_canary_run029_trace021_alias_clipoldq6p_20260610/server.log`
  - `output/q6p_canary_run029b_trace021_alias_clipoldq6p_shorttimeout_20260610/client.log`
  - `output/q6p_canary_run029b_trace021_alias_clipoldq6p_shorttimeout_20260610/server.log`

- Outcome: this local alias variant did not produce a stable strict pass.

## Run 030 - 2026-06-10 13:30 (UTC)

- Goal: Confirm traced-key stability after reverting local custom entry file mapping away from traced key.
- Local config action:
  - reverted `dt-models/custom.json` 10_e_v1 entry `file` back to `10_e_v1_bf16_regen_0_q6p.ckpt`.
- Script path: `tools/run_q6p_canary_once.sh`
- Command:

```bash
bash tools/run_q6p_canary_once.sh \
  --model 10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt \
  --timeout-sec 240 --max-responses 0 \
  --require-complete-stream --require-final-output --final-mode \
  --tag run030_trace021_post_revert_control_20260610
```

- Result:
  - `canary_rc=0`, `post_echo_rc=0`, `RESULT=PASS`
  - full streamed completion with final image output (`responses=10`, `images=1`)

- Artifacts:
  - `output/q6p_canary_run030_trace021_post_revert_control_20260610/client.log`
  - `output/q6p_canary_run030_trace021_post_revert_control_20260610/server.log`

- Outcome: traced q6p strict pass is restored when the local custom entry no longer matches the traced file key.

## Run 031 - 2026-06-10 13:36 (UTC)

- Goal: Derive a minimal safe local custom-entry schema for the traced q6p key by sweeping controlled field combinations and validating with strict final-mode canary.
- New tooling:
  - `tools/run_custom_alias_schema_probe.sh`
  - mutates one named custom entry (`10_e_v1`) across predefined variants
  - runs strict final-mode canary for each variant
  - classifies failures (`loader_crash`, `textencoder_illegal`, `timeout`)
  - restores `dt-models/custom.json` automatically on exit

- Command:

```bash
bash tools/run_custom_alias_schema_probe.sh \
  --tag run031_alias_schema_probe_20260610 \
  --timeout-sec 75
```

- Probe summary (`run031_alias_schema_probe_20260610`):
  - cases: 5
  - pass: 1
  - fail: 4
  - model under test: `10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt`

- Case results:
  - `control_unmatched_oldq6p`
    - entry `file=10_e_v1_bf16_regen_0_q6p.ckpt`
    - entry `clip_encoder=10_e_v1_bf16_regen_0_q6p.ckpt`
    - `RESULT=PASS`, `canary_rc=0`, `responses=10`, `images=1`
  - `match_trace_clip_trace_scale1`
    - entry `file=trace021`, `clip_encoder=trace021`, `default_scale=1`
    - `RESULT=FAIL`, signature `loader_crash`, `canary_rc=124`
  - `match_trace_clip_oldq6p_scale1`
    - entry `file=trace021`, `clip_encoder=old_q6p`, `default_scale=1`
    - `RESULT=FAIL`, signature `loader_crash`, `canary_rc=124`
  - `match_trace_clip_official_scale1`
    - entry `file=trace021`, `clip_encoder=official_1.1_q6p`, `default_scale=1`
    - `RESULT=FAIL`, signature `textencoder_illegal`, `canary_rc=1`
  - `match_trace_clip_trace_scale12`
    - entry `file=trace021`, `clip_encoder=trace021`, `default_scale=12`
    - `RESULT=FAIL`, signature `loader_crash`, `canary_rc=124`

- Artifacts:
  - summary: `output/custom_alias_schema_probe_run031_alias_schema_probe_20260610/summary.md`
  - results table: `output/custom_alias_schema_probe_run031_alias_schema_probe_20260610/results.tsv`
  - per-case logs: `output/custom_alias_schema_probe_run031_alias_schema_probe_20260610/cases/*.log`

- Outcome:
  - In tested dimensions (`file`, `clip_encoder`, `default_scale`), any variant where custom entry `file` matched traced key failed.
  - Only unmatched-file control passed.
  - Current best operational path: keep traced strict validation and usage on non-custom key path until deeper custom-entry path isolation is completed.

## Run 032b - 2026-06-10 14:14 (UTC)

- Goal: Extend custom-entry schema isolation beyond `file`/`clip_encoder`/`default_scale` to test `version`, `modifier`, `objective`, `text_encoder`, and `autoencoder` impact while keeping strict final-mode canary gates.
- Tooling update:
  - `tools/run_custom_alias_schema_probe.sh` now supports:
    - `--matrix core|extended-fields`
    - per-case mutation of `version`, `modifier`, `text_encoder`, `autoencoder`, `objective`
    - periodic case heartbeat lines (`case=<id> status=running`) for unattended long runs
    - additional `missing_file` signature bucket

- Command:

```bash
bash tools/run_custom_alias_schema_probe.sh \
  --matrix extended-fields \
  --tag run032b_alias_schema_extended_fields_20260610 \
  --timeout-sec 75
```

- Probe summary (`run032b_alias_schema_extended_fields_20260610`):
  - cases: 10
  - pass: 1
  - fail: 9
  - model under test: `10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt`

- Case highlights:
  - `control_unmatched_oldq6p`:
    - entry `file=10_e_v1_bf16_regen_0_q6p.ckpt`, `clip_encoder=10_e_v1_bf16_regen_0_q6p.ckpt`
    - `RESULT=PASS`, `canary_rc=0`, `responses=10`, `images=1`
  - All nine `file=trace021` variants failed with the same observed canary outcome:
    - `RESULT=FAIL`, signature `timeout`, `canary_rc=124`, `responses=0`, `images=0`
    - variants swept:
      - baseline matched trace
      - `modifier=none`
      - `modifier=kontext_kv`
      - `version=ltx2`
      - `objective.u.condition_scale=1`
      - `objective` removed
      - `text_encoder` removed
      - `autoencoder` removed
      - `text_encoder` and `autoencoder` both removed

- Artifacts:
  - summary: `output/custom_alias_schema_probe_run032b_alias_schema_extended_fields_20260610/summary.md`
  - results table: `output/custom_alias_schema_probe_run032b_alias_schema_extended_fields_20260610/results.tsv`
  - per-case logs: `output/custom_alias_schema_probe_run032b_alias_schema_extended_fields_20260610/cases/*.log`

- Outcome:
  - Extending schema mutations across `version`/`modifier`/`objective`/`text_encoder`/`autoencoder` did not recover any matched-trace pass.
  - Current evidence strengthens the hypothesis that the destabilizing axis is custom-entry file-key match/resolution itself rather than these secondary entry fields.

## Run 033b - 2026-06-11 (UTC)

- Goal: Rerun the alias-resolution branch after crash interruption and verify whether failure follows custom-entry resolution semantics (not only traced-key naming).
- New tooling:
  - `tools/run_custom_alias_resolution_probe.sh`
  - sweeps model argument forms (file key vs alias name), duplicate alias ordering, and tmpkey hardlink controls
  - auto-restores `dt-models/custom.json` and removes temporary hardlink key on exit/interrupt
  - classifies failure signatures (`loader_crash`, `timeout`, `textencoder_illegal`, `missing_file`)

- Command:

```bash
bash tools/run_custom_alias_resolution_probe.sh \
  --tag run033b_alias_resolution_probe_20260611 \
  --timeout-sec 75
```

- Probe summary (`run033b_alias_resolution_probe_20260611`):
  - cases: 9
  - pass: 2
  - fail: 7
  - models under test:
    - `10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt`
    - `10_e_v1_bf16_regen_0_q6p_trace021_run033_tmpkey.ckpt` (hardlink to same content)

- Case highlights:
  - PASS controls (no custom probe alias entries active):
    - `control_trace_noncustom`: traced file key, `RESULT=PASS`, `responses=10`, `images=1`
    - `control_tmpkey_noncustom`: tmpkey file key, `RESULT=PASS`, `responses=10`, `images=1`
  - FAIL with custom probe entries enabled:
    - `probe_trace_alias_model_filearg`: `loader_crash`, `canary_rc=124`, `post_echo_rc=124`
    - `probe_trace_alias_model_namearg`: `loader_crash`, `canary_rc=124`, `post_echo_rc=124`
    - `probe_trace_alias_clipold_model_filearg`: `timeout`, `canary_rc=124`, `post_echo_rc=0`
    - `probe_trace_dupe_order_ab`: `loader_crash`, `canary_rc=124`, `post_echo_rc=124`
    - `probe_trace_dupe_order_ba`: `loader_crash`, `canary_rc=124`, `post_echo_rc=124`
    - `probe_tmpkey_alias_model_filearg`: `loader_crash`, `canary_rc=124`, `post_echo_rc=124`
    - `probe_tmpkey_alias_model_namearg`: `loader_crash`, `canary_rc=124`, `post_echo_rc=124`

- Artifacts:
  - summary: `output/custom_alias_resolution_probe_run033b_alias_resolution_probe_20260611/summary.md`
  - results table: `output/custom_alias_resolution_probe_run033b_alias_resolution_probe_20260611/results.tsv`
  - per-case logs: `output/custom_alias_resolution_probe_run033b_alias_resolution_probe_20260611/cases/*.log`

- Outcome:
  - Two independent non-custom controls passed for the same traced tensor content (original traced key and tmpkey hardlink).
  - Any probe scenario that introduces a custom alias entry keyed to that same file content failed, regardless of whether the request used file key or alias name.
  - Duplicate alias ordering (`ab` vs `ba`) did not recover a pass.
  - This strengthens the branch conclusion that the dominant trigger is custom-entry resolution/shadowing semantics around file-key lookup, rather than tensor payload validity or simple key-string identity.

## Run 034 - 2026-06-11 (UTC)

- Goal: Test whether alias-triggered instability is file-key-local or global by enabling aliases for one key while requesting a different key.
- Tooling update:
  - `tools/run_custom_alias_resolution_probe.sh` now supports:
    - `--matrix core|cross-file`
    - `alias_both` mode (simultaneous probe aliases for traced key and tmpkey)
- Command:

```bash
bash tools/run_custom_alias_resolution_probe.sh \
  --matrix cross-file \
  --tag run034_alias_resolution_crossfile_20260611 \
  --timeout-sec 75
```

- Probe summary (`run034_alias_resolution_crossfile_20260611`):
  - cases: 9
  - pass: 5
  - fail: 4

- Case highlights:
  - PASS controls:
    - `control_trace_noncustom` (trace key baseline)
    - `control_tmpkey_noncustom` (tmpkey baseline)
  - PASS cross-file probes (single alias active for a different key):
    - `cross_trace_alias_active_request_tmpkey`
    - `cross_tmpkey_alias_active_request_trace`
    - `cross_trace_dupe_active_request_tmpkey`
  - FAIL only when both probe aliases were simultaneously active:
    - `cross_both_aliases_request_trace`
    - `cross_both_aliases_request_tmpkey`
    - `cross_both_aliases_request_trace_name`
    - `cross_both_aliases_request_tmpkey_name`
    - all four failed with `loader_crash`, `canary_rc=124`, `post_echo_rc=124`

- Artifacts:
  - summary: `output/custom_alias_resolution_probe_run034_alias_resolution_crossfile_20260611/summary.md`
  - results table: `output/custom_alias_resolution_probe_run034_alias_resolution_crossfile_20260611/results.tsv`
  - per-case logs: `output/custom_alias_resolution_probe_run034_alias_resolution_crossfile_20260611/cases/*.log`

- Outcome:
  - Single-key alias activation does not globally poison other keys (cross-file single-alias cases passed).
  - Failure reappears when both matching probe aliases are active together, indicating a stronger mapping-collision/selection interaction than simple one-alias presence.
  - This narrows the next isolation target to resolution behavior under multi-entry shadowing rather than broad custom-entry presence.

## Run 035 (2026-06-11): Core Alias-Resolution Matrix with Winner Context

- Tool update:
  - `tools/run_custom_alias_resolution_probe.sh` now emits per-case resolution context columns in `results.tsv`:
    - `arg_source`, `arg_resolved_file`, `arg_match_count`, `arg_winner_name`, `arg_winner_modifier`
    - `trace_match_count`, `trace_winner_name`
    - `tmpkey_match_count`, `tmpkey_winner_name`

- Command:

```bash
bash tools/run_custom_alias_resolution_probe.sh \
  --matrix core \
  --tag run035_alias_resolution_core_ctx_20260611 \
  --timeout-sec 75
```

- Probe summary (`run035_alias_resolution_core_ctx_20260611`):
  - cases: 9
  - pass: 2
  - fail: 7

- Context highlights:
  - PASS controls (`control_trace_noncustom`, `control_tmpkey_noncustom`) both had `arg_match_count=0` and no custom winner.
  - All FAIL cases had `arg_match_count>=1` with an active winning custom entry for the resolved file key.
  - Duplicate-order probes showed winner flip by insertion order (`ab` winner=`probe_trace_alias_b`, `ba` winner=`probe_trace_alias_a`), but both still failed.

- Artifacts:
  - summary: `output/custom_alias_resolution_probe_run035_alias_resolution_core_ctx_20260611/summary.md`
  - results table: `output/custom_alias_resolution_probe_run035_alias_resolution_core_ctx_20260611/results.tsv`
  - per-case logs: `output/custom_alias_resolution_probe_run035_alias_resolution_core_ctx_20260611/cases/*.log`

## Run 036 (2026-06-11): Cross-File Alias-Resolution Matrix with Winner Context

- Command:

```bash
bash tools/run_custom_alias_resolution_probe.sh \
  --matrix cross-file \
  --tag run036_alias_resolution_crossfile_ctx_20260611 \
  --timeout-sec 75
```

- Probe summary (`run036_alias_resolution_crossfile_ctx_20260611`):
  - cases: 9
  - pass: 5
  - fail: 4

- Context highlights:
  - Cross-file single-alias requests remained PASS when request-file had `arg_match_count=0`, even while the other key had active matches (`trace_match_count` or `tmpkey_match_count` > 0).
  - All `alias_both` cases failed; each failure had simultaneous overlap (`trace_match_count=1` and `tmpkey_match_count=1`) with an active arg winner.
  - In `alias_both`, file-arg variants failed as `timeout`, while alias-name variants failed as `loader_crash`.

- Artifacts:
  - summary: `output/custom_alias_resolution_probe_run036_alias_resolution_crossfile_ctx_20260611/summary.md`
  - results table: `output/custom_alias_resolution_probe_run036_alias_resolution_crossfile_ctx_20260611/results.tsv`
  - per-case logs: `output/custom_alias_resolution_probe_run036_alias_resolution_crossfile_ctx_20260611/cases/*.log`

- Outcome:
  - The trigger remains tied to overlapping multi-entry winner selection state, not global contamination from any one alias and not tensor payload validity.

## Run 037 (2026-06-11): Minimal-v1 Custom-Winner Isolation Matrix

- Tool update:
  - `tools/run_custom_alias_resolution_probe.sh` now supports `--matrix minimal-v1`.
  - Added minimal custom-entry modes (`alias_trace_min_v1`, `alias_tmpkey_min_v1`, `alias_both_min_v1`) that create only:
    - `name`, `file`, `prefix=""`, `version="v1"`, `upcast_attention=false`, `default_scale=8`
  - Purpose: isolate winner-presence effects from LTX-specific custom fields.

- Command:

```bash
bash tools/run_custom_alias_resolution_probe.sh \
  --matrix minimal-v1 \
  --tag run037_alias_resolution_minv1_ctx_20260611 \
  --timeout-sec 75
```

- Probe summary (`run037_alias_resolution_minv1_ctx_20260611`):
  - cases: 8
  - pass: 8
  - fail: 0

- Context highlights:
  - All custom-winner cases passed, including:
    - file-arg and alias-name requests with `arg_match_count=1`
    - simultaneous overlap (`alias_both_min_v1`) where `trace_match_count=1` and `tmpkey_match_count=1`
  - All cases returned `canary_rc=0`, `post_echo_rc=0`, `responses=10`, `images=1`.

- Artifacts:
  - summary: `output/custom_alias_resolution_probe_run037_alias_resolution_minv1_ctx_20260611/summary.md`
  - results table: `output/custom_alias_resolution_probe_run037_alias_resolution_minv1_ctx_20260611/results.tsv`
  - per-case logs: `output/custom_alias_resolution_probe_run037_alias_resolution_minv1_ctx_20260611/cases/*.log`

- Outcome:
  - Winner presence (`arg_match_count>=1`) and overlap alone are not sufficient to trigger failure.
  - Prior failures require one or more LTX-style custom-entry fields used in earlier modes (`version=ltx2.3`, `modifier=kontext/none`, `clip_encoder=file`, `default_scale=1`, and related fields).

## Run 038 (2026-06-11): Additive LTX Field Ladder from Minimal-v1 Baseline

- Tool update:
  - `tools/run_custom_alias_resolution_probe.sh` now supports `--matrix field-ladder`.
  - Added additive modes for trace-key custom entry:
    - `alias_trace_ladder_ltx23_min`
    - `alias_trace_ladder_ltx23_modifier`
    - `alias_trace_ladder_ltx23_modifier_scale1`
    - `alias_trace_ladder_ltx23_modifier_scale1_clip`
    - `alias_trace_a` (full base clone)

- Command:

```bash
bash tools/run_custom_alias_resolution_probe.sh \
  --matrix field-ladder \
  --tag run038_alias_resolution_field_ladder_20260611 \
  --timeout-sec 75
```

- Probe summary (`run038_alias_resolution_field_ladder_20260611`):
  - cases: 7
  - pass: 2
  - fail: 5

- Transition results:
  - `control_trace_noncustom`: PASS
  - `probe_trace_minv1` (custom winner, `version=v1` minimal schema): PASS
  - `probe_trace_ladder_v_ltx23` (only `version` switched to `ltx2.3`): FAIL (`textencoder_illegal`)
  - `probe_trace_ladder_v_ltx23_modifier`: FAIL (`textencoder_illegal`)
  - `probe_trace_ladder_v_ltx23_modifier_scale1`: FAIL (`textencoder_illegal`)
  - `probe_trace_ladder_v_ltx23_modifier_scale1_clip`: FAIL (`timeout`)
  - `probe_trace_ladder_full_base`: FAIL (`timeout`)

- Key finding:
  - First failing transition is `version: v1 -> ltx2.3` alone, even with otherwise minimal/default-like entry.
  - Additional LTX-style fields change failure signature (`textencoder_illegal` to timeout) but are not required to trigger failure.

- Artifacts:
  - summary: `output/custom_alias_resolution_probe_run038_alias_resolution_field_ladder_20260611/summary.md`
  - results table: `output/custom_alias_resolution_probe_run038_alias_resolution_field_ladder_20260611/results.tsv`
  - per-case logs: `output/custom_alias_resolution_probe_run038_alias_resolution_field_ladder_20260611/cases/*.log`

## Run 039 (2026-06-11): LTX2.3 Encoder-Path Variants (Version Fixed)

- Tool update:
  - `tools/run_custom_alias_resolution_probe.sh` now supports `--matrix ltx23-encoders`.
  - Added ltx2.3-fixed minimal variants:
    - `alias_trace_ltx23_min_text`
    - `alias_trace_ltx23_min_auto`
    - `alias_trace_ltx23_min_text_auto`
    - `alias_trace_ltx23_min_clip`

- Command:

```bash
bash tools/run_custom_alias_resolution_probe.sh \
  --matrix ltx23-encoders \
  --tag run039_alias_resolution_ltx23_encoders_20260611 \
  --timeout-sec 75
```

- Probe summary (`run039_alias_resolution_ltx23_encoders_20260611`):
  - cases: 7
  - pass: 1
  - fail: 6

- Variant outcomes (all with `version=ltx2.3`):
  - `probe_trace_ltx23_min`: FAIL (`textencoder_illegal`)
  - `probe_trace_ltx23_min_text`: FAIL (`timeout`)
  - `probe_trace_ltx23_min_auto`: FAIL (`textencoder_illegal`)
  - `probe_trace_ltx23_min_text_auto`: FAIL (`timeout`)
  - `probe_trace_ltx23_min_clip`: FAIL (`timeout`)
  - `probe_trace_ltx23_full_base`: FAIL (`timeout`)

- Key finding:
  - Keeping `version=ltx2.3` is sufficient for deterministic failure across encoder variants.
  - Adding `text_encoder` and/or `clip_encoder` shifts signature from immediate `textencoder_illegal` toward timeout-style failure, but does not recover PASS.

- Artifacts:
  - summary: `output/custom_alias_resolution_probe_run039_alias_resolution_ltx23_encoders_20260611/summary.md`
  - results table: `output/custom_alias_resolution_probe_run039_alias_resolution_ltx23_encoders_20260611/results.tsv`
  - per-case logs: `output/custom_alias_resolution_probe_run039_alias_resolution_ltx23_encoders_20260611/cases/*.log`

## Run 040 (2026-06-11): LTX2.3 Encoder Matrix with Resolved File-List Context

- Tool update:
  - Extended `tools/run_custom_alias_resolution_probe.sh` context export with resolved encoder columns:
    - `arg_winner_version`
    - `arg_winner_text_encoder`
    - `arg_winner_clip_encoder`
    - `arg_winner_autoencoder`
    - `arg_text_files_count`
    - `arg_text_file0`
    - `arg_text_file1`
    - `arg_ltx23_textfiles_ok` (1 when non-ltx2.3 or at least two text files)

- Command:

```bash
bash tools/run_custom_alias_resolution_probe.sh \
  --matrix ltx23-encoders \
  --tag run040_alias_resolution_ltx23_encoders_ctx2_20260611 \
  --timeout-sec 75
```

- Probe summary (`run040_alias_resolution_ltx23_encoders_ctx2_20260611`):
  - cases: 7
  - pass: 1
  - fail: 6

- Context-correlated findings:
  - All `textencoder_illegal` failures had:
    - `arg_winner_version=ltx2.3`
    - `arg_text_files_count=1`
    - `arg_ltx23_textfiles_ok=0`
  - These crashes reported:
    - `Program crashed: Illegal instruction`
    - frame at `TextEncoder.encodeLTX2(...)+12255`
  - Variants with two resolved text files (`arg_text_files_count=2`, `arg_ltx23_textfiles_ok=1`) shifted to timeout/loader-crash signatures rather than immediate illegal-instruction.

- Interpretation:
  - Source-level hypothesis is now strongly supported: in ltx2.3 custom path, single-entry text file lists are unsafe for `encodeLTX2` and correlate with deterministic illegal-instruction crashes.
  - Remaining timeout/loader-crash cases are likely downstream effects after satisfying the second-file precondition.

- Artifacts:
  - summary: `output/custom_alias_resolution_probe_run040_alias_resolution_ltx23_encoders_ctx2_20260611/summary.md`
  - results table: `output/custom_alias_resolution_probe_run040_alias_resolution_ltx23_encoders_ctx2_20260611/results.tsv`
  - per-case logs: `output/custom_alias_resolution_probe_run040_alias_resolution_ltx23_encoders_ctx2_20260611/cases/*.log`

## Run 041 (2026-06-11): Controlled LTX2.3 Two-File Companion Matrix

- Tool update:
  - Extended `tools/run_custom_alias_resolution_probe.sh` with `--matrix ltx23-companion`.
  - Added configurable companion clip candidates:
    - `--companion-clip-a` (default `ltx_2.3_22b_distilled_q6p_forcedfix_clipfix2_20260602.ckpt`)
    - `--companion-clip-b` (default `10_e_v1_bf16_regen_0_q6p.ckpt`)
    - `--companion-clip-c` (default `ltx_2.3_22b_distilled_f16.ckpt`)
  - Added ltx2.3 custom modes that force two-file text list construction (`text_encoder` from base + selected `clip_encoder`).

- Command:

```bash
bash tools/run_custom_alias_resolution_probe.sh \
  --matrix ltx23-companion \
  --timeout-sec 75 \
  --tag run041_ltx23_companion
```

- Probe summary (`run041_ltx23_companion`):
  - cases: 7
  - pass: 1
  - fail: 6

- Outcome map:
  - `control_trace_noncustom`: PASS
  - `probe_trace_ltx23_min_text` (ltx2.3 + one text file): FAIL (`textencoder_illegal`, `canary_rc=1`)
  - `probe_trace_ltx23_text_clip_trace` (two-file with traced clip): FAIL (`loader_crash`, `canary_rc=124`, `post_echo_rc=124`)
  - `probe_trace_ltx23_text_clip_companion_a` (clipfix2 companion): FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=0`)
  - `probe_trace_ltx23_text_clip_companion_b` (regen0_q6p companion): FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=0`)
  - `probe_trace_ltx23_text_clip_companion_c` (distilled_f16 companion): FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=0`)
  - `probe_trace_ltx23_full_base`: FAIL (`loader_crash`, `canary_rc=124`, `post_echo_rc=124`)

- Key finding:
  - Run040 precondition is reinforced: one-file ltx2.3 custom text path remains the immediate illegal-instruction branch.
  - Forcing two-file lists avoids immediate `TextEncoder.encodeLTX2` illegal-instruction, but outcomes split by companion choice into timeout vs loader-crash, indicating a downstream failure branch after encoder-list precondition is satisfied.

- Artifacts:
  - summary: `output/custom_alias_resolution_probe_run041_ltx23_companion/summary.md`
  - results table: `output/custom_alias_resolution_probe_run041_ltx23_companion/results.tsv`
  - per-case logs: `output/custom_alias_resolution_probe_run041_ltx23_companion/cases/*.log`

## Run 042 (2026-06-11): LTX2.3 Two-File Ordering Matrix

- Tool update:
  - Extended `tools/run_custom_alias_resolution_probe.sh` with `--matrix ltx23-order`.
  - Added explicit ordering modes to separate `text_encoder` vs `clip_encoder` assignment while keeping requested model fixed to traced q6p key:
    - `alias_trace_ltx23_text_swap_companion_a`
    - `alias_trace_ltx23_text_swap_companion_b`
    - `alias_trace_ltx23_text_swap_companion_c`

- Command:

```bash
bash tools/run_custom_alias_resolution_probe.sh \
  --matrix ltx23-order \
  --timeout-sec 75 \
  --tag run042_ltx23_order
```

- Probe summary (`run042_ltx23_order`):
  - cases: 8
  - pass: 1
  - fail: 7

- Outcome map:
  - `control_trace_noncustom`: PASS
  - `probe_trace_ltx23_min_text` (ltx2.3 + one text file): FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=0`)
  - `probe_trace_ltx23_text_clip_trace`: FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=0`)
  - `probe_trace_ltx23_text_clip_companion_a`: FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=0`)
  - `probe_trace_ltx23_text_swap_companion_a`: FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=0`)
  - `probe_trace_ltx23_text_swap_companion_b`: FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=0`)
  - `probe_trace_ltx23_text_swap_companion_c`: FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=0`)
  - `probe_trace_ltx23_full_base`: FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=0`)

- Context note (important drift):
  - Current local base entry `10_e_v1` resolves `text_encoder=gemma_3_12b_it_qat_q8p.ckpt`.
  - In run042, one-file ltx2.3 case used `arg_text_file0=gemma_3_12b_it_qat_q8p.ckpt` and did not reproduce immediate `textencoder_illegal`; it timed out instead.

- Key finding:
  - Under current base-entry state, all ltx2.3 custom variants converge to timeout before first streamed response, regardless of two-file ordering (`text_encoder` first vs companion first).
  - Ordering across tested companions did not separate timeout vs loader-crash in this run; state-sensitive encoder selection remains a likely confound.

- Artifacts:
  - summary: `output/custom_alias_resolution_probe_run042_ltx23_order/summary.md`
  - results table: `output/custom_alias_resolution_probe_run042_ltx23_order/results.tsv`
  - per-case logs: `output/custom_alias_resolution_probe_run042_ltx23_order/cases/*.log`

## Run 043 (2026-06-11): LTX2.3 Text-Encoder Pin Matrix

- Tool update:
  - Extended `tools/run_custom_alias_resolution_probe.sh` with `--matrix ltx23-textpin`.
  - Added explicit text pins and text+clip pinned modes:
    - `--text-pin-a` (default `clip_vit_l14_f16.ckpt`)
    - `--text-pin-b` (default `gemma_3_12b_it_qat_q8p.ckpt`)
    - `alias_trace_ltx23_text_only_pin_a|b`
    - `alias_trace_ltx23_text_clip_trace_pin_a|b`

- Command:

```bash
bash tools/run_custom_alias_resolution_probe.sh \
  --matrix ltx23-textpin \
  --timeout-sec 75 \
  --tag run043_ltx23_textpin
```

- Probe summary (`run043_ltx23_textpin`):
  - cases: 6
  - pass: 1
  - fail: 5

- Outcome map:
  - `control_trace_noncustom`: PASS
  - `probe_trace_ltx23_text_only_pin_a` (one-file, `text_encoder=clip_vit_l14_f16.ckpt`): FAIL (`textencoder_illegal`, `canary_rc=124`, `post_echo_rc=124`)
  - `probe_trace_ltx23_text_only_pin_b` (one-file, `text_encoder=gemma_3_12b_it_qat_q8p.ckpt`): FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=0`)
  - `probe_trace_ltx23_text_clip_trace_pin_a` (two-file, clip-vs-trace): FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=0`)
  - `probe_trace_ltx23_text_clip_trace_pin_b` (two-file, gemma-vs-trace): FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=0`)
  - `probe_trace_ltx23_full_base`: FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=0`)

- Key finding:
  - Explicit pinning confirms text-encoder identity is a first-order discriminator in one-file ltx2.3 path:
    - one-file + `clip_vit_l14_f16.ckpt` reproduces immediate `TextEncoder.encodeLTX2` illegal-instruction crash.
    - one-file + `gemma_3_12b_it_qat_q8p.ckpt` shifts to timeout branch.
  - Two-file variants remained timeout-only in this run, regardless of text pin A/B.
  - This reconciles run041 vs run042: earlier split behavior is reproducible when text-encoder source is pinned, and local base-entry drift can mask it.

- Artifacts:
  - summary: `output/custom_alias_resolution_probe_run043_ltx23_textpin/summary.md`
  - results table: `output/custom_alias_resolution_probe_run043_ltx23_textpin/results.tsv`
  - per-case logs: `output/custom_alias_resolution_probe_run043_ltx23_textpin/cases/*.log`

## Run 044 (2026-06-11): LTX2.3 Pinned-Companion Boundary Matrix

- Tool update:
  - Extended `tools/run_custom_alias_resolution_probe.sh` with `--matrix ltx23-pinned-companion`.
  - Added pinned companion modes with fixed `text_encoder=$TEXT_PIN_A`:
    - `alias_trace_ltx23_text_clip_companion_a_pin_a`
    - `alias_trace_ltx23_text_clip_companion_b_pin_a`
    - `alias_trace_ltx23_text_clip_companion_c_pin_a`

- Command:

```bash
bash tools/run_custom_alias_resolution_probe.sh \
  --matrix ltx23-pinned-companion \
  --timeout-sec 75 \
  --tag run044_ltx23_pinned_companion
```

- Probe summary (`run044_ltx23_pinned_companion`):
  - cases: 7
  - pass: 1
  - fail: 6

- Outcome map:
  - `control_trace_noncustom`: PASS
  - `probe_trace_ltx23_text_only_pin_a` (one-file, `text_encoder=clip_vit_l14_f16.ckpt`): FAIL (`textencoder_illegal`, `canary_rc=124`, `post_echo_rc=1`)
  - `probe_trace_ltx23_text_clip_trace_pin_a` (two-file, clip_vit + traced clip): FAIL (`loader_crash`, `canary_rc=124`, `post_echo_rc=124`)
  - `probe_trace_ltx23_text_clip_companion_a_pin_a` (two-file, clip_vit + clipfix2 companion): FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=0`)
  - `probe_trace_ltx23_text_clip_companion_b_pin_a` (two-file, clip_vit + regen0_q6p companion): FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=0`)
  - `probe_trace_ltx23_text_clip_companion_c_pin_a` (two-file, clip_vit + distilled_f16 companion): FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=0`)
  - `probe_trace_ltx23_full_base`: FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=0`)

- Key finding:
  - With text identity pinned to clip_vit, run044 restores a deterministic three-branch boundary:
    - one-file => `textencoder_illegal`
    - two-file with traced clip => `loader_crash`
    - two-file with non-trace companions => `timeout`
  - This is the clearest controlled separation so far between immediate encode crash, loader crash, and stall branches.
  - Stage evidence from case logs:
    - all failing variants terminated before streamed signposts (`textEncoded`/`imageEncoded`) were emitted in canary output.
    - crash branches are distinguished by post-echo state: `post_echo_rc=1` (illegal-instruction) vs `post_echo_rc=124` (loader crash), while timeout branch preserves healthy post-echo (`post_echo_rc=0`).

- Artifacts:
  - summary: `output/custom_alias_resolution_probe_run044_ltx23_pinned_companion/summary.md`
  - results table: `output/custom_alias_resolution_probe_run044_ltx23_pinned_companion/results.tsv`
  - per-case logs: `output/custom_alias_resolution_probe_run044_ltx23_pinned_companion/cases/*.log`

## Run 045 (2026-06-11): Instrumented Linux Build + Pinned-Companion Replay

- Build unblock fixes applied (source repo):
  - `Libraries/GRPC/Server/Sources/GRPCServerAdvertiser.swift`
  - `Libraries/GRPC/Server/Sources/GRPCServiceBrowser.swift`
  - `Libraries/LocalImageGenerator/Sources/ImageConverter.swift`
  - `Package.swift` (platform-conditional `DiffusionCoreML` dependency)
  - Result: `swift build -c release --product gRPCServerCLI` succeeded on Linux.

- Patch-bundle sync updates (toolkit repo):
  - Added the above GRPC/ImageConverter files to:
    - `tools/generate_drawthings_quant_patches.sh`
    - `tools/sync_drawthings_patch_bundle.sh`
    - `tools/apply_drawthings_quant_patch.sh`
  - Regenerated: `DRAW_THINGS_PATCH/patches/draw-things-community.patch`.

- Command (instrumented replay using built binary via PATH):

```bash
PATH="/workspaces/drawthings-linux-toolkit/draw-things-community/.build/release:$PATH" \
DT_LTX23_TRACE=1 \
bash tools/run_custom_alias_resolution_probe.sh \
  --matrix ltx23-pinned-companion \
  --timeout-sec 75 \
  --tag run045_ltx23_pinned_companion_trace_20260611
```

- Probe summary (`run045_ltx23_pinned_companion_trace_20260611`):
  - cases: 7
  - pass: 0
  - fail: 7

- Outcome map:
  - `control_trace_noncustom`: FAIL (`unknown`, `canary_rc=1`, `post_echo_rc=1`)
  - `probe_trace_ltx23_text_only_pin_a`: FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=124`)
  - `probe_trace_ltx23_text_clip_trace_pin_a`: FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=124`)
  - `probe_trace_ltx23_text_clip_companion_a_pin_a`: FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=124`)
  - `probe_trace_ltx23_text_clip_companion_b_pin_a`: FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=124`)
  - `probe_trace_ltx23_text_clip_companion_c_pin_a`: FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=124`)
  - `probe_trace_ltx23_full_base`: FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=0`)

- Key finding:
  - Under the locally built instrumented Linux binary, baseline control stability regressed (non-custom control failed).
  - Most failing cases now show server abort behavior in `server.log` (libc crash tail) and no streamed responses.
  - `DT_LTX23_TRACE` evidence was captured in at least one case (`probe_trace_ltx23_full_base`) through:
    - `LocalImageGenerator.textEncoderFiles`
    - `LocalImageGenerator.beforeTextEncode`
    - `encodeLTX2.begin`
    - `encodeLTX2.loading_text_model`
  - This confirms tracing hooks are active, but the broad regression means run045 cannot be compared directly against run044 branch signatures without controlling for source-built runtime instability.

- Artifacts:
  - summary: `output/custom_alias_resolution_probe_run045_ltx23_pinned_companion_trace_20260611/summary.md`
  - results table: `output/custom_alias_resolution_probe_run045_ltx23_pinned_companion_trace_20260611/results.tsv`
  - per-case logs: `output/custom_alias_resolution_probe_run045_ltx23_pinned_companion_trace_20260611/cases/*.log`

## Run 046 (2026-06-11): Binary-Path Control A/B for Source-Build Confound

- Goal:
  - Isolate whether run045 regressions are caused by alias-matrix changes or by binary/runtime path (`/usr/local/bin/gRPCServerCLI` vs local source-built binary).

- Control A (default wrapper binary):

```bash
bash tools/run_q6p_canary_once.sh \
  --model 10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt \
  --final-mode \
  --require-complete-stream \
  --require-final-output \
  --max-responses 0 \
  --timeout-sec 75 \
  --tag run046_control_wrapper_20260611
```

- Control B (local source-built binary + trace env):

```bash
PATH="/workspaces/drawthings-linux-toolkit/draw-things-community/.build/release:$PATH" \
DT_LTX23_TRACE=1 \
bash tools/run_q6p_canary_once.sh \
  --model 10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt \
  --final-mode \
  --require-complete-stream \
  --require-final-output \
  --max-responses 0 \
  --timeout-sec 75 \
  --tag run046_control_sourcebuild_trace_20260611
```

- Results:
  - Control A: PASS (`canary_rc=0`, `post_echo_rc=0`, streamed responses and final image output).
  - Control B: FAIL (`canary_rc=1`, `post_echo_rc=1`, server abort/core dump with libc crash tail, client `UNAVAILABLE: Socket closed`).

- Key finding:
  - The source-built Linux binary path is itself a deterministic failure confound for current strict control.
  - run045 matrix failures cannot be interpreted as pure alias-resolution branch evidence until this binary-path instability is isolated or removed.

- Artifacts:
  - wrapper control: `output/q6p_canary_run046_control_wrapper_20260611/{client.log,server.log}`
  - source-build control: `output/q6p_canary_run046_control_sourcebuild_trace_20260611/{client.log,server.log}`

## Run 047 (2026-06-11): Wrapper-Binary Pinned-Companion Replay

- Goal:
  - Re-run the pinned-companion matrix on the stable wrapper binary path to verify whether run044 branch signatures still hold after run045/run046 confound isolation.

- Command:

```bash
bash tools/run_custom_alias_resolution_probe.sh \
  --matrix ltx23-pinned-companion \
  --timeout-sec 75 \
  --tag run047_wrapper_ltx23_pinned_companion_20260611
```

- Probe summary (`run047_wrapper_ltx23_pinned_companion_20260611`):
  - cases: 7
  - pass: 1
  - fail: 6

- Outcome map:
  - `control_trace_noncustom`: PASS (`canary_rc=0`, `post_echo_rc=0`, `responses=10`, `images=1`)
  - `probe_trace_ltx23_text_only_pin_a`: FAIL (`textencoder_illegal`, `canary_rc=124`, `post_echo_rc=1`)
  - `probe_trace_ltx23_text_clip_trace_pin_a`: FAIL (`loader_crash`, `canary_rc=124`, `post_echo_rc=124`)
  - `probe_trace_ltx23_text_clip_companion_a_pin_a`: FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=0`)
  - `probe_trace_ltx23_text_clip_companion_b_pin_a`: FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=0`)
  - `probe_trace_ltx23_text_clip_companion_c_pin_a`: FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=0`)
  - `probe_trace_ltx23_full_base`: FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=0`)

- Key finding:
  - Wrapper-binary replay reproduces the run044 three-way split and confirms branch logic signal is stable under the non-source-built runtime path.
  - run045 all-fail behavior is therefore attributable to source-built runtime instability rather than a persistent alias-resolution behavior change.

- Artifacts:
  - summary: `output/custom_alias_resolution_probe_run047_wrapper_ltx23_pinned_companion_20260611/summary.md`
  - results table: `output/custom_alias_resolution_probe_run047_wrapper_ltx23_pinned_companion_20260611/results.tsv`
  - per-case logs: `output/custom_alias_resolution_probe_run047_wrapper_ltx23_pinned_companion_20260611/cases/*.log`

## Run 048 (2026-06-11): Source-Build Official-Control Check (Availability Confound)

- Goal:
  - Test source-built runtime on an official control model to check whether run046 crash behavior is global.

- Command:

```bash
PATH="/workspaces/drawthings-linux-toolkit/draw-things-community/.build/release:$PATH" \
DT_LTX23_TRACE=1 \
bash tools/run_q6p_canary_once.sh \
  --model ltx_2.3_22b_distilled_1.1_q6p.ckpt \
  --final-mode \
  --require-complete-stream \
  --require-final-output \
  --max-responses 0 \
  --timeout-sec 75 \
  --tag run048_sourcebuild_official_control_20260611
```

- Result:
  - FAIL (`canary_rc=124`, `post_echo_rc=0`, timeout, no streamed responses).

- Critical trace note:
  - `DT_LTX23_TRACE` shows resolved model path was `ltx_2.3_22b_distilled_1.1_q8p.ckpt` (not requested q6p), with one-file text list.
  - This indicates model-resolution fallback occurred, so run048 is an availability/resolution confound and not a clean official-q6p source-build A/B.

- Key finding:
  - Source-built path remains unstable/confounded; run048 adds evidence that model availability/resolution must be pinned before interpreting source-build behavior.

- Artifacts:
  - `output/q6p_canary_run048_sourcebuild_official_control_20260611/{client.log,server.log}`

## Run 049 (2026-06-11): Canary Preflight Guard Validation

- Goal:
  - Verify fail-fast guard for missing file-like model keys in canary harness.

- Command:

```bash
bash tools/run_q6p_canary_once.sh \
  --model ltx_2.3_22b_distilled_1.1_q6p.ckpt \
  --timeout-sec 75 \
  --max-responses 0 \
  --tag run049_preflight_missing_model_check_20260611
```

- Result:
  - FAIL-FAST (expected):
    - `error: model file not found: /workspaces/drawthings-linux-toolkit/dt-models/ltx_2.3_22b_distilled_1.1_q6p.ckpt`
    - `hint: use --allow-missing-model to permit fallback-resolution tests`

- Key finding:
  - Guardrail works as intended and prevents silent fallback contamination in control canaries.

## Run 050 (2026-06-11): Focused Wrapper-Path Branch Gate

- Goal:
  - Establish a lightweight regression gate for the highest-signal two-file branch split under stable wrapper runtime.

- Harness update:
  - `tools/run_custom_alias_resolution_probe.sh` gained a focused matrix mode:
    - `--matrix ltx23-focused`
  - Cases in this mode:
    - `control_trace_noncustom`
    - `probe_trace_ltx23_text_clip_trace_pin_a`
    - `probe_trace_ltx23_text_clip_companion_a_pin_a`

- Command:

```bash
bash tools/run_custom_alias_resolution_probe.sh \
  --matrix ltx23-focused \
  --timeout-sec 75 \
  --tag run050_wrapper_ltx23_focused_20260611
```

- Probe summary (`run050_wrapper_ltx23_focused_20260611`):
  - cases: 3
  - pass: 1
  - fail: 2

- Outcome map:
  - `control_trace_noncustom`: PASS (`canary_rc=0`, `post_echo_rc=0`, `responses=10`, `images=1`)
  - `probe_trace_ltx23_text_clip_trace_pin_a`: FAIL (`loader_crash`, `canary_rc=124`, `post_echo_rc=124`, `ccv_nnc_tensor_read`)
  - `probe_trace_ltx23_text_clip_companion_a_pin_a`: FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=0`)

- Key finding:
  - Focused matrix reliably reproduces the traced-clip loader-crash vs companion-timeout split with healthy control PASS.
  - This is now the fastest stable regression gate for branch-shape verification before broader matrices.

- Artifacts:
  - summary: `output/custom_alias_resolution_probe_run050_wrapper_ltx23_focused_20260611/summary.md`
  - results table: `output/custom_alias_resolution_probe_run050_wrapper_ltx23_focused_20260611/results.tsv`
  - per-case logs: `output/custom_alias_resolution_probe_run050_wrapper_ltx23_focused_20260611/cases/*.log`

## Run 051 (2026-06-11): Source-Build Control Without Trace Logging

- Goal:
  - Verify whether source-built crash depends on `DT_LTX23_TRACE` instrumentation overhead.

- Command:

```bash
PATH="/workspaces/drawthings-linux-toolkit/draw-things-community/.build/release:$PATH" \
bash tools/run_q6p_canary_once.sh \
  --model 10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt \
  --final-mode \
  --require-complete-stream \
  --require-final-output \
  --max-responses 0 \
  --timeout-sec 75 \
  --tag run051_sourcebuild_notrace_control_20260611
```

- Result:
  - FAIL (`canary_rc=1`, `post_echo_rc=1`, server abort/core dump, client `UNAVAILABLE: Socket closed`).

- Key finding:
  - Source-built runtime abort reproduces even without trace logging; `DT_LTX23_TRACE` is not the primary trigger.

- Artifacts:
  - `output/q6p_canary_run051_sourcebuild_notrace_control_20260611/{client.log,server.log}`

## Run 052 (2026-06-11): Binary/Linkage Forensics (Default vs Source-Build)

- Goal:
  - Compare working default binary and failing source-built binary to find structural runtime differences.

- Binary provenance snapshot:
  - default: `/usr/local/bin/gRPCServerCLI` (size `829,369,480` bytes)
  - source-built: `draw-things-community/.build/x86_64-unknown-linux-gnu/release/gRPCServerCLI` (size `217,319,528` bytes)
  - hashes differ:
    - default: `6d33b751f5b8b72a1e72fb9e70d5c94b6f8b5f7dc2a043deaa5f7d247ec3e717`
    - source: `ff5181f437b33a4b92aed3c68f6bf2f3cee30f316aa8bfa9e223ac24989b8b87`

- Linkage difference summary (`ldd`):
  - default binary links CUDA/CUDNN stack (`libcublas`, `libcudart`, `libcudnn`, `libcurand`, etc.)
  - source-built binary links `libopenblas` and does not link CUDA runtime libraries.

- Build-rule evidence from ccv SwiftPM package:
  - Linux host path in `.build/checkouts/ccv/Package.swift` sets OpenBLAS linker path and excludes GPU/CUDA sources (`gpu`, `cmd/*/gpu`, `cuda`).

- Key finding:
  - Source-built binary is not runtime-equivalent to the working deployed binary; it is effectively a different backend build profile (CPU/OpenBLAS vs CUDA-linked).
  - This is a high-confidence root confound for interpreting source-build canary behavior against wrapper-binary branch results.

## Run 053 (2026-06-11): Focused Wrapper Matrix With Trace Env

- Goal:
  - Re-run focused stable wrapper matrix with `DT_LTX23_TRACE=1` to capture branch behavior under the same runtime class.

- Command:

```bash
DT_LTX23_TRACE=1 \
bash tools/run_custom_alias_resolution_probe.sh \
  --matrix ltx23-focused \
  --timeout-sec 75 \
  --tag run053_wrapper_ltx23_focused_trace_20260611
```

- Probe summary (`run053_wrapper_ltx23_focused_trace_20260611`):
  - cases: 3
  - pass: 1
  - fail: 2

- Outcome map:
  - `control_trace_noncustom`: PASS
  - `probe_trace_ltx23_text_clip_trace_pin_a`: FAIL (`loader_crash`, `canary_rc=124`, `post_echo_rc=124`)
  - `probe_trace_ltx23_text_clip_companion_a_pin_a`: FAIL (`timeout`, `canary_rc=124`, `post_echo_rc=0`)

- Key finding:
  - Focused branch split reproduces under wrapper runtime with trace env enabled.

- Artifacts:
  - summary: `output/custom_alias_resolution_probe_run053_wrapper_ltx23_focused_trace_20260611/summary.md`
  - results table: `output/custom_alias_resolution_probe_run053_wrapper_ltx23_focused_trace_20260611/results.tsv`
  - per-case logs: `output/custom_alias_resolution_probe_run053_wrapper_ltx23_focused_trace_20260611/cases/*.log`

## Run 054 (2026-06-11): Wrapper Text-Pin Matrix Refresh

- Goal:
  - Recheck text-pin branch behavior on stable wrapper runtime.

- Command:

```bash
bash tools/run_custom_alias_resolution_probe.sh \
  --matrix ltx23-textpin \
  --timeout-sec 75 \
  --tag run054_wrapper_ltx23_textpin_20260611
```

- Probe summary (`run054_wrapper_ltx23_textpin_20260611`):
  - cases: 6
  - pass: 1
  - fail: 5

- Key observations:
  - one-file `text_pin_a` stayed `textencoder_illegal`.
  - one-file `text_pin_b` stayed `timeout`.
  - both traced-clip two-file text-pin variants timed out in this run.
  - full-base remained `loader_crash`.

- Key finding:
  - Text identity remains a strong selector for one-file branch behavior; two-file traced-clip signature showed transient drift and required immediate reproducibility recheck.

- Artifacts:
  - summary: `output/custom_alias_resolution_probe_run054_wrapper_ltx23_textpin_20260611/summary.md`
  - results table: `output/custom_alias_resolution_probe_run054_wrapper_ltx23_textpin_20260611/results.tsv`
  - per-case logs: `output/custom_alias_resolution_probe_run054_wrapper_ltx23_textpin_20260611/cases/*.log`

## Run 055 (2026-06-11): Focused Wrapper Recheck

- Goal:
  - Verify whether run054 two-file traced-clip timeout was persistent or transient.

- Command:

```bash
bash tools/run_custom_alias_resolution_probe.sh \
  --matrix ltx23-focused \
  --timeout-sec 75 \
  --tag run055_wrapper_ltx23_focused_recheck_20260611
```

- Probe summary (`run055_wrapper_ltx23_focused_recheck_20260611`):
  - cases: 3
  - pass: 1
  - fail: 2

- Outcome map:
  - `control_trace_noncustom`: PASS
  - `probe_trace_ltx23_text_clip_trace_pin_a`: FAIL (`loader_crash`)
  - `probe_trace_ltx23_text_clip_companion_a_pin_a`: FAIL (`timeout`)

- Key finding:
  - Focused gate reproducibility restored expected split; run054 two-file traced-clip timeout is treated as drift/noise until repeated.

- Artifacts:
  - summary: `output/custom_alias_resolution_probe_run055_wrapper_ltx23_focused_recheck_20260611/summary.md`
  - results table: `output/custom_alias_resolution_probe_run055_wrapper_ltx23_focused_recheck_20260611/results.tsv`
  - per-case logs: `output/custom_alias_resolution_probe_run055_wrapper_ltx23_focused_recheck_20260611/cases/*.log`

## Run 056 (2026-06-11): Loader-Branch Field Isolation Matrix

- Goal:
  - Isolate whether `modifier` and/or `autoencoder` fields activate loader-crash for traced-clip + `text_pin_a` two-file custom entries.

- Harness update:
  - `tools/run_custom_alias_resolution_probe.sh` gained:
    - new matrix: `--matrix ltx23-loaderbranch`
    - new modes:
      - `alias_trace_ltx23_text_clip_trace_pin_a_mod`
      - `alias_trace_ltx23_text_clip_trace_pin_a_auto`
      - `alias_trace_ltx23_text_clip_trace_pin_a_mod_auto`

- Command:

```bash
bash tools/run_custom_alias_resolution_probe.sh \
  --matrix ltx23-loaderbranch \
  --timeout-sec 75 \
  --tag run056_wrapper_ltx23_loaderbranch_20260611
```

- Probe summary (`run056_wrapper_ltx23_loaderbranch_20260611`):
  - cases: 6
  - pass: 1
  - fail: 5

- Outcome map:
  - baseline `text_pin_a + clip=trace` (no modifier, no autoencoder): `timeout`
  - add `modifier=kontext` only: `loader_crash`
  - add `autoencoder` only: `loader_crash`
  - add both: `loader_crash`
  - full-base (`text_pin_b` path): `timeout`

- Key finding:
  - In traced-clip + two-file + `text_pin_a` path, either `modifier` or `autoencoder` is sufficient to flip timeout into loader-crash; full-base still times out due a different field composition (`text_pin_b`).

- Artifacts:
  - summary: `output/custom_alias_resolution_probe_run056_wrapper_ltx23_loaderbranch_20260611/summary.md`
  - results table: `output/custom_alias_resolution_probe_run056_wrapper_ltx23_loaderbranch_20260611/results.tsv`
  - per-case logs: `output/custom_alias_resolution_probe_run056_wrapper_ltx23_loaderbranch_20260611/cases/*.log`

## Run 057 (2026-06-11): Loader-Branch Isolation for Text Pin B

- Goal:
  - Mirror run056 field-isolation on traced-clip two-file path with `text_pin_b` to determine whether the modifier/autoencoder flip is text-identity specific.

- Harness update:
  - `tools/run_custom_alias_resolution_probe.sh` gained:
    - new matrix: `--matrix ltx23-loaderbranch-pinb`
    - new modes:
      - `alias_trace_ltx23_text_clip_trace_pin_b_mod`
      - `alias_trace_ltx23_text_clip_trace_pin_b_auto`
      - `alias_trace_ltx23_text_clip_trace_pin_b_mod_auto`

- Command:

```bash
bash tools/run_custom_alias_resolution_probe.sh \
  --matrix ltx23-loaderbranch-pinb \
  --timeout-sec 75 \
  --tag run057_wrapper_ltx23_loaderbranch_pinb_20260611
```

- Probe summary (`run057_wrapper_ltx23_loaderbranch_pinb_20260611`):
  - cases: 6
  - pass: 1
  - fail: 5

- Outcome map:
  - baseline traced-clip pin-b (no modifier, no autoencoder): `loader_crash`
  - add `modifier=kontext` only: `loader_crash`
  - add `autoencoder` only: `loader_crash`
  - add both: `loader_crash`
  - full-base (`text_pin_b` composition): `loader_crash`

- Key finding:
  - For traced-clip two-file path with `text_pin_b`, loader-crash is already active at baseline and remains unchanged by modifier/autoencoder toggles.
  - This contrasts with run056 `text_pin_a`, where baseline was timeout and modifier/autoencoder activated loader-crash.

- Artifacts:
  - summary: `output/custom_alias_resolution_probe_run057_wrapper_ltx23_loaderbranch_pinb_20260611/summary.md`
  - results table: `output/custom_alias_resolution_probe_run057_wrapper_ltx23_loaderbranch_pinb_20260611/results.tsv`
  - per-case logs: `output/custom_alias_resolution_probe_run057_wrapper_ltx23_loaderbranch_pinb_20260611/cases/*.log`

## Run 058 (2026-06-11): Compact Text-Gate Boundary Matrix

- Goal:
  - Run a compact matrix that toggles only text pin (`A` vs `B`) under matched traced-clip conditions, with and without `modifier+autoencoder` bundle.

- Harness update:
  - `tools/run_custom_alias_resolution_probe.sh` gained matrix:
    - `--matrix ltx23-textgate`
  - Cases:
    - control
    - traced-clip pin-a baseline
    - traced-clip pin-b baseline
    - traced-clip pin-a + mod+auto
    - traced-clip pin-b + mod+auto
    - full-base

- Command:

```bash
bash tools/run_custom_alias_resolution_probe.sh \
  --matrix ltx23-textgate \
  --timeout-sec 75 \
  --tag run058_wrapper_ltx23_textgate_20260611
```

- Probe summary (`run058_wrapper_ltx23_textgate_20260611`):
  - cases: 6
  - pass: 1
  - fail: 5

- Outcome map:
  - pin-a baseline => `loader_crash`
  - pin-b baseline => `timeout`
  - pin-a + mod+auto => `timeout`
  - pin-b + mod+auto => `timeout`
  - full-base => `loader_crash`

- Key finding:
  - Text identity remains a dominant branch gate under traced-clip two-file path, but one additive case (`pin-b + mod+auto`) required reproducibility check.

- Artifacts:
  - summary: `output/custom_alias_resolution_probe_run058_wrapper_ltx23_textgate_20260611/summary.md`
  - results table: `output/custom_alias_resolution_probe_run058_wrapper_ltx23_textgate_20260611/results.tsv`
  - per-case logs: `output/custom_alias_resolution_probe_run058_wrapper_ltx23_textgate_20260611/cases/*.log`

## Run 059 (2026-06-11): Text-Gate Reproducibility Recheck

- Goal:
  - Repeat run058 unchanged to classify stable vs drifting signature cases.

- Command:

```bash
bash tools/run_custom_alias_resolution_probe.sh \
  --matrix ltx23-textgate \
  --timeout-sec 75 \
  --tag run059_wrapper_ltx23_textgate_recheck_20260611
```

- Probe summary (`run059_wrapper_ltx23_textgate_recheck_20260611`):
  - cases: 6
  - pass: 1
  - fail: 5

- Stable cases across run058/run059:
  - pin-a baseline => `loader_crash`
  - pin-b baseline => `timeout`
  - pin-a + mod+auto => `timeout`
  - full-base => `loader_crash`

- Drift case:
  - pin-b + mod+auto switched from `timeout` (run058) to `loader_crash` (run059).

- Key finding:
  - Majority of text-gate signatures are reproducible; one mixed pin-b additive case remains state/noise sensitive and should be treated as non-deterministic until repeated further.

- Artifacts:
  - summary: `output/custom_alias_resolution_probe_run059_wrapper_ltx23_textgate_recheck_20260611/summary.md`
  - results table: `output/custom_alias_resolution_probe_run059_wrapper_ltx23_textgate_recheck_20260611/results.tsv`
  - per-case logs: `output/custom_alias_resolution_probe_run059_wrapper_ltx23_textgate_recheck_20260611/cases/*.log`

## Run 060 (2026-06-11): Pin-B Noise Check (Compact Matrix)

- Goal:
  - Re-test the previously drifting `pin-b + mod+auto` branch using a compact repeat matrix.

- Harness update:
  - `tools/run_custom_alias_resolution_probe.sh` gained matrix:
    - `--matrix ltx23-pinb-noise`
  - Cases:
    - control
    - traced-clip pin-b baseline
    - traced-clip pin-b + mod+auto
    - full-base

- Command:

```bash
bash tools/run_custom_alias_resolution_probe.sh \
  --matrix ltx23-pinb-noise \
  --timeout-sec 75 \
  --tag run060_wrapper_ltx23_pinb_noise_20260611
```

- Probe summary (`run060_wrapper_ltx23_pinb_noise_20260611`):
  - cases: 4
  - pass: 1
  - fail: 3

- Outcome map:
  - pin-b baseline => `loader_crash`
  - pin-b + mod+auto => `loader_crash`
  - full-base => `loader_crash`

- Key finding:
  - No timeout drift in this run; all pin-b traced-clip variants converged to loader-crash.

- Artifacts:
  - summary: `output/custom_alias_resolution_probe_run060_wrapper_ltx23_pinb_noise_20260611/summary.md`
  - results table: `output/custom_alias_resolution_probe_run060_wrapper_ltx23_pinb_noise_20260611/results.tsv`
  - per-case logs: `output/custom_alias_resolution_probe_run060_wrapper_ltx23_pinb_noise_20260611/cases/*.log`

## Run 061 (2026-06-11): Pin-B Noise Check Recheck

- Goal:
  - Repeat run060 unchanged to confirm stabilization.

- Command:

```bash
bash tools/run_custom_alias_resolution_probe.sh \
  --matrix ltx23-pinb-noise \
  --timeout-sec 75 \
  --tag run061_wrapper_ltx23_pinb_noise_20260611
```

- Probe summary (`run061_wrapper_ltx23_pinb_noise_20260611`):
  - cases: 4
  - pass: 1
  - fail: 3

- Outcome map:
  - pin-b baseline => `loader_crash`
  - pin-b + mod+auto => `loader_crash`
  - full-base => `loader_crash`

- Key finding:
  - Reproduces run060 exactly; pin-b + mod+auto now appears stabilized in loader-crash branch (run058 timeout treated as outlier/noise).

- Artifacts:
  - summary: `output/custom_alias_resolution_probe_run061_wrapper_ltx23_pinb_noise_20260611/summary.md`
  - results table: `output/custom_alias_resolution_probe_run061_wrapper_ltx23_pinb_noise_20260611/results.tsv`
  - per-case logs: `output/custom_alias_resolution_probe_run061_wrapper_ltx23_pinb_noise_20260611/cases/*.log`

## Run 062 (2026-06-11): Pin-B Noise Matrix with Trace Env (Wrapper Runtime)

- Goal:
  - Re-run the compact stabilized pin-b matrix with `DT_LTX23_TRACE=1` after adding source-level trace instrumentation.

- Command:

```bash
DT_LTX23_TRACE=1 tools/run_custom_alias_resolution_probe.sh \
  --matrix ltx23-pinb-noise \
  --tag run062_ltx23_pinb_noise_trace
```

- Probe summary (`run062_ltx23_pinb_noise_trace`):
  - cases: 4
  - pass: 1
  - fail: 3

- Outcome map:
  - pin-b baseline => `loader_crash`
  - pin-b + mod+auto => `loader_crash`
  - full-base => `loader_crash`

- Trace check:
  - No `DT_LTX23_TRACE` lines were emitted in run artifacts.
  - Runtime resolution check showed wrapper path still points to deployed binaries:
    - `/usr/local/bin/drawthings-grpc`
    - `/usr/local/bin/gRPCServerCLI`

- Key finding:
  - Wrapper branch behavior remains stable and reproducible, but newly added source markers are not visible on this runtime path.

- Artifacts:
  - summary: `output/custom_alias_resolution_probe_run062_ltx23_pinb_noise_trace/summary.md`
  - results table: `output/custom_alias_resolution_probe_run062_ltx23_pinb_noise_trace/results.tsv`
  - per-case logs: `output/custom_alias_resolution_probe_run062_ltx23_pinb_noise_trace/cases/*.log`

## Run 063 (2026-06-11): Source-Built Trace Smoke (PATH Override)

- Goal:
  - Force source-built `gRPCServerCLI` onto PATH to verify that newly added trace markers are active at runtime.

- Command:

```bash
PATH=/workspaces/drawthings-linux-toolkit/draw-things-community/.build/release:$PATH \
DT_LTX23_TRACE=1 tools/run_q6p_canary_once.sh \
  --model 10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt \
  --timeout-sec 75 \
  --max-responses 0 \
  --require-complete-stream \
  --require-final-output \
  --final-mode \
  --tag run062_source_trace_smoke
```

- Canary result (`run062_source_trace_smoke`):
  - `canary_rc=1`
  - `post_echo_rc=1`
  - client error: `UNAVAILABLE: Socket closed`
  - server terminated with abort/core dump (`libc` stack tail)

- Key finding:
  - Source-built runtime remains backend-confounded/unstable and aborts before useful source-level divergence traces can be harvested in this path.

- Artifacts:
  - client log: `output/q6p_canary_run062_source_trace_smoke/client.log`
  - server log: `output/q6p_canary_run062_source_trace_smoke/server.log`

## Run 064 (2026-06-11): Wrapper Runtime Selector Smoke

- Goal:
  - Validate new explicit runtime selector (`--grpc-bin`) on known-stable wrapper path without PATH overrides.

- Command:

```bash
DT_LTX23_TRACE=1 tools/run_q6p_canary_once.sh \
  --grpc-bin /usr/local/bin/drawthings-grpc \
  --model 10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt \
  --timeout-sec 45 \
  --max-responses 1 \
  --tag run064_wrapper_bin_selector_smoke
```

- Result:
  - `canary_rc=0`, `post_echo_rc=0`, `RESULT=PASS`
  - streamed signpost reached (`response #1`, `textEncoded`) under explicit wrapper binary selection.

- Key finding:
  - Runtime-path control is now explicit and reproducible at script level (no PATH hack required).

- Artifacts:
  - client log: `output/q6p_canary_run064_wrapper_bin_selector_smoke/client.log`
  - server log: `output/q6p_canary_run064_wrapper_bin_selector_smoke/server.log`

## Run 065 (2026-06-11): Source Runtime Selector Smoke

- Goal:
  - Validate `--grpc-bin` against source-built `gRPCServerCLI` and re-check current failure signature using explicit selection.

- Command:

```bash
DT_LTX23_TRACE=1 tools/run_q6p_canary_once.sh \
  --grpc-bin /workspaces/drawthings-linux-toolkit/draw-things-community/.build/release/gRPCServerCLI \
  --model 10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt \
  --timeout-sec 45 \
  --max-responses 1 \
  --tag run065_source_bin_selector_smoke \
  --soft-fail
```

- Result:
  - `canary_rc=1`, `post_echo_rc=1`, client `UNAVAILABLE: Socket closed`
  - server abort/core dump with `libc` crash tail (same source-build confound class).

- Key finding:
  - Explicit selector confirms source-built Linux runtime instability is unchanged; failures are not due to PATH/environment selection ambiguity.

- Artifacts:
  - client log: `output/q6p_canary_run065_source_bin_selector_smoke/client.log`
  - server log: `output/q6p_canary_run065_source_bin_selector_smoke/server.log`

## Run 066 (2026-06-11): Focused Matrix via Explicit Wrapper Selector

- Goal:
  - Exercise the real probe matrix path with explicit wrapper runtime selection (`--grpc-bin`) and verify branch-map stability.

- Command:

```bash
DT_LTX23_TRACE=1 tools/run_custom_alias_resolution_probe.sh \
  --grpc-bin /usr/local/bin/drawthings-grpc \
  --matrix ltx23-focused \
  --tag run066_wrapper_selector_focused_trace
```

- Probe summary (`run066_wrapper_selector_focused_trace`):
  - cases: 3
  - pass: 1
  - fail: 2

- Outcome map:
  - control => `pass`
  - traced-clip pin-a => `loader_crash`
  - companion-a pin-a => `timeout`

- Trace check:
  - No `DT_LTX23_TRACE` markers were present in run logs on wrapper runtime path.

- Key finding:
  - Probe-level binary selector works in full matrix flow, and branch mapping remains consistent with prior focused baseline while wrapper logs remain non-instrumented.

- Artifacts:
  - summary: `output/custom_alias_resolution_probe_run066_wrapper_selector_focused_trace/summary.md`
  - results table: `output/custom_alias_resolution_probe_run066_wrapper_selector_focused_trace/results.tsv`
  - per-case logs: `output/custom_alias_resolution_probe_run066_wrapper_selector_focused_trace/cases/*.log`

## Run 067 (2026-06-11): Source Runtime Tuning Sweep + Key-Path Discriminator

- Goal:
  - Test whether source-build abort behavior can be changed by server launch flags.
  - Compare source trace path for unresolved trace-key requests vs mapped model-key requests.

- Harness update used:
  - `tools/run_q6p_canary_once.sh` now supports:
    - `--server-gpu`
    - `--server-cpu-offload`
    - `--server-no-flash-attention`
    - `--server-weights-cache`
  - `tools/run_custom_alias_resolution_probe.sh` forwards these server options.

- Sweep commands (source runtime, same trace021 model file key):
  - `run067a_source_default`
  - `run067b_source_cpuoffload`
  - `run067c_source_noflash`
  - `run067d_source_wcache1`
  - `run067e_source_combo`

- Sweep result:
  - all five cases failed identically: `canary_rc=1`, `post_echo_rc=1`, `RESULT=FAIL canary rc=1`
  - tuning flags did not change abort class for trace021 file-key path.

- Follow-up discriminator runs:
  - `run067f_source_alias_model` (`--model 10_e_v1`) => timeout branch (`canary_rc=124`, `post_echo_rc=0`)
  - `run067g_source_mapped_file` (`--model 10_e_v1_bf16_regen_0_q6p.ckpt`) => timeout branch (`canary_rc=124`, `post_echo_rc=0`)

- Key finding:
  - Source traces now show a deterministic split by model-resolution path:
    - unresolved trace key path logs repeated `ModelZoo.specificationForModel ... source=miss` then abort class
    - mapped key path (`source=mapping`) advances into:
      - `LocalImageGenerator.textEncoderFiles`
      - `LocalImageGenerator.modelContext`
      - `LocalImageGenerator.textEncoderInit`
      - `encodeLTX2.begin`
      - `encodeLTX2.loading_text_model`
      then stalls/timeouts.

- Artifacts:
  - `output/q6p_canary_run067a_source_default/{client.log,server.log}`
  - `output/q6p_canary_run067b_source_cpuoffload/{client.log,server.log}`
  - `output/q6p_canary_run067c_source_noflash/{client.log,server.log}`
  - `output/q6p_canary_run067d_source_wcache1/{client.log,server.log}`
  - `output/q6p_canary_run067e_source_combo/{client.log,server.log}`
  - `output/q6p_canary_run067f_source_alias_model/{client.log,server.log}`
  - `output/q6p_canary_run067g_source_mapped_file/{client.log,server.log}`

## Run 068 (2026-06-11): Focused Matrix via Explicit Source Selector

- Goal:
  - Re-run focused probe matrix on source-selected runtime to compare control miss-path vs alias-mapped paths under one harness.

- Command:

```bash
DT_LTX23_TRACE=1 tools/run_custom_alias_resolution_probe.sh \
  --grpc-bin /workspaces/drawthings-linux-toolkit/draw-things-community/.build/release/gRPCServerCLI \
  --matrix ltx23-focused \
  --timeout-sec 45 \
  --tag run068_source_selector_focused_trace
```

- Probe summary (`run068_source_selector_focused_trace`):
  - cases: 3
  - pass: 0
  - fail: 3

- Outcome map:
  - control trace noncustom => `canary_rc=1`, `post_echo_rc=1`, signature `unknown`
  - traced-clip pin-a alias => timeout (`canary_rc=124`, `post_echo_rc=124`)
  - companion-a pin-a alias => timeout (`canary_rc=124`, `post_echo_rc=124`)

- Key finding:
  - Under source runtime, focused control still aborts while alias-driven two-file branches shift to timeout class, consistent with run067 miss-vs-mapping split.

- Artifacts:
  - summary: `output/custom_alias_resolution_probe_run068_source_selector_focused_trace/summary.md`
  - results table: `output/custom_alias_resolution_probe_run068_source_selector_focused_trace/results.tsv`
  - per-case logs: `output/custom_alias_resolution_probe_run068_source_selector_focused_trace/cases/*.log`

## Run 069 (2026-06-11): Targeted Same-Arg Source A/B (Trace-Key Mapping Toggle)

- Goal:
  - Validate whether mapping alone flips source-runtime behavior for the same trace021 model argument.

- Method:
  - Keep `--model 10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt` constant in both cases.
  - Case A (`run069c`): baseline `dt-models/custom.json` (no trace-key mapping entry).
  - Case B (`run069d`): temporarily inject one explicit custom entry mapping trace key, then rerun same command.
  - Runtime fixed to source binary selector: `--grpc-bin /workspaces/drawthings-linux-toolkit/draw-things-community/.build/release/gRPCServerCLI`.

- Commands:

```bash
# Case A: unresolved baseline
DT_LTX23_TRACE=1 bash tools/run_q6p_canary_once.sh \
  --grpc-bin /workspaces/drawthings-linux-toolkit/draw-things-community/.build/release/gRPCServerCLI \
  --model 10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt \
  --final-mode --require-complete-stream --require-final-output --max-responses 0 \
  --timeout-sec 75 --soft-fail --tag run069c_source_trace_unresolved

# Case B: same model arg, with temporary trace-key mapping entry present
DT_LTX23_TRACE=1 bash tools/run_q6p_canary_once.sh \
  --grpc-bin /workspaces/drawthings-linux-toolkit/draw-things-community/.build/release/gRPCServerCLI \
  --model 10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt \
  --final-mode --require-complete-stream --require-final-output --max-responses 0 \
  --timeout-sec 75 --soft-fail --tag run069d_source_trace_mapped_filearg
```

- Results:
  - `run069c_source_trace_unresolved`: `canary_rc=1`, `post_echo_rc=1`, abort/core-dump class.
  - `run069d_source_trace_mapped_filearg`: `canary_rc=124`, `post_echo_rc=124`, timeout class.

- Trace discriminator:
  - unresolved case: repeated `ModelZoo.specificationForModel ... source=miss`.
  - mapped case: `source=mapping` with winner `probe_trace021_map_only_run069`, then progression into:
    - `LocalImageGenerator.textEncoderInit`
    - `LocalImageGenerator.beforeTextEncode`
    - `encodeLTX2.begin`
    - `encodeLTX2.loading_text_model`

- Key finding:
  - With model argument held constant, adding/removing trace-key mapping entry alone flips source runtime from miss/abort branch to mapping/timeout branch.

- Artifacts:
  - case A dir: `output/q6p_canary_run069c_source_trace_unresolved`
  - case B dir: `output/q6p_canary_run069d_source_trace_mapped_filearg`
  - captured console summaries: `output/run069c_console.log`, `output/run069d_console.log`

## Run 070 (2026-06-12): Same-Arg Source A/B with Minimal-v1 Mapping Entry

- Goal:
  - Test whether mapping presence alone is sufficient when the mapping entry is minimal (`version=v1`) rather than LTX-shaped.

- Method:
  - Keep request constant in both cases:
    - `--model 10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt`
    - `--grpc-bin /workspaces/drawthings-linux-toolkit/draw-things-community/.build/release/gRPCServerCLI`
  - Case A (`run070a`): baseline `custom.json` (no trace-key mapping entry).
  - Case B (`run070b`): inject temporary minimal mapping entry:
    - `name=probe_trace021_map_minv1_run070`
    - `file=trace021 key`, `prefix=""`, `version="v1"`, `upcast_attention=false`, `default_scale=8`

- Results:
  - `run070a_source_trace_unresolved`: `canary_rc=1`, `post_echo_rc=1` (abort class).
  - `run070b_source_trace_mapped_minv1`: `canary_rc=1`, `post_echo_rc=1` (abort class).

- Trace discriminator:
  - unresolved case: repeated `source=miss`.
  - minimal-v1 mapped case: repeated `source=mapping` with winner `probe_trace021_map_minv1_run070`.
  - Unlike run069 mapped-LTX case, run070 mapped-minv1 case did **not** progress into `LocalImageGenerator.textEncoderInit` / `encodeLTX2.*` before abort.

- Key finding:
  - Mapping presence alone is not sufficient to induce the mapping-timeout branch.
  - Entry shape matters: LTX-style mapped entries can shift to deeper path/timeout (run069), while minimal-v1 mapped entry still aborts.

- Artifacts:
  - case A dir: `output/q6p_canary_run070a_source_trace_unresolved`
  - case B dir: `output/q6p_canary_run070b_source_trace_mapped_minv1`
  - captured console summaries: `output/run070a_console.log`, `output/run070b_console.log`

## Run 071 (2026-06-12): Same-Arg Source Probe with LTX2.3-Minimal Mapping Entry

- Goal:
  - Bridge run069 vs run070 by testing a mapped entry with `version=ltx2.3` but without explicit text/clip/auto fields.

- Method:
  - Same request and source binary as run069/run070:
    - `--model 10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt`
    - `--grpc-bin /workspaces/drawthings-linux-toolkit/draw-things-community/.build/release/gRPCServerCLI`
  - Temporary mapped entry:
    - `name=probe_trace021_map_ltx23min_run071`
    - `file=trace021 key`, `prefix=""`, `version="ltx2.3"`, `upcast_attention=false`, `default_scale=1`

- Result:
  - `run071_source_trace_mapped_ltx23min`: timeout class (`canary_rc=124`, `post_echo_rc=124`).

- Trace discriminator:
  - repeated `source=mapping` winner `probe_trace021_map_ltx23min_run071`.
  - progressed into:
    - `LocalImageGenerator.textEncoderInit` with `resolvedFilePaths=["clip_vit_l14_f16.ckpt"]`
    - `LocalImageGenerator.beforeTextEncode`
    - `encodeLTX2.begin` (`version=ltx2_3`, `modifier=none`)
    - `encodeLTX2.loading_text_model path=clip_vit_l14_f16.ckpt`

- Key finding:
  - `version=ltx2.3` is sufficient to leave the early-abort behavior seen in mapped minimal-v1 (run070) and re-enter deeper encode path/timeout class.
  - This supports schema-dependent branching where LTX-version activation is a primary gate.

- Artifacts:
  - case dir: `output/q6p_canary_run071_source_trace_mapped_ltx23min`
  - captured console summary: `output/run071_console.log`

## Run 072 (2026-06-12): Source Field Ladder (`ltx23_min -> +text -> +clip -> +auto`)

- Goal:
  - Identify the minimum ltx2.3 custom-entry field set that changes source-runtime branch behavior for the same request key.

- Method:
  - Fixed request and runtime across all cases:
    - `--model 10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt`
    - `--grpc-bin /workspaces/drawthings-linux-toolkit/draw-things-community/.build/release/gRPCServerCLI`
    - strict canary gates (`--final-mode --require-complete-stream --require-final-output --max-responses 0 --timeout-sec 75 --soft-fail`)
  - Ladder cases:
    - `run0720_source_trace_control_unresolved`: no temporary mapping entry
    - `run0721_source_trace_mapped_ltx23_min`: `version=ltx2.3`, minimal mapped entry
    - `run0722_source_trace_mapped_ltx23_text`: +`text_encoder=gemma_3_12b_it_qat_q8p.ckpt`
    - `run0723_source_trace_mapped_ltx23_text_clip`: +`clip_encoder=10_e_v1_bf16_regen_0_q6p.ckpt`
    - `run0724_source_trace_mapped_ltx23_text_clip_auto`: +`autoencoder=ltx_2.3_audio_video_vae_f16.ckpt`

- Ladder summary (canonical artifact):
  - `output/run072_ladder_summary.tsv`
  - `run0720`: `canary_rc=1`, `post_echo_rc=1`, `source=miss`, `text_init=0`, `encode=0/0`
  - `run0721`: `canary_rc=124`, `post_echo_rc=124`, `source=mapping`, `text_init=1`, `encode=1/1`
  - `run0722`: `canary_rc=124`, `post_echo_rc=124`, `source=mapping`, `text_init=1`, `encode=1/1`
  - `run0723`: `canary_rc=124`, `post_echo_rc=124`, `source=mapping`, `text_init=1`, `encode=1/1`
  - `run0724`: `canary_rc=124`, `post_echo_rc=0`, `source=mapping`, `text_init=1`, `encode=1/1`
  - provenance note: `run0722` result codes are retained from the completed strict run before later trace-only rerun interruption; trace markers were refreshed, but interrupted reruns should not replace canonical rc/post-echo fields.

- Key findings:
  - `run0720 -> run0721` is the principal branch transition:
    - unresolved control stays miss/abort class.
    - mapped ltx2.3-min immediately enters mapping + deep encode timeout class.
  - Additional fields (`+text`, `+clip`, `+auto`) did not revert the deep-encode branch once `version=ltx2.3` mapping was active.
  - `+auto` changed post-echo behavior (`124 -> 0`) while preserving timeout outcome, indicating downstream-stage modulation rather than branch reversion.

- Artifacts:
  - case dirs:
    - `output/q6p_canary_run0720_source_trace_control_unresolved`
    - `output/q6p_canary_run0721_source_trace_mapped_ltx23_min`
    - `output/q6p_canary_run0722_source_trace_mapped_ltx23_text`
    - `output/q6p_canary_run0723_source_trace_mapped_ltx23_text_clip`
    - `output/q6p_canary_run0724_source_trace_mapped_ltx23_text_clip_auto`
  - console logs:
    - `output/run0720_source_trace_control_unresolved_console.log`
    - `output/run0721_source_trace_mapped_ltx23_min_console.log`
    - `output/run0722_source_trace_mapped_ltx23_text_console.log`
    - `output/run0723_source_trace_mapped_ltx23_text_clip_console.log`
    - `output/run0724_source_trace_mapped_ltx23_text_clip_auto_console.log`

## Run 073 (2026-06-12): Modulation Check (`+auto` vs `+modifier` vs combined)

- Goal:
  - Resolve whether the `run072` `post_echo_rc` flip (`124 -> 0`) is specific to `autoencoder` or reflects broader downstream-field modulation.

- Method:
  - Fixed request, runtime, and strict canary gates across all cases:
    - `--model 10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt`
    - `--grpc-bin /workspaces/drawthings-linux-toolkit/draw-things-community/.build/release/gRPCServerCLI`
    - strict canary gates (`--final-mode --require-complete-stream --require-final-output --max-responses 0 --timeout-sec 75 --soft-fail`)
  - Temporary mapped probe entry base held:
    - `version=ltx2.3`, `text_encoder=gemma_3_12b_it_qat_q8p.ckpt`, `clip_encoder=10_e_v1_bf16_regen_0_q6p.ckpt`
  - Matrix cases:
    - `run0731_source_trace_mapped_ltx23_text_clip_base`: base (`text+clip`)
    - `run0732_source_trace_mapped_ltx23_text_clip_auto`: base + `autoencoder=ltx_2.3_audio_video_vae_f16.ckpt`
    - `run0733_source_trace_mapped_ltx23_text_clip_mod`: base + `modifier=kontext`
    - `run0734_source_trace_mapped_ltx23_text_clip_mod_auto`: base + `modifier=kontext` + `autoencoder`

- Matrix summary (canonical artifact):
  - `output/run073_modulation_summary.tsv`
  - `run0731` base: `canary_rc=124`, `post_echo_rc=124`, `source=mapping`, `text_init=1`, `encode=1/1`
  - `run0732` +auto: `canary_rc=124`, `post_echo_rc=0`, `source=mapping`, `text_init=1`, `encode=1/1`
  - `run0733` +modifier: `canary_rc=124`, `post_echo_rc=0`, `source=mapping`, `text_init=1`, `encode=1/1`
  - `run0734` +modifier+auto: `canary_rc=124`, `post_echo_rc=124`, `source=mapping`, `text_init=1`, `encode=1/1`

- Key findings:
  - All four cases stayed in the same mapping + deep-encode timeout class (`canary_rc=124`, `source=mapping`, `encode=1/1`).
  - `auto` and `modifier` each independently shifted `post_echo_rc` to `0` on this base.
  - Combining `auto+modifier` returned `post_echo_rc` to `124`, indicating non-additive downstream interaction rather than a monotonic single-field effect.
  - Branch-gate conclusion from run071/072 remains unchanged: `version=ltx2.3` is the primary gate into deep encode path; these fields tune downstream behavior only.

- Artifacts:
  - case dirs:
    - `output/q6p_canary_run0731_source_trace_mapped_ltx23_text_clip_base`
    - `output/q6p_canary_run0732_source_trace_mapped_ltx23_text_clip_auto`
    - `output/q6p_canary_run0733_source_trace_mapped_ltx23_text_clip_mod`
    - `output/q6p_canary_run0734_source_trace_mapped_ltx23_text_clip_mod_auto`
  - console logs:
    - `output/run0731_source_trace_mapped_ltx23_text_clip_base_console.log`
    - `output/run0732_source_trace_mapped_ltx23_text_clip_auto_console.log`
    - `output/run0733_source_trace_mapped_ltx23_text_clip_mod_console.log`
    - `output/run0734_source_trace_mapped_ltx23_text_clip_mod_auto_console.log`

## Run 074 (2026-06-29): Repro Check (`+auto`, `+modifier`, `+modifier+auto`) with 2x repeats

- Goal:
  - Test reproducibility of `run073` post-echo modulation behavior with low-footprint repeats and no large artifact growth.

- Method:
  - Fixed request, runtime, and strict canary gates across all six cases:
    - `--model 10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt`
    - `--grpc-bin /workspaces/drawthings-linux-toolkit/draw-things-community/.build/release/gRPCServerCLI`
    - strict canary gates (`--final-mode --require-complete-stream --require-final-output --max-responses 0 --timeout-sec 75 --soft-fail`)
  - Temporary mapped probe entry base held:
    - `version=ltx2.3`, `text_encoder=gemma_3_12b_it_qat_q8p.ckpt`, `clip_encoder=10_e_v1_bf16_regen_0_q6p.ckpt`
  - Repeated variants:
    - `run0741` / `run0742`: `+auto`
    - `run0743` / `run0744`: `+modifier`
    - `run0745` / `run0746`: `+modifier+auto`

- Matrix summary (canonical artifact):
  - `output/run074_repro_summary.tsv`
  - `run0741` `+auto` r1: `canary_rc=124`, `post_echo_rc=0`, `source=mapping`, `text_init=1`, `encode=1/1`
  - `run0742` `+auto` r2: `canary_rc=124`, `post_echo_rc=0`, `source=mapping`, `text_init=1`, `encode=1/1`
  - `run0743` `+modifier` r1: `canary_rc=124`, `post_echo_rc=0`, `source=mapping`, `text_init=1`, `encode=1/1`
  - `run0744` `+modifier` r2: `canary_rc=124`, `post_echo_rc=0`, `source=mapping`, `text_init=1`, `encode=1/1`
  - `run0745` `+modifier+auto` r1: `canary_rc=124`, `post_echo_rc=0`, `source=mapping`, `text_init=1`, `encode=1/1`
  - `run0746` `+modifier+auto` r2: `canary_rc=124`, `post_echo_rc=124`, `source=mapping`, `text_init=1`, `encode=1/1`

- Key findings:
  - All six repeats stayed in the same mapping + deep-encode timeout class (`canary_rc=124`, `source=mapping`, `encode=1/1`).
  - `+auto` and `+modifier` were stable across repeats at `post_echo_rc=0` (2/2 each).
  - `+modifier+auto` showed mixed post-echo behavior (1x `0`, 1x `124`), so the combined-field effect is reproducible as a variability surface, not a fixed single outcome.
  - This reinforces branch-gate separation: `version=ltx2.3` controls deep-path entry; post-echo behavior remains downstream and state-sensitive.

- Artifacts:
  - case dirs:
    - `output/q6p_canary_run0741_source_trace_mapped_ltx23_text_clip_auto_r1`
    - `output/q6p_canary_run0742_source_trace_mapped_ltx23_text_clip_auto_r2`
    - `output/q6p_canary_run0743_source_trace_mapped_ltx23_text_clip_mod_r1`
    - `output/q6p_canary_run0744_source_trace_mapped_ltx23_text_clip_mod_r2`
    - `output/q6p_canary_run0745_source_trace_mapped_ltx23_text_clip_mod_auto_r1`
    - `output/q6p_canary_run0746_source_trace_mapped_ltx23_text_clip_mod_auto_r2`
  - console logs:
    - `output/run0741_source_trace_mapped_ltx23_text_clip_auto_r1_console.log`
    - `output/run0742_source_trace_mapped_ltx23_text_clip_auto_r2_console.log`
    - `output/run0743_source_trace_mapped_ltx23_text_clip_mod_r1_console.log`
    - `output/run0744_source_trace_mapped_ltx23_text_clip_mod_r2_console.log`
    - `output/run0745_source_trace_mapped_ltx23_text_clip_mod_auto_r1_console.log`
    - `output/run0746_source_trace_mapped_ltx23_text_clip_mod_auto_r2_console.log`

## Run 075 (2026-06-29): Focused Repeat Sweep (`+modifier+auto`, 8x)

- Goal:
  - Quantify flip-rate of `post_echo_rc` for the highest-variance branch (`+modifier+auto`) under fixed ltx2.3 mapped conditions.

- Method:
  - Fixed request, runtime, and strict canary gates for all 8 repeats:
    - `--model 10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt`
    - `--grpc-bin /workspaces/drawthings-linux-toolkit/draw-things-community/.build/release/gRPCServerCLI`
    - strict canary gates (`--final-mode --require-complete-stream --require-final-output --max-responses 0 --timeout-sec 75 --soft-fail`)
  - Temporary mapped probe entry was fixed to:
    - `version=ltx2.3`, `text_encoder=gemma_3_12b_it_qat_q8p.ckpt`, `clip_encoder=10_e_v1_bf16_regen_0_q6p.ckpt`, `modifier=kontext`, `autoencoder=ltx_2.3_audio_video_vae_f16.ckpt`

- Repeat summary (canonical artifacts):
  - `output/run075_mod_auto_repeats_summary.tsv`
  - `output/run075_mod_auto_repeats_aggregate.txt`
  - Per-repeat outcomes:
    - `r1`: `canary_rc=124`, `post_echo_rc=0`
    - `r2`: `canary_rc=124`, `post_echo_rc=0`
    - `r3`: `canary_rc=124`, `post_echo_rc=0`
    - `r4`: `canary_rc=124`, `post_echo_rc=124`
    - `r5`: `canary_rc=124`, `post_echo_rc=0`
    - `r6`: `canary_rc=124`, `post_echo_rc=0`
    - `r7`: `canary_rc=124`, `post_echo_rc=0`
    - `r8`: `canary_rc=124`, `post_echo_rc=0`
  - Aggregate counts:
    - `post_echo_rc=0`: `7/8`
    - `post_echo_rc=124`: `1/8`
    - `post_echo_rc=other`: `0/8`

- Key findings:
  - Deep-encode timeout branch was stable for all repeats (`canary_rc=124`, `source=mapping`, `encode=1/1`).
  - `+modifier+auto` post-echo behavior is skewed toward `0` but not deterministic.
  - This confirms a state-sensitive downstream variability surface layered on top of a stable ltx2.3 branch gate.

- Artifacts:
  - case dirs:
    - `output/q6p_canary_run0751_source_trace_mapped_ltx23_text_clip_mod_auto_r1`
    - `output/q6p_canary_run0752_source_trace_mapped_ltx23_text_clip_mod_auto_r2`
    - `output/q6p_canary_run0753_source_trace_mapped_ltx23_text_clip_mod_auto_r3`
    - `output/q6p_canary_run0754_source_trace_mapped_ltx23_text_clip_mod_auto_r4`
    - `output/q6p_canary_run0755_source_trace_mapped_ltx23_text_clip_mod_auto_r5`
    - `output/q6p_canary_run0756_source_trace_mapped_ltx23_text_clip_mod_auto_r6`
    - `output/q6p_canary_run0757_source_trace_mapped_ltx23_text_clip_mod_auto_r7`
    - `output/q6p_canary_run0758_source_trace_mapped_ltx23_text_clip_mod_auto_r8`
  - console logs:
    - `output/run0751_source_trace_mapped_ltx23_text_clip_mod_auto_r1_console.log`
    - `output/run0752_source_trace_mapped_ltx23_text_clip_mod_auto_r2_console.log`
    - `output/run0753_source_trace_mapped_ltx23_text_clip_mod_auto_r3_console.log`
    - `output/run0754_source_trace_mapped_ltx23_text_clip_mod_auto_r4_console.log`
    - `output/run0755_source_trace_mapped_ltx23_text_clip_mod_auto_r5_console.log`
    - `output/run0756_source_trace_mapped_ltx23_text_clip_mod_auto_r6_console.log`
    - `output/run0757_source_trace_mapped_ltx23_text_clip_mod_auto_r7_console.log`
    - `output/run0758_source_trace_mapped_ltx23_text_clip_mod_auto_r8_console.log`

## Run 076 (2026-06-29): Warm-Server Repeat Probe (`+modifier+auto`, 8x)

- Goal:
  - Test whether `post_echo_rc` variability for `+modifier+auto` persists when requests are repeated against one long-lived source-runtime server process (no per-repeat restart).

- Method:
  - One `gRPCServerCLI` process was launched once and reused for all repeats.
  - Fixed request/config across repeats (`256x256`, seed `4242`, steps `4`, same model key).
  - Reusable runner now committed for future replay: `tools/run_q6p_warm_server_mod_auto_repeats.sh`.
  - Temporary mapped entry was injected for request key `10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt` with:
    - `version=ltx2.3`, `text_encoder=gemma_3_12b_it_qat_q8p.ckpt`, `clip_encoder=10_e_v1_bf16_regen_0_q6p.ckpt`, `modifier=kontext`, `autoencoder=ltx_2.3_audio_video_vae_f16.ckpt`
  - Alias was restored out of `dt-models/custom.json` on cleanup.

- Repeat summary (canonical artifacts):
  - `output/run076_warm_server_mod_auto_repeats_summary.tsv`
  - `output/run076_warm_server_mod_auto_repeats_aggregate.txt`
  - `r1`: `canary_rc=124`, `post_echo_rc=124`, `source=mapping`, `text_init=1`, `encode_begin=1`, `encode_loading=1`
  - `r2..r8`: `canary_rc=1`, `post_echo_rc=124`, `source=unknown`, `text_init=0`, `encode_begin=0`, `encode_loading=0`
  - aggregate:
    - `canary_rc=124`: `1/8`
    - `post_echo_rc=124`: `1/8`
    - `post_echo_rc=0`: `0/8`

- Failure evidence captured during run:
  - Source runtime crashed after first repeat with assertion:
    - `ccv_nnc_symbolic_graph_compile.c:1365` -> `Assertion \`memory_type == CCV_TENSOR_CPU_MEMORY\` failed.`
    - crash marker: `*** Program crashed: Aborted ...`
    - see `output/run076_warm_server_mod_auto/server.log`
  - Subsequent client requests (`r2..r8`) failed with connection errors:
    - `gRPC error: UNAVAILABLE ... connection attempt timed out before receiving SETTINGS frame`
    - see `output/run0762_warm_server_mod_auto_r2_client.log` through `output/run0768_warm_server_mod_auto_r8_client.log`

- Key findings:
  - This run did not produce a clean warm-state `post_echo_rc` flip-rate estimate because server stability failed after `r1`.
  - New high-signal observation: repeated warm-process operation on this mapped `ltx2.3 + modifier + autoencoder` setup can trigger a source-runtime crash path distinct from the earlier per-run timeout/variability behavior.
  - Existing run075 interpretation remains valid for cold-start repeats; run076 adds a separate warm-runtime crash confound to investigate.

- Artifacts:
  - `output/run076_warm_server_mod_auto_console.log`
  - `output/run076_warm_server_mod_auto/server.log`
  - `output/run076_warm_server_mod_auto_repeats_summary.tsv`
  - `output/run076_warm_server_mod_auto_repeats_aggregate.txt`

## Run 077 (2026-06-29): Warm-Server Control (`text+clip`, no modifier/autoencoder)

- Goal:
  - Test whether the run076 warm-process crash confound is specific to `+modifier+auto` fields or persists under a control mapped entry without those fields.

- Method:
  - Reused committed runner with control flags:
    - `tools/run_q6p_warm_server_mod_auto_repeats.sh --tag run077_warm_server_text_clip_only --repeats 4 --timeout-sec 75 --modifier none --autoencoder none`
  - Same request key and persistent source runtime pattern as run076.
  - Mapped entry kept `version=ltx2.3`, `text_encoder=gemma_3_12b_it_qat_q8p.ckpt`, `clip_encoder=10_e_v1_bf16_regen_0_q6p.ckpt`, while omitting `modifier` and `autoencoder`.

- Repeat summary (canonical artifacts):
  - `output/run077_warm_server_text_clip_only_repeats_summary.tsv`
  - `output/run077_warm_server_text_clip_only_repeats_aggregate.txt`
  - `r1`: `canary_rc=124`, `post_echo_rc=0`, `source=mapping`, `text_init=1`, `encode_begin=1`, `encode_loading=1`
  - `r2`: `canary_rc=124`, `post_echo_rc=124`, `source=unknown`, no encode markers
  - `r3`: `canary_rc=1`, `post_echo_rc=124`, `source=unknown`, no encode markers
  - `r4`: `canary_rc=1`, `post_echo_rc=124`, `source=unknown`, no encode markers
  - aggregate:
    - `canary_rc=124`: `2/4`
    - `post_echo_rc=0`: `1/4`
    - `post_echo_rc=124`: `3/4`

- Failure evidence captured during run:
  - Source runtime assertion and abort again observed in control run:
    - `ccv_nnc_symbolic_graph_compile.c:1365` -> `Assertion \`memory_type == CCV_TENSOR_CPU_MEMORY\` failed.`
    - crash marker: `*** Program crashed: Aborted ...`
    - see `output/run077_warm_server_text_clip_only/server.log`
  - Post-crash client failures (`r3`, `r4`) show:
    - `gRPC error: UNAVAILABLE ... connection attempt timed out before receiving SETTINGS frame`
    - see `output/run077_warm_server_text_clip_only_r3_client.log` and `output/run077_warm_server_text_clip_only_r4_client.log`

- Key findings:
  - Warm-process crash confound persists even when `modifier` and `autoencoder` are removed from the mapped entry.
  - This weakens the hypothesis that run076 crash behavior is caused specifically by combined `+modifier+auto` fields.
  - Updated working interpretation: warm-process instability is likely tied to repeated-request lifecycle under this mapped ltx2.3 path more broadly, while cold-start repeat behavior (run075) remains a separate signal.

- Artifacts:
  - `output/run077_warm_server_text_clip_only_console.log`
  - `output/run077_warm_server_text_clip_only/server.log`
  - `output/run077_warm_server_text_clip_only_repeats_summary.tsv`
  - `output/run077_warm_server_text_clip_only_repeats_aggregate.txt`

## Run 078 (2026-06-29): Warm-Server Minimal Mapping Control (`version=ltx2.3` only)

- Goal:
  - Test whether warm-process crash behavior persists when all optional mapped fields are removed and only `version=ltx2.3` is supplied in the temporary mapped entry.

- Method:
  - Reused committed runner with minimal-entry control flags:
    - `tools/run_q6p_warm_server_mod_auto_repeats.sh --tag run078_warm_server_ltx23_min --repeats 4 --timeout-sec 75 --entry-version ltx2.3 --text-encoder none --clip-encoder none --modifier none --autoencoder none`
  - Same request key and persistent source runtime pattern as run076/run077.

- Repeat summary (canonical artifacts):
  - `output/run078_warm_server_ltx23_min_repeats_summary.tsv`
  - `output/run078_warm_server_ltx23_min_repeats_aggregate.txt`
  - `r1`: `canary_rc=124`, `post_echo_rc=124`, `source=mapping`, `text_init=1`, `encode_begin=1`, `encode_loading=1`
  - `r2`: `canary_rc=1`, `post_echo_rc=124`, `source=unknown`, no encode markers
  - `r3`: `canary_rc=1`, `post_echo_rc=124`, `source=unknown`, no encode markers
  - `r4`: `canary_rc=1`, `post_echo_rc=124`, `source=unknown`, no encode markers
  - aggregate:
    - `canary_rc=124`: `1/4`
    - `post_echo_rc=0`: `0/4`
    - `post_echo_rc=124`: `4/4`

- Failure evidence captured during run:
  - Source runtime assertion and abort reproduced again:
    - `ccv_nnc_symbolic_graph_compile.c:1365` -> `Assertion \`memory_type == CCV_TENSOR_CPU_MEMORY\` failed.`
    - crash marker: `*** Program crashed: Aborted ...`
    - see `output/run078_warm_server_ltx23_min/server.log`
  - Post-crash client requests fail with connection errors:
    - `gRPC error: UNAVAILABLE ... connection attempt timed out before receiving SETTINGS frame`
    - see `output/run078_warm_server_ltx23_min_r2_client.log` through `output/run078_warm_server_ltx23_min_r4_client.log`

- Key findings:
  - Warm-process crash confound reproduces even when mapped entry is reduced to `version=ltx2.3` only.
  - This further rules out `modifier`, `autoencoder`, and explicit text/clip pin fields as required triggers.
  - Updated working interpretation: instability is tied to repeated warm lifecycle on mapped ltx2.3 path itself (or nearby shared runtime state), not to specific optional companion fields.

- Artifacts:
  - `output/run078_warm_server_ltx23_min_console.log`
  - `output/run078_warm_server_ltx23_min/server.log`
  - `output/run078_warm_server_ltx23_min_repeats_summary.tsv`
  - `output/run078_warm_server_ltx23_min_repeats_aggregate.txt`
