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

## Run 079 (2026-06-29): Cold-Per-Repeat Control (`version=ltx2.3` only)

- Goal:
  - Isolate whether the run076-078 warm-process collapse (`canary_rc=1` + `UNAVAILABLE`) depends on reusing one long-lived server, by restarting server before each repeat under the same minimal mapped entry.

- Method:
  - Reused same minimal-entry control as run078, but enabled per-repeat restart mode:
    - `tools/run_q6p_warm_server_mod_auto_repeats.sh --tag run079_cold_per_repeat_ltx23_min --repeats 4 --timeout-sec 75 --restart-per-repeat --entry-version ltx2.3 --text-encoder none --clip-encoder none --modifier none --autoencoder none`
  - Same request key and prompt/config held constant.

- Repeat summary (canonical artifacts):
  - `output/run079_cold_per_repeat_ltx23_min_repeats_summary.tsv`
  - `output/run079_cold_per_repeat_ltx23_min_repeats_aggregate.txt`
  - `r1`: `canary_rc=124`, `post_echo_rc=124`, `source=mapping`, `text_init=1`, `encode_begin=1`, `encode_loading=1`
  - `r2`: `canary_rc=124`, `post_echo_rc=124`, `source=mapping`, `text_init=1`, `encode_begin=1`, `encode_loading=1`
  - `r3`: `canary_rc=124`, `post_echo_rc=124`, `source=mapping`, `text_init=1`, `encode_begin=1`, `encode_loading=1`
  - `r4`: `canary_rc=124`, `post_echo_rc=124`, `source=mapping`, `text_init=1`, `encode_begin=1`, `encode_loading=1`
  - aggregate:
    - `canary_rc=124`: `4/4`
    - `post_echo_rc=124`: `4/4`
    - `post_echo_rc=0`: `0/4`
    - `source=mapping`: `4/4`

- Failure evidence captured during run:
  - Each per-repeat server instance still hits the same assertion path during generation:
    - `ccv_nnc_symbolic_graph_compile.c:1365` -> `Assertion \`memory_type == CCV_TENSOR_CPU_MEMORY\` failed.`
    - crash marker: `*** Program crashed: Aborted ...`
    - see repeated occurrences in `output/run079_cold_per_repeat_ltx23_min/server.log`
  - Unlike warm-reuse runs, there were no `UNAVAILABLE` connect failures in per-repeat client logs; each repeat reached mapping/deep-timeout class before its server instance aborted.

- Key findings:
  - Per-repeat restart removes the cross-repeat collapse signature (`canary_rc=1` + `UNAVAILABLE`) seen in warm-reuse runs.
  - Core assertion failure remains reproducible per request, indicating two layered failure surfaces:
    - per-request crash/abort on mapped ltx2.3 path,
    - plus an additional warm-reuse collapse mode that manifests as subsequent connection failures when server is not restarted.

- Artifacts:
  - `output/run079_cold_per_repeat_ltx23_min_console.log`
  - `output/run079_cold_per_repeat_ltx23_min/server.log`
  - `output/run079_cold_per_repeat_ltx23_min_repeats_summary.tsv`
  - `output/run079_cold_per_repeat_ltx23_min_repeats_aggregate.txt`

## Run 080 (2026-06-29): Restart-Per-Repeat Text/Clip Matrix (4 cases x 2 repeats)

- Goal:
  - Isolate which text/clip pin combinations preserve versus suppress the per-request assertion/abort path under cold-per-repeat conditions.

- Method:
  - Added script-first matrix runner:
    - `tools/run_q6p_restart_per_repeat_textclip_matrix.sh`
  - Executed matrix:
    - `bash tools/run_q6p_restart_per_repeat_textclip_matrix.sh --tag run080_restart_per_repeat_textclip_matrix --repeats 2 --timeout-sec 75`
  - Fixed controls across all cases:
    - `model=10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt`
    - `entry_version=ltx2.3`
    - `modifier=none`, `autoencoder=none`
    - `--restart-per-repeat` active for every repeat.

- Case matrix:
  - `min_none`: `text_encoder=none`, `clip_encoder=none`
  - `text_clipvit_only`: `text_encoder=clip_vit_l14_f16.ckpt`, `clip_encoder=none`
  - `text_gemma_only`: `text_encoder=gemma_3_12b_it_qat_q8p.ckpt`, `clip_encoder=none`
  - `text_clipvit_clip_traced`: `text_encoder=clip_vit_l14_f16.ckpt`, `clip_encoder=10_e_v1_bf16_regen_0_q6p.ckpt`

- Matrix outcomes (canonical artifact):
  - `output/run080_restart_per_repeat_textclip_matrix/results.tsv`
  - shared outcomes:
    - all cases: `canary_rc_124=2/2`, `canary_rc_1=0/2`, `source_mapping=2/2`, `unavailable_count=0`
  - per-case split:
    - `min_none`: `post_echo_rc_124=2`, `post_echo_rc_0=0`, `assert_count=2`, `abort_count=2`
    - `text_clipvit_only`: `post_echo_rc_124=2`, `post_echo_rc_0=0`, `assert_count=2`, `abort_count=2`
    - `text_gemma_only`: `post_echo_rc_124=0`, `post_echo_rc_0=2`, `assert_count=0`, `abort_count=0`
    - `text_clipvit_clip_traced`: `post_echo_rc_124=2`, `post_echo_rc_0=0`, `assert_count=2`, `abort_count=2`

- Failure evidence and controls:
  - Assertion/abort signature (when present):
    - `ccv_nnc_symbolic_graph_compile.c:1365` with `Assertion \`memory_type == CCV_TENSOR_CPU_MEMORY\` failed.`
    - `*** Program crashed: Aborted ...`
  - No `UNAVAILABLE ... SETTINGS frame` client cascade in this run (restart-per-repeat kept runs in mapping timeout classes).

- Key findings:
  - Under cold-per-repeat conditions, per-request assertion/abort is text-path dependent, not universal across all mapped ltx2.3 entries.
  - `text_gemma_only` is a discriminating non-assertion branch (`post_echo_rc=0`, no abort), while `none` and `clip_vit`-pinned text cases remain on assertion/abort branch (`post_echo_rc=124`).
  - This sharpens the layered model from run079: warm-reuse collapse is separate, and within per-request failures the text/encoder path acts as a strong gate.

- Artifacts:
  - `output/run080_restart_per_repeat_textclip_matrix/results.tsv`
  - `output/run080_restart_per_repeat_textclip_matrix/summary.md`
  - `output/run080_restart_per_repeat_textclip_matrix/cases/min_none.log`
  - `output/run080_restart_per_repeat_textclip_matrix/cases/text_clipvit_only.log`
  - `output/run080_restart_per_repeat_textclip_matrix/cases/text_gemma_only.log`
  - `output/run080_restart_per_repeat_textclip_matrix/cases/text_clipvit_clip_traced.log`

## Run 081 (2026-06-29): Restart-Per-Repeat Text-Gate Clip Companion Matrix (6 cases x 2 repeats)

- Goal:
  - Test whether clip companion choice can override or stabilize the run080 gemma text-path branch boundary, and compare against clip-vit / no-text controls.

- Method:
  - Added focused matrix runner:
    - `tools/run_q6p_restart_per_repeat_textgate_clip_matrix.sh`
  - Executed:
    - `bash tools/run_q6p_restart_per_repeat_textgate_clip_matrix.sh --tag run081_restart_per_repeat_textgate_clip_matrix --repeats 2 --timeout-sec 75`
  - Fixed controls:
    - `model=10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt`
    - `entry_version=ltx2.3`
    - `modifier=none`, `autoencoder=none`
    - `--restart-per-repeat` active for every repeat.

- Case matrix:
  - `text_gemma_only`: `text_encoder=gemma_3_12b_it_qat_q8p.ckpt`, `clip_encoder=none`
  - `text_gemma_clip_traced`: `text_encoder=gemma_3_12b_it_qat_q8p.ckpt`, `clip_encoder=10_e_v1_bf16_regen_0_q6p.ckpt`
  - `text_gemma_clipvit`: `text_encoder=gemma_3_12b_it_qat_q8p.ckpt`, `clip_encoder=clip_vit_l14_f16.ckpt`
  - `text_clipvit_only`: `text_encoder=clip_vit_l14_f16.ckpt`, `clip_encoder=none`
  - `text_clipvit_clip_traced`: `text_encoder=clip_vit_l14_f16.ckpt`, `clip_encoder=10_e_v1_bf16_regen_0_q6p.ckpt`
  - `text_none_clip_traced`: `text_encoder=none`, `clip_encoder=10_e_v1_bf16_regen_0_q6p.ckpt`

- Matrix outcomes (canonical artifact):
  - `output/run081_restart_per_repeat_textgate_clip_matrix/results.tsv`
  - global:
    - all cases: `canary_rc_124=2/2`, `canary_rc_1=0/2`, `source_mapping=2/2`, `unavailable_count=0`
    - `cases_with_any_assertion=4/6`
  - per-case split:
    - `text_gemma_only`: `post_echo_rc_124=1`, `post_echo_rc_0=1`, `assert_count=1`, `abort_count=1`
    - `text_gemma_clip_traced`: `post_echo_rc_124=0`, `post_echo_rc_0=2`, `assert_count=0`, `abort_count=0`
    - `text_gemma_clipvit`: `post_echo_rc_124=0`, `post_echo_rc_0=2`, `assert_count=0`, `abort_count=0`
    - `text_clipvit_only`: `post_echo_rc_124=2`, `post_echo_rc_0=0`, `assert_count=2`, `abort_count=2`
    - `text_clipvit_clip_traced`: `post_echo_rc_124=2`, `post_echo_rc_0=0`, `assert_count=2`, `abort_count=2`
    - `text_none_clip_traced`: `post_echo_rc_124=2`, `post_echo_rc_0=0`, `assert_count=2`, `abort_count=2`

- Failure evidence and controls:
  - Assertion/abort signature when active:
    - `ccv_nnc_symbolic_graph_compile.c:1365` with `Assertion \`memory_type == CCV_TENSOR_CPU_MEMORY\` failed.`
    - `*** Program crashed: Aborted ...`
  - No `UNAVAILABLE ... SETTINGS frame` cascade observed in this run.

- Key findings:
  - Gemma text path is not uniformly non-assert: `text_gemma_only` is mixed (1 assertion repeat, 1 non-assert repeat).
  - Adding clip companion (`clip_traced` or `clip_vit`) to gemma text stabilized both repeats onto the non-assert/post-echo-0 branch.
  - Clip-vit-only and no-text+clip-traced controls remain fully assertion-positive.
  - Updated interpretation: text/companion pairing controls a boundary between assertion-positive and non-assert sub-branches inside the cold-per-repeat mapped ltx2.3 surface.

- Artifacts:
  - `output/run081_restart_per_repeat_textgate_clip_matrix/results.tsv`
  - `output/run081_restart_per_repeat_textgate_clip_matrix/summary.md`
  - `output/run081_restart_per_repeat_textgate_clip_matrix/cases/text_gemma_only.log`
  - `output/run081_restart_per_repeat_textgate_clip_matrix/cases/text_gemma_clip_traced.log`
  - `output/run081_restart_per_repeat_textgate_clip_matrix/cases/text_gemma_clipvit.log`
  - `output/run081_restart_per_repeat_textgate_clip_matrix/cases/text_clipvit_only.log`
  - `output/run081_restart_per_repeat_textgate_clip_matrix/cases/text_clipvit_clip_traced.log`
  - `output/run081_restart_per_repeat_textgate_clip_matrix/cases/text_none_clip_traced.log`

## Run 081b (2026-06-29): Restart-Per-Repeat Text-Gate Companion Repro Pass (6 cases x 4 repeats)

- Goal:
  - Re-test run081 with higher repeats to measure boundary stability and determine whether the observed non-assert branch assignments are reproducible.

- Method:
  - Reused focused matrix runner:
    - `tools/run_q6p_restart_per_repeat_textgate_clip_matrix.sh`
  - Executed:
    - `bash tools/run_q6p_restart_per_repeat_textgate_clip_matrix.sh --tag run081b_restart_per_repeat_textgate_clip_matrix --repeats 4 --timeout-sec 75`
  - Fixed controls:
    - `model=10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt`
    - `entry_version=ltx2.3`
    - `modifier=none`, `autoencoder=none`
    - `--restart-per-repeat` active for every repeat.

- Case matrix:
  - `text_gemma_only`: `text_encoder=gemma_3_12b_it_qat_q8p.ckpt`, `clip_encoder=none`
  - `text_gemma_clip_traced`: `text_encoder=gemma_3_12b_it_qat_q8p.ckpt`, `clip_encoder=10_e_v1_bf16_regen_0_q6p.ckpt`
  - `text_gemma_clipvit`: `text_encoder=gemma_3_12b_it_qat_q8p.ckpt`, `clip_encoder=clip_vit_l14_f16.ckpt`
  - `text_clipvit_only`: `text_encoder=clip_vit_l14_f16.ckpt`, `clip_encoder=none`
  - `text_clipvit_clip_traced`: `text_encoder=clip_vit_l14_f16.ckpt`, `clip_encoder=10_e_v1_bf16_regen_0_q6p.ckpt`
  - `text_none_clip_traced`: `text_encoder=none`, `clip_encoder=10_e_v1_bf16_regen_0_q6p.ckpt`

- Matrix outcomes (canonical artifact):
  - `output/run081b_restart_per_repeat_textgate_clip_matrix/results.tsv`
  - global:
    - all cases: `source_mapping=4/4`
    - `cases_with_any_canary_rc_1=2/6`
    - `cases_with_any_assertion=5/6`
  - per-case split:
    - `text_gemma_only`: `canary_rc_124=4`, `canary_rc_1=0`, `post_echo_rc_124=0`, `post_echo_rc_0=4`, `assert_count=0`, `abort_count=0`, `unavailable_count=0`
    - `text_gemma_clip_traced`: `canary_rc_124=4`, `canary_rc_1=0`, `post_echo_rc_124=2`, `post_echo_rc_0=2`, `assert_count=2`, `abort_count=2`, `unavailable_count=0`
    - `text_gemma_clipvit`: `canary_rc_124=4`, `canary_rc_1=0`, `post_echo_rc_124=1`, `post_echo_rc_0=3`, `assert_count=1`, `abort_count=1`, `unavailable_count=0`
    - `text_clipvit_only`: `canary_rc_124=4`, `canary_rc_1=0`, `post_echo_rc_124=4`, `post_echo_rc_0=0`, `assert_count=4`, `abort_count=4`, `unavailable_count=0`
    - `text_clipvit_clip_traced`: `canary_rc_124=3`, `canary_rc_1=1`, `post_echo_rc_124=3`, `post_echo_rc_0=0`, `assert_count=4`, `abort_count=4`, `unavailable_count=1`
    - `text_none_clip_traced`: `canary_rc_124=2`, `canary_rc_1=2`, `post_echo_rc_124=1`, `post_echo_rc_0=0`, `assert_count=4`, `abort_count=4`, `unavailable_count=2`

- Failure evidence and controls:
  - Assertion/abort signature when active:
    - `ccv_nnc_symbolic_graph_compile.c:1365` with `Assertion \`memory_type == CCV_TENSOR_CPU_MEMORY\` failed.`
    - `*** Program crashed: Aborted ...`
  - Unlike warm-reuse collapse runs, there was no cross-case process cascade; restart-per-repeat kept cases bounded even when individual repeats recorded `canary_rc=1` and `UNAVAILABLE` lines.

- Key findings:
  - `text_gemma_only` moved to a stable non-assert branch in this repro pass (`post_echo_rc_0=4/4`, `assert_count=0`).
  - Gemma + clip companion pairings were not stably non-assert at higher repeats:
    - `text_gemma_clip_traced`: mixed (`post_echo_rc_124=2`, `assert_count=2`)
    - `text_gemma_clipvit`: mostly non-assert but not clean (`post_echo_rc_124=1`, `assert_count=1`)
  - Clip-vit/no-text regimes remained strongly assertion-positive, including `canary_rc=1` and `UNAVAILABLE` events in the two clip-traced control variants.
  - Updated interpretation: the cold-per-repeat mapped ltx2.3 assertion surface is state-sensitive and probabilistic near the gemma boundary; companion choice shifts probabilities but is not a deterministic selector.

- Artifacts:
  - `output/run081b_restart_per_repeat_textgate_clip_matrix/results.tsv`
  - `output/run081b_restart_per_repeat_textgate_clip_matrix/summary.md`
  - `output/run081b_restart_per_repeat_textgate_clip_matrix/cases/text_gemma_only.log`
  - `output/run081b_restart_per_repeat_textgate_clip_matrix/cases/text_gemma_clip_traced.log`
  - `output/run081b_restart_per_repeat_textgate_clip_matrix/cases/text_gemma_clipvit.log`
  - `output/run081b_restart_per_repeat_textgate_clip_matrix/cases/text_clipvit_only.log`
  - `output/run081b_restart_per_repeat_textgate_clip_matrix/cases/text_clipvit_clip_traced.log`
  - `output/run081b_restart_per_repeat_textgate_clip_matrix/cases/text_none_clip_traced.log`

## Run 082 (2026-06-30): Restart-Per-Repeat Gemma-Boundary Stress Pass (3 cases x 8 repeats)

- Goal:
  - Increase repeat depth on the gemma-side boundary only (`text_gemma_only`, `text_gemma_clip_traced`, `text_gemma_clipvit`) to estimate branch frequencies with lower noise than run081/run081b.

- Method:
  - Added focused boundary runner:
    - `tools/run_q6p_restart_per_repeat_textgate_boundary_matrix.sh`
  - Executed:
    - `bash tools/run_q6p_restart_per_repeat_textgate_boundary_matrix.sh --tag run082_restart_per_repeat_textgate_boundary_matrix --repeats 8 --timeout-sec 75`
  - Fixed controls:
    - `model=10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt`
    - `entry_version=ltx2.3`
    - `modifier=none`, `autoencoder=none`
    - `--restart-per-repeat` active for every repeat.

- Case matrix:
  - `text_gemma_only`: `text_encoder=gemma_3_12b_it_qat_q8p.ckpt`, `clip_encoder=none`
  - `text_gemma_clip_traced`: `text_encoder=gemma_3_12b_it_qat_q8p.ckpt`, `clip_encoder=10_e_v1_bf16_regen_0_q6p.ckpt`
  - `text_gemma_clipvit`: `text_encoder=gemma_3_12b_it_qat_q8p.ckpt`, `clip_encoder=clip_vit_l14_f16.ckpt`

- Matrix outcomes (canonical artifact):
  - `output/run082_restart_per_repeat_textgate_boundary_matrix/results.tsv`
  - global:
    - all cases: `canary_rc_124=8/8`, `canary_rc_1=0/8`, `source_mapping=8/8`, `unavailable_count=0`
    - `cases_with_any_assertion=3/3`
  - per-case split:
    - `text_gemma_only`: `post_echo_rc_124=2`, `post_echo_rc_0=6`, `assert_count=2`, `abort_count=2`
    - `text_gemma_clip_traced`: `post_echo_rc_124=1`, `post_echo_rc_0=7`, `assert_count=1`, `abort_count=1`
    - `text_gemma_clipvit`: `post_echo_rc_124=2`, `post_echo_rc_0=6`, `assert_count=2`, `abort_count=2`

- Failure evidence and controls:
  - Assertion/abort signature when active:
    - `ccv_nnc_symbolic_graph_compile.c:1365` with `Assertion \`memory_type == CCV_TENSOR_CPU_MEMORY\` failed.`
    - `*** Program crashed: Aborted ...`
  - No `canary_rc=1` and no client-side `UNAVAILABLE ... SETTINGS frame` in this focused run.

- Key findings:
  - None of the gemma-side variants are deterministically non-assert at 8 repeats; all three showed assertion-positive repeats.
  - All three remained predominantly non-assert (`post_echo_rc_0` majority), but with non-zero assertion probability.
  - Companion choice shifted frequency modestly (`clip_traced` lowest assertion count here), but did not create a hard branch split.
  - Updated interpretation: gemma-boundary behavior is probabilistic/state-sensitive; model comparisons should use repeat-depth statistics rather than binary case labels.

- Artifacts:
  - `output/run082_restart_per_repeat_textgate_boundary_matrix/results.tsv`
  - `output/run082_restart_per_repeat_textgate_boundary_matrix/summary.md`
  - `output/run082_restart_per_repeat_textgate_boundary_matrix/cases/text_gemma_only.log`
  - `output/run082_restart_per_repeat_textgate_boundary_matrix/cases/text_gemma_clip_traced.log`
  - `output/run082_restart_per_repeat_textgate_boundary_matrix/cases/text_gemma_clipvit.log`

## Run 083 (2026-06-30): Restart-Per-Repeat Gemma-Boundary Repro Pass (3 cases x 8 repeats)

- Goal:
  - Re-run the same focused boundary matrix as run082 to test short-horizon reproducibility of gemma-side branch rates.

- Method:
  - Reused focused boundary runner:
    - `tools/run_q6p_restart_per_repeat_textgate_boundary_matrix.sh`
  - Executed:
    - `bash tools/run_q6p_restart_per_repeat_textgate_boundary_matrix.sh --tag run083_restart_per_repeat_textgate_boundary_matrix --repeats 8 --timeout-sec 75`
  - Fixed controls unchanged:
    - `model=10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt`
    - `entry_version=ltx2.3`
    - `modifier=none`, `autoencoder=none`
    - `--restart-per-repeat` active for every repeat.

- Case matrix:
  - `text_gemma_only`: `text_encoder=gemma_3_12b_it_qat_q8p.ckpt`, `clip_encoder=none`
  - `text_gemma_clip_traced`: `text_encoder=gemma_3_12b_it_qat_q8p.ckpt`, `clip_encoder=10_e_v1_bf16_regen_0_q6p.ckpt`
  - `text_gemma_clipvit`: `text_encoder=gemma_3_12b_it_qat_q8p.ckpt`, `clip_encoder=clip_vit_l14_f16.ckpt`

- Matrix outcomes (canonical artifact):
  - `output/run083_restart_per_repeat_textgate_boundary_matrix/results.tsv`
  - global:
    - all cases: `canary_rc_124=8/8`, `canary_rc_1=0/8`, `source_mapping=8/8`, `unavailable_count=0`
    - `cases_with_any_assertion=3/3`
  - per-case split:
    - `text_gemma_only`: `post_echo_rc_124=1`, `post_echo_rc_0=7`, `assert_count=1`, `abort_count=1`
    - `text_gemma_clip_traced`: `post_echo_rc_124=2`, `post_echo_rc_0=6`, `assert_count=2`, `abort_count=2`
    - `text_gemma_clipvit`: `post_echo_rc_124=2`, `post_echo_rc_0=6`, `assert_count=2`, `abort_count=2`

- Failure evidence and controls:
  - Assertion/abort signature when active:
    - `ccv_nnc_symbolic_graph_compile.c:1365` with `Assertion \`memory_type == CCV_TENSOR_CPU_MEMORY\` failed.`
    - `*** Program crashed: Aborted ...`
  - No `canary_rc=1` and no client-side `UNAVAILABLE ... SETTINGS frame` in this run.

- Key findings:
  - Repro pass preserved the same high-level shape as run082: all gemma-side variants remain majority non-assert yet non-deterministic.
  - Case-level ordering shifted versus run082 (`text_gemma_only` 1/8 assertions here vs 2/8 in run082; `text_gemma_clip_traced` 2/8 here vs 1/8 in run082), reinforcing sampling variability.
  - Updated interpretation: cold-per-repeat gemma-boundary behavior is rate-based and state-sensitive; single-run rank ordering across companion variants is not stable enough to treat as a hard rule.

- Artifacts:
  - `output/run083_restart_per_repeat_textgate_boundary_matrix/results.tsv`
  - `output/run083_restart_per_repeat_textgate_boundary_matrix/summary.md`
  - `output/run083_restart_per_repeat_textgate_boundary_matrix/cases/text_gemma_only.log`
  - `output/run083_restart_per_repeat_textgate_boundary_matrix/cases/text_gemma_clip_traced.log`
  - `output/run083_restart_per_repeat_textgate_boundary_matrix/cases/text_gemma_clipvit.log`

## Run 084 (2026-06-30): Pipeline Stage-Gate Check (conversion pass, q6p inference fail)

- Goal:
  - Establish a strict readiness snapshot for the final pipeline objective by gating conversion outputs and quantized variants through both structure validation and final-output inference checks.

- Method:
  - Structural validation (`ltx2_3` profile):
    - `.venv/bin/python tools/dt_validate_converted_ckpt.py --file dt-models/10_e_v1_bf16_regen_0_f16.ckpt --profile ltx2_3`
    - `.venv/bin/python tools/dt_validate_converted_ckpt.py --file dt-models/10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt --profile ltx2_3`
    - `.venv/bin/python tools/dt_validate_converted_ckpt.py --file dt-models/10_e_v1_bf16_regen_0_q6p.ckpt --profile ltx2_3`
  - Strict inference gate (`require_complete_stream` + `require_final_output`, timeout 240s):
    - `bash tools/run_q6p_canary_once.sh --model 10_e_v1_bf16_regen_0_f16.ckpt --tag run084_canary_f16_final_output --timeout-sec 240 --max-responses 0 --require-final-output --require-complete-stream --width 256 --height 256 --steps 8`
    - `bash tools/run_q6p_canary_once.sh --model 10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt --tag run084_canary_q6p_trace_final_output --timeout-sec 240 --max-responses 0 --require-final-output --require-complete-stream --width 256 --height 256 --steps 8`
    - `bash tools/run_q6p_canary_once.sh --model 10_e_v1_bf16_regen_0_q6p.ckpt --tag run084_canary_q6p_regen_final_output --timeout-sec 240 --max-responses 0 --require-final-output --require-complete-stream --width 256 --height 256 --steps 8`

- Stage-gate outcomes:
  - Structural gate:
    - all three files passed `dt_validate_converted_ckpt.py` under `ltx2_3` profile.
  - Inference gate:
    - `f16` model: `RESULT=PASS`, full stream completed, final payload present (`images written: 1`).
    - `q6p_trace021`: `RESULT=FAIL canary rc=1`; server crashed with `Illegal instruction` in `TextEncoder.encodeLTX2`.
    - `q6p_regen_0`: `RESULT=FAIL canary timed out (240s)`; server accepted request but no streamed responses before timeout (`post_echo_rc=0`).

- Key findings:
  - Conversion stage is currently inference-viable on f16 output in strict final-output mode.
  - q6p stage is not production-ready despite structural PASS; two distinct runtime failure classes remain:
    - immediate crash (`Illegal instruction` / `encodeLTX2`) for traced q6p,
    - deep timeout/no-stream for regen q6p.
  - Structural validation is necessary but not sufficient; runtime gating must remain mandatory for codec qualification.

- Artifacts:
  - `output/q6p_canary_run084_canary_f16_final_output/client.log`
  - `output/q6p_canary_run084_canary_f16_final_output/server.log`
  - `output/q6p_canary_run084_canary_q6p_trace_final_output/client.log`
  - `output/q6p_canary_run084_canary_q6p_trace_final_output/server.log`
  - `output/q6p_canary_run084_canary_q6p_regen_final_output/client.log`
  - `output/q6p_canary_run084_canary_q6p_regen_final_output/server.log`

## Run 085 (2026-07-01): q8p Resume Recovery + Strict Runtime Gate Pass

- Goal:
  - Confirm the resumed q8p quantization output is genuinely pipeline-ready (not only structurally intact) after interruption.

- Method:
  - Structural validation (`ltx2_3` profile):
    - `.venv/bin/python tools/dt_validate_converted_ckpt.py --file dt-models/10_e_v1_bf16_regen_0_q8p_20260630.ckpt --profile ltx2_3`
  - Strict inference gate (`require_complete_stream` + `require_final_output`, timeout 240s):
    - `bash tools/run_q6p_canary_once.sh --model 10_e_v1_bf16_regen_0_q8p_20260630.ckpt --tag run085_canary_q8p_final_output --timeout-sec 240 --max-responses 0 --require-complete-stream --require-final-output --width 256 --height 256 --steps 8`
  - Practical one-frame video-path check:
    - start server: `drawthings-grpc --address 127.0.0.1 --port 7861 --gpu 0 --no-tls --model-browser --no-response-compression /workspaces/drawthings-linux-toolkit/dt-models`
    - run: `bash tools/dt_video_test_run.sh 10_e_v1_bf16_regen_0_q8p_20260630.ckpt`

- Stage-gate outcomes:
  - Structural gate:
    - q8p file passed `dt_validate_converted_ckpt.py` under `ltx2_3` profile (`RESULT=PASS`).
  - Inference gate:
    - strict q8p canary passed (`RESULT=PASS`), stream completed, final payload written (`responses=14`, `images written: 1`).
  - Video-path smoke:
    - one-frame video harness completed end-to-end, writing playable outputs (`playable.png`, `playable.gif`, `playable.mp4`).

- Key findings:
  - Resumed q8p artifact is operationally valid under the same strict final-output gate used for f16/q6p qualification.
  - q8p now has both structural and runtime evidence (including media conversion path), closing the resume-recovery uncertainty.
  - Remaining quantized qualification gap is now centered on q4p and broader multi-case repetition, not q8p integrity.

- Artifacts:
  - `output/q6p_canary_run085_canary_q8p_final_output/client.log`
  - `output/q6p_canary_run085_canary_q8p_final_output/server.log`
  - `output/dt_video_20260701_072007/config.bin`
  - `output/dt_video_20260701_072007/image_r0014_01.bin`
  - `output/dt_video_20260701_072007/playable.png`
  - `output/dt_video_20260701_072007/playable.gif`
  - `output/dt_video_20260701_072007/playable.mp4`

## Run 086 (2026-07-01): Official q6p Long Final-Gate Reconfirm PASS

- Goal:
  - Reconfirm a clean official distilled q6p end-to-end pass with enough time budget to avoid timeout false negatives.

- Command:

```bash
bash tools/run_q6p_canary_once.sh \
  --model ltx_2.3_22b_distilled_1.1_q6p.ckpt \
  --width 256 --height 256 --steps 8 \
  --timeout-sec 1800 --max-responses 0 --require-final-output \
  --tag official_q6p_singleframe_wait30_20260701
```

- Outcome:
  - PASS (`canary_rc=0`, `post_echo_rc=0`)
  - stream complete (`responses=15`, `generation stream finished`)
  - final payloads present (`images written: 1`, `audio written: 1`)
  - no crash signature in server log.

- Artifacts:
  - `output/official_q6p_singleframe_wait30_20260701/client.log`
  - `output/official_q6p_singleframe_wait30_20260701/server.log`
  - `output/official_q6p_singleframe_wait30_20260701/official_q6p_singleframe_wait30.png`

## Run 087 (2026-07-01): Custom Raw-Key Runtime Checks (Alias Disabled)

- Goal:
  - Remove custom-entry schema confounds and test custom bytes directly.

- Method summary:
  - Temporarily moved `dt-models/custom.json` out of the way.
  - Stage-gate canary (`max-responses=3`) on:
    - `10_e_v1_bf16_regen_0_q6p.ckpt`
    - `10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt`
    - `ltx_2.3_22b_distilled_q6p_forcedfix_clipfix2_20260602.ckpt`
  - Long final-gate raw-key runs on:
    - `10_e_v1_bf16_regen_0_q6p.ckpt`
    - `10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt`
    - `10_e_v1_bf16_regen_0_q8p_20260630.ckpt` (control)

- Runtime outcome:
  - All stage-gates reached `textEncoded -> imageEncoded -> sampling` and PASSed.
  - Long raw-key runs for custom q6p/q8p completed stream and wrote final image payloads.

- Quality outcome:
  - Decoded PNGs from custom raw-key outputs were visually noisy during manual inspection in this session.
  - Therefore raw-key runtime completion was not accepted as quality success.

- Artifacts:
  - `output/custom_q6p_main_singleframe_wait30_rawkey_20260701/`
  - `output/custom_q6p_trace021_singleframe_wait30_rawkey_20260701/`
  - `output/custom_q8p_control_singleframe_20260701/`

## Run 088 (2026-07-01): Alias Schema Regression and Minimal-v1 Mitigation

- Goal:
  - Re-test `10_e_v1` custom alias path and isolate schema-triggered crashes.

- Outcomes:
  - `10_e_v1` under prior ltx2.3-style entry failed with deterministic illegal-instruction crash:
    - `TextEncoder.encodeLTX2`
    - client `UNAVAILABLE: Socket closed`
  - After changing `10_e_v1` entry to minimal `version=v1` shape, stage-gate PASSed and long run completed.
  - However, decoded PNG from that long alias run remained visually noisy in this session.

- Artifacts:
  - fail: `output/q6p_canary_alias_10_e_v1_stagegate_20260701/`
  - pass (runtime only): `output/q6p_canary_alias_10_e_v1_stagegate_minv1schema_20260701/`
  - long runtime pass with noisy PNG: `output/alias_10_e_v1_singleframe_wait30_minv1schema_20260701/`

## Run 089 (2026-07-01): Custom-Main + Official-Clip Alias Final-Gate Failure

- Goal:
  - Test whether ltx2.3 alias wiring with custom main and official clip recovers quality/stability.

- Commands:
  - stage-gate: `alias_10e_main_official_clip_stagegate_20260701`
  - final-gate: `alias_10e_main_official_clip_finalgate_wait30_20260701`

- Outcome:
  - stage-gate timed out in one run (`canary_rc=124`, request accepted, no stream)
  - long final-gate crashed (`canary_rc=1`) after `imageEncoded`
  - server stack included cuDNN graph frames and `ccv_nnc_tensor_read`.

- Artifacts:
  - `output/q6p_canary_alias_10e_main_official_clip_stagegate_20260701/`
  - `output/q6p_canary_alias_10e_main_official_clip_finalgate_wait30_20260701/`

## Run 090 (2026-07-02): Custom f16 Raw-Key Long Control (Runtime PASS, Visual FAIL)

- Goal:
  - Determine whether quality failure starts before q6p quantization by testing custom f16 directly.

- Method:
  - Temporarily moved `dt-models/custom.json` out of the way.
  - Long raw-key run on `10_e_v1_bf16_regen_0_f16.ckpt` (256x256, steps=8, timeout 1800s).
  - Converted final tensor to PNG for visual inspection.

- Runtime outcome:
  - PASS (`canary_rc=0`, `post_echo_rc=0`)
  - stream complete (`responses=14`, `generation stream finished`)
  - final payload present (`images written: 1`)
  - no crash signature in server log.

- Visual outcome:
  - FAIL for quality objective: PNG is noise/non-coherent.

- Interpretation:
  - Noise appears upstream of q6p quantization; conversion-stage/custom-model semantics are now the primary suspect.

- Artifacts:
  - `output/custom_f16_control_singleframe_wait30_rawkey_20260702/client.log`
  - `output/custom_f16_control_singleframe_wait30_rawkey_20260702/server.log`
  - `output/custom_f16_control_singleframe_wait30_rawkey_20260702/custom_f16_control_singleframe_wait30_rawkey.png`

## Run 091 (2026-07-02): Custom f16 Full-Schema Alias Stage-Gate (Loader Crash)

- Goal:
  - Test whether explicit ltx2.3 alias wiring restores stable custom f16 path.

- Alias tested:
  - `name=10_e_v1_f16_fullschema_test`
  - `version=ltx2.3`, `file=10_e_v1_bf16_regen_0_f16.ckpt`, `clip_encoder=10_e_v1_bf16_regen_0_f16.ckpt`, `text_encoder=gemma_3_12b_it_qat_q8p.ckpt`, `autoencoder=ltx_2.3_audio_video_vae_f16.ckpt`.

- Outcome:
  - FAIL (`canary_rc=1`, `post_echo_rc=1`)
  - crash before first streamed response
  - server crash head: `ccv_nnc_tensor_read -> ccv_cnnp_model_read`

- Interpretation:
  - Full-schema aliasing did not recover custom f16 stability and reintroduced loader-path crash behavior.

- Artifacts:
  - `output/q6p_canary_alias_10e_f16_fullschema_stagegate_20260702/client.log`
  - `output/q6p_canary_alias_10e_f16_fullschema_stagegate_20260702/server.log`

## Run 092 (2026-07-02): Persistent Raw-Key Custom f16 Repro (Immediate Loader Crash)

- Goal:
  - Reproduce the paired persistent-server custom f16 failure under apples-to-apples controls and capture definitive server crash evidence.

- Method:
  - Started visible persistent server with tee log capture:
    - `drawthings-grpc --address 127.0.0.1 --port 7861 --gpu 0 --no-tls --model-browser --no-response-compression dt-models 2>&1 | tee output/persistent_server_logs_20260702_081754/server.log`
  - Built raw-key config and executed custom f16 request:
    - model: `10_e_v1_bf16_regen_0_f16.ckpt`
    - size/steps/seed: `256x256`, `steps=8`, `seed=4242`
    - command output dir: `output/persistent_repro_custom_f16_crash_20260702_081813`

- Outcome:
  - FAIL immediately (`gRPC error: UNAVAILABLE: Socket closed`)
  - no streamed responses (`responses=0`, `last_signpost=none`)
  - post-run health check failed (`server_alive_after_custom=0`)
  - server crashed with SIGSEGV in loader path:
    - `ccv_nnc_tensor_read -> ccv_cnnp_model_read`

- Artifacts:
  - `output/persistent_repro_custom_f16_crash_20260702_081813/client.log`
  - `output/persistent_repro_custom_f16_crash_20260702_081813/config.bin`
  - `output/persistent_server_logs_20260702_081754/server.log`

## Run 093 (2026-07-02): Persistent Official q6p Short Control (Healthy, Deep Sampling)

- Goal:
  - Run a matching official control after Run 092 to confirm behavior split is model-path specific, not a generic persistent-server failure.

- Method:
  - Started fresh visible persistent server with tee log capture:
    - `drawthings-grpc --address 127.0.0.1 --port 7861 --gpu 0 --no-tls --model-browser --no-response-compression dt-models 2>&1 | tee output/persistent_server_logs_20260702_082117/server.log`
  - Built config and executed official raw-key request with same prompt/settings as Run 092:
    - model: `ltx_2.3_22b_distilled_1.1_q6p.ckpt`
    - size/steps/seed: `256x256`, `steps=8`, `seed=4242`
    - bounded window: `timeout 600s`
    - command output dir: `output/persistent_control_official_short_20260702_082130`

- Outcome:
  - PASS for bounded-control objective (no crash, server remained alive)
  - stream progressed deep into sampling before timeout cancellation:
    - client reached `response #9`
    - signposts observed: `textEncoded -> imageEncoded -> sampling`
  - bounded command terminated by timeout at 600s, and server log recorded expected cancellation:
    - `cacncelled image generation`
    - `Image processed cancelled, generated images return nil`
  - post-run health check succeeded (`Received echo from: post_official_short_health`)

- Interpretation:
  - Under matched settings, official q6p remains stable while custom f16 dies pre-stream in loader read path.
  - This isolates the current failure to custom model-load/serialization behavior rather than persistent server orchestration.

- Artifacts:
  - `output/persistent_control_official_short_20260702_082130/client.log`
  - `output/persistent_control_official_short_20260702_082130/config.bin`
  - `output/persistent_control_official_short_20260702_082130/preview_0004.bin`
  - `output/persistent_control_official_short_20260702_082130/preview_0006.bin`
  - `output/persistent_control_official_short_20260702_082130/preview_0008.bin`
  - `output/persistent_control_official_short_20260702_082130/preview_0009.bin`
  - `output/persistent_server_logs_20260702_082117/server.log`

## Run 094 (2026-07-02): Persistent Raw-Key Custom q8p Stage-Gate (Healthy Sampling)

- Goal:
  - Verify whether the immediate persistent-loader crash seen in Run 092 is specific to custom f16 or a broader custom-path failure.

- Method:
  - Reused the active persistent server from Run 093.
  - Built raw-key config and executed bounded stage-gate request:
    - model: `10_e_v1_bf16_regen_0_q8p_20260630.ckpt`
    - size/steps/seed: `256x256`, `steps=8`, `seed=4242`
    - bounded stream: `--max-responses 3`
    - output dir: `output/persistent_probe_custom_q8p_stagegate_20260702_083324`

- Outcome:
  - PASS (`rc=0`, `responses=3`, `last_signpost=sampling`)
  - signpost path: `textEncoded -> imageEncoded -> sampling`
  - expected early-stop behavior observed (`stopping early at max responses: 3`)
  - server remained healthy (`server_alive_after_q8p_stagegate=1`)

- Interpretation:
  - Immediate loader crash is not a universal custom-path behavior under this setup.
  - Current high-signal split is now: custom f16 can crash pre-stream in loader read path, while custom q8p can reach stable sampling in the same persistent environment.

- Artifacts:
  - `output/persistent_probe_custom_q8p_stagegate_20260702_083324/client.log`
  - `output/persistent_probe_custom_q8p_stagegate_20260702_083324/config.bin`
  - `output/persistent_server_logs_20260702_082117/server.log`

## Run 095 (2026-07-02): Persistent Raw-Key Custom q8p Final-Gate (Runtime PASS, Visual FAIL)

- Goal:
  - Execute a full final-output persistent run for custom q8p under the same prompt/settings used in recent A/B checks, then verify visual quality.

- Method:
  - Reused healthy persistent server (`127.0.0.1:7861`).
  - Config/model:
    - model: `10_e_v1_bf16_regen_0_q8p_20260630.ckpt`
    - size/steps/seed: `256x256`, `steps=8`, `seed=4242`
    - sampler/guidance/shift: `17 / 1.0 / 3.0`
  - Full run window: `timeout 1800s`, `--max-responses 0`.
  - Converted final tensor to media:
    - base name: `persistent_final_custom_q8p`

- Outcome:
  - Runtime PASS:
    - `rc=0`
    - `generation stream finished`
    - `responses=14`
    - `images written=1`
    - `audio written=0`
    - server healthy after run (`HELLO after_custom_q8p_final_health`)
  - Visual FAIL:
    - PNG output is still noise/non-coherent.

- Interpretation:
  - Confirms prior pattern: custom pipeline can complete stream and write final payload while still failing visual-quality objective.

- Artifacts:
  - `output/persistent_final_custom_q8p_20260702_095806/client.log`
  - `output/persistent_final_custom_q8p_20260702_095806/config.bin`
  - `output/persistent_final_custom_q8p_20260702_095806/image_r0014_01.bin`
  - `output/persistent_final_custom_q8p_20260702_095806/persistent_final_custom_q8p.png`
  - `output/persistent_final_custom_q8p_20260702_095806/persistent_final_custom_q8p.gif`
  - `output/persistent_final_custom_q8p_20260702_095806/persistent_final_custom_q8p.mp4`
  - `output/persistent_final_custom_q8p_20260702_095806/convert.log`

## Run 096 (2026-07-02): Persistent Official q6p Final-Gate Control (Runtime PASS, Visual PASS)

- Goal:
  - Produce a same-session official control under identical settings to Run 095 for direct visual and runtime comparison.

- Method:
  - Reused same persistent server/session as Run 095.
  - Config/model:
    - model: `ltx_2.3_22b_distilled_1.1_q6p.ckpt`
    - size/steps/seed: `256x256`, `steps=8`, `seed=4242`
    - sampler/guidance/shift: `17 / 1.0 / 3.0`
  - Full run window: `timeout 1800s`, `--max-responses 0`.
  - Converted final tensor to media:
    - base name: `persistent_final_official_q6p`

- Outcome:
  - Runtime PASS:
    - `rc=0`
    - `generation stream finished`
    - `responses=15`
    - `images written=1`
    - `audio written=1`
    - server healthy after run (`server_alive_after_official_q6p_final=1`)
  - Visual PASS:
    - PNG output is coherent (red car on mountain road at sunset).

- Interpretation:
  - Same-session, same-settings A/B now explicitly shows quality divergence:
    - custom q8p: runtime pass + visual fail
    - official q6p: runtime pass + visual pass

- Artifacts:
  - `output/persistent_final_official_q6p_20260702_095927/client.log`
  - `output/persistent_final_official_q6p_20260702_095927/config.bin`
  - `output/persistent_final_official_q6p_20260702_095927/image_r0014_01.bin`
  - `output/persistent_final_official_q6p_20260702_095927/audio_r0015_01.bin`
  - `output/persistent_final_official_q6p_20260702_095927/persistent_final_official_q6p.png`
  - `output/persistent_final_official_q6p_20260702_095927/persistent_final_official_q6p.gif`
  - `output/persistent_final_official_q6p_20260702_095927/persistent_final_official_q6p.mp4`
  - `output/persistent_final_official_q6p_20260702_095927/persistent_final_official_q6p.wav`
  - `output/persistent_final_official_q6p_20260702_095927/convert.log`

## Run 097 (2026-07-02): Persistent Raw-Key Custom q6p Trace021 Final-Gate (Hard Crash)

- Goal:
  - Run a full final-gate raw-key test for custom trace021 under matched settings to determine whether this q6p branch can complete end-to-end.

- Method:
  - Started fresh persistent server and captured logs:
    - `output/persistent_server_logs_20260702_101533/server.log`
  - Built raw-key config and executed full final-gate request:
    - model: `10_e_v1_bf16_regen_0_q6p_trace021_20260610.ckpt`
    - size/steps/seed: `256x256`, `steps=8`, `seed=4242`
    - sampler/guidance/shift: `17 / 1.0 / 3.0`
    - run dir: `output/persistent_final_custom_trace021_q6p_20260702_101649`

- Outcome:
  - Runtime FAIL:
    - client error: `gRPC error: UNAVAILABLE: Socket closed`
    - summary: `custom_rc=1`, `server_alive_after_custom=0`
    - no final payload written (`no_custom_final_image_bin`)
  - Server crash signature:
    - `Program crashed: Illegal instruction`
    - crash site: `TextEncoder.encodeLTX2(...)`

- Interpretation:
  - Custom trace021 q6p currently fails in text-encoder path before stream progression.
  - This is a distinct failure mode from custom q8p final-gate (which can complete stream but still fail visually).

- Artifacts:
  - `output/persistent_final_custom_trace021_q6p_20260702_101649/client.log`
  - `output/persistent_final_custom_trace021_q6p_20260702_101649/config.bin`
  - `output/persistent_final_custom_trace021_q6p_20260702_101649/summary.txt`
  - `output/persistent_final_custom_trace021_q6p_20260702_101649/convert.log`
  - `output/persistent_server_logs_20260702_101533/server.log`

## Run 098 (2026-07-02): Persistent Official q6p Final-Gate Control After Run 097 Crash (Runtime PASS, Visual PASS)

- Goal:
  - Verify whether Run 097 crash is model-path specific by running a same-settings official control on a fresh server immediately after the custom crash.

- Method:
  - Restarted server cleanly and captured logs:
    - `output/persistent_server_logs_20260702_101940_official_control/server.log`
  - Ran full final-gate control with same settings as Run 097:
    - model: `ltx_2.3_22b_distilled_1.1_q6p.ckpt`
    - size/steps/seed: `256x256`, `steps=8`, `seed=4242`
    - sampler/guidance/shift: `17 / 1.0 / 3.0`
    - run dir: `output/persistent_final_official_q6p_20260702_101958_post_custom_crash`

- Outcome:
  - Runtime PASS:
    - `official_rc=0`
    - `generation stream finished`
    - `responses=15`
    - `images written=1`
    - `audio written=1`
    - server healthy after run (`server_alive_after_official=1`)
  - Visual PASS:
    - final PNG is coherent.

- Interpretation:
  - Immediate post-crash control confirms runtime environment remained valid; Run 097 failure is specific to custom trace021 path, not a generic server health issue.

- Artifacts:
  - `output/persistent_final_official_q6p_20260702_101958_post_custom_crash/client.log`
  - `output/persistent_final_official_q6p_20260702_101958_post_custom_crash/config.bin`
  - `output/persistent_final_official_q6p_20260702_101958_post_custom_crash/image_r0014_01.bin`
  - `output/persistent_final_official_q6p_20260702_101958_post_custom_crash/audio_r0015_01.bin`
  - `output/persistent_final_official_q6p_20260702_101958_post_custom_crash/persistent_final_official_q6p.png`
  - `output/persistent_final_official_q6p_20260702_101958_post_custom_crash/persistent_final_official_q6p.gif`
  - `output/persistent_final_official_q6p_20260702_101958_post_custom_crash/persistent_final_official_q6p.mp4`
  - `output/persistent_final_official_q6p_20260702_101958_post_custom_crash/summary.txt`
  - `output/persistent_final_official_q6p_20260702_101958_post_custom_crash/convert.log`
  - `output/persistent_server_logs_20260702_101940_official_control/server.log`

## Run 099 (2026-07-02): Persistent Raw-Key Custom q8p 3-Seed Sweep (Runtime PASS, Visual FAIL 3/3)

- Goal:
  - Determine whether custom q8p noisy output is seed-sensitive or deterministic under the established final-gate profile.

- Method:
  - Started fresh persistent server with dedicated log capture:
    - `output/persistent_server_logs_20260702_110531_q8p_seed_sweep/server.log`
  - Executed three full raw-key final-gate runs using identical settings except seed:
    - model: `10_e_v1_bf16_regen_0_q8p_20260630.ckpt`
    - size/steps: `256x256`, `steps=8`
    - sampler/guidance/shift: `17 / 1.0 / 3.0`
    - seeds: `4242`, `4243`, `4244`
    - root dir: `output/persistent_seed_sweep_custom_q8p_20260702_110812`

- Outcome:
  - Runtime PASS for all seeds (3/3):
    - each run: `rc=0`, `generation stream finished`, `responses=14`, `images written=1`, `audio written=0`
    - server remained healthy after each run (`post_seed_4242_health`, `post_seed_4243_health`, `post_seed_4244_health` echoes observed)
    - no crash markers (`Program crashed`, `Illegal instruction`, `Signal 11`) in sweep server log
  - Visual FAIL for all seeds (3/3):
    - all three PNGs were noise/non-coherent.

- Interpretation:
  - Custom q8p quality failure is robust across nearby seed changes under this profile.
  - This strengthens the hypothesis that the blocker is semantic/model-path corruption rather than a seed-specific sampling artifact.

- Artifacts:
  - `output/persistent_seed_sweep_custom_q8p_20260702_110812/summary.tsv`
  - `output/persistent_seed_sweep_custom_q8p_20260702_110812/seed_4242/client.log`
  - `output/persistent_seed_sweep_custom_q8p_20260702_110812/seed_4242/config.bin`
  - `output/persistent_seed_sweep_custom_q8p_20260702_110812/seed_4242/convert.log`
  - `output/persistent_seed_sweep_custom_q8p_20260702_110812/seed_4242/persistent_seed_4242_custom_q8p.png`
  - `output/persistent_seed_sweep_custom_q8p_20260702_110812/seed_4243/client.log`
  - `output/persistent_seed_sweep_custom_q8p_20260702_110812/seed_4243/config.bin`
  - `output/persistent_seed_sweep_custom_q8p_20260702_110812/seed_4243/convert.log`
  - `output/persistent_seed_sweep_custom_q8p_20260702_110812/seed_4243/persistent_seed_4243_custom_q8p.png`
  - `output/persistent_seed_sweep_custom_q8p_20260702_110812/seed_4244/client.log`
  - `output/persistent_seed_sweep_custom_q8p_20260702_110812/seed_4244/config.bin`
  - `output/persistent_seed_sweep_custom_q8p_20260702_110812/seed_4244/convert.log`
  - `output/persistent_seed_sweep_custom_q8p_20260702_110812/seed_4244/persistent_seed_4244_custom_q8p.png`
  - `output/persistent_server_logs_20260702_110531_q8p_seed_sweep/server.log`

## Run 100 (2026-07-02): Persistent Raw-Key Official q6p 3-Seed Matched Control (Runtime PASS, Visual PASS 3/3)

- Goal:
  - Run the same 3-seed final-gate protocol as Run 099 on official q6p to provide a matched distribution control.

- Method:
  - Started fresh persistent server with dedicated log capture:
    - `output/persistent_server_logs_20260702_120218_official_seed_sweep/server.log`
  - Executed three full raw-key final-gate runs using identical settings except seed:
    - model: `ltx_2.3_22b_distilled_1.1_q6p.ckpt`
    - size/steps: `256x256`, `steps=8`
    - sampler/guidance/shift: `17 / 1.0 / 3.0`
    - seeds: `4242`, `4243`, `4244`
    - root dir: `output/persistent_seed_sweep_official_q6p_20260702_133851`

- Outcome:
  - Runtime PASS for all seeds (3/3):
    - each run: `rc=0`, `generation stream finished`, `responses=15`, `images written=1`, `audio written=1`
    - server remained healthy after each run (`post_official_seed_4242_health`, `post_official_seed_4243_health`, `post_official_seed_4244_health` echoes observed)
    - no crash markers (`Program crashed`, `Illegal instruction`, `Signal 11`) in sweep server log
  - Visual PASS for all seeds (3/3):
    - all three PNGs were coherent/non-noise outputs.

- Interpretation:
  - Compared against Run 099 (custom q8p visual fail 3/3), this matched control strengthens the conclusion that the custom-path quality failure is model-path semantic and not due to seed variance or generic runtime instability.

- Artifacts:
  - `output/persistent_seed_sweep_official_q6p_20260702_133851/summary.tsv`
  - `output/persistent_seed_sweep_official_q6p_20260702_133851/seed_4242/client.log`
  - `output/persistent_seed_sweep_official_q6p_20260702_133851/seed_4242/config.bin`
  - `output/persistent_seed_sweep_official_q6p_20260702_133851/seed_4242/convert.log`
  - `output/persistent_seed_sweep_official_q6p_20260702_133851/seed_4242/persistent_seed_4242_official_q6p.png`
  - `output/persistent_seed_sweep_official_q6p_20260702_133851/seed_4243/client.log`
  - `output/persistent_seed_sweep_official_q6p_20260702_133851/seed_4243/config.bin`
  - `output/persistent_seed_sweep_official_q6p_20260702_133851/seed_4243/convert.log`
  - `output/persistent_seed_sweep_official_q6p_20260702_133851/seed_4243/persistent_seed_4243_official_q6p.png`
  - `output/persistent_seed_sweep_official_q6p_20260702_133851/seed_4244/client.log`
  - `output/persistent_seed_sweep_official_q6p_20260702_133851/seed_4244/config.bin`
  - `output/persistent_seed_sweep_official_q6p_20260702_133851/seed_4244/convert.log`
  - `output/persistent_seed_sweep_official_q6p_20260702_133851/seed_4244/persistent_seed_4244_official_q6p.png`
  - `output/persistent_server_logs_20260702_120218_official_seed_sweep/server.log`

## Run 101 (2026-07-02): Quantitative PNG Metrics for Matched 3-Seed A/B (Custom q8p vs Official q6p)

- Goal:
  - Add objective image-statistics evidence to complement manual visual verdicts from Runs 099 and 100.

- Method:
  - Computed grayscale summary metrics on all six PNGs (custom q8p seeds `4242/4243/4244` and official q6p seeds `4242/4243/4244`):
    - `gray_mean`, `gray_std`, `entropy`, `hf_delta` (neighbor delta), `grad_mean`
  - Wrote per-image rows plus aggregate means:
    - `output/seed_sweep_png_metrics_20260702_135352.tsv`

- Outcome (aggregate means):
  - custom q8p:
    - `gray_mean=0.504858`
    - `gray_std=0.097851`
    - `entropy=6.694596`
    - `hf_delta=0.045156`
    - `grad_mean=0.047160`
  - official q6p:
    - `gray_mean=0.333414`
    - `gray_std=0.308807`
    - `entropy=7.122476`
    - `hf_delta=0.033813`
    - `grad_mean=0.049277`

- Interpretation:
  - Custom q8p outputs are quantitatively clustered into a narrow mid-tone band with low global contrast and lower entropy, consistent with the observed noise/non-coherent quality failure.
  - Official q6p outputs show substantially higher contrast and entropy with coherent scene structure.

- Artifacts:
  - `output/seed_sweep_png_metrics_20260702_135352.tsv`
  - `output/persistent_seed_sweep_custom_q8p_20260702_110812/seed_4242/persistent_seed_4242_custom_q8p.png`
  - `output/persistent_seed_sweep_custom_q8p_20260702_110812/seed_4243/persistent_seed_4243_custom_q8p.png`
  - `output/persistent_seed_sweep_custom_q8p_20260702_110812/seed_4244/persistent_seed_4244_custom_q8p.png`
  - `output/persistent_seed_sweep_official_q6p_20260702_133851/seed_4242/persistent_seed_4242_official_q6p.png`
  - `output/persistent_seed_sweep_official_q6p_20260702_133851/seed_4243/persistent_seed_4243_official_q6p.png`
  - `output/persistent_seed_sweep_official_q6p_20260702_133851/seed_4244/persistent_seed_4244_official_q6p.png`

## Run 102 (2026-07-03): Persistent Raw-Key Custom q6p Main 3-Seed Sweep Attempt (Seed 4242 Hard Crash)

- Goal:
  - Extend the seed-sweep protocol to custom q6p main (`10_e_v1_bf16_regen_0_q6p.ckpt`) under the same final-gate settings used in Runs 099/100.

- Method:
  - Started fresh persistent server with dedicated log capture:
    - `output/persistent_server_logs_20260703_102119_custom_q6p_main_seed_sweep/server.log`
  - Started 3-seed sweep (`4242/4243/4244`) with matched profile:
    - model: `10_e_v1_bf16_regen_0_q6p.ckpt`
    - size/steps: `256x256`, `steps=8`
    - sampler/guidance/shift: `17 / 1.0 / 3.0`
    - root dir: `output/persistent_seed_sweep_custom_q6p_main_20260703_102142`

- Outcome:
  - FAIL on first seed (`4242`) before sweep could continue:
    - client reached `textEncoded` then `imageEncoded`
    - then `gRPC error: UNAVAILABLE: Socket closed`
    - summary row: `4242\t1\t0\t0\t0\t0\t0\t0\t0`
  - Server crash signature:
    - `Program crashed: Bad pointer dereference`
    - stack head includes `libcudnn_graph.so.9.8.0` and `ccv_nnc_tensor_read`
  - No final payloads written and no downstream seeds executed.

- Interpretation:
  - Custom q6p main currently exhibits a hard runtime instability class distinct from the custom q8p case (which is runtime-stable but visually degraded).
  - This adds a third clear custom failure surface in current evidence: (1) text-path illegal-instruction (trace021), (2) loader/cudnn pointer crash (q6p main), (3) runtime-pass visual-noise (q8p/f16).

- Artifacts:
  - `output/persistent_seed_sweep_custom_q6p_main_20260703_102142/summary.tsv`
  - `output/persistent_seed_sweep_custom_q6p_main_20260703_102142/pre_echo.log`
  - `output/persistent_seed_sweep_custom_q6p_main_20260703_102142/seed_4242/client.log`
  - `output/persistent_seed_sweep_custom_q6p_main_20260703_102142/seed_4242/config.bin`
  - `output/persistent_seed_sweep_custom_q6p_main_20260703_102142/seed_4242/config.log`
  - `output/persistent_seed_sweep_custom_q6p_main_20260703_102142/seed_4242/post_echo.log`
  - `output/persistent_seed_sweep_custom_q6p_main_20260703_102142/seed_4242/convert.log`
  - `output/persistent_server_logs_20260703_102119_custom_q6p_main_seed_sweep/server.log`

## Run 103 (2026-07-03): Custom q6p Main Stage-Gate Long-Wait Recheck (1800s Budget, Hard Crash)

- Goal:
  - Validate whether prior custom q6p main failure was merely insufficient wait time by rerunning with a full 1800s timeout budget.

- Method:
  - Fresh server and single-case bounded stage-gate (`--max-responses 3`) with long timeout:
    - model: `10_e_v1_bf16_regen_0_q6p.ckpt`
    - size/steps/seed: `256x256`, `steps=8`, `seed=4242`
    - timeout: `1800s`
    - run dir: `output/run103_longwait_20260703_122344/custom_q6p_main_stagegate_longwait`
    - server log: `output/run103_longwait_20260703_122344/server_custom_q6p_main_stagegate_longwait/server.log`

- Outcome:
  - FAIL quickly despite long timeout budget:
    - client reached `textEncoded -> imageEncoded`
    - then `gRPC error: UNAVAILABLE: Socket closed`
    - summary: `rc=1`, `finished=0`, `responses=0`, `last_signpost=imageEncoded`, `server_alive_after=0`
  - Server crash signature:
    - `Signal 11`
    - `Program crashed: Bad pointer dereference`
    - stack includes `libcudnn_graph.so.9.8.0` and `ccv_nnc_tensor_read`

- Interpretation:
  - This failure is not explained by short timeout windows; crash occurs early in execution path.

- Artifacts:
  - `output/run103_longwait_20260703_122344/custom_q6p_main_stagegate_longwait/summary.txt`
  - `output/run103_longwait_20260703_122344/custom_q6p_main_stagegate_longwait/client.log`
  - `output/run103_longwait_20260703_122344/custom_q6p_main_stagegate_longwait/config.bin`
  - `output/run103_longwait_20260703_122344/custom_q6p_main_stagegate_longwait/config.log`
  - `output/run103_longwait_20260703_122344/custom_q6p_main_stagegate_longwait/pre_echo.log`
  - `output/run103_longwait_20260703_122344/custom_q6p_main_stagegate_longwait/post_echo.log`
  - `output/run103_longwait_20260703_122344/server_custom_q6p_main_stagegate_longwait/server.log`

## Run 104 (2026-07-03): Official q6p Stage-Gate Long-Wait Matched Control (PASS)

- Goal:
  - Run a same-profile long-wait control immediately after Run 103 to determine whether the crash is custom-path specific.

- Method:
  - Fresh server and same bounded stage-gate (`--max-responses 3`) with 1800s timeout:
    - model: `ltx_2.3_22b_distilled_1.1_q6p.ckpt`
    - size/steps/seed: `256x256`, `steps=8`, `seed=4242`
    - run dir: `output/run104_longwait_20260703_123501/official_q6p_stagegate_longwait_control`
    - server log: `output/run104_longwait_20260703_123501/server_official_q6p_stagegate_longwait_control/server.log`

- Outcome:
  - PASS:
    - signposts reached: `textEncoded -> imageEncoded -> sampling`
    - bounded stop observed: `stopping early at max responses: 3`
    - `generation stream finished`
    - summary: `rc=0`, `finished=1`, `responses=3`, `last_signpost=sampling`, `server_alive_after=1`
  - No crash markers in server log.

- Interpretation:
  - Under matched settings and long wait budget, official control remains stable while custom q6p main crashes; this localizes the failure to custom model path, not to insufficient wait or generic runtime instability.

- Artifacts:
  - `output/run104_longwait_20260703_123501/official_q6p_stagegate_longwait_control/summary.txt`
  - `output/run104_longwait_20260703_123501/official_q6p_stagegate_longwait_control/client.log`
  - `output/run104_longwait_20260703_123501/official_q6p_stagegate_longwait_control/config.bin`
  - `output/run104_longwait_20260703_123501/official_q6p_stagegate_longwait_control/config.log`
  - `output/run104_longwait_20260703_123501/official_q6p_stagegate_longwait_control/pre_echo.log`
  - `output/run104_longwait_20260703_123501/official_q6p_stagegate_longwait_control/post_echo.log`
  - `output/run104_longwait_20260703_123501/server_official_q6p_stagegate_longwait_control/server.log`

## Run 105 (2026-07-03): Converter-Side Focused Diagnostics (Custom q6p Main vs Official q6p)

- Goal:
  - Move from runtime symptom checks to serializer/content diagnostics for custom q6p main (`10_e_v1_bf16_regen_0_q6p.ckpt`) against official q6p baseline.

- Method:
  - Intended deep diff run was started but produced no output for multiple minutes (likely heavy-scan stall in this environment):
    - `output/run105_converter_focus_20260703_124301/deep_diff.log`
  - Switched to stable fallback probe path:
    1. row-wise metadata/length probe over all shared tensors
    2. equal-length payload signature probe on first 2500 shared rows (`allow-metadata-mismatch`) to avoid full-scan hangs

- Key outcomes:
  - Row-wise meta/len (`5746` shared, `5745` readable):
    - `meta_mismatch_any=5745`
    - `meta_full_match=0`
    - `meta_metadata_mismatch_type=5745`
    - `meta_data_len_mismatch=5546`
    - `meta_unreadable_only_file=1` (`__text_feature_extractor__[t-video_aggregate_embed-0-0]`)
  - Top mismatch families (row-wise):
    - `__dit__`: `5484/5484`
    - `__text_video_connector__`: `128/128`
    - `__text_audio_connector__`: `128/128`
    - `__text_feature_extractor__`: `3/4`
    - connector learnable registers: `1/1` each
  - Equal-length payload probe (first `2500` rows):
    - `payload_metadata_mismatch=2500`
    - `payload_data_len_equal=23`
    - among equal-length rows: `0` head-signature matches, `23` head-signature mismatches
    - small-hash checks: `21/21` mismatched
    - all sampled payload mismatches concentrated in `__dit__`

- Interpretation:
  - Custom q6p main remains globally non-equivalent to official q6p in both metadata type and data length, dominated by `__dit__` but including connector/text-feature families.
  - The crash class is consistent with broad serializer/content divergence rather than a narrow single-row issue.

- Artifacts:
  - `output/run105_converter_focus_20260703_124301/summary.txt`
  - `output/run105_converter_focus_20260703_124301/summary_metrics.txt`
  - `output/run105_converter_focus_20260703_124301/meta_len_rowwise.json`
  - `output/run105_converter_focus_20260703_124301/meta_len_rowwise.tsv`
  - `output/run105_converter_focus_20260703_124301/meta_len_rowwise.log`
  - `output/run105_converter_focus_20260703_124301/meta_len_mismatch_names.txt`
  - `output/run105_converter_focus_20260703_124301/payload_mismatch_first2500.json`
  - `output/run105_converter_focus_20260703_124301/payload_mismatch_first2500.log`
  - `output/run105_converter_focus_20260703_124301/payload_mismatch_names_first2500.txt`
  - `output/run105_converter_focus_20260703_124301/deep_diff.log`

## Run 106 (2026-07-06): Connector+DIT-Slice Content Alignment Canary (In-Place Candidate Copy)

- Goal:
  - Test whether a surgical content alignment on highest-signal families from Run 105 can mitigate the custom q6p main stage-gate crash class.

- Method:
  - Reused Run 105 mismatch names to build focused patch list:
    - all connector/text-feature rows: `__text_video_connector__`, `__text_audio_connector__`, `__text_feature_extractor__`, and both connector learnable register families
    - plus bounded DIT slice: first `256` `__dit__` rows
  - Name-list sizes:
    - connector rows: `261`
    - DIT slice rows: `256`
    - union: `517`
  - Candidate handling:
    - in-place patch applied to copied custom candidate:
      - `dt-models/10_e_v1_bf16_regen_0_q6p_run106_connector_ditslice.ckpt`
      - `dt-models/10_e_v1_bf16_regen_0_q6p_run106_connector_ditslice.ckpt-tensordata`
  - Alignment tool:
    - `tools/dt_align_ckpt_content_subset.py` with both dim+data names set to the same 517-name list.
  - Validation sequence:
    1. targeted content probe before patch
    2. dry-run + apply
    3. targeted content probe after patch
    4. matched long-wait bounded stage-gate canary (`--timeout-sec 1800 --max-responses 3 --steps 8 --seed 4242`) on patched candidate then official control

- Key outcomes:
  - Pre-patch targeted probe (`517` selected):
    - `metadata_mismatch_type=517`
    - `data_small_sha256_compared=170`
    - `data_small_sha256_mismatch=138`
  - Apply results:
    - `rows_updated=517`
    - `dim_rows_updated=517`
    - `data_rows_updated=517`
    - `post_dim_head_mismatch=0`
    - `post_data_head_mismatch=0`
  - Post-patch targeted probe (`517` selected):
    - `metadata_mismatch_type=517` (unchanged)
    - `data_small_sha256_compared=413`
    - `data_small_sha256_mismatch=0`
  - Stage-gate canary (patched candidate):
    - `canary_rc=0`, `RESULT=PASS`
    - signposts: `textEncoded -> imageEncoded -> sampling`
    - bounded stop at `responses=3`, stream finished, no crash markers.
  - Matched official stage-gate control:
    - `canary_rc=0`, `RESULT=PASS`
    - same signpost progression and bounded-stop behavior.

- Interpretation:
  - Surgical connector+DIT-slice content alignment removed observed payload/head mismatches in the selected subset and avoided the immediate stage-gate crash for this bounded scenario.
  - Metadata type mismatch remains for selected rows, and global mismatch outside the 517-row subset is still expected; this is evidence of partial mitigation, not full model equivalence.

- Artifacts:
  - `output/run106_connector_ditslice_20260703_131008/summary.txt`
  - `output/run106_connector_ditslice_20260703_131008/connector_names.txt`
  - `output/run106_connector_ditslice_20260703_131008/dit_slice_256.txt`
  - `output/run106_connector_ditslice_20260703_131008/patch_names_connector_plus_dit256.txt`
  - `output/run106_connector_ditslice_20260703_131008/targeted_before.log`
  - `output/run106_connector_ditslice_20260703_131008/targeted_before.json`
  - `output/run106_connector_ditslice_20260703_131008/targeted_before.md`
  - `output/run106_connector_ditslice_20260703_131008/content_align_dryrun.log`
  - `output/run106_connector_ditslice_20260703_131008/content_align_apply.log`
  - `output/run106_connector_ditslice_20260703_131008/targeted_after.log`
  - `output/run106_connector_ditslice_20260703_131008/targeted_after.json`
  - `output/run106_connector_ditslice_20260703_131008/targeted_after.md`
  - `output/run106_connector_ditslice_20260703_131008/patched_stagegate.log`
  - `output/run106_connector_ditslice_20260703_131008/official_stagegate.log`
  - `output/q6p_canary_run106_patched_connector_ditslice_stagegate_longwait_20260706_054624/client.log`
  - `output/q6p_canary_run106_patched_connector_ditslice_stagegate_longwait_20260706_054624/server.log`
  - `output/q6p_canary_run106_official_q6p_stagegate_longwait_control_20260706_054624/client.log`
  - `output/q6p_canary_run106_official_q6p_stagegate_longwait_control_20260706_054624/server.log`

## Run 107 (2026-07-06): Full-Gate A/B on Patched Candidate vs Official Control

- Goal:
  - Validate whether Run 106 patched candidate stability holds in strict full-gate mode and compare output quality against official under matched settings.

- Method:
  - Matched full-gate profile on both models:
    - `--timeout-sec 1800 --max-responses 0 --require-complete-stream --require-final-output`
    - `256x256`, `steps=8`, `seed=4242`
  - Models:
    - patched custom: `10_e_v1_bf16_regen_0_q6p_run106_connector_ditslice.ckpt`
    - official control: `ltx_2.3_22b_distilled_1.1_q6p.ckpt`
  - Converted generated tensor outputs to playable media with `tools/dt_tensor_to_playable.py`.
  - Computed grayscale/entropy metrics on representative PNGs.

- Key outcomes:
  - Full-gate runtime status:
    - patched custom: `canary_rc=0`, `post_echo_rc=0`, `RESULT=PASS`
    - official control: `canary_rc=0`, `post_echo_rc=0`, `RESULT=PASS`
  - Stream/output shape:
    - patched custom: `responses=14`, `images written=1`, `audio written=0`, `preview frames=5`
    - official control: `responses=23`, `images written=9`, `audio written=1`, `preview frames=5`
  - Quantitative PNG comparison:
    - patched: `gray_mean=0.476247`, `gray_std=0.130545`, `entropy=7.096774`
    - official: `gray_mean=0.461745`, `gray_std=0.330333`, `entropy=7.483953`
  - Quality interpretation from metrics:
    - patched output remains low-contrast and lower-entropy than official despite full-gate PASS, indicating partial runtime recovery without equivalent generation quality.

- Interpretation:
  - The connector+DIT-slice patch materially improves runtime stability (now passing both stage-gate and full-gate in this profile), but does not yet recover official-like output richness or motion/audio behavior.
  - This further supports a broad content-divergence root cause and suggests additional alignment coverage is required beyond the current 517-row subset.

- Artifacts:
  - `output/run107_fullgate_patched_vs_official_20260706_055513/summary.txt`
  - `output/run107_fullgate_patched_vs_official_20260706_055513/patched_finalgate.log`
  - `output/run107_fullgate_patched_vs_official_20260706_055513/official_finalgate.log`
  - `output/run107_fullgate_patched_vs_official_20260706_055513/patched_playable.log`
  - `output/run107_fullgate_patched_vs_official_20260706_055513/official_playable.log`
  - `output/run107_fullgate_patched_vs_official_20260706_055513/png_metrics_run107.txt`
  - `output/run107_fullgate_patched_vs_official_20260706_055513/png_metrics_run107.json`
  - `output/run107_fullgate_patched_vs_official_20260706_055513/media/patched/patched_run107.png`
  - `output/run107_fullgate_patched_vs_official_20260706_055513/media/patched/patched_run107.gif`
  - `output/run107_fullgate_patched_vs_official_20260706_055513/media/patched/patched_run107.mp4`
  - `output/run107_fullgate_patched_vs_official_20260706_055513/media/official/official_run107.png`
  - `output/run107_fullgate_patched_vs_official_20260706_055513/media/official/official_run107.gif`
  - `output/run107_fullgate_patched_vs_official_20260706_055513/media/official/official_run107.mp4`
  - `output/run107_fullgate_patched_vs_official_20260706_055513/media/official/official_run107.wav`
  - `output/q6p_canary_run107_patched_connector_ditslice_finalgate_20260706_055513/client.log`
  - `output/q6p_canary_run107_patched_connector_ditslice_finalgate_20260706_055513/server.log`
  - `output/q6p_canary_run107_official_q6p_finalgate_control_20260706_055513/client.log`
  - `output/q6p_canary_run107_official_q6p_finalgate_control_20260706_055513/server.log`

## Run 108 (2026-07-06): DIT Coverage Expansion (Connector + DIT1024) Full-Gate Recheck

- Goal:
  - Move directly toward coherent custom output by expanding aligned DIT coverage from `256` to `1024` while keeping connector/text-feature families fully aligned.

- Method:
  - Fresh candidate copy:
    - `dt-models/10_e_v1_bf16_regen_0_q6p_run108_connector_dit1024.ckpt`
  - Patch list built from Run 105 mismatch names:
    - connector/text-feature families: `261`
    - first `1024` DIT names
    - union selected: `1285`
  - Applied `tools/dt_align_ckpt_content_subset.py` with both dim+data names set to the 1285-name list.
  - Validation and runtime sequence:
    1. targeted content probe before patch
    2. dry-run + apply
    3. targeted content probe after patch
    4. matched full-gate A/B (`1800s`, `max-responses=0`, `require-complete-stream`, `require-final-output`, `256x256`, `steps=8`, `seed=4242`) vs official q6p control
    5. media conversion + PNG quality metrics

- Key outcomes:
  - Pre-patch targeted probe (`1285` selected):
    - `metadata_mismatch_type=1285`
    - `data_small_sha256_compared=458`
    - `data_small_sha256_mismatch=426`
  - Apply:
    - `rows_updated=1285`, `dim_rows_updated=1285`, `data_rows_updated=1285`
    - `post_dim_head_mismatch=0`, `post_data_head_mismatch=0`
    - `rows_skipped_dataerror=0`
  - Post-patch targeted probe (`1285` selected):
    - `metadata_mismatch_type=1285` (unchanged)
    - `data_small_sha256_compared=913`
    - `data_small_sha256_mismatch=0`
  - Full-gate A/B runtime:
    - patched custom: `canary_rc=0`, `post_echo_rc=0`, `RESULT=PASS`
    - official control: `canary_rc=0`, `post_echo_rc=0`, `RESULT=PASS`
  - Full-gate output shape remained unchanged from Run 107:
    - patched custom: `responses=14`, `images=1`, `audio=0`, `preview frames=5`
    - official control: `responses=23`, `images=9`, `audio=1`, `preview frames=5`
  - PNG metrics (patched vs official):
    - patched: `gray_mean=0.470583`, `gray_std=0.128570`, `entropy=7.074494`
    - official: `gray_mean=0.461745`, `gray_std=0.330333`, `entropy=7.483953`
  - Run107->Run108 patched delta:
    - `gray_mean=-0.005664`
    - `gray_std=-0.001976`
    - `entropy=-0.022280`

- Interpretation:
  - Increasing DIT alignment coverage to 1024 preserved runtime stability but did not improve final output richness and slightly worsened patched image entropy/contrast metrics.
  - This indicates the remaining quality gap is not solved by this contiguous DIT-prefix expansion strategy alone.

- Artifacts:
  - `output/run108_connector_dit1024_20260706_061243/summary.txt`
  - `output/run108_connector_dit1024_20260706_061243/connector_names.txt`
  - `output/run108_connector_dit1024_20260706_061243/dit_slice_1024.txt`
  - `output/run108_connector_dit1024_20260706_061243/patch_names_connector_plus_dit1024.txt`
  - `output/run108_connector_dit1024_20260706_061243/targeted_before.log`
  - `output/run108_connector_dit1024_20260706_061243/targeted_before.json`
  - `output/run108_connector_dit1024_20260706_061243/targeted_before.md`
  - `output/run108_connector_dit1024_20260706_061243/content_align_dryrun.log`
  - `output/run108_connector_dit1024_20260706_061243/content_align_apply.log`
  - `output/run108_connector_dit1024_20260706_061243/targeted_after.log`
  - `output/run108_connector_dit1024_20260706_061243/targeted_after.json`
  - `output/run108_connector_dit1024_20260706_061243/targeted_after.md`
  - `output/run108_connector_dit1024_20260706_061243/patched_finalgate.log`
  - `output/run108_connector_dit1024_20260706_061243/official_finalgate.log`
  - `output/run108_connector_dit1024_20260706_061243/patched_playable.log`
  - `output/run108_connector_dit1024_20260706_061243/official_playable.log`
  - `output/run108_connector_dit1024_20260706_061243/png_metrics_run108.txt`
  - `output/run108_connector_dit1024_20260706_061243/png_metrics_run108.json`
  - `output/run108_connector_dit1024_20260706_061243/media/patched/patched_run108.png`
  - `output/run108_connector_dit1024_20260706_061243/media/patched/patched_run108.gif`
  - `output/run108_connector_dit1024_20260706_061243/media/patched/patched_run108.mp4`
  - `output/run108_connector_dit1024_20260706_061243/media/official/official_run108.png`
  - `output/run108_connector_dit1024_20260706_061243/media/official/official_run108.gif`
  - `output/run108_connector_dit1024_20260706_061243/media/official/official_run108.mp4`
  - `output/run108_connector_dit1024_20260706_061243/media/official/official_run108.wav`
  - `output/q6p_canary_run108_patched_connector_dit1024_finalgate_20260706_062654/client.log`
  - `output/q6p_canary_run108_patched_connector_dit1024_finalgate_20260706_062654/server.log`
  - `output/q6p_canary_run108_official_q6p_finalgate_control_20260706_062654/client.log`
  - `output/q6p_canary_run108_official_q6p_finalgate_control_20260706_062654/server.log`

## Run 109 (2026-07-06): Stratified DIT1024 Coverage (Non-Prefix) Full-Gate Recheck

- Goal:
  - Test whether DIT row *selection strategy* (distributed across full DIT mismatch space) improves custom output quality, keeping row count fixed at `1024` DIT names plus full connector/text-feature set.

- Method:
  - Fresh candidate copy:
    - `dt-models/10_e_v1_bf16_regen_0_q6p_run109_connector_dit1024_stratified.ckpt`
  - Selection list:
    - connector/text-feature families: `261`
    - DIT names: `1024` chosen by deterministic stratified spacing across all DIT mismatch names (not first-prefix slicing)
    - union selected: `1285`
  - Applied `tools/dt_align_ckpt_content_subset.py` with both dim+data names set to the stratified 1285-name list.
  - Validation and runtime sequence:
    1. targeted content probe before patch
    2. dry-run + apply
    3. targeted content probe after patch
    4. matched full-gate A/B (`1800s`, `max-responses=0`, `require-complete-stream`, `require-final-output`, `256x256`, `steps=8`, `seed=4242`) vs official q6p control
    5. media conversion + PNG quality metrics (including Run108->Run109 delta)

- Key outcomes:
  - Pre-patch targeted probe (`1285` selected):
    - `metadata_mismatch_type=1285`
    - `data_small_sha256_compared=349`
    - `data_small_sha256_mismatch=317`
  - Apply:
    - `rows_updated=1285`, `dim_rows_updated=1285`, `data_rows_updated=1285`
    - `post_dim_head_mismatch=0`, `post_data_head_mismatch=0`
    - `rows_skipped_dataerror=0`
  - Post-patch targeted probe (`1285` selected):
    - `metadata_mismatch_type=1285` (unchanged)
    - `data_small_sha256_compared=777`
    - `data_small_sha256_mismatch=0`
  - Full-gate A/B runtime:
    - patched custom: `canary_rc=0`, `post_echo_rc=0`, `RESULT=PASS`
    - official control: `canary_rc=0`, `post_echo_rc=0`, `RESULT=PASS`
  - Full-gate output shape remained unchanged vs Runs 107/108:
    - patched custom: `responses=14`, `images=1`, `audio=0`, `preview frames=5`
    - official control: `responses=23`, `images=9`, `audio=1`, `preview frames=5`
  - PNG metrics (patched vs official):
    - patched: `gray_mean=0.464466`, `gray_std=0.125271`, `entropy=7.035664`
    - official: `gray_mean=0.461745`, `gray_std=0.330333`, `entropy=7.483953`
  - Run108->Run109 patched delta:
    - `gray_mean=-0.006117`
    - `gray_std=-0.003299`
    - `entropy=-0.038830`

- Interpretation:
  - Changing DIT selection from contiguous-prefix to stratified-coverage did not improve custom first-frame quality and again slightly worsened contrast/entropy metrics.
  - At fixed row count (`1285` union), DIT selection strategy change alone was insufficient to move toward coherent custom output.

- Artifacts:
  - `output/run109_connector_dit1024_stratified_20260706_063708/summary.txt`
  - `output/run109_connector_dit1024_stratified_20260706_063708/connector_names.txt`
  - `output/run109_connector_dit1024_stratified_20260706_063708/dit_all_names.txt`
  - `output/run109_connector_dit1024_stratified_20260706_063708/dit_stratified_1024.txt`
  - `output/run109_connector_dit1024_stratified_20260706_063708/patch_names_connector_plus_dit1024_stratified.txt`
  - `output/run109_connector_dit1024_stratified_20260706_063708/targeted_before.log`
  - `output/run109_connector_dit1024_stratified_20260706_063708/targeted_before.json`
  - `output/run109_connector_dit1024_stratified_20260706_063708/targeted_before.md`
  - `output/run109_connector_dit1024_stratified_20260706_063708/content_align_dryrun.log`
  - `output/run109_connector_dit1024_stratified_20260706_063708/content_align_apply.log`
  - `output/run109_connector_dit1024_stratified_20260706_063708/targeted_after.log`
  - `output/run109_connector_dit1024_stratified_20260706_063708/targeted_after.json`
  - `output/run109_connector_dit1024_stratified_20260706_063708/targeted_after.md`
  - `output/run109_connector_dit1024_stratified_20260706_063708/patched_finalgate.log`
  - `output/run109_connector_dit1024_stratified_20260706_063708/official_finalgate.log`
  - `output/run109_connector_dit1024_stratified_20260706_063708/patched_playable.log`
  - `output/run109_connector_dit1024_stratified_20260706_063708/official_playable.log`
  - `output/run109_connector_dit1024_stratified_20260706_063708/png_metrics_run109.txt`
  - `output/run109_connector_dit1024_stratified_20260706_063708/png_metrics_run109.json`
  - `output/run109_connector_dit1024_stratified_20260706_063708/media/patched/patched_run109.png`
  - `output/run109_connector_dit1024_stratified_20260706_063708/media/patched/patched_run109.gif`
  - `output/run109_connector_dit1024_stratified_20260706_063708/media/patched/patched_run109.mp4`
  - `output/run109_connector_dit1024_stratified_20260706_063708/media/official/official_run109.png`
  - `output/run109_connector_dit1024_stratified_20260706_063708/media/official/official_run109.gif`
  - `output/run109_connector_dit1024_stratified_20260706_063708/media/official/official_run109.mp4`
  - `output/run109_connector_dit1024_stratified_20260706_063708/media/official/official_run109.wav`
  - `output/q6p_canary_run109_patched_connector_dit1024_stratified_finalgate_20260706_064651/client.log`
  - `output/q6p_canary_run109_patched_connector_dit1024_stratified_finalgate_20260706_064651/server.log`
  - `output/q6p_canary_run109_official_q6p_finalgate_control_20260706_064651/client.log`
  - `output/q6p_canary_run109_official_q6p_finalgate_control_20260706_064651/server.log`

## Run 110 (2026-07-06): Semantic CoreAV Targeting (Connector + DIT Core Families)

- Goal:
  - Move beyond fixed-size DIT row geometry changes by targeting a semantic DIT bundle more likely to influence coherent output behavior while preserving full connector coverage.

- Method:
  - Fresh candidate copy:
    - `dt-models/10_e_v1_bf16_regen_0_q6p_run110_semantic_coreav.ckpt`
  - Selection list (`2573` union rows):
    - connector/text-feature families: `261`
    - semantic DIT CoreAV subset: `2312`
      - output/gate families (`*_gate`, `*_out_proj`, selected `*_linear1`)
      - AV q/k/v/o for primary audio/video branches (`t-a_*`, `t-x_*`)
      - branch norms (`*_norm_q`, `*_norm_k` across selected AV/cross branches)
      - embedders (`t-a2v_embedder_*`, `t-a_embedder`, `t-x_embedder`)
  - Applied `tools/dt_align_ckpt_content_subset.py` with both dim+data names set to the `2573` semantic-union names.
  - Validation sequence completed before runtime gate:
    1. targeted content probe before patch
    2. dry-run + apply
    3. targeted content probe after patch

- Key outcomes (content alignment stage):
  - Pre-patch targeted probe (`2573` selected):
    - `metadata_mismatch_type=2573`
    - `data_small_sha256_compared=986`
    - `data_small_sha256_mismatch=954`
  - Apply:
    - `rows_updated=2573`, `dim_rows_updated=2573`, `data_rows_updated=2573`
    - `post_dim_head_mismatch=0`, `post_data_head_mismatch=0`
    - `rows_skipped_dataerror=0`
  - Post-patch targeted probe (`2573` selected):
    - `metadata_mismatch_type=2573` (unchanged)
    - `data_small_sha256_compared=1969`
    - `data_small_sha256_mismatch=0`

- Key outcomes (runtime gate stage):
  - First matched full-gate A/B attempt failed for both patched and official control:
    - patched custom: `canary_rc=1`, `post_echo_rc=1`, `RESULT=FAIL canary rc=1`
    - official control: `canary_rc=1`, `post_echo_rc=1`, `RESULT=FAIL canary rc=1`
  - Official control was retried under multiple profiles and remained failing:
    - standard retries (`retry1`, `retry2`): both failed (`canary_rc=1`)
    - `--server-no-flash-attention`: failed (`canary_rc=1`)
    - `--server-cpu-offload --server-no-flash-attention`: failed (`canary_rc=1`), with repeated CUDA/CUFILE initialization errors in server log.
  - Because official control did not recover, Run 110 did not produce a valid matched A/B quality comparison and generated no new PNG metrics.

- Interpretation:
  - Semantic subset alignment itself worked as intended at selected-row content level (`small_sha256_mismatch -> 0`), but runtime evaluation was blocked by environment-level instability affecting official and patched runs alike.
  - Run 110 should be treated as:
    - content-alignment PASS,
    - runtime-quality verdict INCONCLUSIVE (baseline control unavailable).

- Artifacts:
  - `output/run110_semantic_coreav_20260706_070329/summary.txt`
  - `output/run110_semantic_coreav_20260706_070329/connector_names.txt`
  - `output/run110_semantic_coreav_20260706_070329/dit_semantic_coreav_names.txt`
  - `output/run110_semantic_coreav_20260706_070329/patch_names_connector_plus_dit_semantic_coreav.txt`
  - `output/run110_semantic_coreav_20260706_070329/targeted_before.log`
  - `output/run110_semantic_coreav_20260706_070329/targeted_before.json`
  - `output/run110_semantic_coreav_20260706_070329/targeted_before.md`
  - `output/run110_semantic_coreav_20260706_070329/content_align_dryrun.log`
  - `output/run110_semantic_coreav_20260706_070329/content_align_apply.log`
  - `output/run110_semantic_coreav_20260706_070329/targeted_after.log`
  - `output/run110_semantic_coreav_20260706_070329/targeted_after.json`
  - `output/run110_semantic_coreav_20260706_070329/targeted_after.md`
  - `output/run110_semantic_coreav_20260706_070329/patched_finalgate.log`
  - `output/run110_semantic_coreav_20260706_070329/official_finalgate.log`
  - `output/run110_semantic_coreav_20260706_070329/official_finalgate_retry1.log`
  - `output/run110_semantic_coreav_20260706_070329/official_finalgate_retry2.log`
  - `output/run110_semantic_coreav_20260706_070329/official_finalgate_noflash.log`
  - `output/run110_semantic_coreav_20260706_070329/official_finalgate_cpuoff_noflash.log`
  - `output/q6p_canary_run110_patched_semantic_coreav_finalgate_20260706_084209/client.log`
  - `output/q6p_canary_run110_patched_semantic_coreav_finalgate_20260706_084209/server.log`
  - `output/q6p_canary_run110_official_q6p_finalgate_control_20260706_084241/client.log`
  - `output/q6p_canary_run110_official_q6p_finalgate_control_20260706_084241/server.log`
  - `output/q6p_canary_run110_official_q6p_finalgate_retry1_20260706_084345/client.log`
  - `output/q6p_canary_run110_official_q6p_finalgate_retry1_20260706_084345/server.log`
  - `output/q6p_canary_run110_official_q6p_finalgate_retry2_20260706_084418/client.log`
  - `output/q6p_canary_run110_official_q6p_finalgate_retry2_20260706_084418/server.log`
  - `output/q6p_canary_run110_official_q6p_finalgate_noflash_20260706_084512/client.log`
  - `output/q6p_canary_run110_official_q6p_finalgate_noflash_20260706_084512/server.log`
  - `output/q6p_canary_run110_official_q6p_finalgate_cpuoff_noflash_20260706_084557/client.log`
  - `output/q6p_canary_run110_official_q6p_finalgate_cpuoff_noflash_20260706_084557/server.log`

## Run 110R (2026-07-06): Environment Recovery Check (Post-Run110)

- Goal:
  - Re-establish a valid official control baseline after Run 110 dual-fail outcomes before any further custom quality interpretation.

- Method:
  - Executed official-only canary recovery sweep:
    - official q6p default
    - official q6p no-flash
  - Executed known-good historical custom sanity check:
    - `10_e_v1_bf16_regen_0_q8p_20260630.ckpt`
  - Executed alternate server binary check:
    - `--grpc-bin draw-things-community/.build/release/gRPCServerCLI`
  - Performed runtime environment preflight:
    - `/dev/nvidia*` device presence
    - NVML visibility

- Key outcomes:
  - Official q6p remained failing in both default and no-flash profiles:
    - `canary_rc=1`, `post_echo_rc=1`, `RESULT=FAIL canary rc=1`
  - Historical custom q8p sanity model also failed with the same class:
    - `Signal 11`, bad pointer, `_ccv_nnc_index_select_forw`
  - Alternate source-built binary did not recover generation path:
    - aborted with assertion path (`memory_type == CCV_TENSOR_CPU_MEMORY`) and `RESULT=FAIL canary rc=1`
  - Environment preflight showed GPU device loss in-container:
    - `/dev/nvidia*` missing
    - `nvidia-smi` unusable (`Failed to initialize NVML: N/A`)

- Interpretation:
  - This is an infrastructure/runtime blocker, not a model-content discriminator for current runs.
  - Until GPU device visibility is restored in the container, full-gate A/B results are not trustworthy for converter-quality decisions.

- Artifacts:
  - `output/run110_env_recovery_20260706_092927/summary.txt`
  - `output/run110_env_recovery_20260706_092927/official_q6p_default.log`
  - `output/run110_env_recovery_20260706_092927/official_q6p_noflash.log`
  - `output/run110_env_recovery_20260706_092927/official_q6p_default_stderr_capture.log`
  - `output/run110_env_recovery_20260706_092927/sanity_custom_q8p.log`
  - `output/run110_env_recovery_20260706_092927/altbin_official_q6p.log`
  - `output/q6p_canary_official_q6p_default/client.log`
  - `output/q6p_canary_official_q6p_default/server.log`
  - `output/q6p_canary_official_q6p_noflash/client.log`
  - `output/q6p_canary_official_q6p_noflash/server.log`
  - `output/q6p_canary_official_q6p_default_stderr_capture_20260706_093238/client.log`
  - `output/q6p_canary_official_q6p_default_stderr_capture_20260706_093238/server.log`
  - `output/q6p_canary_sanity_custom_q8p_20260706_093248/client.log`
  - `output/q6p_canary_sanity_custom_q8p_20260706_093248/server.log`
  - `output/q6p_canary_altbin_official_q6p_20260706_093325/client.log`
  - `output/q6p_canary_altbin_official_q6p_20260706_093325/server.log`
