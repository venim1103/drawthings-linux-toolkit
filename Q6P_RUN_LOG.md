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
