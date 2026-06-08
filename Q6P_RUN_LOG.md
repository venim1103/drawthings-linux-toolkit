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
