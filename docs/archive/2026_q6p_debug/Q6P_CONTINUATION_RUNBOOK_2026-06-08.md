# Q6P Continuation Runbook (Low-Space, 2026-06-08)

This runbook is the fastest way to continue q6p remediation under strict storage limits.

## 1) Preconditions

- Work from repo root:

```bash
cd /workspaces/drawthings-linux-toolkit
```

- Ensure target/source files exist:
  - `dt-models/10_e_v1_bf16_regen_0_q6p.ckpt`
  - `dt-models/10_e_v1_bf16_regen_0_f16.ckpt`

- Preferred fallback baseline (official q6p currently absent):
  - `dt-models/ltx_2.3_22b_distilled_q6p_forcedfix_clipfix2_20260602.ckpt`

## 2) Fast rerun (no baseline dependency)

Uses precomputed 770-row patch set and keeps storage usage low.

```bash
bash tools/run_q6p_inplace_dimfix_from_f16.sh \
  --canary-timeout-sec 120 \
  --max-responses 10
```

Outputs are written to:

- `output/q6p_inplace_dimfix_<timestamp>/client.log`
- `output/q6p_inplace_dimfix_<timestamp>/server.log`

## 3) Rebuild names using clipfix2 as baseline (official baseline missing)

### 3.1 Conservative direct rebuild command

If no second working comparator is available, use clipfix2 for both baseline and reference.

```bash
bash tools/run_q6p_inplace_dimfix_from_f16.sh \
  --rebuild-names \
  --baseline-q6p dt-models/ltx_2.3_22b_distilled_q6p_forcedfix_clipfix2_20260602.ckpt \
  --reference-q6p dt-models/ltx_2.3_22b_distilled_q6p_forcedfix_clipfix2_20260602.ckpt \
  --names-file tools/patch_sets/10_e_v1_q6p_dimfix_from_clipfix2_20260608.txt \
  --canary-timeout-sec 120 \
  --max-responses 10
```

Note:

- This broadens selection behavior because `reference_dim_mismatch` tends toward zero when baseline and reference are the same file.

### 3.2 Preferred rebuild when another working q6p comparator is available

Use clipfix2 as baseline and another known-good q6p as reference:

```bash
bash tools/run_q6p_inplace_dimfix_from_f16.sh \
  --rebuild-names \
  --baseline-q6p dt-models/ltx_2.3_22b_distilled_q6p_forcedfix_clipfix2_20260602.ckpt \
  --reference-q6p /path/to/another_working_q6p.ckpt \
  --names-file tools/patch_sets/10_e_v1_q6p_dimfix_from_clipfix2_vs_altref.txt
```

## 4) Split mode for debugging (patch now, canary later)

Patch and validate only:

```bash
bash tools/run_q6p_inplace_dimfix_from_f16.sh --skip-canary
```

Then run canary with custom bounds by rerunning without `--skip-canary` once ready.

Canary-only helper (no patch step):

```bash
bash tools/run_q6p_canary_once.sh \
  --model 10_e_v1_bf16_regen_0_q6p.ckpt \
  --timeout-sec 90 \
  --max-responses 10 \
  --tag manual_canary_YYYYMMDD_HHMMSS
```

Expanded-set branch used in latest attempt:

```bash
bash tools/run_q6p_inplace_dimfix_from_f16.sh \
  --names-file tools/patch_sets/10_e_v1_q6p_dimfix770_plus_extra512_clipfix2base_20260608.txt \
  --tag 20260608_clipfix2_plus_extra512 \
  --canary-timeout-sec 120 \
  --max-responses 10 \
  --min-free-gb 50
```

Row-wise safe residual probe (metadata/length only, no heavy blob signatures):

```bash
/workspaces/drawthings-linux-toolkit/.venv/bin/python tools/dt_probe_ckpt_meta_len_rowwise.py \
  --file dt-models/10_e_v1_bf16_regen_0_q6p.ckpt \
  --baseline dt-models/10_e_v1_bf16_regen_0_f16.ckpt \
  --sample-limit 20 \
  --top-prefix-count 50 \
  --progress-every 500 \
  --out-json output/probe_meta_len_all5746_post_metachunkfix_20260608.json \
  --out-tsv output/probe_meta_len_all5746_post_metachunkfix_20260608.tsv \
  --out-mismatch-names tools/patch_sets/10_e_v1_q6p_post_metachunkfix_mismatch_meta_len_all5746_20260608.txt
```

Metadata-only parity branch (micro-batch mode to avoid D-state/exit-137 stalls):

```bash
mkdir -p tools/patch_sets/run005_meta_chunks_20260608
split -d -l 900 tools/patch_sets/10_e_v1_q6p_run005_mismatch_meta_len_all5746_20260608.txt \
  tools/patch_sets/run005_meta_chunks_20260608/chunk_

# If a chunk stalls/kills, recursively split that chunk (300 -> 100 -> 25)
# and continue applying with the same command:
/workspaces/drawthings-linux-toolkit/.venv/bin/python tools/dt_patch_ckpt_metadata_subset.py \
  --file dt-models/10_e_v1_bf16_regen_0_q6p.ckpt \
  --baseline dt-models/10_e_v1_bf16_regen_0_f16.ckpt \
  --names-file tools/patch_sets/run005_meta_chunks_20260608/chunk_XX \
  --journal-mode delete \
  --progress-every 100
```

Post-metachunkfix canary command:

```bash
bash tools/run_q6p_canary_once.sh \
  --model 10_e_v1_bf16_regen_0_q6p.ckpt \
  --timeout-sec 120 \
  --max-responses 10 \
  --tag 20260608_post_metachunkfix
```

## 5) Triage checklist after each run

1. Check whether `client.log` contains `response #1`.
2. Check server stack in `server.log` for `ccv_nnc_tensor_read -> ccv_cnnp_model_read`.
3. Record run folder name and command variant used.
4. Preserve the names file used for that run (for reproducibility).

## 6) Decision gates

- If run reaches streamed responses:
  - continue increasing `--max-responses` gradually.
- If run times out with no response and same crash stack:
  - treat current candidate set as insufficient and iterate selection strategy.
- If canary returns `124` with request-start only (no stack, no response #1):
  - treat as runtime stall branch; keep batch windows small and verify after each window.
- If official baseline becomes available again:
  - rerun `--rebuild-names` using official baseline for higher-fidelity comparator behavior.

## 7) Related documents

- Main handoff findings: `Q6P_HANDOFF_FINDINGS_2026-06-08.md`
- Long historical report: `CONVERSION_TOOL_FINDINGS_2026-05-28.md`
- Workspace operations manual: `WORKSPACE_MANUAL.md`
- Append-only iteration log: `Q6P_RUN_LOG.md`

## 8) Latest verified state (Run 006)

Run 006 closed the previous residual `data_len_mismatch=2728` branch to zero under row-wise probe metrics.

Key artifacts:

- `output/20260608_run006_payloadfix/probe_meta_len_rowwise_after_leafretry_20260608.json`
- `output/20260608_run006_payloadfix/probe_mismatch_names_after_leafretry_20260608.txt`
- `output/q6p_canary_20260608_run006_post_leafretry/client.log`
- `output/q6p_canary_20260608_run006_post_leafretry/server.log`

Measured post-retry probe state:

- `metadata_mismatch_type=0`
- `metadata_mismatch_format=0`
- `metadata_mismatch_datatype=0`
- `dim_len_mismatch=0`
- `data_len_mismatch=0`
- `mismatch_any=0`
- `full_match=5745`
- `unreadable_both=1`

Canary state after full parity:

- `canary_rc=124`
- `post_echo_rc=0`
- no `response #1` in client log

Implication:

- Continue with byte-level payload semantics investigation (equal-length rows), not additional metadata/length parity sweeps.

## 9) Immediate continuation commands

Reproduce latest canary baseline quickly:

```bash
bash tools/run_q6p_canary_once.sh \
  --model 10_e_v1_bf16_regen_0_q6p.ckpt \
  --timeout-sec 120 \
  --max-responses 10 \
  --tag rerun_post_run006_baseline
```

Reconfirm row-wise parity before/after any next payload experiment:

```bash
/workspaces/drawthings-linux-toolkit/.venv/bin/python tools/dt_probe_ckpt_meta_len_rowwise.py \
  --file dt-models/10_e_v1_bf16_regen_0_q6p.ckpt \
  --baseline dt-models/10_e_v1_bf16_regen_0_f16.ckpt \
  --progress-every 400 \
  --out-json output/probe_meta_len_rowwise_recheck_after_next_step_20260608.json \
  --out-tsv output/probe_meta_len_rowwise_recheck_after_next_step_20260608.tsv \
  --out-mismatch-names output/probe_meta_len_rowwise_recheck_after_next_step_20260608.txt
```
