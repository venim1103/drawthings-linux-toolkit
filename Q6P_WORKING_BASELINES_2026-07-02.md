# Q6P Working Baselines And Resume Notes (2026-07-02)

Purpose:

- Preserve exact known-good references so future sessions do not re-discover baseline behavior.
- Record high-signal failure signatures and confounds observed during 2026-07-01..2026-07-02.
- Provide a restart checklist for continuing custom quantization work.

## 1) Confirmed Working Original Baseline

### Baseline model (official distilled q6p)

- Model key: `ltx_2.3_22b_distilled_1.1_q6p.ckpt`
- Runtime evidence directory:
  - `output/official_q6p_singleframe_wait30_20260701`
- Final-output evidence:
  - `canary_rc=0`
  - `responses=15`
  - `generation stream finished`
  - `images written: 1`
  - `audio written: 1`
  - no `UNAVAILABLE: Socket closed`
  - no SIGSEGV / illegal-instruction in server log
- Media evidence:
  - `output/official_q6p_singleframe_wait30_20260701/official_q6p_singleframe_wait30.png`
  - `output/official_q6p_singleframe_wait30_20260701/official_q6p_singleframe_wait30.gif`
  - `output/official_q6p_singleframe_wait30_20260701/official_q6p_singleframe_wait30.mp4`

Command used for the confirmed pass:

```bash
bash tools/run_q6p_canary_once.sh \
  --model ltx_2.3_22b_distilled_1.1_q6p.ckpt \
  --width 256 --height 256 --steps 8 \
  --timeout-sec 1800 --max-responses 0 --require-final-output \
  --tag official_q6p_singleframe_wait30_20260701
```

## 2) Acceptance Contract To Reuse (Do Not Skip)

A candidate is only considered working if all are true:

1. Stream-level success:
   - client log contains `generation stream finished`
   - at least one final payload (`images written >= 1` or `audio written >= 1`)
2. Runtime health:
   - server log has no crash signature (`Signal 11`, `Program crashed`, `Illegal instruction`)
3. Visual sanity:
   - convert final image tensor to PNG and inspect manually
   - reject preview-only / noise-only outputs even if stream completed

Tensor conversion command:

```bash
/workspaces/drawthings-linux-toolkit/.venv/bin/python tools/dt_tensor_to_playable.py \
  --image-bin <run_dir>/image_rXXXX_01.bin \
  --audio-bin <run_dir>/audio_rXXXX_01.bin \
  --out-dir <run_dir> \
  --base-name <label>
```

## 3) High-Signal Confounds (Observed)

1. Timeout confound:
   - short windows can produce false negatives (for this model, single frame may need ~15 min).
   - use 1800s timeout for final gates.

2. Alias/schema confound (`dt-models/custom.json`):
   - custom entry shape can crash independently of checkpoint bytes.
   - same underlying file can pass when addressed directly, then fail via alias.

3. Stage-gate confound:
   - reaching `textEncoded -> imageEncoded -> sampling` is necessary but not sufficient.
   - multiple candidates passed stage-gate but failed full-output or visual quality.

4. Stream-vs-quality confound:
   - some custom runs completed stream and wrote image payloads but decoded PNGs were visually noisy.
   - treat this as failure for model-quality objective.

## 4) Latest Custom Findings (2026-07-01..2026-07-02)

### What passed technically (stream/runtime)

- Raw-key full runs (with `custom.json` temporarily moved out of the way) completed with final image payloads:
  - `output/custom_q6p_main_singleframe_wait30_rawkey_20260701`
  - `output/custom_q6p_trace021_singleframe_wait30_rawkey_20260701`
  - `output/custom_q8p_control_singleframe_20260701`
- Persistent bounded raw-key stage-gate also passed for custom q8p under the same settings used in custom-f16 crash repro (`256x256`, `steps=8`, `seed=4242`):
   - `output/persistent_probe_custom_q8p_stagegate_20260702_083324`
   - reached `textEncoded -> imageEncoded -> sampling` with server staying alive.

### What failed functionally (quality/stability)

- Visual inspection for custom outputs above showed noisy/non-coherent PNGs during this session.
- Additional control confirmed the same for custom f16:
   - `output/custom_f16_control_singleframe_wait30_rawkey_20260702/custom_f16_control_singleframe_wait30_rawkey.png`
- Persistent A/B reconfirmation (matched `256x256`, `steps=8`, `seed=4242`, same prompt):
   - custom f16 raw-key crashed immediately before first streamed response (`UNAVAILABLE: Socket closed`), with server SIGSEGV at `ccv_nnc_tensor_read -> ccv_cnnp_model_read`:
      - `output/persistent_repro_custom_f16_crash_20260702_081813/`
      - `output/persistent_server_logs_20260702_081754/server.log`
   - official q6p control on fresh server stayed healthy and progressed to sampling (`response #9`) before bounded 600s cancel, with post-run echo healthy:
      - `output/persistent_control_official_short_20260702_082130/`
      - `output/persistent_server_logs_20260702_082117/server.log`
- Same-session full final-gate visual A/B (matched `256x256`, `steps=8`, `seed=4242`, same prompt):
   - custom q8p raw-key completed stream and wrote final image, but PNG remained noise/non-coherent:
      - `output/persistent_final_custom_q8p_20260702_095806/persistent_final_custom_q8p.png`
   - official q6p completed stream and wrote coherent image+audio (visual pass):
      - `output/persistent_final_official_q6p_20260702_095927/persistent_final_official_q6p.png`
- Seed-sensitivity check on custom q8p (same profile, seeds `4242/4243/4244`) was runtime-stable but visual-fail in all cases:
   - root: `output/persistent_seed_sweep_custom_q8p_20260702_110812/`
   - all three runs finished with final image payloads (`responses=14`, `images written=1`, `server_alive_after=1`), yet each decoded PNG was noise/non-coherent:
      - `seed_4242/persistent_seed_4242_custom_q8p.png`
      - `seed_4243/persistent_seed_4243_custom_q8p.png`
      - `seed_4244/persistent_seed_4244_custom_q8p.png`
   - sweep server remained healthy with no crash signatures:
      - `output/persistent_server_logs_20260702_110531_q8p_seed_sweep/server.log`
- Matched official q6p control sweep using the same profile and seeds (`4242/4243/4244`) was runtime-pass and visual-pass in all cases:
   - root: `output/persistent_seed_sweep_official_q6p_20260702_133851/`
   - all three runs finished with final image+audio payloads (`responses=15`, `images written=1`, `audio written=1`, `server_alive_after=1`)
   - all three decoded PNGs were coherent/non-noise:
      - `seed_4242/persistent_seed_4242_official_q6p.png`
      - `seed_4243/persistent_seed_4243_official_q6p.png`
      - `seed_4244/persistent_seed_4244_official_q6p.png`
   - control sweep server remained healthy with no crash signatures:
      - `output/persistent_server_logs_20260702_120218_official_seed_sweep/server.log`
- Quantitative image-statistics discriminator on the six matched-seed PNGs also separates custom vs official:
   - metrics file: `output/seed_sweep_png_metrics_20260702_135352.tsv`
   - aggregate means:
      - custom q8p: `gray_mean=0.504858`, `gray_std=0.097851`, `entropy=6.694596`
      - official q6p: `gray_mean=0.333414`, `gray_std=0.308807`, `entropy=7.122476`
   - interpretation: custom outputs are low-contrast, mid-tone clustered, and lower-entropy versus official controls, matching visual-noise findings.
- Additional same-settings raw-key discriminator (trace021 q6p vs official control):
   - custom trace021 q6p hard-crashed immediately (`UNAVAILABLE: Socket closed`) with server `Illegal instruction` in `TextEncoder.encodeLTX2` and no final payload:
      - `output/persistent_final_custom_trace021_q6p_20260702_101649/`
      - `output/persistent_server_logs_20260702_101533/server.log`
   - immediate fresh-server official control under the same settings passed end-to-end (`responses=15`, image+audio written, post-run echo healthy) with coherent PNG:
      - `output/persistent_final_official_q6p_20260702_101958_post_custom_crash/`
      - `output/persistent_server_logs_20260702_101940_official_control/server.log`
- Alias `10_e_v1` under `version=ltx2.3` schema produced deterministic `Illegal instruction` in `TextEncoder.encodeLTX2`:
  - artifact: `output/q6p_canary_alias_10_e_v1_stagegate_20260701`
- Experimental alias `10_e_v1_main_official_clip_test` timed out in one stage-gate and segfaulted in long final-gate:
  - artifact: `output/q6p_canary_alias_10e_main_official_clip_stagegate_20260701`
  - artifact: `output/q6p_canary_alias_10e_main_official_clip_finalgate_wait30_20260701`
  - crash head included `ccv_nnc_tensor_read` with cuDNN graph stack.
- Experimental full-schema custom f16 alias also crashed in loader path:
   - artifact: `output/q6p_canary_alias_10e_f16_fullschema_stagegate_20260702`
   - crash head: `ccv_nnc_tensor_read -> ccv_cnnp_model_read`.

### Working-but-limited alias tweak

- Changing `10_e_v1` custom entry to a minimal `version=v1` style removed immediate alias crash and allowed stream completion:
  - stage-gate artifact: `output/q6p_canary_alias_10_e_v1_stagegate_minv1schema_20260701`
  - long artifact: `output/alias_10_e_v1_singleframe_wait30_minv1schema_20260701`
- But decoded PNG from that long run was still visually noisy in this session.

## 5) Crash Signature Map (Quick Triage)

1. `Illegal instruction` + `TextEncoder.encodeLTX2`:
   - usually points to text/clip path interactions (alias/schema or checkpoint semantics).
   - first isolation step: bypass alias and run direct model key.

2. `Signal 11` + cuDNN graph + `ccv_nnc_tensor_read`:
   - usually points to loader/runtime data-path instability.
   - verify with long timeout and clean no-interference run.

3. `UNAVAILABLE: Socket closed` alone:
   - not root cause by itself; inspect server log for corresponding crash class.

## 6) Restart Checklist For Next Session

1. Reconfirm official baseline first (must pass before custom debugging):

```bash
bash tools/run_q6p_canary_once.sh \
  --model ltx_2.3_22b_distilled_1.1_q6p.ckpt \
  --width 256 --height 256 --steps 8 \
  --timeout-sec 1800 --max-responses 0 --require-final-output \
  --tag official_q6p_reconfirm_<date>
```

2. Convert and visually verify official PNG.

3. For custom candidate bytes, run raw-key tests with alias disabled:
   - temporarily move `dt-models/custom.json` out of the way,
   - run candidate by direct file key,
   - restore `custom.json` after run.

4. Only after raw-key quality is coherent, test alias schema variants.

5. Keep all candidate decisions evidence-based:
   - stream complete + no crash + coherent PNG.

## 7) Current Plan Continuation Point

- Focus next on restoring custom output quality (not just stream completion).
- Current strongest hypothesis: custom conversion/model semantics are broken upstream of q6p quantization, with two observed failure surfaces in this session: (a) noisy PNGs despite runtime completion (custom f16/q8p/q6p raw-key branches, including q8p 3-seed sweep visual fail 3/3 while matched official q6p control visual-pass 3/3) and (b) immediate text-path hard crash for trace021 q6p (`Illegal instruction` at `TextEncoder.encodeLTX2`).
- Priority order:
  1. establish one coherent raw-key custom output,
  2. then reintroduce alias fields one at a time,
  3. reject any configuration that reintroduces `encodeLTX2` illegal-instruction or noisy outputs.
