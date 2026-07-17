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
- Custom q6p main matched-seed sweep attempt (`10_e_v1_bf16_regen_0_q6p.ckpt`) failed on first seed before continuation:
   - sweep root: `output/persistent_seed_sweep_custom_q6p_main_20260703_102142/`
   - seed `4242` reached `textEncoded -> imageEncoded` then client `UNAVAILABLE: Socket closed`
   - server crashed with bad-pointer dereference in cuDNN graph path with `ccv_nnc_tensor_read` in stack head:
      - `output/persistent_server_logs_20260703_102119_custom_q6p_main_seed_sweep/server.log`
- Long-wait recheck (1800s timeout budget) confirms this is not a short-timeout artifact:
   - custom q6p main still crashed quickly after `imageEncoded` with `Signal 11` / bad-pointer dereference and `ccv_nnc_tensor_read`:
      - `output/run103_longwait_20260703_122344/custom_q6p_main_stagegate_longwait/summary.txt`
      - `output/run103_longwait_20260703_122344/server_custom_q6p_main_stagegate_longwait/server.log`
   - matched official long-wait control under same profile passed to `sampling` and exited cleanly at bounded stop (`responses=3`, server alive):
      - `output/run104_longwait_20260703_123501/official_q6p_stagegate_longwait_control/summary.txt`
      - `output/run104_longwait_20260703_123501/server_official_q6p_stagegate_longwait_control/server.log`
- Converter-side focus run (custom q6p main vs official q6p) shows broad serializer/content divergence, not a narrow edge case:
   - run root: `output/run105_converter_focus_20260703_124301/`
   - row-wise meta/len over all shared tensors:
      - `mismatch_any=5745/5745 readable`, `full_match=0`, `metadata_mismatch_type=5745`, `data_len_mismatch=5546`
      - dominant families: `__dit__` (`5484/5484`), `__text_video_connector__` (`128/128`), `__text_audio_connector__` (`128/128`), `__text_feature_extractor__` (`3/4`)
   - equal-length payload sample probe (first 2500 rows):
      - `metadata_mismatch=2500`, `data_len_equal=23`, `head-signature mismatches=23/23`, `small_sha256 mismatches=21/21`
      - sampled payload mismatches were `__dit__`-dominated
   - note: full deep-diff path produced no output in this environment and was bypassed in favor of stable row-wise probes.
- Connector+DIT-slice surgical content alignment canary (Run 106) produced partial mitigation in bounded stage-gate:
   - run root: `output/run106_connector_ditslice_20260703_131008/`
   - selected patch set from Run 105 mismatch names:
      - connector/text-feature families (`261` names) + first `256` DIT names, union `517`
   - targeted probe before patch (`517` selected):
      - `metadata_mismatch_type=517`, `small_sha256_mismatch=138/170 compared`
   - after `dt_align_ckpt_content_subset.py` apply:
      - `rows_updated=517`, `post_dim_head_mismatch=0`, `post_data_head_mismatch=0`
      - targeted post-probe: `small_sha256_mismatch=0/413 compared` (for selected rows)
      - metadata type mismatch remained for selected rows (`metadata_mismatch_type=517`)
   - matched long-wait bounded stage-gate (`--timeout-sec 1800 --max-responses 3`, seed `4242`) passed for both patched custom candidate and official control:
      - patched custom: `output/q6p_canary_run106_patched_connector_ditslice_stagegate_longwait_20260706_054624/`
      - official control: `output/q6p_canary_run106_official_q6p_stagegate_longwait_control_20260706_054624/`
   - interpretation: targeted content surgery can suppress the immediate stage-gate crash in this bounded profile, but does not establish global equivalence or full-run quality.
- Full-gate follow-up (Run 107) confirms stability improvement but persistent quality gap:
   - run root: `output/run107_fullgate_patched_vs_official_20260706_055513/`
   - matched full-gate settings: `--timeout-sec 1800 --max-responses 0 --require-complete-stream --require-final-output`, `256x256`, `steps=8`, `seed=4242`
   - runtime: both patched custom and official control PASSed (`canary_rc=0`, `post_echo_rc=0`, `RESULT=PASS`)
   - output shape diverged materially:
      - patched custom: `responses=14`, `images=1`, `audio=0`
      - official control: `responses=23`, `images=9`, `audio=1`
   - PNG metrics still show patched-quality deficit vs official:
      - patched: `gray_std=0.130545`, `entropy=7.096774`
      - official: `gray_std=0.330333`, `entropy=7.483953`
   - interpretation: surgical patch lifted crash/stability behavior into PASS for this profile but did not recover official-like richness/coherence.
- DIT expansion follow-up (Run 108, connector + DIT1024) did not improve quality despite larger aligned subset:
   - run root: `output/run108_connector_dit1024_20260706_061243/`
   - selected rows increased from `517` (Run 106/107) to `1285` (`261` connector/text-feature + `1024` DIT)
   - targeted selected-row payload mismatch still reached zero post-patch (`small_sha256_mismatch=0/913 compared`), with runtime PASS maintained in matched full-gate
   - full-gate output shape remained the same as Run 107 for patched custom (`responses=14`, `images=1`, `audio=0`)
   - patched PNG metrics did not improve and were slightly worse than Run 107:
      - Run 107 patched: `gray_std=0.130545`, `entropy=7.096774`
      - Run 108 patched: `gray_std=0.128570`, `entropy=7.074494`
   - interpretation: contiguous early-DIT expansion alone is not sufficient to close the quality gap to official.
- Non-prefix DIT strategy follow-up (Run 109, connector + stratified DIT1024) also did not improve quality:
   - run root: `output/run109_connector_dit1024_stratified_20260706_063708/`
   - same selected row count as Run 108 (`1285` union), but DIT names were distributed across full DIT mismatch list instead of first-prefix slicing
   - selected-row payload mismatch again reached zero post-patch (`small_sha256_mismatch=0/777 compared`), with matched full-gate runtime PASS on both patched custom and official control
   - patched output shape remained unchanged (`responses=14`, `images=1`, `audio=0`)
   - patched PNG metrics again moved slightly worse:
      - Run 108 patched: `gray_std=0.128570`, `entropy=7.074494`
      - Run 109 patched: `gray_std=0.125271`, `entropy=7.035664`
   - interpretation: at fixed coverage size, switching DIT selection from contiguous to stratified does not recover quality.
- Semantic targeting follow-up (Run 110, connector + DIT CoreAV bundle) completed content alignment but did not yield a valid quality verdict due control instability:
   - run root: `output/run110_semantic_coreav_20260706_070329/`
   - selected rows increased to `2573` (`261` connector/text-feature + `2312` semantic DIT rows)
   - selected-row payload mismatch reached zero post-patch (`small_sha256_mismatch=0/1969 compared`)
   - runtime stage failed for both patched and official in matched full-gate profile (`canary_rc=1` / `RESULT=FAIL` on each)
   - official control remained failing across retries and mitigation variants (`retry1`, `retry2`, `--server-no-flash-attention`, `--server-cpu-offload --server-no-flash-attention`)
   - CPU-offload/no-flash official log surfaced repeated CUDA/CUFILE initialization failures (`Cuda version not Supported`, `CUFILE - NVFS driver initialization error`)
   - interpretation: Run 110 is content-alignment PASS but runtime-quality INCONCLUSIVE because baseline official generation was unavailable in-session.
- Post-Run110 environment recovery check (Run 110R) found a container-level runtime blocker:
   - official q6p default and no-flash controls both failed immediately (`canary_rc=1`, `post_echo_rc=1`)
   - previously known-good custom q8p sanity model also failed with the same `_ccv_nnc_index_select_forw` bad-pointer crash class
   - source-built alternate server binary also failed (assert path), so wrapper-vs-source switch did not restore generation
   - container preflight showed missing GPU devices (`/dev/nvidia*` absent; `nvidia-smi` unavailable with NVML init failure)
   - interpretation: current session is environment-blocked; do not use these failed gates to infer model-quality regressions.
- Post-GPUfix strict revalidation (Run 110S) restored matched A/B execution in-session:
   - run root: `output/run110_fullgate_patched_vs_official_after_gpufix_20260706_121602/`
   - official q6p control PASSed under strict profile with explicit runtime library path:
      - `LD_LIBRARY_PATH=/usr/local/swift/usr/lib/swift/linux:/usr/lib/wsl/lib:/usr/local/cuda/targets/x86_64-linux/lib`
      - `responses=23`, `images=9`, `audio=1`
   - Run110 patched candidate PASSed under the same strict profile:
      - `responses=14`, `images=1`, `audio=0`
   - PNG metrics remain strongly separated (patched lower contrast/entropy):
      - patched: `gray_std=0.124541`, `entropy=7.030824`
      - official: `gray_std=0.330278`, `entropy=7.488716`
   - interpretation: environment gate is currently recovered for this profile, but Run110 patched quality deficit vs official is still unresolved.
- Run 110T seed sweep on the same patched candidate confirmed corruption persistence across seeds:
   - run root: `output/run110_patched_seed_sweep_after_gpufix_20260706_123216/`
   - seeds `4242/4243/4244` all PASSed runtime with the same stream shape (`responses=14`, `images=1`, `audio=0`)
   - decoded PNGs remained visually garbled/noise-like for all three seeds
   - entropy/contrast stayed in low-information band (gray_std `0.131004 -> 0.106963`, entropy `7.103863 -> 6.812831`)
   - interpretation: this failure mode is not seed-specific in tested range and likely reflects remaining structural content divergence.
- Run 117 high-limit final-row repair eliminated the last known unreadable shared tensor and reached full tracked row parity without fixing visual quality:
   - run root: `output/run117_highlimit_single_row_retry_20260706_131612/`
   - repaired row: `__text_feature_extractor__[t-video_aggregate_embed-0-0]` (`row_patch_changes=1`, quick_check pre/post `ok`)
   - post-probe over `5746` tracked shared names reached full parity (`full_match=5746`, all metadata/dim/data mismatch counters zero)
   - strict runtime still PASSed but with unchanged patched shape (`responses=14`, `images=1`, `audio=0`)
   - output PNG remained RGB-noise/garbled
   - interpretation: known row-level mismatch/unreadable defects are no longer sufficient to explain the quality gap.
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

0. Verify runtime prerequisites before any model A/B:

```bash
ls -l /dev/nvidia*
nvidia-smi
```

   - If `/dev/nvidia*` is missing or NVML is unavailable, stop and restore container GPU device access first; model-level conclusions are invalid until this passes.

1. Reconfirm official baseline first (must pass before custom debugging):

```bash
env LD_LIBRARY_PATH=/usr/local/swift/usr/lib/swift/linux:/usr/lib/wsl/lib:/usr/local/cuda/targets/x86_64-linux/lib \
bash tools/run_q6p_canary_once.sh \
  --model ltx_2.3_22b_distilled_1.1_q6p.ckpt \
  --width 256 --height 256 --steps 8 \
  --timeout-sec 1800 --max-responses 0 --require-final-output \
  --tag official_q6p_reconfirm_<date>
```

    - If this fails, treat as environment/runtime blocker first (collect server log and resolve CUDA/cuFile initialization state) before drawing model-quality conclusions.

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
- Current strongest hypothesis: custom conversion/model semantics are broken upstream of q6p quantization, with three observed custom failure surfaces: (a) noisy/low-information PNGs despite runtime completion (custom f16/q8p raw-key branches, including q8p 3-seed sweep visual fail 3/3 while matched official q6p control visual-pass 3/3), (b) immediate text-path hard crash for trace021 q6p (`Illegal instruction` at `TextEncoder.encodeLTX2`), and (c) cuDNN/`ccv_nnc_tensor_read` bad-pointer crash for unpatched custom q6p main. Long-wait matched controls and Run 105 converter probes indicated broad metadata/payload divergence (especially `__dit__` + connector families), and Runs 106-113 removed those tracked mismatches progressively while preserving runtime PASS. Run 117 then repaired the final known unreadable shared row via high-limit sqlite and achieved full parity across the tracked 5746 shared names, but output remained visually garbled. This raises confidence that the remaining blocker is outside currently tracked row-level parity defects (for example config/semantic path mismatch not represented by these probes).
- Priority order:
  1. establish one coherent raw-key custom output,
  2. then reintroduce alias fields one at a time,
  3. reject any configuration that reintroduces `encodeLTX2` illegal-instruction or noisy outputs.
