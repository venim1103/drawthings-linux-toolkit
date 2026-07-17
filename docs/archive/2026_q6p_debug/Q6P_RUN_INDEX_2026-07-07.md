# Q6P Run Index (Through Run 117)

Purpose:

- Quick scan index of all run sections with direct links and first recorded outcome line.
- Source of truth remains Q6P_RUN_LOG.md; this file is a navigation appendix.

## How To Use

- Click any Run link to jump to the canonical entry in Q6P_RUN_LOG.md.
- Use Outcome to quickly identify which branches failed, passed, or were inconclusive.

## Run Table

| Run | Link | First Recorded Outcome |
|---|---|---|
| Run 001 | [Run 001 - 2026-06-08 08:00 (UTC)](Q6P_RUN_LOG.md#L56) | In-place 770-row patch insufficient. |
| Run 002 | [Run 002 - 2026-06-08 09:26 (UTC)](Q6P_RUN_LOG.md#L90) | Expanded 1282-row in-place patch still did not reach first streamed response. |
| Run 003 | [Run 003 - 2026-06-08 09:31 (UTC)](Q6P_RUN_LOG.md#L129) | Crash signature unchanged after expanded 1282-row patch. |
| Run 004 | [Run 004 - 2026-06-08 10:39 (UTC)](Q6P_RUN_LOG.md#L167) | Full 2756-row patch still failed to produce first streamed response. |
| Run 005 | [Run 005 - 2026-06-08 12:25 (UTC)](Q6P_RUN_LOG.md#L206) | metadata parity reached (`metadata_mismatch_type/format/datatype=0`) but runtime still produced no streamed response; residual `data_len_mismatch=2728` remains. |
| Run 006 | [Run 006 - 2026-06-08 13:21 (UTC)](Q6P_RUN_LOG.md#L256) | Row-wise metadata/length parity reached full-match state (`mismatch_any=0`, `full_match=5745`, `unreadable_both=1`), but runtime still timed out pre-stream (`canary_rc=124`, no `response #1`). |
| Run 007 | [Run 007 - 2026-06-08 14:11 (UTC)](Q6P_RUN_LOG.md#L317) | Scripted evidence indicates the `10_e_v1_bf16_regen_0_f16.ckpt` conversion is structurally valid and runtime-usable; active blocker remains q6p runtime behavior. |
| Run 008 | [Run 008 - 2026-06-09 10:55 (UTC)](Q6P_RUN_LOG.md#L355) | Repairing all currently detected equal-length data-head/small-hash mismatches (26 rows) did not restore runtime streaming; q6p still times out before first response. |
| Run 009 | [Run 009 - 2026-06-09 11:02 (UTC)](Q6P_RUN_LOG.md#L404) | Wider small-hash probing remains unstable in this environment at current chunk shape; need a lighter targeted branch. |
| Run 010 | [Run 010 - 2026-06-09 11:09 (UTC)](Q6P_RUN_LOG.md#L434) | Even copying all 192 DIT gate `*-1` payloads did not restore runtime streaming; q6p still times out before first response. |
| Run 011 | [Run 011 - 2026-06-09 11:33 (UTC)](Q6P_RUN_LOG.md#L484) | DIT head/mid/tail signature probe did not surface actionable equal-length payload mismatches. |
| Run 012 | [Run 012 - 2026-06-09 11:47 (UTC)](Q6P_RUN_LOG.md#L522) | No signature mismatch in readable text-feature rows; one unreadable row remained unresolved. |
| Run 013 | [Run 013 - 2026-06-09 11:52 (UTC)](Q6P_RUN_LOG.md#L559) | Unreadable leaf row correlates with deterministic model-load crash path. |
| Run 014 | [Run 014 - 2026-06-09 12:18 (UTC)](Q6P_RUN_LOG.md#L595) | High-limit patch removes the crash path for the unreadable row but does not restore streaming response; runtime still stalls before first response. |
| Run 015 | [Run 015 - 2026-06-09 12:33 (UTC)](Q6P_RUN_LOG.md#L623) | No evidence of mismatches in processed ranges, but run incomplete; cannot treat as final coverage result. |
| Run 016 | [Run 016 - 2026-06-09 12:52 (UTC)](Q6P_RUN_LOG.md#L655) | Extending timeout to final-mode (15 min) did not address the failure; this branch fails due deterministic server crash, not timeout. |
| Run 017 | [Run 017 - 2026-06-09 13:28 (UTC)](Q6P_RUN_LOG.md#L688) | Even full text-feature family high-limit patch does not prevent the runtime crash; failure remains pre-stream with the same loader crash signature. |
| Run 018 | [Run 018 - 2026-06-09 14:29 (UTC)](Q6P_RUN_LOG.md#L736) | Phase 1 baseline is now locked with strict completion gating and still reproduces the deterministic q6p crash path. |
| Run 019 | [Run 019 - 2026-06-10 07:43 (UTC)](Q6P_RUN_LOG.md#L786) | first-divergence instrumentation now provides deterministic stage localization; in this controlled run divergence is isolated to q6p stage while f16 remains parity-clean. |
| Run 020 | [Run 020 - 2026-06-10 08:20 (UTC)](Q6P_RUN_LOG.md#L834) | full official-vs-custom comparison is model-identity dominated at f16 payload/signature level, but q6p still introduces a narrow additional mismatch slice isolated to text connector families. |
| Run 021 | [Run 021 - 2026-06-10 11:21 (UTC)](Q6P_RUN_LOG.md#L887) | traced regenerated q6p candidate passed strict runtime gate end-to-end in this controlled run; this is the first clear pass shift against the previously deterministic crash branch. |
| Run 022 | [Run 022 - 2026-06-10 11:38 (UTC)](Q6P_RUN_LOG.md#L967) | with identical f16 inputs, first divergence isolates strictly to q6p stage and indicates broad q6p-row rewrite between old and traced candidates. |
| Run 023 | [Run 023 - 2026-06-10 11:32 (UTC)](Q6P_RUN_LOG.md#L1023) | traced q6p candidate now satisfies the same strict matrix gates that previously failed, including final-output completion. |
| Run 024 | [Run 024 - 2026-06-10 12:02 (UTC)](Q6P_RUN_LOG.md#L1069) | traced q6p candidate passes strict runtime gates across multi-seed and multi-size coverage in this matrix. |
| Run 025 | [Run 025 - 2026-06-10 13:10 (UTC)](Q6P_RUN_LOG.md#L1108) | second-model strict matrix failed 6/6 with deterministic early crash pattern. |
| Run 026 | [Run 026 - 2026-06-10 13:20 (UTC)](Q6P_RUN_LOG.md#L1141) | failure is not explained by non-final-mode-only behavior; it persists with `--final-mode` and reproduces across two non-traced q6p artifacts. |
| Run 027 | [Run 027 - 2026-06-10 13:22 (UTC)](Q6P_RUN_LOG.md#L1174) | traced q6p stability became sensitive after local alias experiment; this prompted an isolation control. |
| Run 028 | [Run 028 - 2026-06-10 13:25 (UTC)](Q6P_RUN_LOG.md#L1200) | traced checkpoint content remains viable; instability is tied to the model-key/custom-entry path, not raw tensor content. |
| Run 029 | [Run 029 - 2026-06-10 13:25 (UTC)](Q6P_RUN_LOG.md#L1231) | this local alias variant did not produce a stable strict pass. |
| Run 030 | [Run 030 - 2026-06-10 13:30 (UTC)](Q6P_RUN_LOG.md#L1255) | traced q6p strict pass is restored when the local custom entry no longer matches the traced file key. |
| Run 031 | [Run 031 - 2026-06-10 13:36 (UTC)](Q6P_RUN_LOG.md#L1281) | (not recorded in entry) |
| Run 032b | [Run 032b - 2026-06-10 14:14 (UTC)](Q6P_RUN_LOG.md#L1333) | (not recorded in entry) |
| Run 033b | [Run 033b - 2026-06-11 (UTC)](Q6P_RUN_LOG.md#L1384) | (not recorded in entry) |
| Run 034 | [Run 034 - 2026-06-11 (UTC)](Q6P_RUN_LOG.md#L1433) | (not recorded in entry) |
| Run 035 | [Run 035 (2026-06-11): Core Alias-Resolution Matrix with Winner Context](Q6P_RUN_LOG.md#L1479) | (not recorded) |
| Run 036 | [Run 036 (2026-06-11): Cross-File Alias-Resolution Matrix with Winner Context](Q6P_RUN_LOG.md#L1511) | (not recorded in entry) |
| Run 037 | [Run 037 (2026-06-11): Minimal-v1 Custom-Winner Isolation Matrix](Q6P_RUN_LOG.md#L1540) | (not recorded in entry) |
| Run 038 | [Run 038 (2026-06-11): Additive LTX Field Ladder from Minimal-v1 Baseline](Q6P_RUN_LOG.md#L1577) | (not recorded) |
| Run 039 | [Run 039 (2026-06-11): LTX2.3 Encoder-Path Variants (Version Fixed)](Q6P_RUN_LOG.md#L1620) | (not recorded) |
| Run 040 | [Run 040 (2026-06-11): LTX2.3 Encoder Matrix with Resolved File-List Context](Q6P_RUN_LOG.md#L1661) | (not recorded) |
| Run 041 | [Run 041 (2026-06-11): Controlled LTX2.3 Two-File Companion Matrix](Q6P_RUN_LOG.md#L1707) | (not recorded) |
| Run 042 | [Run 042 (2026-06-11): LTX2.3 Two-File Ordering Matrix](Q6P_RUN_LOG.md#L1749) | (not recorded) |
| Run 043 | [Run 043 (2026-06-11): LTX2.3 Text-Encoder Pin Matrix](Q6P_RUN_LOG.md#L1795) | (not recorded) |
| Run 044 | [Run 044 (2026-06-11): LTX2.3 Pinned-Companion Boundary Matrix](Q6P_RUN_LOG.md#L1839) | (not recorded) |
| Run 045 | [Run 045 (2026-06-11): Instrumented Linux Build + Pinned-Companion Replay](Q6P_RUN_LOG.md#L1886) | (not recorded) |
| Run 046 | [Run 046 (2026-06-11): Binary-Path Control A/B for Source-Build Confound](Q6P_RUN_LOG.md#L1942) | (not recorded) |
| Run 047 | [Run 047 (2026-06-11): Wrapper-Binary Pinned-Companion Replay](Q6P_RUN_LOG.md#L1987) | (not recorded) |
| Run 048 | [Run 048 (2026-06-11): Source-Build Official-Control Check (Availability Confound)](Q6P_RUN_LOG.md#L2024) | (not recorded) |
| Run 049 | [Run 049 (2026-06-11): Canary Preflight Guard Validation](Q6P_RUN_LOG.md#L2057) | (not recorded) |
| Run 050 | [Run 050 (2026-06-11): Focused Wrapper-Path Branch Gate](Q6P_RUN_LOG.md#L2080) | (not recorded) |
| Run 051 | [Run 051 (2026-06-11): Source-Build Control Without Trace Logging](Q6P_RUN_LOG.md#L2121) | (not recorded) |
| Run 052 | [Run 052 (2026-06-11): Binary/Linkage Forensics (Default vs Source-Build)](Q6P_RUN_LOG.md#L2149) | (not recorded) |
| Run 053 | [Run 053 (2026-06-11): Focused Wrapper Matrix With Trace Env](Q6P_RUN_LOG.md#L2172) | (not recorded) |
| Run 054 | [Run 054 (2026-06-11): Wrapper Text-Pin Matrix Refresh](Q6P_RUN_LOG.md#L2205) | (not recorded) |
| Run 055 | [Run 055 (2026-06-11): Focused Wrapper Recheck](Q6P_RUN_LOG.md#L2238) | (not recorded) |
| Run 056 | [Run 056 (2026-06-11): Loader-Branch Field Isolation Matrix](Q6P_RUN_LOG.md#L2270) | (not recorded) |
| Run 057 | [Run 057 (2026-06-11): Loader-Branch Isolation for Text Pin B](Q6P_RUN_LOG.md#L2312) | (not recorded) |
| Run 058 | [Run 058 (2026-06-11): Compact Text-Gate Boundary Matrix](Q6P_RUN_LOG.md#L2355) | (not recorded) |
| Run 059 | [Run 059 (2026-06-11): Text-Gate Reproducibility Recheck](Q6P_RUN_LOG.md#L2400) | (not recorded) |
| Run 060 | [Run 060 (2026-06-11): Pin-B Noise Check (Compact Matrix)](Q6P_RUN_LOG.md#L2436) | (not recorded) |
| Run 061 | [Run 061 (2026-06-11): Pin-B Noise Check Recheck](Q6P_RUN_LOG.md#L2477) | (not recorded) |
| Run 062 | [Run 062 (2026-06-11): Pin-B Noise Matrix with Trace Env (Wrapper Runtime)](Q6P_RUN_LOG.md#L2509) | (not recorded) |
| Run 063 | [Run 063 (2026-06-11): Source-Built Trace Smoke (PATH Override)](Q6P_RUN_LOG.md#L2546) | (not recorded) |
| Run 064 | [Run 064 (2026-06-11): Wrapper Runtime Selector Smoke](Q6P_RUN_LOG.md#L2578) | (not recorded) |
| Run 065 | [Run 065 (2026-06-11): Source Runtime Selector Smoke](Q6P_RUN_LOG.md#L2605) | (not recorded) |
| Run 066 | [Run 066 (2026-06-11): Focused Matrix via Explicit Wrapper Selector](Q6P_RUN_LOG.md#L2633) | (not recorded) |
| Run 067 | [Run 067 (2026-06-11): Source Runtime Tuning Sweep + Key-Path Discriminator](Q6P_RUN_LOG.md#L2668) | (not recorded) |
| Run 068 | [Run 068 (2026-06-11): Focused Matrix via Explicit Source Selector](Q6P_RUN_LOG.md#L2717) | (not recorded) |
| Run 069 | [Run 069 (2026-06-11): Targeted Same-Arg Source A/B (Trace-Key Mapping Toggle)](Q6P_RUN_LOG.md#L2750) | (not recorded) |
| Run 070 | [Run 070 (2026-06-12): Same-Arg Source A/B with Minimal-v1 Mapping Entry](Q6P_RUN_LOG.md#L2799) | (not recorded) |
| Run 071 | [Run 071 (2026-06-12): Same-Arg Source Probe with LTX2.3-Minimal Mapping Entry](Q6P_RUN_LOG.md#L2831) | (not recorded) |
| Run 072 | [Run 072 (2026-06-12): Source Field Ladder (`ltx23_min -> +text -> +clip -> +auto`)](Q6P_RUN_LOG.md#L2863) | (not recorded) |
| Run 073 | [Run 073 (2026-06-12): Modulation Check (`+auto` vs `+modifier` vs combined)](Q6P_RUN_LOG.md#L2910) | (not recorded) |
| Run 074 | [Run 074 (2026-06-29): Repro Check (`+auto`, `+modifier`, `+modifier+auto`) with 2x repeats](Q6P_RUN_LOG.md#L2953) | (not recorded) |
| Run 075 | [Run 075 (2026-06-29): Focused Repeat Sweep (`+modifier+auto`, 8x)](Q6P_RUN_LOG.md#L3001) | (not recorded) |
| Run 076 | [Run 076 (2026-06-29): Warm-Server Repeat Probe (`+modifier+auto`, 8x)](Q6P_RUN_LOG.md#L3056) | (not recorded) |
| Run 077 | [Run 077 (2026-06-29): Warm-Server Control (`text+clip`, no modifier/autoencoder)](Q6P_RUN_LOG.md#L3099) | (not recorded) |
| Run 078 | [Run 078 (2026-06-29): Warm-Server Minimal Mapping Control (`version=ltx2.3` only)](Q6P_RUN_LOG.md#L3142) | (not recorded) |
| Run 079 | [Run 079 (2026-06-29): Cold-Per-Repeat Control (`version=ltx2.3` only)](Q6P_RUN_LOG.md#L3184) | (not recorded) |
| Run 080 | [Run 080 (2026-06-29): Restart-Per-Repeat Text/Clip Matrix (4 cases x 2 repeats)](Q6P_RUN_LOG.md#L3226) | (not recorded) |
| Run 081 | [Run 081 (2026-06-29): Restart-Per-Repeat Text-Gate Clip Companion Matrix (6 cases x 2 repeats)](Q6P_RUN_LOG.md#L3277) | (not recorded) |
| Run 081b | [Run 081b (2026-06-29): Restart-Per-Repeat Text-Gate Companion Repro Pass (6 cases x 4 repeats)](Q6P_RUN_LOG.md#L3336) | (not recorded) |
| Run 082 | [Run 082 (2026-06-30): Restart-Per-Repeat Gemma-Boundary Stress Pass (3 cases x 8 repeats)](Q6P_RUN_LOG.md#L3398) | (not recorded) |
| Run 083 | [Run 083 (2026-06-30): Restart-Per-Repeat Gemma-Boundary Repro Pass (3 cases x 8 repeats)](Q6P_RUN_LOG.md#L3448) | (not recorded) |
| Run 084 | [Run 084 (2026-06-30): Pipeline Stage-Gate Check (conversion pass, q6p inference fail)](Q6P_RUN_LOG.md#L3497) | (not recorded) |
| Run 085 | [Run 085 (2026-07-01): q8p Resume Recovery + Strict Runtime Gate Pass](Q6P_RUN_LOG.md#L3535) | (not recorded) |
| Run 086 | [Run 086 (2026-07-01): Official q6p Long Final-Gate Reconfirm PASS](Q6P_RUN_LOG.md#L3571) | (not recorded in entry) |
| Run 087 | [Run 087 (2026-07-01): Custom Raw-Key Runtime Checks (Alias Disabled)](Q6P_RUN_LOG.md#L3597) | (not recorded) |
| Run 088 | [Run 088 (2026-07-01): Alias Schema Regression and Minimal-v1 Mitigation](Q6P_RUN_LOG.md#L3626) | (not recorded) |
| Run 089 | [Run 089 (2026-07-01): Custom-Main + Official-Clip Alias Final-Gate Failure](Q6P_RUN_LOG.md#L3643) | (not recorded in entry) |
| Run 090 | [Run 090 (2026-07-02): Custom f16 Raw-Key Long Control (Runtime PASS, Visual FAIL)](Q6P_RUN_LOG.md#L3661) | (not recorded) |
| Run 091 | [Run 091 (2026-07-02): Custom f16 Full-Schema Alias Stage-Gate (Loader Crash)](Q6P_RUN_LOG.md#L3688) | (not recorded in entry) |
| Run 092 | [Run 092 (2026-07-02): Persistent Raw-Key Custom f16 Repro (Immediate Loader Crash)](Q6P_RUN_LOG.md#L3709) | (not recorded in entry) |
| Run 093 | [Run 093 (2026-07-02): Persistent Official q6p Short Control (Healthy, Deep Sampling)](Q6P_RUN_LOG.md#L3734) | (not recorded in entry) |
| Run 094 | [Run 094 (2026-07-02): Persistent Raw-Key Custom q8p Stage-Gate (Healthy Sampling)](Q6P_RUN_LOG.md#L3771) | (not recorded in entry) |
| Run 095 | [Run 095 (2026-07-02): Persistent Raw-Key Custom q8p Final-Gate (Runtime PASS, Visual FAIL)](Q6P_RUN_LOG.md#L3799) | (not recorded in entry) |
| Run 096 | [Run 096 (2026-07-02): Persistent Official q6p Final-Gate Control (Runtime PASS, Visual PASS)](Q6P_RUN_LOG.md#L3837) | (not recorded in entry) |
| Run 097 | [Run 097 (2026-07-02): Persistent Raw-Key Custom q6p Trace021 Final-Gate (Hard Crash)](Q6P_RUN_LOG.md#L3879) | (not recorded in entry) |
| Run 098 | [Run 098 (2026-07-02): Persistent Official q6p Final-Gate Control After Run 097 Crash (Runtime PASS, Visual PASS)](Q6P_RUN_LOG.md#L3913) | (not recorded in entry) |
| Run 099 | [Run 099 (2026-07-02): Persistent Raw-Key Custom q8p 3-Seed Sweep (Runtime PASS, Visual FAIL 3/3)](Q6P_RUN_LOG.md#L3953) | (not recorded in entry) |
| Run 100 | [Run 100 (2026-07-02): Persistent Raw-Key Official q6p 3-Seed Matched Control (Runtime PASS, Visual PASS 3/3)](Q6P_RUN_LOG.md#L3996) | (not recorded in entry) |
| Run 101 | [Run 101 (2026-07-02): Quantitative PNG Metrics for Matched 3-Seed A/B (Custom q8p vs Official q6p)](Q6P_RUN_LOG.md#L4038) | (not recorded) |
| Run 102 | [Run 102 (2026-07-03): Persistent Raw-Key Custom q6p Main 3-Seed Sweep Attempt (Seed 4242 Hard Crash)](Q6P_RUN_LOG.md#L4076) | (not recorded in entry) |
| Run 103 | [Run 103 (2026-07-03): Custom q6p Main Stage-Gate Long-Wait Recheck (1800s Budget, Hard Crash)](Q6P_RUN_LOG.md#L4114) | (not recorded in entry) |
| Run 104 | [Run 104 (2026-07-03): Official q6p Stage-Gate Long-Wait Matched Control (PASS)](Q6P_RUN_LOG.md#L4149) | (not recorded in entry) |
| Run 105 | [Run 105 (2026-07-03): Converter-Side Focused Diagnostics (Custom q6p Main vs Official q6p)](Q6P_RUN_LOG.md#L4181) | (not recorded) |
| Run 106 | [Run 106 (2026-07-06): Connector+DIT-Slice Content Alignment Canary (In-Place Candidate Copy)](Q6P_RUN_LOG.md#L4229) | (not recorded) |
| Run 107 | [Run 107 (2026-07-06): Full-Gate A/B on Patched Candidate vs Official Control](Q6P_RUN_LOG.md#L4301) | (not recorded) |
| Run 108 | [Run 108 (2026-07-06): DIT Coverage Expansion (Connector + DIT1024) Full-Gate Recheck](Q6P_RUN_LOG.md#L4353) | (not recorded) |
| Run 109 | [Run 109 (2026-07-06): Stratified DIT1024 Coverage (Non-Prefix) Full-Gate Recheck](Q6P_RUN_LOG.md#L4435) | (not recorded) |
| Run 110 | [Run 110 (2026-07-06): Semantic CoreAV Targeting (Connector + DIT Core Families)](Q6P_RUN_LOG.md#L4518) | (not recorded) |
| Run 110R | [Run 110R (2026-07-06): Environment Recovery Check (Post-Run110)](Q6P_RUN_LOG.md#L4601) | (not recorded) |
| Run 110S | [Run 110S (2026-07-06): Post-GPUfix Full-Gate Revalidation (Matched Official vs Run110 Patched)](Q6P_RUN_LOG.md#L4651) | (not recorded) |
| Run 110T | [Run 110T (2026-07-06): Post-GPUfix Seed Sweep On Run110 Patched Candidate](Q6P_RUN_LOG.md#L4695) | (not recorded) |
| Run 111A | [Run 111A (2026-07-06): Full-Copy All-Names Branch Attempt (Aborted)](Q6P_RUN_LOG.md#L4730) | (not recorded) |
| Run 111 | [Run 111 (2026-07-06): Remaining-Set In-Place Patch On Run110 Candidate](Q6P_RUN_LOG.md#L4745) | (not recorded) |
| Run 112 | [Run 112 (2026-07-06): Full Run105 Meta/Len Remaining-Set Patch](Q6P_RUN_LOG.md#L4776) | (not recorded) |
| Run 113 | [Run 113 (2026-07-06): Metadata Alignment Over Full 5745 Row Set](Q6P_RUN_LOG.md#L4804) | (not recorded) |
| Run 114 | [Run 114 (2026-07-06): Single Missing-Row Content Patch Attempt](Q6P_RUN_LOG.md#L4837) | (not recorded) |
| Run 115 | [Run 115 (2026-07-06): Direct SQLite Row-Replace Attempt (Schema Mismatch)](Q6P_RUN_LOG.md#L4859) | (not recorded) |
| Run 116 | [Run 116 (2026-07-06): Direct SQLite Row-Replace Attempt (Blob Limit)](Q6P_RUN_LOG.md#L4881) | (not recorded) |
| Run 117 | [Run 117 (2026-07-06): High-Limit Repair Of Final Unreadable Tensor Row](Q6P_RUN_LOG.md#L4898) | (not recorded) |
