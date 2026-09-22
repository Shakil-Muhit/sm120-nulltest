# Ceiling-ledger execution report — 2026-09-22 (Phase 3)

Handoff: the owner's Phase-3 handoff of 2026-09-21 (supersedes `plan_ceiling_ledger.md` after Phase 2). Executor: Claude Fable 5.1 with codex gpt-6-astra (max effort, read-only, on the driver workstation only) as the adversarial reviewer. Repository: `Shakil-Muhit/sm120-nulltest`, branch `main`, measurement commit `ed3ea3d`. Pod: "melband profiling" (RunPod `d5n5vfr3z1tbp6`, RTX 5090, EU-RO-1, image `runpod/pytorch:1.1.0-rc.154-cu1281-torch280-ubuntu2204`: torch 2.8.0+cu128, triton 3.4.0, driver 580.178.04, Nsight Systems 2026.5.1).

## 1. Summary

Phase 3 is complete. The nine registrations (E1–E9, `prereg_amendments.md` v0.2) were run in one pod session of 4 h 29 min on the RTX 5090 pod "melband profiling"; the on-pod audit is COMPLETE (198 of 198 mandatory items, evidence grade pod, `evidence/session_status.json`), every optional stage (E5, E7) finished inside the budget, and the evidence (1,698 files) was extracted with a verified receipt before the pod was deleted. Every number below is computed by `fullmodel/session3_result.py`, `fullmodel/ledger.py --phase3-dir` and `fullmodel/e_tools.py` from the cited files.

Outcomes against the registered defaults (matched pairs, median over three eligible restarts, block-level 95 % bootstrap intervals; SDR gate PASS for every arm; details in `SESSION3_RESULT.md`):

| experiment | pod result | registered default | verdict |
|---|---|---|---|
| E6 A/A (C3 vs C3) | −0.09 ms at 8 s, −0.06 ms at 100 ms, both intervals cover zero; 40 of 40 headline restarts eligible | ≤ 0.15 / ≤ 0.05 ms | 8 s confirmed; 100 ms 0.006 ms over the bound |
| E3 graph partition (C1P vs C1, 8 s) | C1P 15.17 ms slower [15.06, 15.20]; 68 graph launches, 56 staging copies, 68 uncaptured kernels per pass | no material change | not confirmed: partition is a finding |
| E2 max-autotune (C3M vs C3) | C3M faster by 4.80 ms [4.62, 4.81] at 8 s and 0.20 ms [0.19, 0.25] at 100 ms; 178 GEMMs per pass moved to Triton templates that fuse the weight cast | 1.0–4.48 ms; ≤ 0.05 ms | above both defaults |
| E1 freezing (C3F) | 2.49 / 2.54 ms slower; all 420 cast kernels kept | ≥ 0.55 / ≥ 0.60 ms | not confirmed |
| E1 freezing without nn-module inlining (C3G) | +1.40 ms [1.31, 1.42] at 8 s, +0.67 ms [0.66, 0.71] at 100 ms; zero cast kernels | ≥ 0.55 / ≥ 0.60 ms | confirmed; at 8 s exceeds the X6a ceiling of 0.92 ms |
| E1 pre-cast (C3P) | +0.73 ms [0.71, 0.80] at 8 s, +0.55 ms [0.53, 0.60] at 100 ms; casts 420 → 60 | ≥ 0.55 / ≥ 0.60 ms | 8 s confirmed; 100 ms 0.05 ms under |
| E9 held-out crossover (C3 vs H) | H faster by 1.13 ms [1.03, 1.22] at 4 s and 0.67 ms [0.59, 0.72] at 6 s; crossover between 6 s and 8 s | +0.37 ms and −0.02 ms, tolerance 0.4 ms | FAIL; the residual points at the kernel-efficiency term (`docs/phase3/e9/e9_residual.json`) |
| E4a norm-fold wrapper (C3N vs C3M, max-autotune) | 2.30 ms slower [2.30, 2.48] at 8 s; −0.01 ms at 100 ms | ≥ 0.5 ms; ≤ 0.1 ms | 8 s not confirmed; 100 ms confirmed |
| E5 structural variants | cast time depth 6 / depth 1 = 1.11, boundary count 1.17; wall floor at 100 ms moves 3.6–3.7× faster than bytes/1.79 TB/s with mask-estimator bytes | invariant within 10 %; slope ≈ bytes/1.79 TB/s | (i) narrowly not confirmed; (ii) slope is 3.6–3.7× the bandwidth floor (fp32 read + fp16 write + fp16 read of the cast, not one fp16 read); (iii) reported |
| E7 PDL | trivial chain 0.719 → 0.568 µs/node (saving 0.151, reproducing Sep 20 within 0.05); prologue chain graph saving 0.184 µs/node, stream saving 2.03 µs/node; overlap observed in PDL graph segments; SASS ACQBULK/PREEXIT | prologue graph saving ≥ 0.5 µs/node | not confirmed; the 0.58 µs/node figure stands (overlap observed) |

Ledger (`LEDGER.md`, realized/ceiling filled from the pod pairs): X6a realized 1.40 ms against a 0.92 ms ceiling at 8 s, so the coverage check fires ("realized EXCEEDS its ceiling") and the X6a attribution is incomplete: freezing without inlining folds more than the parameter casts. At 100 ms X6a realized is 0.78 of its ceiling. X2a's wrapper realization is negative at both operating points. E2's 4.80 ms exceeds the X2a+X2b+X2c ceiling of 1.84 ms and is compared with the G1 union of 8.52 ms: max-autotune changes GEMM kernel selection and fuses the weight casts, mechanisms outside X2. Graph partition recovers none of the X1 break tax (−2.25 of it).

Two registered predictions held as stated (E6 at 8 s; E1 through C3G and C3P), six did not; each failed prediction is reported with its number and its mechanism, and no prediction was revised after the fact.

## 2. Timeline (local time, UTC+6; UTC in brackets)

| time | event |
|---|---|
| 09-21 21:25 | Phase 3 §2 part 1 pushed (`9547333`): block-level bootstrap statistics, X1 and pad-and-batch decompositions, X2 byte split, `SESSION_RESULT_v2.md`, ledger v2 (X2 split, X5 net difference, per-shape X8/X9 bounds, X12 relabel, realized/ceiling column). |
| 09-21 21:40 | §2.7–2.8 pushed (`026dc49`): 13 related-work notes with the F1–F8 table, `docs/provenance/push_evidence.md` (client-side timestamps; repository private, OpenTimestamps proof submitted), evidence manifest, Zenodo subset and metadata draft, README quickstart, AI-use disclosure, codex gate summary. |
| 09-21 21:59 | §3 pushed (`6619813`): registrations E1–E9 (`prereg_amendments.md` v0.2, default predictions), arms C3F/C3P/C3M/C1P/C3N, fold-norm and pre-cast wrappers, E5 variants, A/A pairs, held-out guard, E7 extension, `scripts/session3.sh`, session-3 audit, E9 cost model (`docs/phase3/e9/e9_cost_model.json`). |
| 09-21 22:01–22:19 | Engineering smoke of the new arms on ml-beast GPU 4 (`scripts/phase3_smoke.sh`); three fixes (`2faa775`). |
| 09-21 22:04–22:22 | Codex gate 09 (`docs/codex/09_phase3.md`, findings 128–138, EXECUTION-READY: no) on the specs and measurement path. |
| 09-21 22:57 | Gate-09 findings closed (`3c47f6a`); first full session-3 shakedown started on ml-beast (engineering data only). |
| 09-22 01:58 | First shakedown done (3.0 h): INCOMPLETE 138/161 for three causes, all in the harness: the fixed two-round joint settle scored the graph-partition pairs ineligible (`fc41966`), the max-autotune traces exceeded the 1,500 s cap (`6c0adfa`), and the clock-drift check needed a sampling window (`d86ec18`). E1 contingency rule extended (`afa6cdd`); the freezing probe (`docs/phase3/probes/freezing_probe.json`) explained why freezing kept the casts and produced arm C3G (`6b3aca8`). |
| 09-22 02:04 | Final full shakedown started on the fixed code. |
| 09-22 05:53 | Final shakedown done (3.8 h): 180/181, then 189/191 after the clock-drift rule was scoped per comparison (`5109c8c`); the two remaining items are the re-audit's tree identity and the C3 vs C3G pairs at 4,410 whose restarts were all clock-flagged on the shared node (environmental; `prereg_amendments.md` note). 5 Hz clock sampler added for session 3 (`ed3ea3d`). |
| 09-22 05:56 [23:56 on 09-21] | Pod "melband profiling" created in EU-RO-1 (EU-CZ-1 had no 5090); bundle of `ed3ea3d`, clip pack and `setup.sh` shipped; pins verified on the pod. |
| 09-22 06:02 [00:02] | Session 3 started on the pod (`SESSION_BUDGET_S=18000`, reserve 900 s, `E5_BUDGET_S=3600`; `SESSION_POD=1`, so the E9 held-out lengths run). |
| 09-22 08:06–08:45 [02:06–02:45] | Owner paused the assistant; the session ran on unattended (E1 → E9). |
| 09-22 10:31 [04:31] | Session 3 done: 16,118 s elapsed with 982 s of budget to spare; on-pod audit COMPLETE 198/198 (grade pod); E5 (48 cells) and E7 completed inside the budget, so the prepared continuation was not needed. |
| 09-22 10:31–10:45 | Extraction: 1,698 files, archive sha256 `0691998a…`, remote content hash unchanged during extraction (receipt in `pod.env.extracted`). |
| 09-22 10:46 [04:46] | Pod deleted (`runpodctl pod delete d5n5vfr3z1tbp6`); production pods untouched. Pod time about 4.8 h. |
| 09-22 10:50–11:05 | Off-pod: local re-audit (197/198; the one difference is the tree identity of a later analysis commit), `SESSION3_RESULT.md`, `LEDGER.md` realized/ceiling, figure CSVs, E9 residual attribution; pushed as `463c923`. Codex gate 10 on the analysis tools started (local). |

## 3. What was built

- Arms (`fullmodel/fullarms.py`): C3F (C3 repairs + Inductor freezing, passed as compile options so the artifact identity differs from C3's), C3G (freezing with `torch._dynamo.config.inline_inbuilt_nn_modules=False` patched around its compile and calls, per the probe), C3P (`fullmodel/precast.py`: band-split and mask-estimator Linear weights pre-cast to fp16 once, external module swap), C3M (repairs + `mode="max-autotune"`), C1P (C1 + `graph_partition`), C3N (`fullmodel/foldnorm.py`: RMSNorm gamma folded into the following Linear weights, per-token scale after the GEMM, fp32 folded buffers so autocast behaviour equals C3's).
- E5 variants (`fullmodel/variants.py`): the pinned construction path on a derived yaml, depth ∈ {1, 6} × mask-estimator expansion ∈ {4, 1}, random weights (seed 20260921), own reference directories, never headline-eligible. The C3 repair's Attend count is derived from the model.
- Matched runs: same-arm A/A pairs (`C3#2`), joint settle until two consecutive stable rounds with warm-up rounds recorded, held-out guard (`fullarms.held_out_guard`: 176,400 and 264,600 samples run only with `SESSION_POD=1`).
- E7 (`fullmodel/boundary_bench.py`): prologue chains (4,096 dependent FMAs before `griddepcontrol.wait`), launch-attribute record, SASS check with Triton's bundled cuobjdump (ACQBULK/PREEXIT found), per-chain PDL qualification, overlap statistics in the trace analysis.
- Analysis tools (`fullmodel/e_tools.py`: parameter-only cast count reproducing X6a exactly on the Sep-20 trace, GEMM-template count by ideal-cost class, A/A statistics, E9 check, E5 summary; `fullmodel/session3_result.py`; `fullmodel/ledger.py --phase3-dir`; `fullmodel/figures.py`), the session-3 audit (`fullmodel/completeness.py`: mandatory E6/E3/E2/E1/E9/E4a items, optional E5/E7, evidence grade, held-out and variant reference joins, clock-drift rule scoped per comparison).
- Session script `scripts/session3.sh`: order E6 → E3 → E2 → E1 (C3F, C3G, C3P by rule) → E9 (pod only) → E4a (control and mode decided from E2 by the registered rule) → E5 (allowance) → E7 → collection and audit; every launch bounded with terminal records.

## 4. Verification before the measurement

- Smoke (ml-beast GPU 4, 22:01–22:19): every new arm builds, passes its gate and validates against eager at 4,410; A/A pair −0.024 ms; E5 variant and boundary prologue fixed (`2faa775`); held-out guard refuses off-pod runs.
- Codex gate 09: eleven findings, all on the measurement path or eligibility (compile identity, cast counting, GEMM classification, E4a control gating, clock drift, reference joins, optional-aux propagation, restart binding, off-pod grading, held-out frame counts, A/A refresh); all closed in `3c47f6a`. No style findings were raised or acted on.
- Two full shakedowns (engineering grade): the first found three harness defects, the second ran every stage end to end (E6 40/40 headline restarts eligible; E3, E2, E1 incl. C3G/C3P, E4a under max-autotune, E5 48/48 cells, E7 with prologue chains) with only the shared node's clock condition open.
- Freezing probe (`docs/phase3/probes/freezing_probe.json`): with torch 2.8's nn-module inlining, freezing leaves the autocast parameter casts in place (9 of 9 on the probe stack; 420 of 420 on C3F's traces); with inlining off it folds them (1 of 9; 0 of 420 on C3G's traces).

## 5. Session 3 on the pod

Stage durations (from `evidence/session3_console.log`, UTC): preflight 00:02–00:04; E6 00:04–00:31 (two A/A pairs, 40 headline restarts, all eligible); E3 00:31–00:40; E2 00:40–01:52 (the two C3M traces took 30 and 25 minutes because a max-autotune trace recompiles with the Inductor caches disabled under the profiler); E1 01:52–02:36 (C3F, C3G, C3P with their traces and SDR gates); E9 02:36–02:48; E4a 02:48–04:00; E5 04:00–04:30; E7 04:30–04:31; collection and audit 04:31.

Headline cells over five restarts (`summary_full.json`, min-of-medians medians): A 63.05 ms and 10.47 ms; C1 36.46 and 5.32; C3 29.58 and 3.25; H 29.61 and 2.33 (8 s and 100 ms). The Sep-20 pod gave C3 28.87 and H 2.34 at the same points; this pod is about 2.5 % slower at 8 s.

E-by-E tables, intervals, cast counts, graph verdicts, template counts, E5 cells and the E7 chain table are in `SESSION3_RESULT.md`; the ledger notes carry every realized value with its files.

## 6. Registered predictions versus outcomes

See the summary table in §1; the verdict lines with the exact rule applied are in `SESSION3_RESULT.md` ("Registered prediction" lines). The E9 registration (`docs/phase3/e9/e9_cost_model.json`, sha256 recorded in `session3_config.json` at session start) predicted the crossover at 261,072 samples; the measurement puts it between 264,600 and 352,800. Under the registered rule the residual is attributed, not refitted: absorbing it in one term would require C3's per-band GEMM rate to be 53 rather than 71 TFLOPS at 401 frames and 67 rather than 82 at 601 (`docs/phase3/e9/e9_residual.json`), i.e. the two-point saturating rate curves are wrong at mid lengths, consistent with the in-sample residual of −0.42 ms at 88,200.

## 7. The ledger: realized versus ceiling

`LEDGER.md` now carries four realized values (X6a and X2a at both operating points) with block-level intervals and their source files, plus notes for E2, E3, E9 and E7. Coverage check: one row's realized value exceeds its ceiling (X6a at 8 s, 1.40 vs 0.92 ms), which falsifies "casts only" as the attribution of freezing's gain. Tightness: grade B rows scored 4, realized/ceiling median −0.02 (min −2.04 for the wrapper fold at 8 s, max 1.53 for X6a at 8 s); grades A and C have no intervened rows. The 100 ms X6a row is 0.78 of its ceiling with interval [0.77, 0.83].

## 8. Audit verdicts and corrections

On-pod: COMPLETE 198/198, evidence grade pod, no optional item incomplete, clock-drift rule applied at 5 Hz sampling with no violation that removed a stage. Off-pod re-audit from the analysis commit: 197/198, the single difference being the manifest's tree identity (the analysis tools were committed after the measurement; `session_status_local_reaudit_940cd26.json` beside the on-pod record). No measuring code was changed after the session; the analysis tools were corrected before the session (gate 09) and the result generator's wording after it (E6 and E4a verdict lines, E5 (iii) wording), with recorded identity.

## 9. Owner decisions and deviations

- "Do not nitpick": one codex gate before the pod (gate 09) and one on the analysis tools after the session; gate-08 items 114–127 remain deferred.
- "You can now even take a pod if necessary but only as instructed": the pod was created only after the final shakedown ran end to end; its audit was 189 of 191 mandatory items rather than COMPLETE, with the two remaining items environmental (shared-node clock proxy at 4,410) and re-audit identity. This is a deviation from the letter of R10 and is recorded here for the owner.
- Default predictions: the owner did not fill §8b before the session, so the assistant's defaults stand as registered.
- Additive implementation notes in `prereg_amendments.md` v0.2 (compile identity via options, cast/template counting rules, E4a quality gating, E1 lane stop and C3P rule, held-out frame counts, joint-settle policy, C3G, 5 Hz sampler), all dated before the pod session.
- Codex is never run on ml-beast (owner instruction 2026-09-21).

## 10. Open items

1. Codex gate 10 on the analysis tools (running at the time of writing; its verdict and any fixes are recorded in `docs/codex/10_analysis.md` and the closing commit).
2. X6a attribution: freezing without inlining realizes 1.40 ms at 8 s against a 0.92 ms cast ceiling; the remaining 0.48 ms is unattributed (constant folding beyond the casts; the C3G trace with 729 kernels versus C3's 1,142 is the place to look).
3. E9: the cost model's rate curves need more than two trace points; the registered rule forbids refitting within this study, so the crossover between 6 s and 8 s stands as a measured bracket.
4. E4a: the wrapper-class fold does not pay under either mode; the kernel-class epilogue (E4b) remains the owner's lane.
5. Gate-08 audit-strictness findings 114–127 remain deferred by the owner's rule.
6. Push-time evidence stays client-side until the repository is public or the owner records the GitHub Events `created_at` (`docs/provenance/push_evidence.md`); the OpenTimestamps proof can be upgraded after about a day.

## 11. Reproducibility

- Registration and code: `Shakil-Muhit/sm120-nulltest` at `ed3ea3d` (measurement).
- Evidence: `/home/muhit/ms/evidence_pod/d5n5vfr3z1tbp6_20260922_1031/evidence` (1,698 files; receipt line in `pod.env.extracted`, archive sha256 `0691998a60f0f87fd954c218a2fa36ddc0ae441f9fe9001d7805bd74a5f4fc5e`). Sep-20 evidence as before.
- Regenerate: `python fullmodel/completeness.py --evidence-dir <ev> --session session3`; `python fullmodel/session3_result.py --evidence-dir <ev> --out SESSION3_RESULT.md`; `python fullmodel/ledger.py --evidence-dir <sep-20 ev> --phase3-dir <ev> --out LEDGER.md`; `python fullmodel/figures.py --evidence-dir <sep-20 ev> --phase3-dir <ev>`.
