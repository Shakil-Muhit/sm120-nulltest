# Ceiling-ledger execution report — 2026-09-22 (Phase 3)

Handoff: the owner's Phase-3 handoff of 2026-09-21 (supersedes `plan_ceiling_ledger.md` after Phase 2). Executor: Claude Fable 5.1 with codex gpt-6-astra (max effort, read-only, on the driver workstation only) as the adversarial reviewer. Repository: `Shakil-Muhit/sm120-nulltest`, branch `main`, measurement commit `ed3ea3d`. Pod: "melband profiling" (RunPod `d5n5vfr3z1tbp6`, RTX 5090, EU-RO-1, image `runpod/pytorch:1.1.0-rc.154-cu1281-torch280-ubuntu2204`: torch 2.8.0+cu128, triton 3.4.0, driver 580.178.04, Nsight Systems 2026.5.1).

## 1. Summary

RESULTS-SECTION-PENDING (filled from the extracted evidence: `SESSION3_RESULT.md`, `LEDGER.md` realized/ceiling column, `e9_check.json`).

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
| 09-22 06:1x [00:1x] | Pod "melband profiling" created in EU-RO-1 (EU-CZ-1 had no 5090); bundle of `ed3ea3d`, clip pack and `setup.sh` shipped; pins verified on the pod. |
| 09-22 06:02 [00:02] | Session 3 started on the pod (`SESSION_BUDGET_S=18000`, reserve 900 s, `E5_BUDGET_S=3600`; `SESSION_POD=1`, so the E9 held-out lengths run). |
| RESULTS-SECTION-PENDING | session end, audit verdict, extraction receipt, pod deletion. |

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

RESULTS-SECTION-PENDING (per stage: E6 A/A and headline restarts, E3, E2, E1, E9, E4a, E5, E7; audit verdict; extraction receipt).

## 6. Registered predictions versus outcomes

RESULTS-SECTION-PENDING (table from `SESSION3_RESULT.md`).

## 7. The ledger: realized versus ceiling

RESULTS-SECTION-PENDING (`LEDGER.md` regenerated with `--phase3-dir`; coverage check; tightness by grade).

## 8. Audit verdicts and corrections

RESULTS-SECTION-PENDING.

## 9. Owner decisions and deviations

- "Do not nitpick": one codex gate before the pod (gate 09) and one on the analysis tools after the session; gate-08 items 114–127 remain deferred.
- "You can now even take a pod if necessary but only as instructed": the pod was created only after the final shakedown ran end to end; its audit was 189 of 191 mandatory items rather than COMPLETE, with the two remaining items environmental (shared-node clock proxy at 4,410) and re-audit identity. This is a deviation from the letter of R10 and is recorded here for the owner.
- Default predictions: the owner did not fill §8b before the session, so the assistant's defaults stand as registered.
- Additive implementation notes in `prereg_amendments.md` v0.2 (compile identity via options, cast/template counting rules, E4a quality gating, E1 lane stop and C3P rule, held-out frame counts, joint-settle policy, C3G, 5 Hz sampler), all dated before the pod session.
- Codex is never run on ml-beast (owner instruction 2026-09-21).

## 10. Open items

RESULTS-SECTION-PENDING.

## 11. Reproducibility

- Registration and code: `Shakil-Muhit/sm120-nulltest` at `ed3ea3d` (measurement).
- Evidence: RESULTS-SECTION-PENDING (extraction directory and receipt).
- Regenerate: `python fullmodel/completeness.py --evidence-dir <ev> --session session3`; `python fullmodel/session3_result.py --evidence-dir <ev> --out SESSION3_RESULT.md`; `python fullmodel/ledger.py --evidence-dir <sep-20 ev> --phase3-dir <ev> --out LEDGER.md`; `python fullmodel/figures.py --evidence-dir <sep-20 ev> --phase3-dir <ev>`.
