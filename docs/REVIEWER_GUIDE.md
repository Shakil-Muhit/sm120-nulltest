# Reviewer guide (for an evaluating agent, 2026-09-22)

Start here. This repository holds a pre-registered measurement study of launch-side tooling (CUDA Graphs, `torch.compile`) and structural interventions on the MelBandRoformer separation model on an RTX 5090 (SM120). Three pod sessions were run (two on 2026-09-20, one on 2026-09-22); every number in the documents is computed by a tool from the evidence files it cites.

## Read in this order
1. `docs/EXECUTION_REPORT_2026-09-20.md` — Phases 0–2: what was built, the two sessions, headline results, the ledger.
2. `docs/EXECUTION_REPORT_2026-09-22.md` — Phase 3: the nine registered experiments E1–E9, their outcomes, audits, deviations.
3. `docs/contribution_test.md` — the one central finding, the four tests, the closest prior work, the venue verdict.
4. `SESSION3_RESULT.md` — E-by-E results with intervals, cast counts, graph verdicts, template counts, E5 cells, E7 chains.
5. `LEDGER.md` — the ceiling ledger with the realized/ceiling column filled from session 3; `LEDGER.json` is the same as data.
6. `SESSION_RESULT_v2.md` — the Sep-20 results with statistics, decompositions and corrections; `SESSION_RESULT.md` is the Sep-20 original.
7. `prereg_amendments.md` — the registration: v0.1 (Phases 1–2) and v0.2 (E1–E9 with dated, additive implementation notes).
8. `docs/related_work/README.md` and the notes 01–16 — the literature positioning, the F1–F8 three-column table, and the three sources closest to the central finding.
9. `docs/FUTURE_PLAN_2026-09-22.md` — proposals, none started.
10. `docs/codex/` — the ten adversarial review gates (prompts, verdicts, transcripts) and their closures.

## Evidence
- `evidence_session3_core/` (in the handover package): the session-3 pod records (run and matched JSON, traces' derived JSON, console and clock logs, the on-pod audit `session_status.json` and the corrected re-audit `session_status_reaudit_gate10.json`). Large raw exports (`.nsys-rep`, `.sqlite`, per-kernel trace CSVs, `.pt` references, compile dumps) are omitted; they exist locally and are listed in `docs/release/evidence_manifest.csv` (Sep 20) and can be provided.
- `evidence_sep20_core/`: the same selection for the Sep-20 sessions.
- Regeneration: `python fullmodel/completeness.py --evidence-dir <ev> --session session3`; `python fullmodel/session3_result.py --evidence-dir <ev> --out SESSION3_RESULT.md`; `python fullmodel/ledger.py --evidence-dir <sep20 ev> --phase3-dir <ev> --out LEDGER.md`; `python fullmodel/figures.py --evidence-dir <sep20 ev> --phase3-dir <ev>`.

## Conventions to know
- "Matched" differences are a − b in ms, median over eligible restarts of per-repeat block medians recomputed from samples, with block-level bootstrap intervals (10,000 resamples, seed 20260921); + means the second arm is faster.
- Headline walls are unprofiled; trace-derived numbers are MODELLED and carry a perturbation caveat; UNRESOLVED means a missing input, never a proxy.
- Engineering data (ml-beast shakedowns) is never evidence; the audit stamps it.
- The registered 5 % clock rule, applied at 5 Hz, flags the three compiled 100 ms headline cells of session 3; they are published clock-flagged (see the execution report §8 and the future plan A1).

## Questions the reviewer is asked to answer
1. Does the central finding pass the four tests as stated, and is the venue verdict right?
2. Are the failed predictions (E3, E2 upper bound, E1 freezing-as-shipped, E9, E4a, E7 prologue) reported without post-hoc rescue?
3. Is any published number not traceable to its cited file, or any eligibility rule silently relaxed?
4. Which of the future-plan items C1–C5 would most change the paper, if any?
