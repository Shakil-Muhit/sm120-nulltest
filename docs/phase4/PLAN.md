# Phase 4 plan (prospective; written 2026-09-22 before any Phase-4 measurement)

Working revision: branch `phase4-followup` (main and every Phase-1–3 artifact unchanged). New measurements go to a new session directory (session 4) with their own manifest (hardware UUID, software, code hash, commands, inputs, seeds, timing policy, quality checks).

## Experiments, in order, with dependencies

| # | experiment | arms and pairs | operating points | depends on |
|---|---|---|---|---|
| P0 | A/A control | C3 vs C3 (matched, one restart) | 44,100; 176,400; 264,600; 352,800; 4,410 | — |
| PA | C3–H ranking under matched conditions | C3 vs H, 3 restarts each | 44,100; 176,400; 264,600; 352,800 | C3 operator qualification at 101 frames |
| PB | C1P runner discrepancy | (i) standalone unprofiled C1P, 3 restarts; (ii) matched C1 vs C1P, 3 restarts, and matched C1P vs C1 (build order swapped), 2 restarts; (iii) node-level profile of the matched process (one block, `--profile-matched`), and of a standalone C1P block, with perturbation = profiled vs unprofiled wall in the same context | 352,800 | — |
| PC1 | inlining knob alone | C3 vs C3I (inlining off, freezing off), 3 restarts | 4,410; 352,800 | — |
| PC2 | freezing given inlining off | C3I vs C3G, 3 restarts | 4,410; 352,800 | — |
| PC3 | composition with max-autotune | C3M vs C3MG, 3 restarts | 4,410; 352,800 | — |
| PC4 | freezing on the pad-and-batch arm | H vs HG, 3 restarts | 4,410; 352,800 | — |
| PQ | quality and deployment checks for new arms | C3I, C3MG, HG: r1 with the SDR gate at both operating points; node traces at 4,410 for all three and at 352,800 for C3I and HG (C3MG at 352,800 if budget remains: its trace recompiles with caches off, ~25 min) | — | PC |

## Statistics, fixed before measuring
- Primary statistic per pair and operating point: median over restarts of the per-restart median of the per-repeat paired difference (a − b of block medians; 5 repeats × 50 passes per restart), with a two-level cluster bootstrap interval (restarts, then repeats; 10,000 draws, seed 20260921). Restart medians and their range are reported beside it (`fullmodel/stats_v2.py`).
- Eligibility: the shared validator (`qualify.pair_problems`), manifest binding, and the registered 5 % clock rule applied per repeat (flagged repeats excluded; a stage needs at least one eligible restart with clean repeats).
- Smallest practically meaningful difference (declared now): 0.30 ms at 352,800 / 264,600 / 176,400 and 0.10 ms at 44,100 / 4,410, i.e. about 1 % of the 8 s wall and 4 % of the 100 ms wall. A pair is reported as: "a faster" or "b faster" when the interval excludes zero and |point| ≥ the threshold; "practical tie" when the interval lies within ±threshold; "unresolved" otherwise. No measurement is repeated to reach a preferred outcome; the repetition schedule is fixed above.
- Ranking claims are per measured length only; no crossover location is interpolated between lengths.
- Order policy: within a matched process the control arm is built first and blocks alternate a, b, a, b, …; the same policy applies to the A/A control, which measures order/position bias. PB additionally swaps the build order for C1/C1P to test the build-order interaction directly.

## Stopping and resource limits
- Session budget 4.5 h (`SESSION_BUDGET_S=16200`, reserve 900 s), stages ordered P0 → PA → PB → PC1–PC4 → PQ; cells the deadline cuts are recorded UNTESTED. Hard cap 6 pod-hours including setup and extraction.
- An arm stops on a correctness failure (validation FAIL or SDR gate FAIL) or an invalid setup (graph gate FAIL for a zero-break arm); independent stages continue.
- Estimated pod time (estimate, not a commitment): P0 15 min; PA 55 min; PB 45 min; PC1–PC4 90 min (max-autotune pairs dominate); PQ 60 min; audit 5 min: about 4.5 h. Engineering time before the pod: smoke of the three new arms and a full shakedown on the engineering host (about 4 h wall, no cost).

## What this session does not do
- It does not refit any predictor on 4 s / 6 s observations; E9 remains a failed registered test. PA is replication at sampled lengths, not validation.
- It does not rerun the historical matrix, and it adds no models, GPUs, quantization or megakernel work.
