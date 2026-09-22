# Analysis revision (Phase 4, 2026-09-22; off-pod part — the session-4 results section is appended after the measurement)

This revision corrects interpretation, not history: every Phase-3 artifact stands as published; corrected statements are given beside the originals with their evidence.

## 1. E9: what the registered predictor was, and what its failure means
- Original (registered): variant `sweep_refit_THNone_TCNone`, presented with saturating per-arm GEMM rate curves; predictions D(176,400) = +0.368 ms, D(264,600) = −0.015 ms; verdict FAIL (observed +1.129 [+0.958, +1.303] and +0.672 [+0.581, +0.718] ms, cluster intervals).
- Corrected reading: with unit tile factors the model is exactly affine in frames, D(F) = 1.136476 − 0.001915801·F ms (`docs/phase4/e9_affine_check.json`). Its four calibration residuals (+0.18, +0.14, −0.42, +0.09 ms) come from fitting a line to non-monotone values; a different line meets the 0.4 ms tolerance on those values (max 0.288 ms), so the miss is not an inevitability of affine forms. Three things are confounded in the held-out miss and are not separated by the evidence: the representational restriction (a line), the fitted coefficients (sweep refit with soft trace constraints), and transfer to another physical card and session. The Phase-3 sentence "the fixed terms transfer and the efficiency term does not" is withdrawn: `e9_residual.json` shows only that changing one rate would absorb the residual, which is sensitivity, not attribution. E9 remains a failed registered test; nothing is refitted.
- Retrospective comparators recorded for any future predictor work (labelled retrospective): the calibration-only minimax line; a piecewise-linear interpolation through the four sweep points gives +0.616 ms at 176,400 and +0.062 ms at 264,600 (closer to the observations than the registered line, but selected after seeing them, so it is not a validated predictor).

## 2. Statistics: estimand, unit, interval
- Estimand: per-repeat paired difference of block medians (a-block minus b-block, alternating within one process); statistic: median over restarts of per-restart medians. Unit of inference: the process restart (compile + settle + 5 repeats); repeats are clustered within restarts. Scope: this card, this session, this software configuration.
- Revised interval: two-level cluster bootstrap (restarts, then repeats), 10,000 draws, seed 20260921, same statistic. With three restarts the interval cannot be narrower than the spread of restart medians, which are reported beside it. Historical intervals (pooled-repeat median bootstrap) are kept in `docs/phase4/stats_v2_session3.json` and labelled as estimating a different statistic. No Phase-3 sign or verdict changes under the revision (table in `stats_v2_session3.json`).
- The 100 ms A/A control missed its 0.05 ms threshold (−0.058 ms) with an interval including zero; every 100 ms claim carries that, plus the registered clock rule's flags (three 100 ms headline cells of session 3 clock-flagged; flagged repeats excluded from pair aggregates).

## 3. Provenance and the crossover claim
| comparison | card / session | design | value |
|---|---|---|---|
| C3 vs H at 8 s | GPU-6bf4cb32…, Sep 20 (pod 9tblar2zwo97ln), commit e523ef9 | matched, 3 restarts | C3 faster by 0.495 ms (paired) |
| C3 vs H at 4 s and 6 s | GPU-7d47d5c2…, Sep 22 (pod d5n5vfr3z1tbp6), commit ed3ea3d | matched, 3 restarts | H faster by 1.129 and 0.672 ms (paired) |
| C3 and H at 8 s | GPU-7d47d5c2…, Sep 22 | independent headline cells, 5 restarts each | 29.582 vs 29.611 ms (unpaired near-tie; not an equivalence result) |
Same software pins, driver, timing procedure and registration lineage; different physical cards. Supported statement: the historical ranking change was observed under its original conditions; a stable crossover on the current baseline is unconfirmed. PA (session 4) measures C3 vs H matched at 1, 4, 6 and 8 s on one card in one session; no location between measured lengths will be claimed.

## 4. E2 (C3 vs C3M at 8 s): descriptive class deltas and template provenance
Traced device time per pass (node-level, perturbed; `docs/phase4/inventory_diff_session3.json`): GEMM 15.939 → 13.556 ms (−2.384), INDUCTOR_FUSED 8.471 → 5.899 ms (−2.571), SDPA −0.033, other classes ≈ 0; sum −4.988 ms versus the unprofiled matched gain of 4.797 ms. From the generated-kernel source: 178 template kernels per pass; the largest are the band GEMMs with the activation cast and scale fused (`_to_copy, bmm, div, mul`, 3.51 ms), the feed-forward GEMM with GELU fused (`addmm, gelu`, 3.48 ms; the standalone GELU kernel of 2.13 ms disappears), plain `mm` (1.03 ms) and the mask-estimator MLP with tanh fused (`addmm, tanh`, 0.48 ms). The 420 parameter casts persist (0.905 ms). Supported: the gain is kernel selection plus elementwise work absorbed into templates; not supported: any exact GEMM-versus-epilogue split of the unprofiled 4.8 ms, and any cast-removal share.

## 5. E1 (freezing without inlining): what changed beyond the casts
At 8 s (`inventory_diff_session3.json`): the 420 parameter casts (0.915 ms) disappear; the attention kernels lose `cos/sin` from their fused ops (the rotary tables are constant-folded; time 1.352 → 1.384 ms, unchanged); the norm-plus-cast kernels shorten by 0.355 ms; launches 1143 → 729; class deltas sum 1.11 ms against an unprofiled 1.40 ms gain, leaving 0.29 ms unexplained by device class time. At 100 ms: casts 0.853 → 0, GEMM class 0.368 ms slower (frozen weights change the GEMM kernels chosen), sum 0.49 vs 0.67 ms unprofiled. The Phase-3 phrase "freezing folds more than the casts" is kept only for the rotary tables, which the source shows; the rest is reported as unexplained remainder. PC1/PC2 (inlining alone; freezing given inlining off) separate the two knobs.

## 6. E3 (C1P): the regression is a matched-context effect of the partition arm
Standalone profiled C1P: block median 36.59 ms, GPU span 36.18 ms, 8 graph breaks, 40 recorded cudagraphs. Matched C1P (built after C1 in the same process): 51.7 ms, 5 graph breaks, 71 recorded cudagraphs, 59 "empty CUDA graph" warnings; the Phase-4 smoke with the build order swapped gives C1 5 breaks when built second. Every other second-built arm compiled like its standalone run and ran within 0.4 ms of it. Supported: the second compile of a graph-break arm in a matched process differs from its standalone compile, and the regression appears only in that context. Not supported: that the missing time is host-side, or any cause. PB measures standalone unprofiled C1P, both build orders, and profiles the matched context with aligned per-pass windows (GPU busy, GPU idle within the window, launches, copies, API unions reported separately).

## 7. Audit status carried forward
On-pod session 3: COMPLETE 198/198 (tools at ed3ea3d). Corrected re-audit: INCOMPLETE 195/199 (tree identity off-pod; three clock-flagged 100 ms headline cells). Both remain in the record; the 8 s results are unaffected.

## 8. Session-4 results
(appended after the measurement from `SESSION4_RESULT.md`)
