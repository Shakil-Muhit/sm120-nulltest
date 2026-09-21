# Chen, "Memory-Bound but Not Bandwidth-Limited" (arXiv 2605.30571v1, May 2026; not peer-reviewed)

**Reading-order item 9.** Tag: F5, F6, ledger, methodology OVERLAP.

## Research question
When a decode step is memory-bound by arithmetic intensity, is it actually bandwidth-limited in practice, and does a preregistered CUDA-graph A/B recover the gap?

## Main contribution (authors' claims, as read)
An analytic floor t_floor = (W + K) / B_peak (weights plus KV bytes over peak bandwidth), a ratio R_floor of floor to measured time, and a preregistered CUDA-graph A/B with kill criteria and confidence intervals. Results: CUDA graphs give 1.259× (95 % CI [1.253, 1.267], N = 10) on H100 versus 1.028× on L4; R_floor 0.27 on H100 versus 0.81 on L4, read as launch-side overhead dominating on the faster GPU. 44 cells in the study.

## Evidence and scope
LLM decode, two GPUs (H100, L4), preregistered protocol with CIs; single author; arXiv only.

## Exact version and date read
arXiv 2605.30571v1 HTML; read 2026-09-20.

## Relevant source locations
Floor definition; R_floor table; preregistration and kill criteria; CUDA-graph A/B table.

## Relationship to this study
- Overlap: nearest design cousin. Preregistered predictions with kill rules, an analytic floor, a CUDA-graph A/B with intervals. The study's registration lives in `prereg_amendments.md` (commit e523ef9, pushed before session 1; `docs/provenance/push_evidence.md`) and its intervals in `docs/phase3/stats/bootstrap_headlines.csv` and `bootstrap_pairs.csv` (10,000 block-level resamples).
- Difference: Chen's floor is a bytes-over-bandwidth point estimate for the whole step; the ledger is per-class and per-mechanism with confidence grades. Chen measures the same GPU class as Hazy; this study measures SM120.
- Scientifically meaningful: the R_floor idea supports F6's boundary-removal wording (launch-side terms matter when the floor is small); it is not the study's.

## Consequence for this work
Cite Chen for the preregistered-A/B form and for R_floor; describe the study's registration as self-registered with public time evidence, never as a registered report.

## Unresolved
Chen's exact N per cell beyond the H100 A/B; peer-review status.
