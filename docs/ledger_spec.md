# LEDGER.md specification (plan Phase 2) — formulas, inputs, confidence, stacking
### v0.1 · 2026-09-20 · written BEFORE session data exists; every ceiling computed from it is a registered prediction (plan §Phase 3.5 scores realized/ceiling)

`Y(X, O) = T_attributable(X, O) × f_recoverable(X)` [ms], `Z = Y / T_wall(best arm, O)`,
O ∈ {352800 (8 s), 4410}, plus the sweep where stated. `T_wall(best arm, O)` = the lowest
HEADLINE latency (median over r1–r3 of per-process min-of-medians) among headline-eligible
arms at O (evidence/summary_full.json). Every `T_attributable` comes from a node-level trace
(evidence/traces/…) and is labelled **MODELLED** wherever it is divided by an unprofiled wall;
the perturbation factor of that trace (evidence/perturbation_and_source_map.json) is carried as a
caveat when > 5 %. Confidence: **A** measured directly (an arm exists that realizes X);
**B** measured inputs + simple model; **C** model with an untested assumption (stated).
"Would be lower if …" is mandatory per row. Unattributable inputs → the row is `UNRESOLVED`,
never a proxy.

| X | T_attributable(X, O) — evidence | f_recoverable — model | conf. | effort | quality risk | stacking group |
|---|---|---|---|---|---|---|
| X1 graph-break removal | matched pair (C1, C3) at 352800: median over blocks of C1−C3 (evidence/matched_C1_C3_c352800_m*.json); at 4410: headline C1 − headline C3 | 1.0 (measured, not modelled) | A | wrapper | none (validated) | — |
| X2 epilogue fusion into GEMMs | C3 trace: INDUCTOR_FUSED kernels adjacent (wrapper order) to extern GEMMs whose fused ops are bias/GELU/residual/norm (kernel_source_map + inductor ledger); their device time per pass | `1 − ideal_fused_epilogue_bytes / current_bytes` from ideal_costs rows (bytes read+written vs the bytes a fused epilogue would move: output once); upper bound = their whole time | B | config→kernel | numerical | G1 (shares kernels with X6, X13) |
| X3 attention backend per axis | C3 trace: SDPA class time per pass split by axis (6 time-axis + 6 freq-axis calls; kernel names/ordering from the trace) | `1 − t_best / t_model` per axis from attn_backend_bench.json (**model-layout** cases; contiguous cases reported separately) | B | config | numerical (backend-dependent rounding) | — |
| X4 head-merge copy elimination | C3 trace: the 12 pure-copy kernels after SDPA (source map: pure clone/copy ops between SDPA and to_out GEMM) — plan's prior: 0.887 % at 8 s (axial study) | 1.0 | A/B (A if the axial Phase-2 wrapper is re-measured) | wrapper | none | — |
| X5 band-split + mask-estimator restructuring | H arm: matched pair (C3, H) at 352800 (median over blocks of C3−H) and headline C3−H at 4410; the band_split + mask-estimator kernels' time in the C3 trace (source map: kernels between band gather and first transformer; and after the last transformer to the scatter) | measured (H) vs modelled residual: padded-FLOPs waste = (ΣK_max·… − ΣK_i·…)/(ΣK_max·…) at roofline for the grouped alternative | A (H) / C (grouped residual) | wrapper/kernel | none (validated) | — |
| X6 elementwise/norm/RoPE/gating traffic | C3 trace: all INDUCTOR_FUSED pointwise/reduction kernels not counted in X2/X13 | per kernel `recoverable_share_hbm_model` from ideal_costs (= max(0, 1 − ideal_hbm/measured); rows flagged `bound_not_tight` contribute 0; the L2-rate variant is reported alongside, never summed into Y) | B | kernel | numerical | G1 |
| X6a autocast weight re-casts (sub-row of X6, registered 2026-09-20 from shakedown structure, before session data) | C3 trace: `to_copy` INDUCTOR_FUSED kernels whose only input is a graph parameter (wrapper `arg*_1`, fp32 → fp16), per pass — expected 60 mask-estimator weight casts | 1.0 (cast once at load; shape-invariant across O, so the share is largest at 4410) | B | wrapper (weights pre-cast outside the pinned clone) | numerical (fp16 weights identical to autocast's cast) | G1 |
| X7 STFT / iSTFT | A-NVTX + C3 traces: FFT class + windowing/framing kernels (front-end and tail ranges) | `1 − FFT roofline / current` where roofline = bytes/BW for the FFT working set (2·2·1025·T·8 B complex fp32 in+out) | C (assumes the FFT is memory-bound) | kernel | numerical | — |
| X8 FP8 GEMMs (offline) | C3 trace: GEMM class time per pass | `1 − rate_fp16 / rate_fp8` = 0.5 on GB202 at FP32 accumulate (registered constant; NOT measured) | C | kernel/config | **task-level (SDR pre-check required)** | G2 (alternative to X9) |
| X9 fp16-accumulate GEMMs | GEMM class time | ≤ 0.5 (GeForce full rate) | C | kernel | **task-level** | G2 |
| X10 kernel-count reduction at 4410 | best arm at 4410: N_kernels per pass (graph_verify / kern_sum) | `c_boundary × (N_current − N_floor)`, c_boundary = the measured serial inter-kernel gap of the matching mode from boundary_bench_trace_analysis.json (graph mode for graphed arms) — NEVER the per-node service time (which includes execution); N_floor = kernels that cannot fuse without a megakernel (registered estimate: one kernel per GEMM/SDPA/FFT call + one per fused elementwise group = counted from the source map) | C | kernel | numerical | G3 (with X11, X12) |
| X11 concurrency at 4410 | G/F traces: three timing views (evidence/traces/timing_views_*.json) | `1 − observed-order critical path / summed durations` (upper bound; the true dependency path is not exported) | C | wrapper | none | G3 |
| X12 persistent megakernel at 4410 | wall(best arm, 4410) − critical-path compute (observed-order bound) − sync floor (boundary graph-mode gap × 1) | upper bound only; realistic range 1.5–3× over graphs quoted from Hazy/Mirage, not derived here | C | architecture | numerical | G3 |
| X13 mask-estimator MLP chain fusion | C3 trace: mask-estimator kernels (source map: tail range) — GEMMs + Tanh/GLU epilogues | roofline: `1 − ideal(FLOPs/P, bytes/BW) / current` from ideal_costs gemm_sites in the tail + fused epilogue rows | B | kernel | numerical | G1 |
| X14 cross-chunk batching (throughput) | whole pass: throughput_ms_per_pass_best vs latency (5a secondary block) for the best arm | Amdahl over fixed per-pass costs: `1 − t_fixed / T`, t_fixed = the launch/graph-launch floor from boundary_bench (graph mode intercept) — reported as THROUGHPUT, separately | C | wrapper | none | — |

Stacking note (mandatory in LEDGER.md): G1 rows (X2, X6, X13) share kernels — sum at most the
union of their attributable time; G2 rows (X8, X9) are alternatives — never summed; G3 rows
(X10, X11, X12) describe the same launch-bound regime from three models — report the max, not
the sum; X1 and X5 are measured deltas and may be summed only with rows attributed from the C3
trace (they are already "inside" C3's wall).

Thresholds (plan 3.2, applied mechanically in LEDGER.md): offline 8 s — Z ≥ 5 % with A/B, or
Z ≥ 10 % with C; streaming 4410 — Y ≥ 2× on wall over the best arm (reported, pursued only if the
lane reopens); X8/X9 require an SDR pre-check before any kernel work.
