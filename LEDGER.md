# LEDGER.md — MelBandRoformer ceiling ledger (plan Phase 2)
evidence: manifest 9bd6639d-582d-4e98-9bff-d195d4bcd924 · repo e523ef9318e6 · msst ea7eb9c20ea0 · ckpt 87201f4d31af · GPU NVIDIA GeForce RTX 5090 GPU-6bf4cb32-2adf-770d-783f-8fff8de28903

`Y(X,O) = T_attributable × f_recoverable` [ms]; `Z = Y / T_wall(best arm, O)`. MODELLED = trace-derived time over an unprofiled wall. UNRESOLVED rows cite the missing input; they are never proxied.

## O = 352800 (8 s) — best arm C3: T_wall = 28.8714 ms
session verdicts: session1 COMPLETE, session2 COMPLETE (an INCOMPLETE session keeps every row that cites a missing item UNRESOLVED)

| X | T_attr (ms) | f | Y (ms) | Z (%) | conf | effort | quality risk | group | label | evidence | would be lower if | caveats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| X1 graph-break removal | 6.7412 | 1.000 | 6.7412 | 23.35 | A | wrapper | none (validated) | — | MEASURED | matched_C1_C3_c352800_m1..m3.json (median over alternating blocks) | C1 recovered part of its breaks without the vendored repair | — |
| X2 epilogue fusion into GEMMs | 4.4806 | 1.000 | 4.4806 | 15.52 | C | config→kernel | numerical | G1 | MODELLED | traces/ideal_costs_C3_352800.json rows + dump wrapper order: 11 kernels | the epilogue kernels are already partly overlapped or the fused GEMM slows down | — · upper bound (whole time): rows carry no read/write split |
| X3 attention backend (time axis → cudnn) | 2.7353 | 0.069 | 0.1877 | 0.65 | B | config | numerical (backend rounding) | — | MODELLED | trace_C3_352800_trace SDPA per axis + attn_backend_bench.json c352800_time_model | the model layout needs an extra copy for the faster backend | — |
| X3 attention backend (freq axis → cudnn) | 0.8429 | 0.148 | 0.1251 | 0.43 | B | config | numerical (backend rounding) | — | MODELLED | trace_C3_352800_trace SDPA per axis + attn_backend_bench.json c352800_freq_model | the model layout needs an extra copy for the faster backend | — |
| X4 head-merge copy elimination | 0.0101 | 1.000 | 0.0101 | 0.03 | B | wrapper | none | — | MODELLED | dump wrapper order: 1 pure-copy kernels after SDPA; ideal_costs rows | the copies are needed by the faster attention backend layout | — |
| X5 band-split + mask-estimator restructuring (H) | -0.4954 | 1.000 | -0.4954 | -1.72 | A | wrapper | none (validated) | — | MEASURED | matched_C3_H / headline C3 − H | the grouped alternative pads more than measured | — |
| X5 residual pool (front + tail fused kernels) | 5.5372 | — | — | — | C | kernel | none | — | MODELLED | dump wrapper order ranges; ideal_costs rows | n/a | — · padded-FLOP residual model not derived (UNRESOLVED f); pool reported only |
| X6 elementwise/norm/RoPE/gating traffic | 0.2962 | 0.237 | 0.0702 | 0.24 | B | kernel | numerical | G1 | MODELLED | traces/ideal_costs_C3_352800.json: 44 INDUCTOR_FUSED rows (HBM model; bound_not_tight rows = 0) | kernels already hit L2 (the L2-rate variant gives Y = 0.054 ms, reported, not summed) | — |
| X6a autocast weight re-casts | 0.9198 | 1.000 | 0.9198 | 3.19 | B | wrapper | numerical (identical fp16 weights) | G1 | MODELLED | dump wrapper order: 117 to_copy kernels reading only graph parameters | some casts are of activations mis-identified as parameters | — |
| X7 STFT / iSTFT | 0.0783 | 0.625 | 0.0489 | 0.17 | C | kernel | numerical | — | MODELLED | ideal_costs_C3_352800.json FFT rows + framing/windowing fused rows (4); roofline bytes 52545600 | the FFTs are compute-bound at this size (assumption: memory-bound) | — |
| X8 FP8 GEMMs (offline) | 16.0835 | 0.500 | 8.0417 | 27.85 | C | kernel/config | TASK-LEVEL (SDR pre-check required) | G2 | MODELLED | ideal_costs_C3_352800.json class_totals GEMM; f = registered constant 0.5 (not measured) | fp8 needs scaling kernels and the SDR pre-check fails | — |
| X9 fp16-accumulate GEMMs | 16.0835 | 0.500 | 8.0417 | 27.85 | C | kernel/config | TASK-LEVEL (SDR pre-check required) | G2 | MODELLED | ideal_costs_C3_352800.json class_totals GEMM; f = registered constant 0.5 (not measured) | GeForce fp16-accumulate rate is not fully realised | — |
| X13 mask-estimator MLP chain fusion | 6.4088 | 0.069 | 0.4392 | 1.52 | B | kernel | numerical | G1 | MODELLED | ideal_costs_C3_352800.json gemm_sites (tail: 172 sites, GEMM time apportioned by FLOPs) + 142 tail fused rows | the tail GEMM time share differs from its FLOP share | — · GEMM class time apportioned by FLOP share (cuBLAS names carry no site) |
| X14 cross-chunk batching (THROUGHPUT, reported separately) | 28.6478 | 0.000 | 0.0024 | 0.01 | C | wrapper | none | — | MEASURED | boundary_bench.json fits trivial/graph intercept 2.398 µs; throughput_ms_per_pass_best of C3 | fixed per-pass costs are larger than the graph-launch floor | — · throughput headroom, not latency; Z here is relative to the throughput wall |
| G1 union of attributable time (X2 ∪ X6 ∪ X6a ∪ X13 fused) | 8.5161 | — | — | — | — | — | — | G1 | MODELLED | ideal_costs rows (distinct kernels) | n/a | — · stacking bound: G1 rows may sum to at most this |

**Thresholds (plan 3.2, mechanical):**
- X1 graph-break removal: Z = 23.35 % vs 5.0 % (A) → PURSUE
- X2 epilogue fusion into GEMMs: Z = 15.52 % vs 10.0 % (C) → PURSUE
- X3 attention backend (time axis → cudnn): Z = 0.65 % vs 5.0 % (B) → below threshold
- X3 attention backend (freq axis → cudnn): Z = 0.43 % vs 5.0 % (B) → below threshold
- X4 head-merge copy elimination: Z = 0.03 % vs 5.0 % (B) → below threshold
- X5 band-split + mask-estimator restructuring (H): Z = -1.72 % vs 5.0 % (A) → below threshold
- X6 elementwise/norm/RoPE/gating traffic: Z = 0.24 % vs 5.0 % (B) → below threshold
- X6a autocast weight re-casts: Z = 3.19 % vs 5.0 % (B) → below threshold
- X7 STFT / iSTFT: Z = 0.17 % vs 10.0 % (C) → below threshold
- X8 FP8 GEMMs (offline): Z = 27.85 % vs 10.0 % (C) → PURSUE
- X9 fp16-accumulate GEMMs: Z = 27.85 % vs 10.0 % (C) → PURSUE
- X13 mask-estimator MLP chain fusion: Z = 1.52 % vs 5.0 % (B) → below threshold

## O = 4410 (0.1 s streaming) — best arm H: T_wall = 2.3386 ms
session verdicts: session1 COMPLETE, session2 COMPLETE (an INCOMPLETE session keeps every row that cites a missing item UNRESOLVED)

| X | T_attr (ms) | f | Y (ms) | Z (%) | conf | effort | quality risk | group | label | evidence | would be lower if | caveats |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| X1 graph-break removal | 1.7109 | 1.000 | 1.7109 | 73.16 | A | wrapper | none (validated) | — | MEASURED | summary_full.json: headline C1 − headline C3 | C1 recovered part of its breaks without the vendored repair | — |
| X2 epilogue fusion into GEMMs | 0.0646 | 1.000 | 0.0646 | 2.76 | C | config→kernel | numerical | G1 | MODELLED | traces/ideal_costs_C3_4410.json rows + dump wrapper order: 11 kernels | the epilogue kernels are already partly overlapped or the fused GEMM slows down | C3 trace perturbation ×1.052 (node-level) · upper bound (whole time): rows carry no read/write split |
| X3 attention backend (time axis → efficient) | 0.0691 | 0.464 | 0.0321 | 1.37 | B | config | numerical (backend rounding) | — | MODELLED | trace_C3_4410_trace SDPA per axis + attn_backend_bench.json c4410_time_model | the model layout needs an extra copy for the faster backend | C3 trace perturbation ×1.052 (node-level) |
| X3 attention backend (freq axis → cudnn) | 0.0332 | 0.425 | 0.0141 | 0.60 | B | config | numerical (backend rounding) | — | MODELLED | trace_C3_4410_trace SDPA per axis + attn_backend_bench.json c4410_freq_model | the model layout needs an extra copy for the faster backend | C3 trace perturbation ×1.052 (node-level) |
| X4 head-merge copy elimination | 0.0095 | 1.000 | 0.0095 | 0.41 | B | wrapper | none | — | MODELLED | dump wrapper order: 1 pure-copy kernels after SDPA; ideal_costs rows | the copies are needed by the faster attention backend layout | C3 trace perturbation ×1.052 (node-level) |
| X5 band-split + mask-estimator restructuring (H) | 0.9367 | 1.000 | 0.9367 | 40.05 | A | wrapper | none (validated) | — | MEASURED | matched_C3_H / headline C3 − H | the grouped alternative pads more than measured | — |
| X5 residual pool (front + tail fused kernels) | 1.3196 | — | — | — | C | kernel | none | — | MODELLED | dump wrapper order ranges; ideal_costs rows | n/a | C3 trace perturbation ×1.052 (node-level) · padded-FLOP residual model not derived (UNRESOLVED f); pool reported only |
| X6 elementwise/norm/RoPE/gating traffic | 0.0769 | 0.912 | 0.0701 | 3.00 | B | kernel | numerical | G1 | MODELLED | traces/ideal_costs_C3_4410.json: 44 INDUCTOR_FUSED rows (HBM model; bound_not_tight rows = 0) | kernels already hit L2 (the L2-rate variant gives Y = 0.029 ms, reported, not summed) | C3 trace perturbation ×1.052 (node-level) |
| X6a autocast weight re-casts | 0.8574 | 1.000 | 0.8574 | 36.66 | B | wrapper | numerical (identical fp16 weights) | G1 | MODELLED | dump wrapper order: 117 to_copy kernels reading only graph parameters | some casts are of activations mis-identified as parameters | C3 trace perturbation ×1.052 (node-level) |
| X7 STFT / iSTFT | 0.0082 | 0.951 | 0.0078 | 0.33 | C | kernel | numerical | — | MODELLED | ideal_costs_C3_4410.json FFT rows + framing/windowing fused rows (4); roofline bytes 721600 | the FFTs are compute-bound at this size (assumption: memory-bound) | C3 trace perturbation ×1.052 (node-level) |
| X8 FP8 GEMMs (offline) | 1.2314 | 0.500 | 0.6157 | 26.33 | C | kernel/config | TASK-LEVEL (SDR pre-check required) | G2 | MODELLED | ideal_costs_C3_4410.json class_totals GEMM; f = registered constant 0.5 (not measured) | fp8 needs scaling kernels and the SDR pre-check fails | C3 trace perturbation ×1.052 (node-level) |
| X9 fp16-accumulate GEMMs | 1.2314 | 0.500 | 0.6157 | 26.33 | C | kernel/config | TASK-LEVEL (SDR pre-check required) | G2 | MODELLED | ideal_costs_C3_4410.json class_totals GEMM; f = registered constant 0.5 (not measured) | GeForce fp16-accumulate rate is not fully realised | C3 trace perturbation ×1.052 (node-level) |
| X10 kernel-count reduction | 0.0213 | 1.000 | 0.0213 | 0.91 | C | kernel | numerical | G3 | MODELLED | graph_verify_H_4410.json (N_current 249), H wrapper (N_floor 175), boundary_bench_trace_analysis trivial/graph/N1000 gap 0.288 µs | boundary gaps overlap with execution (gap is a serial upper bound) | — |
| X11 concurrency (G trace) | 3.7520 | 0.077 | 0.2899 | 12.40 | C | wrapper | none | G3 | MODELLED | timing_views_G_4410.json (observed-order critical path vs summed durations) | true dependencies are longer than the observed-order chain | — |
| X11 concurrency (F trace) | 3.3541 | 0.000 | 0.0000 | 0.00 | C | wrapper | none | G3 | MODELLED | timing_views_F_4410.json (observed-order critical path vs summed durations) | true dependencies are longer than the observed-order chain | — |
| X12 persistent megakernel (upper bound) | 0.2019 | 1.000 | 0.2019 | 8.63 | C | architecture | numerical | G3 | MODELLED | wall(best=H) − critical path (timing_views_H_4410) − one graph-mode gap | the megakernel cannot reach the observed-order critical path (realistic 1.5–3× over graphs per Hazy/Mirage) | — |
| X13 mask-estimator MLP chain fusion | 1.0903 | 0.058 | 0.0637 | 2.72 | B | kernel | numerical | G1 | MODELLED | ideal_costs_C3_4410.json gemm_sites (tail: 172 sites, GEMM time apportioned by FLOPs) + 143 tail fused rows | the tail GEMM time share differs from its FLOP share | C3 trace perturbation ×1.052 (node-level) · GEMM class time apportioned by FLOP share (cuBLAS names carry no site) |
| X14 cross-chunk batching (THROUGHPUT, reported separately) | 2.2709 | 0.001 | 0.0024 | 0.10 | C | wrapper | none | — | MEASURED | boundary_bench.json fits trivial/graph intercept 2.398 µs; throughput_ms_per_pass_best of H | fixed per-pass costs are larger than the graph-launch floor | — · throughput headroom, not latency; Z here is relative to the throughput wall |
| G1 union of attributable time (X2 ∪ X6 ∪ X6a ∪ X13 fused) | 1.1519 | — | — | — | — | — | — | G1 | MODELLED | ideal_costs rows (distinct kernels) | n/a | C3 trace perturbation ×1.052 (node-level) · stacking bound: G1 rows may sum to at most this |

**Thresholds (plan 3.2, mechanical):**
- X1 graph-break removal: Y = 1.7109 ms vs 1.1693 ms needed for 2× → meets 2× on wall
- X2 epilogue fusion into GEMMs: Y = 0.0646 ms vs 1.1693 ms needed for 2× → below 2× on wall (reported only)
- X3 attention backend (time axis → efficient): Y = 0.0321 ms vs 1.1693 ms needed for 2× → below 2× on wall (reported only)
- X3 attention backend (freq axis → cudnn): Y = 0.0141 ms vs 1.1693 ms needed for 2× → below 2× on wall (reported only)
- X4 head-merge copy elimination: Y = 0.0095 ms vs 1.1693 ms needed for 2× → below 2× on wall (reported only)
- X5 band-split + mask-estimator restructuring (H): Y = 0.9367 ms vs 1.1693 ms needed for 2× → below 2× on wall (reported only)
- X6 elementwise/norm/RoPE/gating traffic: Y = 0.0701 ms vs 1.1693 ms needed for 2× → below 2× on wall (reported only)
- X6a autocast weight re-casts: Y = 0.8574 ms vs 1.1693 ms needed for 2× → below 2× on wall (reported only)
- X7 STFT / iSTFT: Y = 0.0078 ms vs 1.1693 ms needed for 2× → below 2× on wall (reported only)
- X8 FP8 GEMMs (offline): Y = 0.6157 ms vs 1.1693 ms needed for 2× → below 2× on wall (reported only)
- X9 fp16-accumulate GEMMs: Y = 0.6157 ms vs 1.1693 ms needed for 2× → below 2× on wall (reported only)
- X10 kernel-count reduction: Y = 0.0213 ms vs 1.1693 ms needed for 2× → below 2× on wall (reported only)
- X11 concurrency (G trace): Y = 0.2899 ms vs 1.1693 ms needed for 2× → below 2× on wall (reported only)
- X11 concurrency (F trace): Y = 0.0000 ms vs 1.1693 ms needed for 2× → below 2× on wall (reported only)
- X12 persistent megakernel (upper bound): Y = 0.2019 ms vs 1.1693 ms needed for 2× → below 2× on wall (reported only)
- X13 mask-estimator MLP chain fusion: Y = 0.0637 ms vs 1.1693 ms needed for 2× → below 2× on wall (reported only)

## Stacking note (mandatory)
- G1 (X2, X6, X6a, X13) share kernels: sum at most the "G1 union of attributable time" row of each O, never the arithmetic sum.
- G2 (X8, X9) are alternatives: never summed; both require an SDR pre-check before any kernel work.
- G3 (X10, X11, X12) model the same launch-bound regime three ways: report the max, not the sum.
- X1 and X5 are measured deltas already inside the C3 wall; they may be summed only with rows attributed from the C3 trace.
- X14 is throughput headroom and is never added to latency rows.

## Conventions
Headline walls are unprofiled (summary_full.json). Trace-derived numbers are MODELLED and carry the node-level perturbation factor of their trace as a caveat when > 5 %. Every number above is computed by fullmodel/ledger.py from the cited files; nothing is transcribed by hand.
