# Ceiling-ledger Phase 1 — SESSION_RESULT (filled mechanically from evidence/)

repo: e523ef9318e6df04ceff4dc237eea0ffc810450f   registration: prereg_amendments.md @ e523ef9 (sha256 c6cc561d7bab; pushed 2026-09-20 ~11:00 UTC (local push log; client-side timestamp, see docs/provenance/push_evidence.md))   msst @ ea7eb9c20ea0   ckpt sha256 87201f4d31af
torch 2.8.0+cu128 · triton 3.4.0 · driver 580.178.04 · GPU NVIDIA GeForce RTX 5090 GPU-6bf4cb32-2adf-770d-783f-8fff8de28903 · pod melband_profiling · manifest 9bd6639d-582d-4e98-9bff-d195d4bcd924 · created 2026-09-20T11:01:24.957652+00:00
clocks session1: SM median 2805 MHz (range 240–2947) · memory 13801 MHz · 3253 samples — evidence/clocks_session1.csv
clocks session2: SM median 2520 MHz (range 232–2947) · memory 13801 MHz · 3260 samples — evidence/clocks_session2.csv

## Decision inputs (mechanical)
| input | value | evidence |
|---|---|---|
| planted fault tight (0.001) | CAUGHT max_abs_diff [0.0010000001639127731, 0.0010000001639127731] | evidence/full_A_c4410_planted_tight.json |
| planted fault loose (0.05) | CAUGHT max_abs_diff [0.05000000074505806, 0.05000000074505806] | evidence/full_A_c4410_planted_loose.json |
| C3 operator qualification | T=11: PASS, T=51: PASS, T=201: PASS, T=801: PASS | evidence/c3_operator_qualification.json |

### Arms × operating points (headline = median over r1–r3 of per-process min-of-medians; unprofiled)
| arm | O | headline ms | restart spread ms | best throughput ms/pass | tier | SDR (mean ΔSDR_ref dB) | variant | breaks | status / reasons |
|---|---|---|---|---|---|---|---|---|---|
| A | 352800 | 62.2750 | 0.1033 | 62.2765 | tight | PASS (+0.000) | capture_scope=None|capture_adapter_active=None|compile_mode=None|stream_count=None | None | HEADLINE  |
| C3 | 352800 | 28.8714 | 0.0910 | 28.6478 | loose | PASS (-0.000) | capture_scope=None|capture_adapter_active=None|compile_mode=reduce-overhead|stream_count=None | 0 | HEADLINE  |
| C1 | 352800 | 35.6203 | 0.0326 | 35.5999 | loose | PASS (-0.000) | capture_scope=None|capture_adapter_active=None|compile_mode=reduce-overhead|stream_count=None | 8 | HEADLINE  |
| H | 352800 | 29.3624 | 0.1181 | 29.3217 | loose | PASS (-0.000) | capture_scope=None|capture_adapter_active=None|compile_mode=reduce-overhead|stream_count=None | 0 | HEADLINE  |
| G | 352800 | 60.8794 | 0.0189 | 60.8649 | tight | PASS (+0.000) | capture_scope=whole_forward_to_istft|capture_adapter_active=dc_hoist+istft_boundary|compile_mode=None|stream_count=4 | None | HEADLINE  |
| B | 352800 | 61.5100 | 0.6025 | 60.9116 | tight | PASS (+0.000) | capture_scope=whole_forward_to_istft|capture_adapter_active=dc_hoist+istft_boundary|compile_mode=None|stream_count=None | None | HEADLINE  |
| E | 352800 | 61.1685 | 0.0593 | 61.0971 | tight | PASS (+0.000) | capture_scope=whole_forward_to_istft|capture_adapter_active=dc_hoist+istft_boundary|compile_mode=None|stream_count=None | None | HEADLINE  |
| D | 352800 | 61.7564 | 0.0429 | 61.6919 | tight | PASS (+0.000) | capture_scope=None|capture_adapter_active=None|compile_mode=None|stream_count=None | None | HEADLINE  |
| F | 352800 | 63.2221 | 0.1009 | 63.1842 | tight | PASS (+0.000) | capture_scope=None|capture_adapter_active=None|compile_mode=None|stream_count=4 | None | HEADLINE  |
| C1D | 352800 | 35.6648 | 0.0512 | 35.6264 | loose | PASS (-0.000) | capture_scope=None|capture_adapter_active=None|compile_mode=reduce-overhead|stream_count=None | 8 | HEADLINE  |
| C2 | 352800 | 30.1211 | 0.0575 | 30.0653 | loose | PASS (+0.000) | capture_scope=None|capture_adapter_active=None|compile_mode=max-autotune|stream_count=None | 8 | HEADLINE  |
| ANVTX | 352800 | UNRESOLVED | UNRESOLVED | UNRESOLVED |  | NONE (UNRESOLVED) |  |  | INCOMPLETE r1 missing; r2 missing; r3 missing; restarts do not share one session manifest: []; effective variants differ: []; restarts were not validated against one linea |
| A | 4410 | 9.5200 | 0.3151 | 9.3038 | tight | PASS (+0.000) | capture_scope=None|capture_adapter_active=None|compile_mode=None|stream_count=None | None | HEADLINE  |
| C3 | 4410 | 3.2752 | 0.0064 | 3.1168 | loose | PASS (-0.000) | capture_scope=None|capture_adapter_active=None|compile_mode=reduce-overhead|stream_count=None | 0 | HEADLINE  |
| C1 | 4410 | 4.9862 | 0.1419 | 4.8345 | loose | PASS (-0.000) | capture_scope=None|capture_adapter_active=None|compile_mode=reduce-overhead|stream_count=None | 8 | HEADLINE  |
| H | 4410 | 2.3386 | 0.0043 | 2.2709 | loose | PASS (-0.001) | capture_scope=None|capture_adapter_active=None|compile_mode=reduce-overhead|stream_count=None | 0 | HEADLINE  |
| G | 4410 | 4.4437 | 0.0018 | 4.4222 | tight | PASS (+0.001) | capture_scope=whole_forward_to_istft|capture_adapter_active=dc_hoist+istft_boundary|compile_mode=None|stream_count=4 | None | HEADLINE  |
| B | 4410 | 5.0726 | 0.0039 | 5.0185 | tight | PASS (+0.000) | capture_scope=whole_forward_to_istft|capture_adapter_active=dc_hoist+istft_boundary|compile_mode=None|stream_count=None | None | HEADLINE  |
| E | 4410 | 4.7368 | 0.0103 | 4.7121 | tight | PASS (+0.001) | capture_scope=whole_forward_to_istft|capture_adapter_active=dc_hoist+istft_boundary|compile_mode=None|stream_count=None | None | HEADLINE  |
| D | 4410 | 9.1456 | 0.4673 | 8.7685 | tight | PASS (+0.001) | capture_scope=None|capture_adapter_active=None|compile_mode=None|stream_count=None | None | HEADLINE  |
| F | 4410 | 9.9801 | 0.4036 | 9.7052 | tight | PASS (+0.000) | capture_scope=None|capture_adapter_active=None|compile_mode=None|stream_count=4 | None | HEADLINE  |
| C2 | 4410 | 4.9599 | 0.0398 | 4.9334 | loose | PASS (+0.000) | capture_scope=None|capture_adapter_active=None|compile_mode=max-autotune|stream_count=None | 8 | HEADLINE  |
| ANVTX | 4410 | UNRESOLVED | UNRESOLVED | UNRESOLVED |  | NONE (UNRESOLVED) |  |  | INCOMPLETE r1 missing; r2 missing; r3 missing; restarts do not share one session manifest: []; effective variants differ: []; restarts were not validated against one linea |
| A | 22050 | 10.0014 | 0.0612 | 9.9230 | tight | PASS (+0.000) | capture_scope=None|capture_adapter_active=None|compile_mode=None|stream_count=None | None | HEADLINE  |
| C3 | 22050 | 4.2173 | 0.0206 | 3.6413 | loose | PASS (-0.001) | capture_scope=None|capture_adapter_active=None|compile_mode=reduce-overhead|stream_count=None | 0 | HEADLINE  |
| H | 22050 | 3.3184 | 0.0034 | 3.2473 | loose | PASS (+0.000) | capture_scope=None|capture_adapter_active=None|compile_mode=reduce-overhead|stream_count=None | 0 | HEADLINE  |
| G | 22050 | 6.8340 | 0.0026 | 6.8231 | tight | PASS (-0.001) | capture_scope=whole_forward_to_istft|capture_adapter_active=dc_hoist+istft_boundary|compile_mode=None|stream_count=4 | None | HEADLINE  |
| A | 88200 | 15.1916 | 0.0381 | 15.1795 | tight | PASS (+0.000) | capture_scope=None|capture_adapter_active=None|compile_mode=None|stream_count=None | None | HEADLINE  |
| C3 | 88200 | 8.6252 | 0.0104 | 8.3035 | loose | PASS (-0.000) | capture_scope=None|capture_adapter_active=None|compile_mode=reduce-overhead|stream_count=None | 0 | HEADLINE  |
| H | 88200 | 7.4558 | 0.0326 | 7.3530 | loose | PASS (-0.001) | capture_scope=None|capture_adapter_active=None|compile_mode=reduce-overhead|stream_count=None | 0 | HEADLINE  |
| G | 88200 | 13.0910 | 0.0097 | 13.0827 | tight | PASS (-0.000) | capture_scope=whole_forward_to_istft|capture_adapter_active=dc_hoist+istft_boundary|compile_mode=None|stream_count=4 | None | HEADLINE  |

### Matched pairs (plan 1.2; median over alternating blocks, a − b)
| pair | m1 | m2 | m3 | median ms | % of a | status / reasons |
|---|---|---|---|---|---|---|
| C1_vs_C1A_c352800 | -0.1502 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | INCOMPLETE m2 missing; m3 missing; C1A: SDR qualification at 352800 is NONE |
| C1_vs_C1D_c352800 | -0.0510 | -0.0520 | -0.0448 | -0.0510 | -0.14 | HEADLINE  |
| C1_vs_C3_c352800 | 6.7412 | 6.7491 | 6.7316 | 6.7412 | +18.89 | HEADLINE  |
| C3_vs_H_c352800 | -0.5122 | -0.4857 | -0.4954 | -0.4954 | -1.71 | HEADLINE  |

### Graph deployment (node-level traces)
| arm | O | verdict | graph launches/pass | captured | staging | uncaptured | unresolved | qualified |
|---|---|---|---|---|---|---|---|---|
| ANVTX | 352800 | NO_REPLAY_EVIDENCE | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | True |
| ANVTX | 4410 | NO_REPLAY_EVIDENCE | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | True |
| A | 352800 | NO_REPLAY_EVIDENCE | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | True |
| A | 4410 | NO_REPLAY_EVIDENCE | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | True |
| B | 352800 | NOT_FULLY_GRAPHED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | True |
| B | 4410 | NOT_FULLY_GRAPHED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | True |
| C1 | 352800 | NOT_FULLY_GRAPHED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | True |
| C1 | 4410 | NOT_FULLY_GRAPHED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | True |
| C2 | 4410 | NOT_FULLY_GRAPHED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | True |
| C3 | 352800 | FULLY_GRAPHED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | True |
| C3 | 4410 | FULLY_GRAPHED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | True |
| E | 352800 | NOT_FULLY_GRAPHED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | True |
| E | 4410 | NOT_FULLY_GRAPHED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | True |
| F | 352800 | NO_REPLAY_EVIDENCE | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | True |
| F | 4410 | NO_REPLAY_EVIDENCE | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | True |
| G | 352800 | NOT_FULLY_GRAPHED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | True |
| G | 4410 | NOT_FULLY_GRAPHED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | True |
| H | 352800 | FULLY_GRAPHED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | True |
| H | 4410 | FULLY_GRAPHED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | True |

### Profiling perturbation (one block each; the only profiled/unprofiled ratios)
| arm@O | unprofiled ms | graph/unprofiled | node/unprofiled | caveat | status |
|---|---|---|---|---|---|
| A_352800 | 62.0771 | 1.024 | 1.023 | False | ok |
| A_4410 | 9.2476 | 1.573 | 1.565 | True | ok |
| C3_352800 | 28.8182 | 1.014 | 1.013 | False | ok |
| C3_4410 | 3.2658 | 1.049 | 1.052 | True | ok |
| G_352800 | 60.7352 | 1.003 | 1.004 | False | ok |
| G_4410 | 4.4373 | 1.019 | 1.021 | False | ok |

### Attention backend (per axis, per O; model layout = deployed layout)
| case | best backend | best µs | deployed backend | deployed µs | ratio best/deployed | status |
|---|---|---|---|---|---|---|
| c352800_freq_contiguous | cudnn | 108.37 | flash | 129.89 | 0.834 | ok |
| c352800_freq_model | cudnn | 111.43 | flash | 130.84 | 0.852 | ok |
| c352800_time_contiguous | cudnn | 408.16 | flash | 440.45 | 0.927 | ok |
| c352800_time_model | cudnn | 407.76 | flash | 437.80 | 0.931 | ok |
| c4410_freq_contiguous | cudnn | 3.22 | flash | 5.69 | 0.566 | ok |
| c4410_freq_model | cudnn | 3.26 | flash | 5.67 | 0.575 | ok |
| c4410_time_contiguous | efficient | 6.39 | flash | 12.00 | 0.532 | ok |
| c4410_time_model | efficient | 6.39 | flash | 11.93 | 0.536 | ok |

### Boundary cost (T(N) fits; serial gap from the node trace)
| mode | slope µs/node | intercept µs | trace gap µs (N1000) | PDL PTX |
|---|---|---|---|---|
| elem37/graph | 12.905 | 2.724 | UNRESOLVED (n/a) | True |
| elem37/stream | 14.335 | 2.782 | UNRESOLVED (n/a) | True |
| gemm/graph | 7.850 | 2.187 | UNRESOLVED (n/a) | True |
| gemm/stream | 10.241 | 4.884 | UNRESOLVED (n/a) | True |
| trivial/graph | 0.732 | 2.398 | 0.288 (ok) | True |
| trivial/pdl_graph | 0.579 | -3.791 | 0.160 (ok) | True |
| trivial/pdl_stream | 4.732 | 1.567 | 5.472 (ok) | True |
| trivial/stream | 4.883 | 6.466 | 5.407 (ok) | True |
L2-resident bandwidth: 6.000 TB/s — evidence/boundary_bench.json

### Three timing views (kernels only; memops separate)
| arm | O | sum µs | union µs | observed-order critical path µs | span µs | streams |
|---|---|---|---|---|---|---|
| ANVTX | 352800 | 58865.4 | 58865.4 | 58865.4 | 63602.3 |  |
| ANVTX | 4410 | 3352.9 | 3352.9 | 3352.9 | 14533.9 |  |
| A | 352800 | 58863.5 | 58863.5 | 58863.5 | 63633.6 |  |
| A | 4410 | 3354.1 | 3354.1 | 3354.1 | 14292.4 |  |
| B | 352800 | 59940.0 | 59940.0 | 59940.0 | 61479.1 |  |
| B | 4410 | 3844.9 | 3844.9 | 3844.9 | 5083.9 |  |
| C1 | 352800 | 33785.6 | 33785.6 | 33785.6 | 35292.6 |  |
| C1 | 4410 | 3044.7 | 3044.7 | 3044.7 | 5729.4 |  |
| C2 | 4410 | 2710.8 | 2710.8 | 2710.8 | 5674.6 |  |
| C3 | 352800 | 28322.9 | 28322.9 | 28322.9 | 29089.9 |  |
| C3 | 4410 | 2514.4 | 2514.4 | 2514.4 | 3261.1 |  |
| E | 352800 | 59700.0 | 59700.0 | 59700.0 | 61119.1 |  |
| E | 4410 | 3632.4 | 3632.4 | 3632.4 | 4750.8 |  |
| F | 352800 | 58862.1 | 58860.1 | 58860.1 | 64775.8 |  |
| F | 4410 | 3354.1 | 3354.1 | 3354.1 | 15097.0 |  |
| G | 352800 | 60014.3 | 59612.2 | 59578.5 | 60847.5 |  |
| G | 4410 | 3752.0 | 3490.8 | 3462.1 | 4452.0 |  |
| H | 352800 | 29481.6 | 29481.6 | 29481.6 | 29696.6 |  |
| H | 4410 | 2136.4 | 2136.4 | 2136.4 | 2301.5 |  |

### Per-kernel ideal costs (class totals µs/pass; GEMM/SDPA ideal vs measured)
| arm | O | measurement | variant | GEMM | SDPA | INDUCTOR_FUSED | COPY | FFT | OTHER | SDPA ideal µs | GEMM ideal µs (sum of sites) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ANVTX | 352800 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| ANVTX | 4410 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| A | 352800 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| A | 4410 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| B | 352800 | ok | n/a | 14214.4 | 3505.4 | 0.0 | 11910.3 | 17.2 | 30282.2 | 2427.1 | UNRESOLVED |
| B | 4410 | ok | n/a | 1210.4 | 102.6 | 0.0 | 1367.4 | 4.9 | 1161.8 | 18.1 | UNRESOLVED |
| C1 | 352800 | ok | UNRESOLVED | 15763.2 | 3435.5 | 9703.0 | 2499.1 | 17.8 | 2357.4 | 2427.1 | UNRESOLVED |
| C1 | 4410 | ok | UNRESOLVED | 1322.1 | 103.8 | 1223.5 | 271.4 | 5.0 | 118.6 | 18.1 | UNRESOLVED |
| C2 | 4410 | ok | UNRESOLVED | 1095.6 | 102.0 | 1121.0 | 269.3 | 4.9 | 118.6 | 18.1 | UNRESOLVED |
| C3 | 352800 | ok | ok | 16083.5 | 3561.1 | 8516.1 | 41.5 | 16.4 | 53.8 | 2427.1 | 13770.2 |
| C3 | 4410 | ok | ok | 1231.4 | 102.2 | 1151.9 | 7.1 | 4.9 | 16.5 | 18.1 | 625.5 |
| E | 352800 | ok | n/a | 14228.5 | 3509.5 | 0.0 | 11912.0 | 17.1 | 30026.0 | 2427.1 | UNRESOLVED |
| E | 4410 | ok | n/a | 1209.2 | 102.7 | 0.0 | 1366.7 | 4.7 | 949.3 | 18.1 | UNRESOLVED |
| F | 352800 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| F | 4410 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED |
| G | 352800 | ok | n/a | 14298.5 | 3503.9 | 0.0 | 12063.9 | 17.3 | 30126.5 | 2427.1 | UNRESOLVED |
| G | 4410 | ok | n/a | 1248.4 | 102.5 | 0.0 | 1425.6 | 4.7 | 970.9 | 18.1 | UNRESOLVED |
| H | 352800 | ok | ok | 16572.3 | 3587.2 | 8782.9 | 349.8 | 16.3 | 55.3 | 2427.1 | 14352.0 |
| H | 4410 | ok | ok | 810.0 | 102.2 | 1190.4 | 12.9 | 4.5 | 16.3 | 18.1 | 682.2 |

## Predictions vs measurements
| registered claim (prereg_amendments.md) | outcome | evidence |
|---|---|---|
| §1.2 planted faults FAIL both seeds, seed-invariant diff | tight [0.0010000001639127731, 0.0010000001639127731] loose [0.05000000074505806, 0.05000000074505806] | evidence/full_A_c4410_planted_*.json |
| §1 C1A compile failure expected | C1A build OK, record status VALID, pair_eligible True | evidence/matched_C1_C1A_c352800_m1.json |
| §1 C3 zero graph breaks | {352800: ['0'], 4410: ['0'], 22050: ['0'], 88200: ['0']} | evidence/full_C3_*_r*.json graph_gate |
| §6 exploratory: G/H faster than best verified-replay compiled arm at 4410 | best compiled (2.3385630920529366, 'H_4410'); G/H [(2.3385630920529366, 'H_4410'), (4.443662939593196, 'G_4410')] | evidence/summary_full.json |
| §5 5a serialization delta > 5 % at 4410 (escalation) | max per arm: [('A', 0.4429126432566846), ('B', 1.096232472972215), ('C1', 1.0731774572862622), ('C2', 0.6699187028720436), ('C3', 5.149094825417591), ('D', 0.15299755794830697), ('E', 0.44022143487650645), ('F', 0.021981354690736043), ('G', 0.4981527920574448), ('H', 3.0924296875348225)] | evidence/full_*_c4410_r1.json timing.serialization_delta_pct |

## What failed / UNTESTED / INCOMPLETE
- session1: **COMPLETE** (100/100 mandatory items) — evidence/session_status.json
- session2: **COMPLETE** (172/172 mandatory items) — evidence/session_status.json
- cell ANVTX_352800: r1 missing; r2 missing; r3 missing; restarts do not share one session manifest: []; effective variants differ: []; restarts were not validated against one lineage-hashed reference set; SDR qualification (r1) at this operating point: NONE (no SDR record in r1)
- cell ANVTX_4410: r1 missing; r2 missing; r3 missing; restarts do not share one session manifest: []; effective variants differ: []; restarts were not validated against one lineage-hashed reference set; SDR qualification (r1) at this operating point: NONE (no SDR record in r1)

## Conventions
Headline walls unprofiled; PERTURBATION ratios are the only profiled/unprofiled ratios; anything trace-derived divided by an unprofiled wall is labelled MODELLED (feeds LEDGER.md, never a measured speedup). This file is generated by fullmodel/session_result.py; every number cites its evidence file.
