# Ceiling-ledger Phase 1 — SESSION_RESULT (template; filled mechanically from evidence/)

repo: <sm120-nulltest commit>   registration: prereg_amendments.md @ <commit> (pushed <UTC>)   msst @ ea7eb9c2…   ckpt sha256 87201f4d…
torch <ver> · triton <ver> · driver <ver> · GPU <name> <uuid> · pod melband_profiling (<DC>) · sessions: 1 = <date/time window>, 2 = <…>
clocks during timed blocks: SM median <MHz> (range) · memory <MHz> — evidence/clocks_session*.csv

## Decision inputs (mechanical)
| input | value | evidence |
|---|---|---|
| planted-fault checks (tight 1e-3 / loose 5e-2) | CAUGHT / NOT CAUGHT | evidence/full_A_c4410_planted_*.json |
| C3 operator qualification (T ∈ {11,51,201,801}) | PASS/FAIL per T | evidence/c3_operator_qualification.json |
| arms × operating points: headline (median over 3 eligible restarts of min-of-medians), spread, throughput, tier, SDR gate, scope/variant | table | evidence/summary_full.json |
| matched pairs (plan 1.2): C1−C3, C1−C1D, C3−H at 352800 (median over alternating blocks) | ms (%) | evidence/matched_*.json |
| chunk sweep A/C3/G/H × {4410, 22050, 88200, 352800} | table | evidence/full_*_c22050_*.json, …c88200… |
| graph deployment (C1, C3, H, B, E, G × 2 OPs): graph launches/pass, uncaptured model kernels, staging, fully_graphed | table | evidence/traces/graph_verify_*.json |
| profiling perturbation (A, C3, G × 2 OPs): node/unprofiled, graph/unprofiled | ratios | evidence/full_*_pert_*.json + traces |
| attention backend (per axis, per OP): best backend, ratio best/model | table | evidence/attn_backend_bench.json |
| boundary cost: T(N) slope/intercept per mode; serial gap from trace; PDL availability | table | evidence/boundary_bench*.json |
| three timing views for F/G (sum / union / critical path) | table | evidence/traces/timing_views_*.json |
| per-kernel ideal costs (class totals; GEMM/SDPA ideal vs measured) | table | evidence/traces/ideal_costs_*.json |

## Predictions vs measurements
| registered claim (prereg_amendments.md) | outcome | evidence |
|---|---|---|
| §1.2 planted faults FAIL both seeds, seed-invariant diff | | |
| §1 C1A compile failure expected | | |
| §1 C3 zero graph breaks | | |
| §6 exploratory: G/H faster than best verified-replay compiled arm at 4410 | | |
| §5 5a serialization delta > 5 % at 4410 (escalation) | | |

## Per-layer / per-class tables (from traces)
<A-NVTX sublayer table at both OPs; C3 generated-code segments; class totals per arm>

## What failed / UNTESTED / INCOMPLETE
<every cell with its machine-readable reason; deadline hits; timeouts; the session-completeness verdict>

## Conventions
Headline walls unprofiled; PERTURBATION ratios are the only profiled/unprofiled ratios; anything trace-derived divided by an unprofiled wall is labelled MODELLED (feeds LEDGER.md, never a measured speedup).
