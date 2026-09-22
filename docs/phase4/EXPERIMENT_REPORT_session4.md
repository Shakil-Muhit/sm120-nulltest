# Phase-4 experiment report — session 4 (2026-09-22, pod "melband profiling", id nsfwv548ifydon)

Companion files: `SESSION4_RESULT.md` (generated tables), `docs/phase4/PLAN.md` (prospective plan, registered before the session), `docs/phase4/stats_v2_session4.json` (primary statistics and intervals), `docs/phase4/inventory_diff_session4.json` (trace attribution), `docs/phase4/e9_retrospective_session4.json`, `docs/phase4/revision/` (verification table, analysis revision, change log, missing evidence). Evidence: `~/ms/evidence_pod/nsfwv548ifydon_20260922_2034/evidence` (1,349 files; extraction receipt in `pod.env.extracted`). Every number below is read from a record in that directory or from the files named here.

## 1. What was run and what the audit says

Session 4 executed `scripts/session4.sh` at commit 06ea672 on one RTX 5090 (GPU-01ffe7fc…), 11:39:21–14:33:22 UTC (10,441 s). Stages: P0 A/A controls at five lengths; PA matched C3 vs H at 44,100 / 176,400 / 264,600 / 352,800 (3 restarts each); PB C1P diagnostics (3 standalone unprofiled restarts with the SDR gate, 3 matched C1→C1P, 2 matched C1P→C1, one matched process under the profiler, standalone node traces of C1P and C1); PQ correctness gates (validation, graph gate, SDR) for the seven PC arms at both operating points; PC1–PC4 (24 matched pairs); node traces of C3I, HG, C3G, C3 at both operating points and of C3MG at both (the 8 s one optional, completed); statistics; collection; audit.

Audit: on-pod verdict INCOMPLETE 169/170 (`session_status.json`, kept unchanged) because one audit item was mis-specified — it demanded the shared validator's eligibility flag from the profiled matched record, which declares itself "trace record; never a headline" by design (its data were complete: repeat 0 captured, exports populated, aligned analysis ok). The item was corrected in `fullmodel/completeness.py` (an analysis tool; measuring code is unchanged and its hashes still match the manifest) and the re-audit in the pinned environment on ml-beast (torch 2.8.0+cu128, triton 3.4.0, pinned MSST and checkpoint) is COMPLETE 170/170 with no optional item missing (`session_status_reaudit_gate12.json`, written with `--status-out` so the on-pod file is untouched; `SESSION4_RESULT.md` was generated with `--status session_status_reaudit_gate12.json` and says so in its header). Registered checks inside the session: both planted faults CAUGHT; C3 operator qualification PASS for frames 11, 51, 201, 801, 401, 601, 101; clock sampler at 5 Hz for the whole session; 250 clock comparisons, 15 flagged (11 pair records lost one repeat each — mostly the first 100 ms block after settle, which ran 5–17 % below the next block's SM clock — and every pair kept ≥ 4 clean repeats per restart; four standalone 100 ms gate cells carry an informational cross-restart flag).

## 2. Provenance

| item | Sep-20 pod (sessions 1–2) | Sep-22 pod (session 3) | Sep-22 pod (session 4, this report) |
|---|---|---|---|
| RunPod id / datacenter | 9tblar2zwo97ln / EU-RO-1 | d5n5vfr3z1tbp6 / EU-RO-1 | nsfwv548ifydon / EU-CZ-1 |
| GPU (nvidia-smi UUID) | GPU-6bf4cb32-2adf-770d-783f-8fff8de28903 | GPU-7d47d5c2-999a-837a-e3fe-bc273ae4bad6 | GPU-01ffe7fc-795d-1679-654e-ff7559c69fd7 |
| NVIDIA driver | 580.178.04 | 580.178.04 | **580.159.03** (recorded, not pinned by the harness) |
| torch / triton / CUDA build | 2.8.0+cu128 / 3.4.0 / 12.8 | same | same |
| MSST commit / checkpoint sha256 / config sha256 | ea7eb9c2… / 87201f4d… / f63f38eb… | same | same (verified at setup and in the manifest) |
| repository commit | e523ef93 | ed3ea3d3 | 06ea672b (branch `phase4-followup`; `main` at 9298a6a holds Phase 3 unchanged) |
| session manifest | 9bd6639d… (s1) | f79ad489… | 86d6ba20-ae68-48c5-868d-1b7c6e706ef8 |
| session window (UTC) | 11:01–11:55 (s1), 11:57–12:51 (s2) | 00:02–04:31 | 11:39–14:33 |
| nsys | 2026.5.1 | 2026.5.1 | 2026.5.1.161 |

Three physically different RTX 5090s, three sessions, three code revisions, two driver builds. Every comparison in this report is within one session unless a row is explicitly labelled cross-session, and no cross-session difference is reported as a paired result.

## 3. Runtime verification

- **On-pod smoke (engineering only, never evidence).** Before the session, the new arms ran once outside the evidence tree with `--no-manifest`: A references at 4,410 and 44,100; C3I at 4,410 with the SDR gate; HG and C3MG at 4,410; C3 at 44,100; matched C3 vs H at 44,100 (2 repeats); one node trace of C3I at 4,410 (1,149 kernels per pass inside one graph launch; 420 parameter-cast kernels, 0.85 ms). All graph gates PASS with 0 breaks; all validations PASS at the loose tier with max |diff| ≈ 1–2 × 10⁻⁷. Records kept off-pod (`smoke_engineering_not_evidence/` in the handover); not used anywhere.
- **First launch aborted at preflight (no measurement).** The 11:36:07 UTC launch stopped at the manifest step (`session requires a committed, clean tree`): the manifest's cleanliness check counts untracked files and the smoke directory was inside the repository. The directory was moved out; the tree had 0 porcelain lines at the 11:39:21 launch. Console log kept (`aborted_first_launch/`).
- **Budget knob (owner-rule decision, recorded).** The registered plan set `SESSION_BUDGET_S=16200`; session-3 per-cell durations put the full workload near 4.8 h, so the session was launched with `SESSION_BUDGET_S=25200 EXTRACT_RESERVE_S=900` under the owner's standing instruction to raise the deadline rather than cut registered stages. Stage order, protocol, thresholds and stopping rules were unchanged; the session used 10,441 s.
- **Option names and arm identities (recorded in every run record, `inductor_config` / `compile_mode`):** C3I `reduce-overhead+noinline` = `{dynamo.inline_inbuilt_nn_modules: False}`; C3G / HG `reduce-overhead+freezing+noinline` = `{freezing: True, dynamo.inline_inbuilt_nn_modules: False}`; C3MG `max-autotune+freezing+noinline` = `{max_autotune: True, triton.cudagraphs: True, coordinate_descent_tuning: True, freezing: True, dynamo.inline_inbuilt_nn_modules: False}` (the full `list_mode_options('max-autotune')` preset plus freezing); C1P `reduce-overhead+graph_partition`.

## 4. Statistics (declared before measurement, `docs/phase4/PLAN.md`)

Matched difference = first − second arm in ms (+ = second arm faster). Per repeat: median of 50 block latencies of arm A minus the same for arm B, blocks alternating in one process after a joint settle. Primary statistic: median over restarts of the per-restart median of the per-repeat differences. Interval: two-level cluster bootstrap (restart, then repeat; 10,000 resamples) of that same statistic (`fullmodel/stats_v2.py`); the historical pooled-repeat interval is also stored next to it but never used for a verdict. Eligibility: registered protocol 5×50, validation PASS at the arm's tier, joint settle affirmatively stable for two rounds, one session manifest, clock-flagged repeats excluded (≥ 1 clean repeat per restart required). Smallest meaningful difference: 0.30 ms at 4/6/8 s, 0.10 ms at 1 s and 100 ms. Verdict rule (plan line 21, implemented in `fullmodel/session4_result.py`): "second (or first) arm faster" when the interval excludes zero and |point| ≥ threshold; "practical tie" when the interval lies inside ±threshold; otherwise "unresolved". Three restarts (two for C1P→C1, one for each A/A) carry the between-restart information; the interval resamples restarts and then repeats, so it is wider than the restart-median range where within-restart variation is large (PC1 at 8 s: restart range [−0.214, −0.154], interval [−0.329, −0.049]).

## 5. PA — matched C3 vs H on one GPU (Priority A)

| length | A/A control (C3 − C3#2) | C3 − H: restart medians | primary statistic | cluster CI95 | repeats used | verdict |
|---|---|---|---|---|---|---|
| 44,100 (1 s) | −0.042 [−0.044, +0.022] | +1.530, +1.512, +1.503 | **+1.512** | [+1.499, +1.540] | 12 | H faster |
| 176,400 (4 s) | −0.069 [−0.155, −0.047] | +0.903, +0.846, +0.839 | **+0.846** | [+0.829, +0.922] | 15 | H faster |
| 264,600 (6 s) | −0.072 [−0.221, +0.118] | +0.455, +0.600, +0.357 | **+0.455** | [+0.318, +0.601] | 15 | H faster |
| 352,800 (8 s) | −0.009 [−0.130, +0.069] | −0.297, −0.331, −0.332 | **−0.331** | [−0.389, −0.232] | 15 | C3 faster (interval excludes zero, |point| ≥ 0.30; the interval itself reaches −0.23) |

Reading. On this GPU, H (pad-and-batch) is faster than C3 at 1, 4 and 6 s by margins that clear the declared thresholds with room; at 8 s C3 is faster by 0.33 ms — the registered rule is met (interval excludes zero, |point| ≥ 0.30), while the point estimate exceeds the 0.30 ms threshold by only 0.03 ms (restart medians 0.297–0.332) and the interval reaches −0.23, so the margin over the declared threshold is thin. All five A/A controls are practical ties (the 4 s A/A interval excludes zero but sits inside ±0.30 ms, which is what the tie rule measures). No crossover length is claimed: the grid has no point between 6 s and 8 s, the differences are not assumed linear, and nothing is refitted. The Sep-20 paired 8 s result (H slower by 0.495 ms on GPU-6bf4cb32), the Phase-3 4 s / 6 s pairs (GPU-7d47d5c2: H faster by 1.13 / 0.67 ms) and this session are three separately labelled observations; the direction at 4 s and 6 s replicates across the two sessions that measured them.

## 6. PB — the C1P "runner discrepancy" (Priority B)

| record | built first / second | wall (min-of-medians, ms) | graph breaks | Δ first − second |
|---|---|---|---|---|
| standalone unprofiled C1P r1/r2/r3 (r1 SDR PASS, +0.0002 dB) | — | 36.377 / 36.358 / 36.797 (HEADLINE-grade cell, clock-clean) | 8 | — |
| matched C1 vs C1P m1–m3 | C1 / C1P | 36.78–36.83 / 52.04–52.05 | 8 / 5 | **−15.259** [−15.299, −15.219] |
| matched C1P vs C1 m1–m2 (order swapped) | C1P / C1 | 36.91, 36.79 / 52.05, 52.00 | 8 / 5 | **−15.220** [−15.291, −15.167] |
| standalone traced C1P / C1 (same session) | — | block medians 36.98 / 37.01 | 8 / 8 | — |

Direct observation: the slow configuration follows the build position, not the arm. Standalone unprofiled C1P runs at 36.36–36.80 ms (8 breaks, NOT_FULLY_GRAPHED as always for this family), in line with this session's traced runs (C1 37.01 ms, C1P 36.98 ms under the profiler, caches disabled) — a same-session unprofiled C1 baseline was not part of the PB design, so "C1P equals C1 standalone" is stated only for the traced pair; whichever of C1/C1P is compiled second in a matched process compiles with 5 breaks instead of 8 and runs ≈ 15.2 ms slower. The Phase-3 statement "C1P is 15 ms slower than C1" is withdrawn as an arm property and restated as a matched-runner artefact for arms that compile with graph breaks (the swapped order makes C1 the slow arm); the Phase-3 records are unchanged and re-labelled ("compiled C1 vs partially compiled C1P"). C3-family arms show no such effect (A/A −0.009 ms at 8 s; identical break counts in both positions).

Composition from the profiled matched process (repeat 0 of both arms in one nsys capture, host NVTX windows, 50 passes each; `traces/matched_trace_C1_C1P_352800_p1.json`): first-built C1 wall 37.45 ms, GPU busy 34.70, GPU span 37.12, 1,354 kernels, 96 graph launches; second-built C1P wall 52.89 ms, GPU busy 49.34, GPU span 52.49, 1,569 kernels, 51 graph launches. The 15.4 ms wall difference is 14.6 ms of additional GPU-busy time (215 more kernels per pass launched as fewer, larger graphs); GPU idle inside the window is 2.76 vs 3.52 ms. No span subtraction was used; the "all host-side" reading is refuted. Profiler perturbation on these blocks: 0.02 ms (C1) and 0.24 ms (C1P) relative to the same record's unprofiled repeats.

Mechanism, supported by the torch logs of this session in both build orders and absent from every standalone run: the two model instances share code objects, so the second build re-enters the first build's dynamo frames and adds guard-failure recompiles instead of tracing fresh frames; every C1-family build exhausts the recompile limit on the einops `rearrange` frame (`[3/8] torch._dynamo hit config.recompile_limit (8)`; part of the known 8-break structure), and only the second build in a matched process also exhausts it on the model's top-level `forward` frame (`[7/8] torch._dynamo hit config.recompile_limit (8)`, function `forward` at mel_band_roformer.py:127, last reason "tensor 'x' size mismatch at index 0. expected 60, actual 801"; present in `torchlogs_matched_C1_C1P_352800_m*.log` and `torchlogs_matched_C1P_C1_352800_m*.log`, absent from `torchlogs_C1P_352800_r*.log` and the standalone trace logs), after which dynamo stops compiling that frame: the top-level forward of the second-built model runs eager and only the frames it calls are compiled as they are reached, which is a different graph/break structure. That is consistent with the profiled composition (more, smaller kernels; fewer graph launches; more GPU time) and for the 5-vs-8 break count. The link limit → eager fallback → 15 ms is torch's documented behaviour and is recorded, but it was not tested by intervention (raising `torch._dynamo.config.recompile_limit` for the second build); that intervention is the smallest follow-up if the matched runner is ever used again for graph-breaking arms. Standalone unprofiled C1 was not part of the registered PB design (C1 was measured standalone in sessions 1–3); the result generator lists it as absent.

## 7. PC1–PC4 — configuration interactions (Priority C)

Every arm passed validation, the graph gate (0 breaks) and the SDR gate (|mean ΔSDR| ≤ 0.0008 dB) at both operating points before any pair ran; no pair was stopped.

| pair (first → second) | O | restart medians | primary statistic | cluster CI95 | repeats | threshold | verdict |
|---|---|---|---|---|---|---|---|
| PC1 C3 → C3I (inlining off, no freezing) | 352,800 | −0.154, −0.206, −0.214 | −0.206 | [−0.329, −0.049] | 15 | 0.30 | unresolved (small consistent cost) |
| PC1 C3 → C3I | 4,410 | −0.084, −0.091, −0.078 | −0.084 | [−0.094, −0.062] | 12 | 0.10 | practical tie (consistent 0.08 ms cost) |
| PC2 C3I → C3G (freezing added, inlining off) | 352,800 | +1.296, +1.317, +1.155 | **+1.296** | [+1.145, +1.369] | 15 | 0.30 | C3G faster |
| PC2 C3I → C3G | 4,410 | +0.743, +0.741, +0.748 | **+0.743** | [+0.740, +0.750] | 13 | 0.10 | C3G faster |
| PC3 C3M → C3MG (max-autotune + freezing/no-inlining) | 352,800 | +1.308, +1.316, +1.266 | **+1.308** | [+1.243, +1.339] | 15 | 0.30 | C3MG faster |
| PC3 C3M → C3MG | 4,410 | +1.022, +1.029, +1.027 | **+1.027** | [+1.022, +1.043] | 15 | 0.10 | C3MG faster |
| PC4 H → HG (pad-and-batch + freezing/no-inlining) | 352,800 | +1.352, +1.370, +1.359 | **+1.359** | [+1.280, +1.390] | 15 | 0.30 | HG faster |
| PC4 H → HG | 4,410 | +1.033, +1.034, +1.046 | **+1.034** | [+1.025, +1.048] | 13 | 0.10 | HG faster |

Reading, without adding gains. (i) Turning inlining off by itself does not help: −0.2 ms at 8 s and −0.08 ms at 100 ms, both below threshold but consistent in sign. (ii) Freezing, with inlining off, is the effect: +1.30 ms at 8 s and +0.74 ms at 100 ms on the reduce-overhead base. (iii) The same freezing + no-inlining change is worth +1.31 ms on the max-autotune base and +1.36 ms on the pad-and-batch base at 8 s, and +1.03 / +1.03 ms at 100 ms. At 8 s the gain is of one size on three bases; at 100 ms the reduce-overhead base gains less (0.74) than the other two (1.03). PC3 and PC4 measure their compositions directly; nothing is summed. Standalone context (r1 min-of-medians, same session, unpaired): at 8 s C3 29.79, C3I 29.89, C3G 28.56, C3M 24.80, **C3MG 23.56**, H 30.11, HG 28.73, C1P 36.38 ms; at 100 ms C3 3.50, C3I 3.57, C3G 2.83, C3M 3.36, C3MG 2.25, H 2.44, **HG 1.39** ms.

## 8. PQ — quality and deployment checks of the new arms

| arm | O | SDR (r1) | graph verify | kernels/pass | cast kernels/pass (ms) | Triton template GEMMs / extern GEMMs per pass |
|---|---|---|---|---|---|---|
| C3I | 4,410 / 352,800 | PASS / PASS | FULLY_GRAPHED / FULLY_GRAPHED | 1,149 / 1,142 | 420 (0.857) / 420 (0.923) | 0 / 328 and 0 / 323 |
| C3G | 4,410 / 352,800 | PASS / PASS | FULLY_GRAPHED / FULLY_GRAPHED | 736 / 729 | 0 / 0 | 0 / 328 and 0 / 323 |
| C3MG | 4,410 / 352,800 | PASS / PASS | FULLY_GRAPHED / FULLY_GRAPHED | 557 / 556 | 0 / 0 | 291 / 14 and 285 / 15 |
| HG | 4,410 / 352,800 | PASS / PASS | FULLY_GRAPHED / FULLY_GRAPHED | 192 / 192 | 0 / 0 | 0 / 64 and 0 / 64 |
| C3 (control) | 4,410 / 352,800 | PASS / PASS | FULLY_GRAPHED / FULLY_GRAPHED | 1,150 / 1,143 | 420 (0.857) / 420 (0.925) | 0 / 328 and 0 / 323 |

C3M and H were not traced in session 4 (not in the plan), so their PQ trace views are UNRESOLVED in the generated table by construction. Their historical traces are on other GPUs and sessions: C3M at 8 s and 100 ms from session 3 (GPU-7d47d5c2); H at 8 s and 100 ms from the Sep-20 pod, session 2 (GPU-6bf4cb32); session 3 holds only H's 100 ms structural-variant traces. Every 50-pass trace has a complete node-level export, all passes validated by the aggregate reconciliation (0 kernels outside the 1 % tolerance).

## 9. Attribution from the traces (descriptive; §4 of the task)

Node-level device time per pass by class (caches disabled, profiler on), `docs/phase4/inventory_diff_session4.json`; deltas are first − second in ms.

| pair | O | INDUCTOR_FUSED | GEMM | SDPA | COPY | sum of class deltas | casts first → second | launches/pass | unprofiled matched Δ |
|---|---|---|---|---|---|---|---|---|---|
| C3 − C3I | 352,800 | +0.061 | −0.011 | −0.004 | +0.014 | +0.060 | 0.925 → 0.923 | 1,143 → 1,142 | −0.206 |
| C3 − C3I | 4,410 | +0.009 | +0.003 | 0.000 | +0.003 | +0.015 | 0.857 → 0.857 | 1,150 → 1,149 | −0.084 |
| C3I − C3G | 352,800 | +1.251 | −0.130 | +0.025 | 0.000 | +1.144 | 0.923 → 0 | 1,142 → 729 | +1.296 |
| C3I − C3G | 4,410 | +0.857 | −0.374 | 0.000 | 0.000 | +0.482 | 0.857 → 0 | 1,149 → 736 | +0.743 |
| C3 − C3G | 352,800 | +1.311 | −0.141 | +0.021 | +0.013 | +1.203 | 0.925 → 0 | 1,143 → 729 | (C3→C3I→C3G: −0.206, +1.296) |
| C3 − C3G | 4,410 | +0.866 | −0.372 | 0.000 | +0.002 | +0.497 | 0.857 → 0 | 1,150 → 736 | — |

What freezing removes and changes (C3I → C3G, same in the C3 → C3G view): the 420 parameter-cast kernels (0.92 ms at 8 s, 0.86 ms at 100 ms) disappear; the rotary cos/sin are folded (the fused SDPA-side signature `…add,cos,mul,sin` is replaced by `…add,mul` at equal time, 1.36 → 1.39 ms at 8 s); the norm+cast kernels shrink by 0.35 ms at 8 s; the launch count falls by 413 per pass. What freezing costs: the GEMM class is slower after freezing, by 0.13 ms at 8 s and by **0.37 ms at 100 ms** — at 100 ms the GEMM slow-down offsets 44 % of the cast saving in device time, which is why the device-time class sum (0.48) is well below the cast time removed (0.86). The device-time class sums (1.14 ms at 8 s, 0.48 ms at 100 ms) are smaller than the unprofiled wall gains (1.30, 0.74); the gap (≈ 0.16 and ≈ 0.26 ms) is not assigned to anything here. The launch-count reduction (413 fewer graph nodes per pass) is a candidate, but the Phase-3 boundary-bench slopes (0.568 µs/node for a trivial PDL graph, 0.719 µs/node for a trivial graph) are total service times including kernel execution, so they cannot be added to device-time deltas without double counting; no such addition is made and the gap stays unresolved. Inlining off alone changes almost nothing in device time (+0.06 ms class sum at 8 s; +0.06 ms on the GELU kernel) while costing 0.2 ms of wall time; that wall cost is not visible in the traces and stays unresolved. HG (192 kernels per pass, 0 casts, 64 extern GEMMs) and C3MG (556 kernels, 0 casts, 285 Triton template GEMMs + 15 cuBLAS GEMMs at 8 s; 291 + 14 at 100 ms) are documented as deployed; the comparison with the historical control traces — C3M from session 3 (GPU-7d47d5c2: 178 templates + 145 extern GEMMs at 8 s) and H from the Sep-20 session 2 (GPU-6bf4cb32) — is cross-session and descriptive only: with freezing on the max-autotune base most GEMMs are Triton templates rather than cuBLAS entry points.

## 10. E9 retrospective (no refit; the registered verdict stays FAIL)

Stored affine predictor D(F) = 1.136476 − 0.001915801·F (F = L//441 + 1), from `docs/phase4/e9_affine_check.json`, against the session-4 PA statistics (`docs/phase4/e9_retrospective_session4.json`):

| length | F | affine prediction (ms) | session-4 observed (CI95) | error | within 0.4 ms | sign |
|---|---|---|---|---|---|---|
| 44,100 | 101 | +0.943 | +1.512 [+1.499, +1.540] | −0.569 | no | agrees |
| 176,400 | 401 | +0.368 | +0.846 [+0.829, +0.922] | −0.478 | no | agrees |
| 264,600 | 601 | −0.015 | +0.455 [+0.318, +0.601] | −0.470 | no | **disagrees** |
| 352,800 | 801 | −0.398 | −0.331 [−0.389, −0.232] | −0.067 | yes | agrees |

The predictor under-estimates H's advantage by ≈ 0.5 ms at every length below 8 s on this GPU, exactly as it did on the Phase-3 GPU (failed held-out test), and gets the sign wrong at 6 s. This is a retrospective comparison on a third GPU; 4 s / 6 s were the Phase-3 registered held-out points, 8 s was one of its calibration points (calibration set 4,410 / 22,050 / 88,200 / 352,800) and 1 s is an additional length never used before, so nothing here validates a model out of sample.

## 11. Failure cases, exclusions and qualifications

- First launch aborted at preflight (clean-tree check; no measurement). Recorded.
- Budget knob raised before the launch (owner rule). Recorded.
- 15 clock-drift flags; 11 pair records lost one repeat each (excluded from every published aggregate by the registered rule); no pair or restart lost eligibility. Four 100 ms standalone gate cells (C3I, C3M, C3, HG at 4,410) carry an informational cross-restart clock flag; they are correctness gates in this session, not headline timings.
- The 8 s PA verdict "C3 faster" rests on |point| = 0.331 vs the 0.30 ms threshold with restart medians 0.297–0.332; the margin over the declared threshold is thin and is stated as such.
- PC1 at 8 s is "unresolved" (interval [−0.33, −0.05]); the sign is consistent.
- The profiled matched record and every traced run are ineligible as headlines by design; profiler perturbation at 8 s was 0.02–0.24 ms.
- Standalone unprofiled C1 was not in the PB design; C3M and H were not traced in this session.
- On-pod audit item for the profiled record mis-specified (169/170); corrected tool → 170/170 re-audit in the pinned environment; both status files kept.
- Driver 580.159.03 on this pod vs 580.178.04 before: confounded with the physical GPU; A/A controls behave alike across sessions; no claim depends on a cross-session difference.

## 12. Conclusion

**Observed effects (direct, matched, same GPU):** H faster than C3 at 1 s (+1.51 ms), 4 s (+0.85) and 6 s (+0.45); C3 faster at 8 s (−0.33, registered rule met with a thin margin over the 0.30 ms threshold). The 15 ms "C1P regression" is a build-order artefact of the matched runner (−15.26 with C1 first, −15.22 with C1P first; standalone unprofiled C1P 36.4–36.8 ms, traced C1 and C1P within 0.03 ms of each other). Freezing with inlining off gains +1.30 (8 s) / +0.74 (100 ms) on reduce-overhead, +1.31 / +1.03 on max-autotune, +1.36 / +1.03 on pad-and-batch; inlining off alone costs 0.2 / 0.08 ms. All new arms are numerically valid, fully graphed and pass the SDR gate.

**Explanations supported by discriminating evidence:** the second-built arm's slow-down is GPU-busy time after dynamo's recompile limit is exhausted on the model's top-level forward frame in the second build (logs in both orders, never standalone; profiled composition). Freezing's gain comes with the removal of the 420 parameter casts, rotary folding, norm+cast shrinkage and 413 fewer launches, and against a GEMM-class slow-down that is largest at 100 ms (−0.37 ms), a possible contributor to the smaller 100 ms gain on this base (the other two bases have no session-4 control trace, so the cross-base difference is not explained).

**Plausible but unverified:** that the launch-count reduction accounts for the gap between device-time deltas and wall gains (not testable with the available per-node figures, which include kernel execution); that raising the recompile limit would remove the matched-runner artefact; that freezing's shift of GEMMs into Triton templates on the max-autotune base is what makes C3MG the fastest configuration (23.56 ms at 8 s, cross-session template counts).

**Unresolved:** the location of any C3/H crossover between 6 s and 8 s (not sampled; no model); the 0.2 ms wall cost of inlining off, invisible in device time; why freezing slows the GEMM class (kernel selection with constant weights is the obvious suspect; not tested); the driver-version effect (confounded).

## 13. Resource use

| pod | created → deleted (UTC) | wall hours | sessions | price |
|---|---|---|---|---|
| 9tblar2zwo97ln (Sep 20) | ≈ 10:15 → ≈ 13:00 (from evidence timestamps; RunPod does not list deleted pods) | ≈ 2.8 | 1 (54 min), 2 (54 min) | secure-cloud RTX 5090 listed $0.99/h |
| d5n5vfr3z1tbp6 (Sep 22) | ≈ 00:20 → 05:05 | ≈ 4.75 | 3 (4.48 h) | same |
| nsfwv548ifydon (Sep 22) | 11:18:18 → 14:36:06 (`runpodctl` records) | 3.30 | smoke (13 min), aborted preflight (3 min), 4 (2.90 h) | $0.99/h (`costPerHr` from the API); ≈ $3.3 if billed by the minute; actual invoice not retrieved |

Five creation attempts in EU datacenters failed for lack of RTX 5090 stock before EU-CZ-1 accepted the pod on the second pass (no cost). ml-beast (five RTX 5090s, shared) was used only for functional smokes before the session and for the pinned-environment re-audit after it; its GPU 4 was occupied by another user's job during the Phase-4 development window, so no engineering timing from it enters this report.

## 14. Reproduction (off-pod; evidence directory E = `~/ms/evidence_pod/nsfwv548ifydon_20260922_2034/evidence`)

```
python fullmodel/completeness.py --evidence-dir E --session session4 --status-out session_status_reaudit_gate12.json   # pinned env (torch 2.8.0+cu128, triton 3.4.0, MSST + checkpoint pins); never overwrites the on-pod session_status.json
python fullmodel/session4_result.py --evidence-dir E --status session_status_reaudit_gate12.json --out SESSION4_RESULT.md   # --status session_status.json reproduces the on-pod 169/170 header
python fullmodel/stats_v2.py --evidence-dir E --out docs/phase4/stats_v2_session4.json --pairs C3:H:44100 C3:H:176400 C3:H:264600 C3:H:352800 C1:C1P:352800 C1P:C1:352800 C3:C3I:4410 C3:C3I:352800 C3I:C3G:4410 C3I:C3G:352800 C3M:C3MG:4410 C3M:C3MG:352800 H:HG:4410 H:HG:352800 C3:C3:44100 C3:C3:176400 C3:C3:264600 C3:C3:352800 C3:C3:4410
python docs/phase4/probes/inventory_diff_pairs.py E docs/phase4/inventory_diff_session4.json C3:C3I:352800 C3:C3I:4410 C3I:C3G:352800 C3I:C3G:4410 C3:C3G:352800 C3:C3G:4410 H:HG:352800 H:HG:4410   # H rows UNRESOLVED (H not traced in session 4)
python docs/phase4/probes/e9_retrospective_session4.py docs/phase4/stats_v2_session4.json docs/phase4/e9_affine_check.json docs/phase4/e9_retrospective_session4.json
grep -c "hit config.recompile_limit" E/torchlogs_matched_C1_C1P_352800_m1.log E/torchlogs_C1P_352800_r1.log     # 2 vs 1
```
