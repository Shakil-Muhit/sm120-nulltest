# Pre-registration amendments — ceiling-ledger measurement session
### v0.1 · 2026-09-20 · registered by the executing agent for the owner's audit · governs Phase 1 of `plan_ceiling_ledger.md`

This file amends `prereg_null_test.md.pdf` for the *ceiling-ledger* session
(plan §Phase 0, item 0.1). Rule carried over verbatim: **everything below is
committed and pushed before the pod session that tests it**; the GitHub push
timestamp is the time evidence. Where this file and the plan disagree, the plan
wins and the disagreement is a bug to report; where this file and the original
prereg disagree, this file is the amendment and says so explicitly. The owner
may strike any item before the session by editing this file (a struck item is
kept, marked `STRUCK`, never deleted).

Language for every arm that is new here, adopted from the external review:
**"prospectively specified replications informed by exploratory observations."**
The exploratory observations are the 2026-08-17 smoke run of `sm120-nulltest`,
the 2026-08-17 production A6000 compile A/B, and the completed
`axial-layout-scoping` study (C1 = 35.403 ms, C3 = 30.306 ms at 8 s on
ml-beast GPU 4, report `axial-layout-scoping@a6f3e6f RESULT.md`).

---

## 0. Reading of the plan's arm matrix (owner: confirm or strike before the session)

The plan's Phase-1 matrix (`A, B, C1, C2, C3, D, E, F, G, H × {352800, 4410}`)
is read as a matrix of **full-model arms**: the pinned public MelBandRoformer
(msst `ea7eb9c2…`, checkpoint sha256 `87201f4d…`, KJ vocals config) executed
end-to-end (audio in → separated vocals out), with the `sm120-nulltest`
arm *mechanics* (manual capture, fused RMSNorm, multi-stream band loop,
compile modes) applied to the full model as **external module swaps**.
Reasons this is the only coherent reading: (i) the task-quality gate (SDR vs
reference stems) needs audio outputs; (ii) `T_wall(best arm, O)` in the
ledger formula and every X row (SDPA, head-merge copies, mask estimator,
STFT/iSTFT) are full-forward quantities; (iii) the front-end is 1.2 % of the
8 s forward (P8, smoke 2026-08-17), so front-end-only walls cannot rank X's.
The original front-end-only arms (`arms/run_arm.py`) remain runnable and
unchanged; they are **not** part of this session's headline matrix.

If the owner intended the front-end-only arms, strike this section and stop:
the front-end-only matrix cannot feed the ledger's end-to-end denominators or
the quality gate, so no automatic fallback is registered (design review
gate 01, `docs/codex/01_design.md`).

## 1. Arms (full model; the clone is never edited; every intervention is an external wrapper)

Common to all arms: model built through the public construction path
(`utils.settings.get_model_from_config` → `torch.load(weights_only=False,
map_location='cpu')` → `load_start_checkpoint(type_='inference')`, mirroring
upstream `inference.py`), `.eval()`, inference under the repo's own precision
`torch.autocast('cuda', enabled=config.training.use_amp)` (= True; fp32
would abort — flash-only SDPA), input `torch.randn(1, 2, chunk)` from a CPU
generator (timing seed 0; validation seeds 1001/2002 with the different-data
assertion — `arms/common.py::validate`, reused).

| arm | definition | tolerance class (rule: compiler-regenerated kernels present → loose) |
|---|---|---|
| **A** | eager, stock forward | reference (tight vs itself is vacuous; A validates the others) |
| **A-NVTX** | A + external per-sublayer NVTX hooks (`scripts/p8_trace.py::install_nvtx_hooks`, unchanged: frontend / L{i}_time_attn / L{i}_rearrange / L{i}_freq_attn / tail). **Attribution only** — never a headline wall; hooks are never installed on compiled arms (they change Dynamo's graphs; compiled arms are attributed from generated code) | tight (1e-4 / 1e-5) |
| **B** | A + manual whole-forward `torch.cuda.CUDAGraph` capture, static I/O buffers, warmup / input recording / capture on a side stream, all under the repo's autocast with **`cache_enabled=False`** (torch CUDA-Graphs/AMP guidance; the eager arms keep the default cache). Capture ladder — every rung's traceback is kept, the stream/context is recovered before the next rung, and the JSON records `capture_scope` AND the active adapter separately: (1) stock forward as-is (expected to fail: `torch.tensor(0, device='cuda')` in the DC filter constructs host data and takes a blocking, synchronizing host→device transfer — verified on ml-beast 2026-09-20); (2) the DC-index hoist: an AST-verified external forward copy whose only change is that expression → a pre-allocated device tensor (no arithmetic change; the pattern C3 already established); (2b) rung 2 plus the **iSTFT boundary**: capture still fails with the index hoisted because `torch.istft` performs a host-side window-envelope check (a device→host read) — the forward's single `torch.istft(...)` call is replaced by a same-signature stand-in that, during capture only, stashes its input (the graph's static pre-iSTFT spectrogram) and returns a placeholder of the correct shape; every replay is followed by the SAME public `torch.istft` call run eagerly on that static buffer and the forward's own two post-iSTFT reshapes. Scope label `whole_forward_to_istft`: everything up to the iSTFT is graphed; the iSTFT and two view ops are eager. No arithmetic changes; validation is tight. (3) if (2b) fails: the adapters are REMOVED (stock forward restored) and only `band_split` is captured (prereg P4 fallback, scope `band_split_only`). | tight |
| **C1** | `torch.compile(model, mode='reduce-overhead', dynamic=False)` — verbatim arm-C1 call. Known at the pin: graph breaks (legacy `sdp_kernel` context, DC-filter decomposition) | loose (1e-2 / 1e-3) |
| **C2** | `mode='max-autotune'`, otherwise C1. Runs LAST in session 1 under a 25-minute per-process cap; a cap → `UNTESTED(timeout)`, never a silent skip | loose |
| **C3** | C1 call on the model with the two external compatibility repairs of `axial-layout-scoping@a6f3e6f extensions/c3/` copied verbatim (sha256 recorded in the files' headers): (i) `Attend.flash_attn` external replacement using `torch.nn.attention.sdpa_kernel` with the same backend list incl. the legacy API's implicit cuDNN default; (ii) the DC-filter `Tensor.index_fill` kept opaque via a custom op (no new kernel, no arithmetic change). **Documented adaptation** (`fullmodel/c3_fix/native_tail_adapted.py`, own sha256): the vendored original registers the op for the single 8-s shape `(2, 1025, 801)`; this session's frame counts are T ∈ {11, 51, 201, 801} (chunks 4410, 22050, 88200, 352800), so the op's shape/stride checks are parametrised by T over exactly that set and the op name differs; the AST substitution and the native call are unchanged. Session step [3] qualifies the op per T (schema/fake checks, bitwise native equality, capture/replay, isolated compile) before any C3/H measurement. Gate: **zero graph breaks** (`TORCH_LOGS=graph_breaks,recompiles` + Dynamo counters) or the arm is INVALID | loose |
| **C1A** | C1 + attention repair only. *Registered expectation from the extension's attempt 2: compilation fails at the DC-filter AOT assertion once the attention break is gone.* Recorded as FAILED with the traceback if so; timed if it compiles | loose |
| **C1D** | C1 + DC-filter boundary only (attention breaks remain) | loose |
| **D** | A + fused RMSNorm (`frontend_rms_fused.rms_fused`, the owner's Triton kernel) in `band_split` via the `FusedBandSplit` swap from `arms/common.py`; eager elsewhere | tight |
| **E** | D + whole-forward capture (B's ladder) | tight |
| **F** | A + multi-stream band loop in `band_split` (`arms/streams.py`, 4 streams, fork/join events) | tight |
| **G** | D + F + capture: the stream-ordered structure captured into ONE graph (B's ladder) | tight |
| **H** | C3 + **pad-and-batch band modules** (§1.1). Compiled with the C3 call. Generic reimplementation of the band-bmm idea; production code is **not used** (not available to the agent; the owner may substitute it and say so) | loose |

`C1D` runs at 352800 only (it isolates the DC-filter repair at the deployed
operating point; plan 1.2) — a stated scope restriction: each repair's
contribution at 4410 is not isolated. `C1A` is attempted ONCE (the matched
pair (C1, C1A) in §5) so its registered compile failure is recorded, and is
not repeated in the matrix; a compile failure is evidence of failure, never
a timing or a tax estimate. All other arms run at both operating points.
The chunk sweep (22050, 88200) runs A, C3, G, H only (plan 1.1).

### 1.1 Arm H — pad-and-batch: the exact tensor program (`fullmodel/padbatch.py`)
Notation: bands i = 0..59, widths K_i (`dim_inputs`), S = ΣK_i, o_i = Σ_{h<i}K_h,
K_max = max K_i, T frames, D = 384, hidden 1536.
- `band_split` input `(1, T, S)` → append one zero column → `(T, S+1)`;
  gather with `I[i,j] = o_i + j` for `j < K_i`, else `S` (the zero column) →
  `(T, 60, K_max)` → **transpose to band-major `(60, T, K_max)`**;
  `F.normalize(dim=-1)` (fp32, torch's default eps) × per-band scale
  `(60,1,1) = K_i**0.5` × zero-padded gamma `(60, 1, K_max)`; then
  `baddbmm(b_pad (60,1,D), x_norm (60,T,K_max), W_pad (60,K_max,D))` with
  `W_pad[i, :K_i, :] = W_iᵀ` and zero rows beyond K_i, `b_pad[i] = b_i`;
  transpose back to `(1, T, 60, D)` = the stock `torch.stack(dim=-2)` order.
  Algebra: zero entries add nothing to the L2 norm, so the padded normalize
  equals the per-band normalize; zero weight rows contribute 0.
- `mask_estimators[·]` input `(1, T, 60, D)` → band-major `(60, T, D)`;
  hidden layers j: `tanh(baddbmm(b_j (60,1,H), h, W_j (60,·,H)))` with every
  original bias kept; last layer: `baddbmm(b_last, h, W_last (60, 1536, 2K_max))`
  where, for the original last weight `W_iᵀ (1536, 2K_i)` and bias `b_i`,
  original columns `j < K_i` go to column `j` and original columns `K_i + j`
  go to column `K_max + j` (same mapping for the bias; other columns zero);
  `F.glu(dim=-1)` on width `2K_max` then equals the per-band GLU for `j < K_i`
  and yields `0·σ(0) = 0` in pad columns; time-major flatten `(T, 60·K_max)`
  and gather `J = concat_i(i·K_max + arange(K_i))` → `(1, T, S)` = the stock
  `torch.cat(dim=-1)` band-concatenation order (duplicated frequencies
  included; the forward's later overlap scatter/averaging is untouched).
- Autocast: `baddbmm` autocasts all three operands to fp16 exactly as
  `nn.Linear` autocasts `F.linear`, so hidden/output dtypes match the stock
  path (fp16 after every affine; fp32 normalize). Padded weights/biases are
  derived buffers built once from the original parameters (no parameter is
  modified). Claim: **algebraic equivalence**, verified numerically at the
  loose tier (H is compiled; reduction trees and fusion differ) — never
  bitwise. A validation failure makes H INVALID and is reported, not tuned.

### 1.2 Planted-fault validator checks (registered predictions; run first)
Arm A with `--planted-fault`: the wrapper adds a constant to the output
audio. Two runs: **tight** comparator, constant `1e-3`; **loose** comparator
(`--force-tier loose`), constant `5e-2`. **Prediction: both FAIL on both
seeds with a seed-invariant `max_abs_diff` equal to the constant (to fp32
rounding).** A validator that passes either run invalidates every validation
verdict of the session (the session aborts). Shakedown 2026-09-20 on
ml-beast: the tight run failed as predicted with `max_abs_diff` 1.0000002e-3
on both seeds (engineering data, not evidence).

## 2. Task-quality gate (plan 0.2)
- 8 fixed clips, 8.000 s each (= one 352800-sample chunk), from the
  MUSDB18-HQ **test** split via the public HF mirror
  `roro128/musdb18-hq-flac` at revision `32fd4e850b45ea0b635be067b921661984b53029`
  (lossless FLAC re-encode of `danjacobellis/musdb18HQ`; CC BY-NC-SA 4.0;
  research use only; **audio is never committed** — `configs/sdr_clips.json`
  records track, split, clip start sample, and the sha256 of every derived
  wav; `scripts/prepare_sdr_clips.py` regenerates the pack deterministically:
  the first 8 tracks of shards `test-00000/00001` in shard order that have a
  usable segment; clip start = the largest multiple of 441 ≤ 30 % of the
  track length, advanced in 1-second steps until the 8-s **vocals** segment
  has RMS ≥ 0.01 — a silent-vocals reference makes SDR degenerate; the first
  draft of this rule, without the non-silence scan, produced two silent
  clips and was corrected before registration).
- **Every compared operating point is gated, paired against A at that same
  operating point.** Reconstruction policy: each 8.000-s clip is processed as
  non-overlapping consecutive chunks of the operating point's size (1×352800,
  4×88200, 16×22050, 80×4410) and the chunk outputs are concatenated (no
  overlap, no crossfade). Short-chunk `SDR_ref` values are therefore NOT
  quality claims about the model; the gate is the paired regression under the
  identical policy.
- Per clip and arm: separated vocals `est`; `SDR(ref, est) = 10·log10(‖ref‖²
  / ‖ref − est‖²)` (plain, no BSS-eval projections — never presented as
  MUSDB BSS-eval SDR) and `SI-SDR(ref, est)` (both signals zero-mean; `s =
  ⟨est,ref⟩/‖ref‖²·ref`; `10·log10(‖s‖²/‖est−s‖²)`); stereo is flattened to
  one vector; float64 accumulation; a perfect match (e.g. `SDR(A_out, A_out)`)
  is reported as `inf` (the gate's non-finite rule is the SDR_ref-terms rule
  stated below, which governs). Computed (a) vs the reference vocals stem and (b) vs the saved
  eager-A output for the same clip at the same operating point
  (`SDR(A_out, est)` — output fidelity).
- **Gate (common to every arm compared on speed):** mean over the 8 clips of
  `ΔSDR_ref = SDR_ref(arm) − SDR_ref(A) ≥ −0.1 dB`. The per-clip minimum,
  SI-SDR and the vs-A numbers are reported alongside; the gate is the mean
  and stays the mean. Tolerance tiers remain the numerical-equivalence check.
- Membership: the eight tracks are distinct and are canonical MUSDB18 test-set
  titles (the mirror's `test` split re-exposes the upstream 50-track test
  set; names checked). The KJ vocals checkpoint's training data is not
  established by its model card, so the gate claims **paired implementation
  fidelity**, not held-out generalization.
- Non-finite semantics: `10·log10(num/den)` with `num>0, den=0` is `inf`
  (perfect match), `num=0` is `undefined` (zero reference / projection). The
  gate uses the `SDR_ref` terms only: a paired row whose arm or A term is
  non-finite makes the gate ungateable → FAIL. Output-fidelity terms vs A may
  be `inf` (A vs itself by construction) and are kept and counted, never
  averaged away.
- The gate runs in restart `r1` of each arm at each operating point, AFTER
  the timed region (the driver's lifecycle: settle → gate → validate → time →
  SDR); A's per-operating-point outputs come from the session's reference
  step, so no extra model builds are needed.

## 3. Graph-deployment verification (plan 0.3)
"Zero graph breaks" is not accepted as evidence of full capture. For C1, C3,
H (Inductor cudagraph trees) and B, E, G (manual graphs), record per pass,
from the node-level nsys export of session 2 (`cuda_api_trace` +
`cuda_gpu_trace`, capture window = the 5a latency block):
`cudaGraphLaunch` calls/pass, `cudaLaunchKernel`-family calls/pass outside
graph replay (eager launches), memcpy/memset API calls/pass, distinct kernel
names inside vs outside replay; plus, for the Inductor arms, the
`torch._inductor` cudagraph counters (`cudagraph_skips` and reasons,
`TORCH_LOGS=cudagraphs`) and the cudagraph-trees recorded-graph count.
Launch counts include runtime AND driver entry points (`cudaLaunchKernel`,
`cudaLaunchKernelExC`, `cuLaunchKernel`, `cuLaunchKernelEx`; `cudaGraphLaunch`
/ `cuGraphLaunch`). This nsys version's GPU trace carries no graph-id column
even under `--cuda-graph-trace=node`, so GPU events are attributed by
CorrId correlation: a kernel whose CorrId belongs to a graph-launch API call
is captured; to a kernel-launch call, eager; neither → UNRESOLVED (counted,
never assumed captured). API calls are windowed by the CPU-side NVTX pass
range, kernels by the GPU-projected range. Fourfold classification per pass: (i) captured
model computation (kernels inside replay); (ii) uncaptured model computation;
(iii) external input staging (cudagraph-trees copies of non-static inputs
into the static pool — `_foreach_copy_`/copy kernels/`cudaMemcpyAsync`);
(iv) other uncaptured copies/memsets. Also reported: recorded-graph count
(Inductor counters), graphs visited per pass, replay launches per pass, and
compile/recording counter stability across settle → validation → timing.
An arm is **fully graphed** only with positive replay evidence (≥1 graph
launch in EVERY captured pass) AND zero uncaptured model kernels AND zero
unresolved kernels in every pass (medians hide a defective minority); the
captured pass count must equal the requested 50. Multiple graphs per pass
are legitimate; staging (`_foreach_copy_`/multi_tensor_apply, direct-copy
kernels, memcpy) is timed overhead but not uncaptured model work; bare
elementwise kernels are model work.
Arms verified: C1, C3, H (Inductor) and B, E, G (manual), from their
session-2 traces at both operating points. No expectation is registered.

## 4. Boundary-cost microbenchmark (plan 0.4; design registered, owner reviews)
`fullmodel/boundary_bench.py`: chains of N **dependent** kernels, N ∈ {10,
100, 1000, 2000}: (trivial) one Triton program loads ONE int32 element, adds
1, stores it back to the same buffer — kernel k+1 depends on kernel k, and
the final counter must equal the total launch count (exact-integer
dependency proof); (elem37) an in-place value-preserving fp16 elementwise op
on a 37 MB tensor, N ∈ {10, 100}; (gemm) alternating dependent GEMMs
660×384×1536 / 660×1536×384 fp16 into preallocated ping-pong outputs with
1/√K-scaled weights (bounded values), N ∈ {10, 100}. Modes: stream launches;
one CUDA graph of the chain (replayed); programmatic dependent launch for the
trivial chain (Triton 3.4 `launch_pdl=True`; `gdc_wait()` before the
dependent load, `gdc_launch_dependents()` after the store; the PTX is checked
for `griddepcontrol`), on a stream and inside a graph; real-kernel PDL is
UNSUPPORTED through cuBLAS/ATen launches and recorded as such. Note: the
triton 3.4.0 wheel's `language/extra/cuda/gdc.py` carries the pre-3.4
`_builder=` signature and fails to compile under the 3.4 code generator
(`_semantic=`); the bench therefore defines the same two one-line PTX
intrinsics with the 3.4 signature (identical asm strings) — recorded here
so the deviation from the library call is explicit. The elem37 chain's
working set (37 MB) fits the RTX 5090's 96 MiB L2, so that chain measures
the L2-resident rate, which is reported as the plan-1.4 L2 bandwidth input. Timing:
unprofiled device time via CUDA events around the chain, median of 20
repeats with min/max → `T(N)` per (chain, mode), plus least-squares slope
and intercept over N. **`T(N)/N` and the slope are effective per-node service
times INCLUDING kernel execution — not boundary costs.** A separate
node-level trace of the same script yields, per chain segment (NVTX), the
median kernel duration and the median serial inter-kernel gap; for non-PDL
modes the serial gap is the boundary estimate; for PDL modes kernels overlap
and no subtraction is attempted (pure boundary `UNRESOLVED`); the traced
run's perturbation vs the unprofiled `T(N)` is reported. Prior calibration
points to compare against: Hazy (H100 ≈ 2.1 µs stream, ≈ 1.3 µs graph);
NVIDIA's CUDA 12.6 figure ≈ 0.55 µs/node is empty-kernel graph device
service time (RTX 3060), not CPU launch cost. **Until measured, no
per-boundary number is used anywhere**, and the ledger's X10 must not count
kernel execution as removable boundary overhead. Trace-derived gaps are
accepted only for segments whose qualification case exists in the traced
run's own JSON with a passed dependency proof and PTX verdict and whose
NVTX segment holds exactly N × 20 kernels (a missing qualification file
yields no calibration value at all); the perturbation runs (plan 1.6) are
accepted only when all three runs are VALID, validation PASS, one block
holding exactly 50 recorded passes, manifest-bound with the same recorded
effective variant, and (compiled arms) affirmatively recording-stable —
otherwise EVERY derived ratio and the caveat are cleared to UNRESOLVED. Any
unavailable mode is recorded `UNTESTED(reason)`.

## 5. Session structure, protocol, budget
- **Protocol 5a**, reused from `arms/common.py::timed_run` (per-iteration
  synchronize latency block of 50 passes inside the cudaProfilerApi range;
  one free-running block of 50 passes outside it), **five repeats per
  process**; per-process statistic = min-of-medians and median spread; per
  (arm × chunk) headline = **median over 3 fresh-process restarts of the
  per-process min-of-medians**, every per-iteration list preserved
  (an explicit extension of `sm120` README deviation 5; spread = max − min of
  the repeat medians; within-process and between-process variation are both
  preserved). **Plan 1.2 matched comparisons** use `fullmodel/matched_run.py`:
  two arms built, settled and validated in ONE fresh process, then their 5a
  blocks ALTERNATE (a, b, a, b, …, five each) so every block pair is matched
  in time and clocks; repeated in three fresh processes. Registered pairs at
  352800: (C1, C3) the graph-break tax; (C1, C1D) the DC-filter repair
  alone; (C3, H) the pad-and-batch delta; (C1, C1A) once, to record the
  expected compile failure. The matrix restarts are additionally round-major
  (r1 of every arm, then r2, then r3). Clocks logged in the background at
  1 Hz with timestamp and GPU UUID (`nvidia-smi --query-gpu=timestamp,uuid,
  clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu -l 1`) for
  the whole session; timed blocks carry wall timestamps.
  **Matched-pair acceptance:** both builds OK, both graph gates PASS, no
  capped settle, a JOINT settle of the actual alternation (counters stable
  across two full alternation rounds), both validations PASS after the joint
  settle, counters stable across every block, and — joined at collection —
  SDR PASS for both arms' cells at that operating point; otherwise the pair
  is ineligible with reasons.
  **Compile/recording stability:** a compiled run is eligible only if, across
  settle → validation → timing → SDR, there are no new graph breaks or
  unimplemented ops, the unique-graph count is unchanged, and every Inductor
  cudagraph counter (recordings, skips, non-static-input recordings) is
  unchanged; all snapshots are preserved in the run JSON.
  **Provenance gate:** every process verifies the msst clone (HEAD = pin,
  clean), the checkpoint sha256, the registered config path and the
  explicit `use_amp` before building; the session preflight writes ONE
  immutable `session_manifest.json` (repo commit + clean tree, sha256 of
  this registration, of every measuring file, the clip manifest, pins,
  environment) that every run and every reference is bound to; a mismatch
  refuses the run.
  **Headline rule:** a cell publishes a headline only if restarts r1, r2 AND
  r3 all exist and are eligible (EXACTLY the 5×50 protocol, validation PASS,
  recording state stable, manifest-bound, not a planted/traced/attribution
  run), share ONE session manifest and ONE effective variant (capture scope
  / adapter / compile mode / stream count), and the cell's SDR qualification
  — taken only from the r1 record, same manifest and variant — is PASS; a
  failed required restart is never replaced by a later tag. A matched pair
  publishes a pair headline only if m1, m2 AND m3 are pair-eligible (5×50,
  three matched restarts) under one manifest with one variant pair and both
  arms' cells hold that SDR qualification; the matched runner re-checks each
  arm's recording state after the joint settle and after validation.
  **Recording observer:** compiled runs snapshot the cudagraph-trees
  manager identity, the identity set of its recorded graph nodes (roots and
  descendants) and its function-id set together with the dynamo/inductor
  counters at every phase boundary; stability means every one of these is
  unchanged (sets compared, never counts); a missing observer leaves
  stability UNRESOLVED and the run ineligible.
  **Independent qualification:** every publisher (driver, matched runner,
  collector, completeness audit) re-derives eligibility from the record
  itself through one versioned validator (`fullmodel/qualify.py`); stored
  eligibility flags are never trusted, and a record whose stored flag
  contradicts the validator — in either direction — is ineligible. The
  validator checks payloads, not labels: both registered validation seeds
  with finite per-pass differences under the arm's registered tier
  tolerances, the SDR gate recomputed from exactly eight distinct finite
  clip rows against the registered threshold, recording stability
  recomputed from the stored phase snapshots (and, for matched pairs, from
  every phase transition: joint settle, validation, every block), the
  zero-break gate's counter for C3/H. Every published statistic (block
  medians, min-of-medians, restart spread, throughput, matched deltas) is
  recomputed from the recorded samples, which must be finite and positive;
  a stored summary that differs from its samples disqualifies the record.
  Each run persists the sha256 of every reference artifact its lineage
  check saw; the three restarts of a cell must have been validated against
  the same hashed validation references (r1 alone additionally checks the
  eager-A clip outputs). Validation passes persist a re-derivable tolerance
  verdict (violating-element count, operand shapes and finiteness, the
  pass's own tolerances); the SDR gate is recomputed from the paired
  SDR_ref terms of the registered clip indices 0–7; recording snapshots
  must match the run's exact phase schema with complete observer fields;
  the graph gate is re-checked from its counters, suppress_errors and the
  after-settle snapshot; a tier override disqualifies any non-planted
  record; every nested stored summary (block medians, extrema, timing
  summaries, matched deltas) must equal its samples. One-block traced and
  perturbation runs are qualified by the same validator (`trace_problems`).
  The completeness audit re-derives every cell and pair from the run
  records through the same validator (never from the collector's summary),
  enumerates the required reference artifacts per operating point, seed and
  clip, re-hashes them and joins every hash a run recorded to its artifact,
  verifies the manifest's identity against the tree, the registration and
  the pins (commit, registration hash, clip manifest hash, measuring-code
  hashes, runtime pins, msst/checkpoint/config pins re-verified where the
  pins are present) and every run's recorded pins against it, requires the
  planted-fault checks and the C3 operator qualification (every registered
  component: opcheck, bitwise equality, capture/replay, isolated full-graph
  compile, output shape) bound to the manifest in BOTH sessions (session 2
  regenerates missing prerequisites under supervision), re-derives every
  reduction it certifies (graph-verification verdict from the per-pass
  evidence, timing-view inequalities and medians, ideal-cost row
  consistency and class totals, perturbation ratios from the run samples,
  source-map coverage), and requires this session's clock log — samples of
  this GPU with numeric fields — to cover every timing block attributed to
  this session by registered tag and operating point (matched blocks
  included) with sample gaps of at most 5 s. Matched records must show the
  registered alternation in chronological order with finite, non-overlapping
  block timestamps, and every stored matched summary must equal its samples.
  The attribution arm ANVTX is validated at the tight tier like A.
  **Runtime pins and manifest scope:** every process checks torch ==
  2.8.0+cu128 and triton == 3.4.0 before building; the manifest binds the
  complete measuring set (fullmodel/**, arms/*, scripts/*, configs/*,
  frontend.py, frontend_rms_fused.py, setup.sh, this registration, the
  runtime identity) and is compared field by field on every load; every
  reference artifact (validation outputs, eager-A clip outputs) carries its
  own lineage sidecar (manifest id + sha256) checked before use.
- **Unprofiled timing is the primary result.** Session-2 traces are for
  mechanism; profiled and unprofiled numbers never share a ratio. Profiling
  perturbation (plan 1.6): A, C3, G at 4410 and 352800, wall unprofiled vs
  `--cuda-graph-trace=graph` vs `=node`; >5 % inflation → every mechanism
  number from that trace carries the factor as a caveat.
- **Session 1 (≤ 3 h, RunPod secure-cloud RTX 5090, pod name
  `melband_profiling`, image `runpod/pytorch:1.1.0-rc.154-cu1281-torch280-ubuntu2204`
  = the pinned torch 2.8.0+cu128 / triton 3.4.0 matrix):** in this order —
  setup (idempotent `setup.sh` + clip pack + nsys check); eager-A references
  at both operating points; the two planted-fault checks; C3 operator
  qualification per frame count; the matched pairs (§5, plan 1.2); the arm
  matrix at 352800 and 4410 (three round-major restarts, SDR gate in r1 at
  both points); C2 last. Budget guards (registered): a hard session deadline
  (3 h) with a 15-min extraction reserve — a cell that cannot start before
  the reserve is `UNTESTED(deadline)` and the session is INCOMPLETE, never
  salvaged by weakening validation or timing; per-process caps (25 min
  compile arms, 15 min others); an **aggregate C2 allowance of 40 min**, and
  a timed-out cold C2 attempt is not retried at that operating point (the
  remaining restarts are `UNTESTED`; a timeout means unavailable
  performance, not evidence that C2 is slow; a C2 cell whose three restarts
  are terminal `UNTESTED` records citing the allowance / no-retry / timeout
  rule is permitted terminal unavailability and does not make the session
  INCOMPLETE — every other missing or failed registered item does). Every run's JSON carries
  `headline_eligible` with machine-readable reasons; a headline per (arm ×
  operating point) requires three complete eligible restarts with the same
  effective variant (capture scope / adapter / compile mode), else the cell
  is `INCOMPLETE`. At the end of each session `fullmodel/completeness.py`
  audits the registered mandatory items BY CONTENT — session 1: the 21
  matrix cells, the three matched-pair families (m1–m3), the once-only
  (C1, C1A) attempt's terminal record, both planted-fault records (FAIL at
  the planted magnitude), the C3 operator qualification (all four frame
  counts PASS); session 2: the eight sweep cells, the 19 node-level traces
  (traced run VALID, one block of 50 passes, manifest-bound, exports
  complete), graph verification and timing views per traced arm, ideal
  costs (C3 requires a unique variant and validated windows), the 18
  perturbation runs and six resolved ratios, source-map coverage for C3,
  the attention bench (eight cases, model-layout ratios established), the
  boundary bench (terminal status per registered chain) and its qualified
  trace analysis; both sessions: manifest, collection summary, reference
  lineage (directory marker and every sidecar), clock log, and every
  auxiliary terminal record OK — and writes the mechanical session verdict
  (COMPLETE / INCOMPLETE with per-item reasons) to `session_status.json`.
  Collection and the audit are themselves supervised; a previous verdict is
  invalidated before every finalisation and any collection or audit failure
  or timeout writes an explicit INCOMPLETE; every measurement or trace
  attempt invalidates its own previous artifacts (run JSON, report, exports,
  derived reductions) before it starts, so no stale success can survive a
  failed attempt. Every on-pod reduction (nsys
  exports, graph verification, timing views, ideal costs, perturbation) is
  bounded by the session deadline, invalidates its stale outputs before the
  attempt (the main-trace reductions belong to the `trace` attempt only;
  perturbation traces invalidate their own products), even when the attempt
  is then skipped for the deadline, and leaves a terminal `aux_<name>.json`
  (OK / FAILED / UNTESTED); supervisor messages go to stderr. A failed or
  timed-out C3 operator qualification records C3 and H work as UNTESTED and
  the session continues with the unaffected arms (finalised INCOMPLETE). Pod setup (image boot,
  `setup.sh`) precedes the session clock. Evidence extraction (validated
  archive, receipt bound to the pod id AND to the content hash of the pod's
  evidence, taken before and after the transfer with no session running and
  required to equal the extracted archive's own content manifest;
  termination takes a lock on the pod that refuses any new session launch,
  refuses while a session runs, and re-checks the content hash under the
  lock) before terminate. A missing reference or a failed planted-fault
  check never aborts a session: dependent measurements are recorded
  UNTESTED and the session is finalised INCOMPLETE. Ideal-cost inventories
  exclude memops; an observed pass count other than 50 is
  UNREGISTERED_PROTOCOL.
- **Session 2 (≤ 3 h):** eager-A references + SDR at the sweep points; chunk
  sweep (A, C3, G, H × {22050, 88200}, 3 restarts, SDR gate in r1);
  node-level traces (ONE captured 50-pass block per traced run, outer NVTX
  `pass_<i>` ranges, captured pass count checked against the requested 50
  before attribution; generated-code dump from the SAME process for compiled
  arms) of A, A-NVTX, C3, G, H **and C1, B, E, F** (graph verification §3;
  timing views, plan 1.7) at both operating points, plus **C2 at 4410 only**
  (the §6 comparator point; a C2 trace at 352800 is not budgeted);
  profiling-perturbation runs (A, C3, G: one matching block each,
  unprofiled vs `--cuda-graph-trace=graph` vs `=node`, reduced to ratios
  with the >5 % caveat flag by `fullmodel/perturbation.py`);
  attention-backend microbench (plan 1.5: time-axis batch 60 × seq T and
  freq-axis batch T × seq 60, heads 8, dim_head 64, fp16, default scale,
  dropout 0; TWO layout families — the model's actual SDPA input layouts
  read from the C3 generated code (q/k as (b,n,h,d)-contiguous rotary
  outputs viewed as (b,h,n,d); v a strided view into the fused qkv output)
  and, as a separate experiment, contiguous (b,h,n,d); flash / efficient /
  cuDNN / math forced one at a time, fp32 MATH-forced reference, CUDA-graph-
  replayed device timing, unsupported cases recorded; the model's deployed
  backend established PER operating point AND axis from the C3 node traces
  — the 12 attention executions per pass in (time, freq) × 6 order, Triton
  preprocessing kernels excluded — and a case whose deployed backend is not
  established from a trace has NO ratio and is UNRESOLVED, never assumed);
  graph verification requires an affirmative node-level export record and
  exactly the registered 50 passes (any other expected count is an
  unregistered shakedown; a missing record is UNRESOLVED); for the eager
  and multi-stream arms (A, D, F) the expected qualified verdict is
  NO_REPLAY_EVIDENCE; timing views count kernels only (memcpy/memset rows
  are reported separately as memops); the recording observer assigns
  lifetime-safe identities through a weak registry (address-based
  identities leave stability UNRESOLVED);
  boundary microbench (§4) plus its node-level trace and traced-vs-unprofiled
  perturbation; per-kernel ideal costs (plan 1.4) computed only from the
  compiled variant(s) whose ORDERED per-pass Triton sequence is reproduced
  by a UNIQUE decomposition into compiled-graph call sequences (graphs may
  repeat; graphs are interchangeable only when their complete pricing
  semantics agree — kernel body hashes, access sites, iteration literals,
  argument shapes/strides/types and every extern/fallback call in order; graphs without Triton kernels are invisible to the
  decomposition and are stated as such; SDPA call count reconciled exactly,
  GEMM events ≥ extern calls, captured pass count = 50; zero or several
  decompositions → UNRESOLVED, expected for graph-broken arms such as C1/C2
  whose per-graph kernel numbering makes names non-unique). Inductor GEMM
  template kernels (`triton_tem_*`, max-autotune) are counted in the GEMM
  class and are not priced by the byte model. Registered deviations from a literal
  per-kernel FLOPs/bytes table: cuBLAS kernel names carry no shapes, so GEMM
  ideals are computed per extern call site from the wrapper and compared at
  class level; SDPA ideals are class-level (12 calls/pass at the model
  shapes); Inductor fused pointwise/reduction kernels are priced by bytes
  only (bandwidth bound, zero FLOPs) with per-call L2 residency tests. Byte
  model for a fused kernel call: per pointer argument, element size from the
  kernel's own ABI signature (`triton_meta`), traffic = min(distinct
  elements of the argument's strided footprint, load/store sites × the
  call's iteration space) × element size, reads and writes charged
  separately, where the iteration space is the product of the call's
  positional integer literals bound one-to-one to the kernel's numeric
  parameters (any other call form is UNRESOLVED); the L2 residency test
  uses the union of the distinct base allocations the call touches, each at
  its full allocated size (an over-estimate: a call passes the test only
  when its whole working set provably fits — allocation size is the address
  extent from the base allocation's own shape, stride and dtype, unknown →
  no L2 ideal); every measured value comes from exactly 50 non-empty
  validated per-pass windows (the aggregate kernel summary is reconciled
  against them by name set, count and duration, and the numerical
  discrepancy is always reported), and an ideal-cost run whose requested or
  observed pass count is not the registered 50 publishes an explicit
  UNREGISTERED_PROTOCOL record with no rows; wrapper calls the parser does
  not recognise (assigned or not, any `torch.ops` namespace) are recorded
  verbatim and make graphs non-interchangeable; boundary calibration
  derives the expected dependency counter from N, mode, warm-up and
  repeats and requires both recorded counters to equal it;
  boundary calibration accepts a segment only when its case carries a passed
  dependency-proof counter (trivial chains) and, for PDL modes, the PTX
  qualification — labels alone never qualify; source-map coverage credits
  only exact, unambiguous kernel names with resolved fused ops; graph-input dtypes from the wrapper's benchmark section;
  fallback-op outputs from their `assert_size_stride`. This over-counts
  slice reads, so a row whose HBM-model ideal exceeds its measured time is
  flagged `bound_not_tight` and its recoverable share under the HBM model is
  0 (never negative, never inflated); eager ATen fallbacks, cuFFT and copy
  kernels without wrapper shapes stay UNRESOLVED. A kernel → fused-op source
  map with its coverage share accompanies the table; exports; evidence
  extraction. Session 2 runs under the
  same deadline/reserve/terminal-record rules as session 1 (qualification
  and microbenchmarks included).
  **Plan 1.7 dependency model (registered):** the exports carry no event or
  graph edges, so the critical-path view is the OBSERVED-ORDER longest chain
  of pairwise non-overlapping kernels (edge a→b iff a ended before b
  started); it is an upper bound on the true dependency path, satisfies
  `critical_path ≤ union ≤ span` by construction (asserted), and is labelled
  as such; the union and the summed views are exact.
- Engineering shakedown of the harness on ml-beast GPU 4 is permitted before
  session 1 (the `sm120` smoke rule): its outputs are never session evidence
  and are never cited.
- Budget guard: every process runs under `timeout`; a timed-out or crashed
  run is INVALID/UNTESTED in its JSON and the session continues (loud in the
  console log, never silent).

## 6. Exploratory hypothesis (labelled exploratory; directional only)
"At the 4410 operating point, at least one of G / H has lower qualified
latency (headline-eligible, tier PASS, SDR PASS) than the fastest of C1 / C2
/ C3 with verified graph replay (§3)." B (manual capture) is reported
separately and is not a comparator here. G and H are bundles, so this ranking
attributes nothing causally to "fusion" or "concurrency" on its own. No
numeric ratio is registered (no derivation exists yet; §4 and the session-2
kernel counts are the inputs one would need). If a required comparator is
unavailable, the restricted comparison is reported as such.

## 7. Deliverables and hygiene
`evidence/` (git-ignored except `*.md`), `SESSION_RESULT.md` in the fixed
format of the scoping study's `RESULT.md` (provenance block; decision inputs
with thresholds applied mechanically; predictions vs measurements; per-arm
tables with tier validation, SDR gate, capture scope / graph verification;
per-layer tables from the traces; what failed), then `LEDGER.md` (plan
Phase 2). Every number cites its file; unattributable = `UNRESOLVED`.
Provenance pins, the pre-commit integrity gate, and the never-edit-the-clone
rule are unchanged. Profiled/unprofiled convention: headline walls are
unprofiled; the perturbation ratios of plan 1.6 are the only ratios mixing
the two and are labelled `PERTURBATION`; every ledger ceiling that divides
trace-derived attributable time by an unprofiled wall is labelled `MODELLED`
and is never presented as a measured speedup. Plan 0.5 (researcher
outreach) is an owner action; `docs/outreach_draft.md` holds the draft and
becomes the audit record (date sent, attachments, replies).

## 8. Owner's dissent / strike lines (fill before the session or leave blank)
- Section 0 (full-model reading): ______
- Arm H generic reimplementation vs production code: ______
- SDR threshold (−0.1 dB mean): ______
