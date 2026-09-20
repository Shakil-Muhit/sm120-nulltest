BLOCKERS: 5

§0 is the **more likely intended reading**, but “the only coherent reading” overstates it. A coherent alternative is a two-level study: retain frontend arms as local mechanism experiments, then measure their effects in full-model wrappers for the ledger and SDR gate. However, every X row uses full-forward attribution or an end-to-end denominator, and the quality gate requires audio outputs. The author most likely intended the registered full-model matrix. Remove the automatic fallback to `scripts/run_all.sh`; it would not satisfy that ledger. ([Registration:24](/home/muhit/ms/sm120-nulltest/prereg_amendments.md:24), [plan:69](/home/muhit/ms/plan_ceiling_ledger.md:69), [plan:111](/home/muhit/ms/plan_ceiling_ledger.md:111))

Findings, ordered by severity:

1. **[BLOCKER] §1 — C3 cannot be vendored verbatim across the registered chunks.**  
   `native_tail.py` asserts `(2,1025,801)` in both real and fake implementations and hardcodes the corresponding strides. C3/H at 4410 and the sweep therefore fail by construction. Preserve the upstream files and hashes, but register a documented adaptation supporting frame counts `{11,51,201,801}`. Verify native output metadata, fake metadata, eager equivalence, capture/replay, and isolated compilation for every supported shape. Do not merely delete the assertions. ([Registration:59](/home/muhit/ms/sm120-nulltest/prereg_amendments.md:59), [native_tail.py:17](/home/muhit/ms/axial-layout-scoping/extensions/c3/native_tail.py:17))

2. **[BLOCKER] §1.1 — H’s written tensor program is incomplete and, literally implemented, incorrect.**  
   The mask chain omits all three Linear biases. Also, `(60,1,1)` scaling requires **band-major** `(60,T,Kmax)` input, whereas the preceding description specifies `(T,60,Kmax)`. Register the intervening permutation and every affine bias.

   Make the indexing explicit. With `S=ΣK_i` and `o_i=Σ_{h<i}K_h`:
   - Input gather: `I[i,j]=o_i+j` for `j<K_i`, otherwise `S`, indexing an appended zero column.
   - Last Linear: transpose each original weight, place original columns `j` at `j`, and original columns `K_i+j` at `Kmax+j`; apply the identical mapping to its bias.
   - After GLU, permute to time-major, flatten `(band,Kmax)`, then gather `J=concat(i*Kmax+arange(K_i))`.

   This restores **band-concatenation order**, including duplicated frequencies. Leave the public forward’s subsequent overlap scatter/averaging untouched. ([Registration:73](/home/muhit/ms/sm120-nulltest/prereg_amendments.md:73))

3. **[BLOCKER] §2 — The quality gate does not cover the claimed speed matrix.**  
   Chunk-independent source transformations do not imply chunk-independent compiled programs, numerical errors, or quality. Two random-input tolerance checks do not establish SDR on real short-chunk audio.

   Gate each compared operating point against **A at that same operating point**. One concrete design: process each fixed eight-second clip as 80×4410, 16×22050, 4×88200, or 1×352800 chunks, with a registered reconstruction policy. Otherwise, explicitly narrow quality-qualified conclusions to 352800 and mark other quality results `UNTESTED`; the existing exemption cannot establish plan compliance. ([Registration:118](/home/muhit/ms/sm120-nulltest/prereg_amendments.md:118), [plan:51](/home/muhit/ms/plan_ceiling_ledger.md:51), [plan:69](/home/muhit/ms/plan_ceiling_ledger.md:69))

4. **[BLOCKER] §3 — Zero host kernel-launch calls is not a sufficient graph-coverage criterion.**  
   Count runtime **and driver** launches, including `cuLaunchKernelEx`, and correlate GPU events with graph identities. Distinguish:
   - captured model computation;
   - uncaptured model computation;
   - external input staging;
   - other uncaptured copies/memsets.

   Cudagraph trees copy nonstatic inputs before replay, using `_foreach_copy_`; staging can itself involve kernels. Conversely, zero uncaptured kernels does not establish that model copies/memsets are captured. `cudaMemcpyAsync` for input staging is compatible with “model computation captured,” but remains timed overhead.

   Require positive replay evidence, complete event attribution, and stable capture/recording state. Report recorded graphs, graphs visited per pass, and replay launches separately; multiple graphs per pass are legitimate. ([Registration:124](/home/muhit/ms/sm120-nulltest/prereg_amendments.md:124), [Torch 2.8 trees implementation](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/_inductor/cudagraph_trees.py))

5. **[BLOCKER] §4 — `chain_time/N` does not isolate boundary cost.**  
   It includes kernel execution, launch/endpoint costs, and potentially Python submission starvation. Subtracting a separately profiled kernel duration from unprofiled event timing does not repair that attribution.

   Register unprofiled `T(N)`, its intercept/slope, and **effective per-node service time**. Separately report kernel durations and serial inter-kernel gaps from the *same trace*, with perturbation quantified. A pure boundary estimate needs an identified execution control; otherwise leave that component `UNRESOLVED`. PDL overlap also invalidates naïvely subtracting summed kernel durations. X10 must not double-count execution as removable boundary overhead. ([Registration:147](/home/muhit/ms/sm120-nulltest/prereg_amendments.md:147))

6. **[MAJOR] §1.1 — H’s “only cuBLAS summation order” claim is false under AMP.**  
   Zero padding preserves the mathematical L2 norm with the same epsilon and `sqrt(K_i)` scale; it can still change the reduction tree. Compilation adds further rounding/fusion differences.

   Crucially, autocast makes `bmm` return fp16, while adding a multidimensional fp32 bias promotes its result to fp32. Subsequent Tanh/GLU then differ from the original autocast Linear path. Specify an autocast-compatible affine implementation, such as `baddbmm`, and verify intermediate/output dtypes. Preserve fp32 normalization, original epsilon, all biases, and parameter values. Claim algebraic equivalence, then require numerical validation—not bitwise exactness. ([Registration:75](/home/muhit/ms/sm120-nulltest/prereg_amendments.md:75), [Torch 2.8 autocast rules](https://docs.pytorch.org/docs/2.8/amp.html#cuda-op-specific-behavior))

7. **[MAJOR] §1 — The capture ladder is legitimate, but needs precise variant and failure semantics.**  
   The constant hoist qualifies as an external adapter: instance-local forward replacement, mechanically restricted AST change, unchanged arithmetic, untouched clone. C3 already establishes this pattern.

   The scalar-construction failure mode is real in Torch 2.8: `torch.tensor(0, device='cuda')` constructs host data and performs a blocking device transfer; the copy path synchronizes the stream. Describe that specific path instead of a blanket claim about every pageable copy. ([Constructor](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/csrc/utils/tensor_new.cpp), [copy/synchronization helper](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/c10/cuda/CUDAFunctions.h))

   Record `adapter=stock|dc_hoist` separately from capture scope. Specify whether band-only fallback restores the original forward. Apply AMP during **warmup, input recording, and capture**, with capture-safe cache handling. Preserve each traceback and recover the stream/context before retrying. Hoisting does not guarantee capture of the remaining forward. ([Registration:56](/home/muhit/ms/sm120-nulltest/prereg_amendments.md:56), [common.py:217](/home/muhit/ms/sm120-nulltest/arms/common.py:217))

8. **[NOTE] §1 — Arm mechanics and tolerance assignments otherwise match the audited harness.**  
   D is fused frontend RMSNorm; E adds replay; F adds streams; G combines fused frontend, streams, and replay. E/G contain no Inductor merely because they use Triton or CUDA graphs. Tight A/B/D/E/F/G and loose C1/C2/C3/C1A/C1D/H follow the classification rule. H is loose because it is compiled, not because batching independently grants relaxed tolerance. Full-model AMP still requires fresh validation; frontend smoke results confer no automatic PASS. ([common.py:75](/home/muhit/ms/sm120-nulltest/arms/common.py:75), [common.py:256](/home/muhit/ms/sm120-nulltest/arms/common.py:256))

9. **[MAJOR] §2 — SDR choices are defensible, but metric and dataset details remain underspecified.**  
   Plain SDR is acceptable for a paired regression gate; it must not be presented as MUSDB BSS-eval SDR. Mean paired `ΔSDR ≥ −0.1 dB` is a legitimate preregistered threshold, though improvements can compensate for individual regressions. Keep the per-clip minimum; do not silently change the gate to a minimum afterward.

   Specify stereo aggregation, SI-SDR centering/projection, float64 accumulation, silence handling, and representation of perfect-match infinities. `SDR(A,A)` is infinite by construction.

   Freeze the eight **distinct track IDs**, mixture/vocals pairing, offsets, sample format, and hashes before measurement. The mirror remaps an upstream `validation` split to `test`; verify canonical MUSDB test membership. KJ vocals is appropriate for a vocals regression check, but its public model card does not establish training-set exclusion. Claim paired implementation fidelity, not held-out generalization. ([Registration:100](/home/muhit/ms/sm120-nulltest/prereg_amendments.md:100), [dataset card](https://huggingface.co/datasets/roro128/musdb18-hq-flac), [checkpoint card](https://huggingface.co/KimberleyJSN/melbandroformer))

10. **[MAJOR] §4 — PDL and real-kernel controls need a reproducible protocol.**  
    Triton 3.4 supports the launch option and GDC intrinsics. Require `gdc_wait()` before dependent loads, register trigger placement, verify generated instructions and captured dependency behavior, and validate the final value. Use an integer scalar or reset a precisely representable value; avoid fp16 increment saturation. A trivial chain offers little independent work, so no speedup would not disprove PDL’s usefulness. ([Triton launch implementation](https://raw.githubusercontent.com/triton-lang/triton/v3.4.0/third_party/nvidia/backend/driver.py), [GDC implementation](https://github.com/triton-lang/triton/blob/v3.4.0/third_party/nvidia/language/cuda/gdc.py), [NVIDIA PDL semantics](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html))

    Preallocate ping-pong GEMM outputs, preserve dependency, bound values against overflow, and specify exact bytes/cache conditions for the elementwise case. Ordinary PyTorch/cuBLAS GEMMs do not acquire PDL through a Triton launch flag; mark unsupported real-kernel PDL separately. The NVIDIA ≈0.55 µs calibration is empty-kernel graph device service time, not CPU launch cost. ([Registration:139](/home/muhit/ms/sm120-nulltest/prereg_amendments.md:139), [NVIDIA measurement](https://developer.nvidia.com/blog/?p=88631))

11. **[NOTE] §5 — The nested timing statistic is acceptable.**  
    Five repeats per process, each containing 50 synchronized and 50 free-running passes, with per-process minimum latency median and spread, matches the plan. Median across three process minima is a reasonable explicitly registered cross-process statistic. It extends—not verbatim reproduces—README deviation 5. Define spread as `max(repeat_medians)-min(repeat_medians)` and preserve both within-process and between-process variation. ([Registration:158](/home/muhit/ms/sm120-nulltest/prereg_amendments.md:158), [README:187](/home/muhit/ms/sm120-nulltest/README.md:187), [common_timing.py:103](/home/muhit/ms/axial-layout-scoping/scoping/common_timing.py:103))

12. **[MAJOR] §5 — Interleaved restarts are not interleaved repeats.**  
    Running five C1 blocks, unrelated arms/compiles, then five C3 blocks does not meet plan §1.2’s matched-repeat requirement. Use separate, settled C1/C3 worker processes and alternate individual blocks, with GPU activity serialized; repeat with fresh workers for each restart. Include the single-fix comparisons in an explicit matched schedule. Log timestamp and GPU identity in the clock CSV—the proposed query contains neither. ([Registration:164](/home/muhit/ms/sm120-nulltest/prereg_amendments.md:164), [plan:73](/home/muhit/ms/plan_ceiling_ledger.md:73))

13. **[MAJOR] §§3/5 — Required traces are absent from the schedule.**  
    §3 promises verification for C1/B/E, but Session 2 only schedules A/A-NVTX/C3/G/H. F also lacks the trace needed for plan §1.7. Add dedicated verification/mechanism passes for C1/B/E/F at both points. These can be shorter evidence runs, separately identified from headline timing. ([Registration:126](/home/muhit/ms/sm120-nulltest/prereg_amendments.md:126), [registration:181](/home/muhit/ms/sm120-nulltest/prereg_amendments.md:181))

14. **[MAJOR] §5 — Five-repeat tracing has a denominator trap.**  
    The reused trace command ends collection after the first profiler range. Calling `timed_run` five times then dividing captured events by all 250 latency iterations produces incorrect per-pass counts. Register either one captured 50-pass block or explicitly repeated capture ranges, and store the **actual captured pass IDs/count**. Perturbation ratios must compare matching blocks, not an unprofiled minimum against an arbitrary traced block. ([traces.sh:76](/home/muhit/ms/sm120-nulltest/analysis/traces.sh:76))

15. **[MINOR] §1 — C1A/C1D at 352800 only is an explicit scope restriction.**  
    Plan §1.2 does not explicitly repeat the chunk list, so this is defensible, but it does not isolate each repair’s contribution at 4410. State that limitation; preferably include the short-point controls if budget permits. C1A’s expected compile failure is evidence of failure, not an estimated timing or tax. ([Registration:60](/home/muhit/ms/sm120-nulltest/prereg_amendments.md:60))

16. **[MAJOR] §5 — Per-process C2 caps do not enforce the session budget.**  
    Six processes permit 150 minutes of C2 alone. Add an aggregate C2 allowance, a session deadline, an evidence-extraction reserve, and a registered rule against repeating an identical timed-out cold attempt. Timeout means unavailable performance—not evidence that C2 is slow. ([Registration:58](/home/muhit/ms/sm120-nulltest/prereg_amendments.md:58), [registration:191](/home/muhit/ms/sm120-nulltest/prereg_amendments.md:191))

17. **[MAJOR] §6 — The exploratory comparison is ambiguous and misattributes mechanisms.**  
    B is manual capture, not compiled execution; C2 is omitted; G/H are bundles rather than isolated tests of “fusion and concurrency.” Specify whether either or both must win. A concrete wording: “Exploratory: at least one of G/H has lower qualified latency than the fastest C1/C2/C3 arm with verified graph replay at 4410.” Report B separately and avoid causal attribution from this ranking. If required comparators are unavailable, report the restricted comparison explicitly. ([Registration:195](/home/muhit/ms/sm120-nulltest/prereg_amendments.md:195))

Session 1 is **possible on the favorable path, but not credibly bounded to three hours as registered**. Using your estimates and the registered matrix:

| Component | Planning allowance |
|---|---:|
| 66 processes × 10 s model load | 11 min |
| Eight non-C2 cold compile attempts × 3–10 min | 24–80 min |
| Two initial C2 attempts × 15–25 min | 30–50 min |
| 66 timing runs × 15–40 s, conservatively applying the eight-second allowance | 16.5–44 min |
| Setup, validation/capture, SDR, extraction allowance | 15–30 min |
| Total before repeated failed C2 attempts or unexpected cache overhead | **≈97–215 min** |

Short-chunk runs and an early C1A failure can reduce this. Warm caches still require process initialization, graph reconstruction/capture, and settling. Building a separate eager reference every process can add another substantial load cost.

Prioritize these changes:

1. Prepare dependencies, clip pack, manifests, and extraction checks before renting; perform CPU exports/postprocessing off-pod where possible. Reuse verified eager reference outputs to avoid redundant reference builds.
2. Run the planted fault and source/operator qualification first, then matched C1/C3 controls, A, and the principal G/H cells. Preserve gates, five repeats, and three restarts.
3. Put C2 last under an **aggregate** allowance; after a deterministic failure/timeout, do not spend two more identical caps.
4. Defer complete lower-priority cells to Session 2 only with reserved capacity and appropriate same-session comparison controls. Keep C1/C3 matched together.
5. Enforce a hard session deadline with extraction time reserved. Unfinished mandatory cells remain `UNTESTED`; the session is incomplete. Do not salvage completeness by reducing validation or timing requirements.

For `fullmodel/`, use a small registry-driven harness:

| Module | Responsibility |
|---|---|
| `loader.py` | Public construction/loading path; checkpoint/config/source verification; strict AMP configuration |
| `arms.py`, `capture.py`, `band_batch.py` | Declarative arm registry, explicit capture ladder, H transformations |
| `vendor/c3/`, adapted C3 module | Untouched upstream files plus separately hashed/documented adaptation |
| `protocol.py`, `run_full_arm.py` | Warm/settle/validate/time lifecycle; one arm/chunk/restart; optional matched-worker interface |
| `sdr_gate.py` | Fixed clip manifest, reconstruction, metric definitions, eligibility |
| `graph_verify.py` | Trace correlation, graph identities, uncaptured work/staging classification |
| `boundary_bench.py`, `attention_bench.py` | Independent calibration protocols and capability failures |
| `session*.sh`, `collect.py` | Deadline-aware scheduling, durable records, mechanical eligibility and aggregation |

Reuse the `common.py` timing body verbatim under an explicit AMP context, numerical tolerances, seeds/input-generation mechanics, fused-band wrappers, and audited stream/event machinery. Adapt the full-model loader, arm builder, capture orchestration, validation, and JSON schema. Do not import `build_stock_frontend` as the model path.

Validation needs exact shape and finiteness checks from [common_timing.py:137](/home/muhit/ms/axial-layout-scoping/scoping/common_timing.py:137), plus the audited different-data and borrowed-output discipline: **step → synchronize → clone immediately**. Preserve capture-final ordering. Fail closed on unsettled compilation and check compilation/recording stability again after validation. The `1e-3` planted fault checks the tight path; it is not an adequate planted failure for the loose tier. ([build_spec.md:8](/home/muhit/ms/sm120-nulltest/build_spec.md:8), [common.py:333](/home/muhit/ms/sm120-nulltest/arms/common.py:333))

Extend run JSONs with:

- Schema/session/run IDs; code, registration, model, checkpoint, config and adapter hashes; push-timestamp evidence.
- Effective arm variant, complete capture attempts, AMP/dtypes, backend flags, cache configuration, GPU UUID.
- Phase status/error/exit code; requested/completed repeats and passes; every timing block and clock window.
- Validation shapes/finiteness/errors/reference paths; per-clip SDR values, gate result and qualifying operating point.
- Compile/recording counter snapshots, graph verification, trace mode, captured pass IDs, source-dump hashes.
- Explicit `headline_eligible` and machine-readable rejection reasons.

A supervisor must write a terminal record even when the child is killed by `timeout`. Compile failure must not fall back to eager; capture may use only the registered ladder. Validation/SDR failure excludes speed comparisons while retaining diagnostic evidence. Provenance failure or a failed planted-fault check stops dependent acceptance. Require three complete valid restarts with the same effective variant before publishing a headline—the old collector does not enforce this. ([collect.py:85](/home/muhit/ms/sm120-nulltest/analysis/collect.py:85))

The attention bench should preserve actual Q/K/V shapes **and strides**, dtype, scale and dropout; force one backend at a time and report unsupported cases. Identify the model’s actual backend from its trace, then measure both numerator and denominator of the backend ratio under the same microbenchmark protocol.

Lay out evidence mechanically:

```text
evidence/<session-id>/
  manifest.json
  runs/<arm>/c<chunk>/r<restart>/{run.json,stdout.log,stderr.log,output_code/}
  traces/<arm>/c<chunk>/<mode>/{run.json,trace.nsys-rep,trace.sqlite,events.csv,graph_verify.json}
  quality/c<chunk>/{manifest.json,metrics.json}
  benches/{boundary,attention}/
  derived/{matrix.csv,perturbation.csv,kernel_ledger.csv,ideal_costs.csv,ledger.csv,claims.json}
```

Give each derived row stable IDs and source paths plus JSON pointers/CSV row IDs. Keep event-to-source mappings, unresolved ownership, and the selected steady generated graph explicit. Archive hashes and restoration instructions must make git-ignored evidence retrievable. Generate report tables from those records; reuse the scoping report’s structure, **not its old layout thresholds**.

The following plan obligations remain binding but are missing or insufficiently operationalized:

- **§0.5:** researcher outreach and its audit record.
- **§1.3:** complete kernel classification and verified compiled source/provenance mapping; existing P8 hooks alone do not identify every requested sublayer.
- **§1.4:** per-kernel FLOPs, minimum bytes, execution dtype, device/L2 properties, and both applicable roofline definitions.
- **§1.7:** F/G summed durations, interval union, and dependency-critical-path estimate with the dependency model stated.
- **Phase 2:** formulas and measured inputs, confidence, effort, prior work, quality risk, limiting assumptions, overlap/stacking notes, and mechanically applied thresholds. Merely naming `LEDGER.md` does not specify these evidence dependencies. ([Plan:65](/home/muhit/ms/plan_ceiling_ledger.md:65), [plan:75](/home/muhit/ms/plan_ceiling_ledger.md:75), [plan:97](/home/muhit/ms/plan_ceiling_ledger.md:97))

One inherited ambiguity also needs an explicit convention: the plan forbids mixing profiled/unprofiled numbers in ratios while requiring perturbation ratios and trace-derived ceilings normalized by unprofiled wall. Name those diagnostic/modelled uses explicitly; do not silently present raw profiled attributable time divided by unprofiled wall as an unprofiled speedup.