BLOCKERS: 10

I read every requested file and rechecked the files changed during review, including the corrected boundary-counter arithmetic. Verification included Bash/Python syntax checks, CPU checks using Torch 2.8.0+cu128, synthetic acceptance-gate reproductions, and inspection of existing trace/compiler artifacts. I performed no new GPU measurements.

| Gate-01 finding | Status | Closure evidence or remaining defect |
|---|---|---|
| 1. C3 frame-count adaptation | CLOSED | [native_tail_adapted.py:25](/home/muhit/ms/sm120-nulltest/fullmodel/c3_fix/native_tail_adapted.py:25) supports all four frame counts; [session1.sh:138](/home/muhit/ms/sm120-nulltest/scripts/session1.sh:138) qualifies them in a separate process. |
| 2. H tensor program | CLOSED | [padbatch.py:50](/home/muhit/ms/sm120-nulltest/fullmodel/padbatch.py:50) implements padded input indices; [padbatch.py:132](/home/muhit/ms/sm120-nulltest/fullmodel/padbatch.py:132) implements both GLU halves, biases, and output ordering. |
| 3. Quality gate at every operating point | PARTIALLY-CLOSED | [sdr_gate.py:93](/home/muhit/ms/sm120-nulltest/fullmodel/sdr_gate.py:93) implements chunked reconstruction and [session1.sh:159](/home/muhit/ms/sm120-nulltest/scripts/session1.sh:159) requests SDR at both points; final eligibility does not require quality qualification—finding 20. |
| 4. Graph-coverage acceptance | NOT-CLOSED | [graph_verify.py:157](/home/muhit/ms/sm120-nulltest/fullmodel/graph_verify.py:157) requires positive replay, but classification and aggregation still admit false positives—finding 22. |
| 5. Boundary versus service time | PARTIALLY-CLOSED | [prereg_amendments.md:212](/home/muhit/ms/sm120-nulltest/prereg_amendments.md:212) corrects the interpretation; [boundary_bench.py:202](/home/muhit/ms/sm120-nulltest/fullmodel/boundary_bench.py:202) reports trace gaps, but verification and perturbation handling remain incomplete. |
| 6. H autocast and numerical claims | PARTIALLY-CLOSED | [padbatch.py:80](/home/muhit/ms/sm120-nulltest/fullmodel/padbatch.py:80) and [padbatch.py:148](/home/muhit/ms/sm120-nulltest/fullmodel/padbatch.py:148) fix affine dtypes; line 17 still falsely says the only difference is cuBLAS summation order. |
| 7. Capture ladder semantics | CLOSED | [capture.py:216](/home/muhit/ms/sm120-nulltest/fullmodel/capture.py:216) implements the explicit iSTFT boundary; [capture.py:236](/home/muhit/ms/sm120-nulltest/fullmodel/capture.py:236) restores stock forward for rung 3. |
| 8. Arm mechanics and tolerance classes | CLOSED | [fullarms.py:33](/home/muhit/ms/sm120-nulltest/fullmodel/fullarms.py:33) correctly separates compiled, captured, and tight/loose classes. |
| 9. SDR metrics and frozen clips | PARTIALLY-CLOSED | [configs/sdr_clips.json:20](/home/muhit/ms/sm120-nulltest/configs/sdr_clips.json:20) freezes eight distinct tracks and hashes; [sdr_gate.py:68](/home/muhit/ms/sm120-nulltest/fullmodel/sdr_gate.py:68) still mishandles degenerate metrics. |
| 10. PDL and dependency controls | PARTIALLY-CLOSED | [boundary_bench.py:84](/home/muhit/ms/sm120-nulltest/fullmodel/boundary_bench.py:84) places wait/load/store/trigger correctly, and line 296 fixes launch arithmetic; failed proofs still enter accepted fits—finding 24. |
| 11. Nested timing statistic | CLOSED | [timing.py:151](/home/muhit/ms/sm120-nulltest/fullmodel/timing.py:151) preserves samples and repeat spread; [collect_full.py:64](/home/muhit/ms/sm120-nulltest/fullmodel/collect_full.py:64) computes the cross-restart median. |
| 12. Matched repeats | PARTIALLY-CLOSED | [matched_run.py:104](/home/muhit/ms/sm120-nulltest/fullmodel/matched_run.py:104) alternates blocks, but qualification, shared compiler state, and block timestamps remain deficient—finding 19. |
| 13. Missing C1/B/E/F traces | PARTIALLY-CLOSED | [session2.sh:97](/home/muhit/ms/sm120-nulltest/scripts/session2.sh:97) schedules them; the compiled node-trace launcher fails—finding 28. |
| 14. Trace denominator | PARTIALLY-CLOSED | [session2.sh:53](/home/muhit/ms/sm120-nulltest/scripts/session2.sh:53) requests one block, but observed pass IDs/count are not checked against the requested count before attribution. |
| 15. Single-fix scope restriction | CLOSED | [prereg_amendments.md:70](/home/muhit/ms/sm120-nulltest/prereg_amendments.md:70) explicitly limits C1A/C1D to 352800 and distinguishes compilation failure from timing evidence. |
| 16. Session/C2 budget enforcement | PARTIALLY-CLOSED | [session1.sh:43](/home/muhit/ms/sm120-nulltest/scripts/session1.sh:43) and [session1.sh:167](/home/muhit/ms/sm120-nulltest/scripts/session1.sh:167) add deadline/C2 controls; qualification and session 2 remain outside equivalent enforcement. |
| 17. Exploratory comparison | CLOSED | [prereg_amendments.md:292](/home/muhit/ms/sm120-nulltest/prereg_amendments.md:292) resolves the ranking, qualification, and attribution wording; obtaining C2’s replay evidence remains outstanding. |

Several scrutinized implementation details are sound:

- **iSTFT boundary:** `hop*(frames−1)` gives exactly 4410, 22050, 88200, and 352800 for the registered centered transforms. The closure retains the graph, spectrogram, and captured window through `st`; eager iSTFT follows replay on the current stream. I found no dangling-buffer or stale-window defect in that path. Rung 3 removes the external forward.
- **H:** input gathering, band-major normalization, affine biases, GLU half placement, and final band-concatenation ordering match §1.1. Small CPU comparisons also confirmed affine output dtype preservation under autocast.
- **C3:** both registered stride forms are correct for every T. The vendored bodies match their stated original hashes. `verify_operator()` resets compiler state, but session 1 invokes it in its own process; it does not reset the later matched models.
- **Planted loose check:** the amplitude check allows `6e-6` around `0.05`, which accommodates fp32 rounding. The runner actually uses the forced loose comparator. The top-level JSON `tier` nevertheless remains incorrectly reported as `tight`.

18. **[BLOCKER] Session 1 crashes at its first reference process.**  
    [session1.sh:66](/home/muhit/ms/sm120-nulltest/scripts/session1.sh:66) sets `TORCH_LOGS=""` for eager/capture arms. With the installed pinned Torch 2.8.0+cu128, I reproduced:
    `AttributeError: 'dict' object has no attribute 'get_log_level_pairs'` during `import torch`. Session 2 has the same issue.

    **Minimal fix:** unset `TORCH_LOGS` for these arms; set it only when nonempty. Exercise the actual shell launcher in the shakedown.

19. **[BLOCKER] Matched results bypass required acceptance gates.**  
    [matched_run.py:100](/home/muhit/ms/sm120-nulltest/fullmodel/matched_run.py:100) proceeds whenever both builds return, even if validation, graph qualification, or settling failed. At [matched_run.py:117](/home/muhit/ms/sm120-nulltest/fullmodel/matched_run.py:117), eligibility checks validation alone. A mocked reproduction with **both graph gates FAIL and both settles capped** produced `status=VALID`, two eligible arms, and a matched delta.

    Two resident models are not inherently incorrect, but `counters.clear()` does not isolate Dynamo caches or the per-device cudagraph-tree manager. The first arm is validated before building the second, and the eventual alternating execution pattern is never jointly settled or checked for new recording. Blocks also lack wall timestamps. No SDR qualification is joined to the pair.

    **Minimal fix:** after both builds, settle the actual alternation, validate both again, snapshot compiler/tree state across every block, and require every numerical/graph/settling gate plus corresponding operating-point SDR qualification. Emit an explicit pair eligibility verdict and block timestamps. The existing median of paired differences is otherwise a reasonable statistic.

20. **[BLOCKER] Matrix headline acceptance is incomplete.**  
    [run_full_arm.py:196](/home/muhit/ms/sm120-nulltest/fullmodel/run_full_arm.py:196) does not require the registered repeat count, an applicable SDR qualification, or exclusion of attribution/profiled runs. [collect_full.py:61](/home/muhit/ms/sm120-nulltest/fullmodel/collect_full.py:61) accepts any three eligible-looking `rN` records.

    I reproduced `HEADLINE` from three **one-block runs with no SDR result**, and from a failed-SDR `r1` followed by eligible-looking `r2/r3/r4`.

    **Minimal fix:** independently enforce the registered restart set and complete timing protocol, matching variants and provenance, all numerical/compiler gates, and an explicit cell-level SDR PASS. A failed required restart cannot be discarded in favor of later tags.

21. **[BLOCKER] “Counters stable” does not establish recording stability.**  
    [timing.py:78](/home/muhit/ms/sm120-nulltest/fullmodel/timing.py:78) compares graph breaks, unique Dynamo graphs, and Inductor keys containing `skip`. It ignores recording changes such as `cudagraph_recorded_non_static_inputs`. I changed that counter from 1 to 2 with unchanged graph counts; `compile_stable()` returned `True`.

    [run_full_arm.py:134](/home/muhit/ms/sm120-nulltest/fullmodel/run_full_arm.py:134) snapshots before SDR, while SDR actually runs before timing despite the registered after-timing placement. The after-validation and final snapshots are not preserved in the result JSON.

    **Minimal fix:** record version-specific tree graph counts and recording counters, preserve all phase snapshots, and check the actual timed boundaries. Put SDR after timing as registered, or explicitly amend and qualify the changed lifecycle.

22. **[BLOCKER] Graph verification can certify uncaptured computation as fully graphed.**  
    [graph_verify.py:28](/home/muhit/ms/sm120-nulltest/fullmodel/graph_verify.py:28) classifies any `elementwise_kernel` as staging. This includes actual multiplication and other model arithmetic. Additional defects at [graph_verify.py:118](/home/muhit/ms/sm120-nulltest/fullmodel/graph_verify.py:118) and [graph_verify.py:157](/home/muhit/ms/sm120-nulltest/fullmodel/graph_verify.py:157):

    - Nonempty `"0"` graph IDs count as captured.
    - Graph-node IDs are counted as distinct graphs.
    - Missing launch correlation implicitly means captured.
    - A median hides uncaptured work in a minority of passes.
    - GPU-projected NVTX windows are also used for host APIs, clipping launch calls.

    Synthetic cases containing an eager multiplication, a zero graph ID, or one defective pass all returned `fully_graphed=True`.

    **Minimal fix:** use original CPU windows for APIs and projected windows for GPU events; correlate explicit graph identities; distinguish nodes from graphs; prove staging ownership; classify missing attribution as unresolved; require coverage for **every expected pass**.

23. **[BLOCKER] Provenance is not an enforced acceptance gate.**  
    [loader.py:52](/home/muhit/ms/sm120-nulltest/fullmodel/loader.py:52) accepts environment-selected checkpoint/config paths; [loader.py:69](/home/muhit/ms/sm120-nulltest/fullmodel/loader.py:69) checks only existence. The session preflight neither verifies those resolved identities nor records a complete immutable session manifest. References are unbound `.pt` files in a reusable directory.

    Consequently, references and candidates from the same unintended checkpoint/config can pass every numerical and SDR comparison. Running `setup.sh` does not verify arbitrary `CKPT_FILE` overrides or establish reference lineage.

    **Minimal fix:** verify resolved source commit/cleanliness, checkpoint/config hashes, precision and runtime pins before producing references. Bind references and runs to one immutable manifest containing code/registration/adapter identities and push evidence; reject mismatches.

24. **[BLOCKER] Boundary qualification failures still enter accepted results.**  
    [boundary_bench.py:296](/home/muhit/ms/sm120-nulltest/fullmodel/boundary_bench.py:296) is now correct: stream/PDL-stream execute **25N**, graph/PDL-graph execute **27N**. Capture records work without executing it. [PyTorch’s capture semantics](https://docs.pytorch.org/docs/2.8/notes/cuda.html#cuda-graphs)

    However, [boundary_bench.py:299](/home/muhit/ms/sm120-nulltest/fullmodel/boundary_bench.py:299) only records a boolean; a failed dependency proof retains `status="ok"` and enters the fit at line 313. The PTX test at line 107 accepts a kernel containing **only** `griddepcontrol.wait`; missing or false PTX verification also does not invalidate PDL results.

    **Minimal fix:** require exact counter equality and both expected PTX instructions before accepting a PDL case. Exclude failed/unresolved qualifications from fits and derived calibration values.

25. **[BLOCKER] Ideal costs can be numerically wrong while marked resolved.**  
    [ideal_costs.py:120](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:120) sums every compiler dump, including superseded variants. Using the existing C3 artifact as a parser fixture, the full dump produced **600 GEMM calls**, while its selected active dump produced **300**.

    Further, missing shapes/dtypes flow through `nbytes()` defaults into “wrapper access bytes”; each distinct SDPA kernel row receives the entire model’s twelve-call workload; and execution-rate selection always uses the fp16 rate. Per-pass traffic is also used as the L2 residency test, rather than the relevant working set.

    **Minimal fix:** select and verify the executed steady variants, reconcile call multiplicities with trace events, resolve dtype/access metadata strictly, and assign costs to actual call sites. Unresolved metadata must remain `UNRESOLVED`; it cannot receive default fp16 pricing.

26. **[BLOCKER] The attention benchmark’s model comparison uses the wrong layouts and an assumed backend.**  
    [attn_backend_bench.py:70](/home/muhit/ms/sm120-nulltest/fullmodel/attn_backend_bench.py:70) creates contiguous `(b,h,n,d)` Q/K/V. The existing [C3 generated call](/home/muhit/ms/axial-layout-scoping/evidence/extensions/c3/raw/phase1/active_dump/output_code.py:14318) shows Q/K strides `(410112,64,512,1)` and a separately strided V view—not those contiguous inputs. Line 95 hardcodes flash as the model denominator, although the repaired context also enables cuDNN. The purported fp32-math reference at line 75 does not force math.

    **Minimal fix:** reproduce actual Q/K/V strides, offsets, scale and dropout; determine the deployed backend from the corresponding trace; force math for that reference. Retain contiguous cases only as explicitly separate experiments.

27. **[BLOCKER] The critical-path estimator invents impossible dependencies.**  
    [timing_views.py:115](/home/muhit/ms/sm120-nulltest/fullmodel/timing_views.py:115) makes a side-stream kernel depend on a dominant-stream kernel merely because that kernel started earlier—even when they overlap.

    A CPU counterexample with main kernels spanning `[0,100]` and `[100,200]`, plus an independent side kernel `[50,60]`, returns **210** for a trace whose entire span and union are **200**. This cannot supply the registered dependency-critical-path input.

    **Minimal fix:** derive fork/join edges from actual event/graph dependencies. If unavailable, report the dependency path as unresolved. Enforce temporal consistency and `critical_path ≤ union ≤ span`.

28. **[MAJOR] Compiled node-trace commands never launch.**  
    [session2.sh:49](/home/muhit/ms/sm120-nulltest/scripts/session2.sh:49) expands `${dump:+TORCH_COMPILE_DEBUG=1 ...}` in command position. Bash does not reinterpret expanded words as assignment syntax. I reproduced `TORCH_COMPILE_DEBUG=1: command not found`.

    **Minimal fix:** construct an environment-argument array and invoke it through `env`. A successful `bash -n` does not test this behavior.

29. **[MAJOR] Ideal-cost postprocessing cannot import its vendored parser.**  
    [ideal_costs.py:35](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:35) imports `kernel_map`, whose [line 48](/home/muhit/ms/sm120-nulltest/fullmodel/attrib/kernel_map.py:48) imports missing `evidence_io`. Even importing `ideal_costs` fails with `ModuleNotFoundError`.

    **Minimal fix:** vendor the dependency with provenance, or provide a documented adapter that removes that dependency. Smoke-test all analysis entry points from an isolated checkout.

30. **[MAJOR] The advertised session deadline is incomplete.**  
    [session1.sh:138](/home/muhit/ms/sm120-nulltest/scripts/session1.sh:138) runs qualification without timeout or remaining-budget enforcement. Setup occurs before the script’s clock starts. [session2.sh:28](/home/muhit/ms/sm120-nulltest/scripts/session2.sh:28) has no session deadline/reserve and does not reliably write terminal JSONs after timeouts; its benchmarks are also uncapped.

    The C2 aggregate arithmetic and first-attempt no-retry mechanism are substantially correct. But requiring the entire per-process cap to remain before admission can strand almost 25 minutes of usable measurement time.

    **Minimal fix:** use a shared deadline-aware supervisor, including qualification and benchmarks, with terminal records. Cap a process by remaining usable time; preserve complete measurement requirements and mark unfinished work incomplete.

31. **[MAJOR] Nonfinite SDR semantics remain inconsistent.**  
    [sdr_gate.py:68](/home/muhit/ms/sm120-nulltest/fullmodel/sdr_gate.py:68) reports `0/0` as `"inf"` and raises on a zero numerator with positive denominator. A zero estimate therefore receives infinite SI-SDR in one degenerate case. Line 154 silently drops perfect-match infinities when averaging a mixture of finite and infinite fidelity scores.

    The registration’s “non-finite value in a paired row fails” wording also needs narrowing: literal application includes `SDR(A,A)` and would reject A itself.

    **Minimal fix:** define zero/undefined cases explicitly; check finiteness before arithmetic; apply rejection to the registered SDR-ref comparison terms; preserve expected fidelity infinities without excluding them from an alleged mean.

32. **[MAJOR] Required verification/postprocessing remains unscheduled.**  
    [session2.sh:97](/home/muhit/ms/sm120-nulltest/scripts/session2.sh:97) omits C2, although §6 requires replay-qualified C2 as a comparator. The perturbation runs at [session2.sh:108](/home/muhit/ms/sm120-nulltest/scripts/session2.sh:108) are never reduced into matched ratios or the >5% caveats. Boundary perturbation is likewise not computed. Importing the attribution parser for shapes does not emit the promised verified kernel-to-source mapping.

    **Minimal fix:** schedule C2 replay verification when available, and provide explicit CPU postprocessing for perturbation, coverage-qualified source attribution, and their acceptance flags before producing ledger claims.

For session 1, the current script contains **20 + 20 + 20 non-C2 matrix runs, then 6 C2 runs**, not 21 per round. Including two references, two planted checks, one qualification process, and ten matched processes gives **81 Python processes and 90 full-model loads**. C1A is attempted four times: once matched and three times in the matrix. [Schedule](/home/muhit/ms/sm120-nulltest/scripts/session1.sh:113)

After the blockers are fixed, this is a reasonable planning envelope—not a measured runtime prediction:

| Component | Allowance |
|---|---:|
| Pod setup/preparation | 10–20 min |
| 84 model loads before C2, assuming 10–20 s each | 14–28 min |
| Seven successful non-C2 cold arm/chunk variants, assuming 3–10 min each | 21–70 min |
| Four C1A failure attempts | 2–8 min |
| Matrix and matched steady timing before C2 | 15–23 min |
| Qualification, graph construction, settling, validation and SDR | 6–15 min |
| Entire C2 group, including its loads/timing | 10–40 min |
| Extraction reserve | 15 min |
| **Total** | **93–219 min** |

**Three hours is plausible on the favorable cache path; it is not a credible promise of completing the full schedule.** The supplied steady-state timings do not establish cold compilation, cache reuse, or joint matched-model settling costs.

Cut first, prospectively in the registration:

1. Stop repeating C1A after its expected deterministic failure has been recorded.
2. Reduce or defer the C2 allowance; report the resulting comparator restriction.
3. Defer complete lower-priority matrix cells while preserving A, C1/C3 matching, and the principal G/H measurements.
4. Prepare dependencies/audio off-pod and move CPU exports/postprocessing off-pod where practical.

Preserve numerical validation, operating-point SDR gates, five repeats, and three required restarts. Session 2 already has substantial work; it is not unallocated overflow capacity.

EXECUTION-READY: no