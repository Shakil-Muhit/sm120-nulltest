BLOCKERS: 9

The tree is **not ready for approval as the measurement registration or for renting the measurement pod**.

Reviewed HEAD `4b9f74e` and every requested file. `graph_verify.py` changed during review; I re-read and tested that current version. It remains uncommitted. Verification included syntax checks, CPU counterexamples, saved compiler artifacts, and existing trace/SQLite data. I performed no new GPU measurements.

| Finding | Status | Closing implementation and remaining issue |
|---|---|---|
| 18. Empty `TORCH_LOGS` | CLOSED | [session_lib.sh:50](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:50) only adds nonempty logging settings. Default eager launch leaves it unset. |
| 19. Matched acceptance | PARTIALLY-CLOSED | [matched_run.py:118](/home/muhit/ms/sm120-nulltest/fullmodel/matched_run.py:118) checks initial gates; joint settling and timestamps exist. Protocol and lifecycle gaps remain—35. |
| 20. Headline acceptance | PARTIALLY-CLOSED | [run_full_arm.py:207](/home/muhit/ms/sm120-nulltest/fullmodel/run_full_arm.py:207), [collect_full.py:53](/home/muhit/ms/sm120-nulltest/fullmodel/collect_full.py:53) enforce minimum timing and required restart names. Qualification joins remain leaky—34. |
| 21. Recording stability | PARTIALLY-CLOSED | [timing.py:78](/home/muhit/ms/sm120-nulltest/fullmodel/timing.py:78) now checks all cudagraph counters; [run_full_arm.py:177](/home/muhit/ms/sm120-nulltest/fullmodel/run_full_arm.py:177) puts SDR after timing. Actual recording changes can still escape—36. |
| 22. Graph verification | PARTIALLY-CLOSED | [graph_verify.py:92](/home/muhit/ms/sm120-nulltest/fullmodel/graph_verify.py:92) implements CorrId membership and separate CPU/GPU windows; every-pass checks exist. False positives remain—37. |
| 23. Provenance | PARTIALLY-CLOSED | [provenance.py:51](/home/muhit/ms/sm120-nulltest/fullmodel/provenance.py:51) verifies model pins; manifest/reference binding exists. Manifest enforcement is incomplete—33. |
| 24. Boundary qualification | PARTIALLY-CLOSED | [boundary_bench.py:303](/home/muhit/ms/sm120-nulltest/fullmodel/boundary_bench.py:303), [boundary_bench.py:320](/home/muhit/ms/sm120-nulltest/fullmodel/boundary_bench.py:320) exclude failed proofs/PTX from fits. Trace-derived calibration bypasses qualification—40. |
| 25. Ideal costs | PARTIALLY-CLOSED | [ideal_costs.py:76](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:76) selects variants and rejects unresolved metadata. Selection, denominator and L2 defects remain—38, 42. |
| 26. Attention benchmark | PARTIALLY-CLOSED | [attn_backend_bench.py:40](/home/muhit/ms/sm120-nulltest/fullmodel/attn_backend_bench.py:40) fixes layouts; line 140 forces fp32 math. Deployed-backend attribution remains wrong—39. |
| 27. Impossible critical path | CLOSED | [timing_views.py:91](/home/muhit/ms/sm120-nulltest/fullmodel/timing_views.py:91) computes the explicitly registered observed-order bound and asserts its inequalities. |
| 28. Compiled trace launcher | CLOSED | [session_lib.sh:97](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:97) constructs an array and launches through `env`. |
| 29. Missing parser dependency | CLOSED | [evidence_io.py:1](/home/muhit/ms/sm120-nulltest/fullmodel/attrib/evidence_io.py:1) vendors the dependency. Imports succeed; all three attribution files’ body hashes match their headers. |
| 30. Session deadline | PARTIALLY-CLOSED | [session_lib.sh:24](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:24), [session_lib.sh:131](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:131) bound measurement/auxiliary commands. Setup and postprocessing remain outside enforcement—43. |
| 31. Nonfinite SDR | CLOSED | [sdr_gate.py:68](/home/muhit/ms/sm120-nulltest/fullmodel/sdr_gate.py:68), [sdr_gate.py:166](/home/muhit/ms/sm120-nulltest/fullmodel/sdr_gate.py:166), [collect_full.py:41](/home/muhit/ms/sm120-nulltest/fullmodel/collect_full.py:41) propagate ungateable SDR-ref terms as FAIL without crashing collection. Fidelity infinities are counted separately. |
| 32. Missing postprocessing | PARTIALLY-CLOSED | [session2.sh:66](/home/muhit/ms/sm120-nulltest/scripts/session2.sh:66), [session2.sh:114](/home/muhit/ms/sm120-nulltest/scripts/session2.sh:114) schedule C2 and reductions. Perturbation acceptance remains unchecked—41; source mapping inherits 38. |

Several specific checks passed:

- Environment-array expansion and cap arithmetic work. A 1,500-second requested cap with 300 usable seconds becomes 300; exhausted time is rejected before launching.
- Manifest code hashing excludes `evidence/`, so the session’s generated forwards, dumps and results do **not** invalidate its code hash.
- CorrId membership is supported by the saved C3 trace: **57,100 kernels correlate to 50 graph launches—1,142 kernels per replay**. The current verifier also handles the saved C1 trace’s **96 graph launches per pass** without unresolved kernels. This agrees with [CUPTI’s API/activity correlation semantics](https://docs.nvidia.com/cupti/13.3.0/api/group__CUPTI__ACTIVITY__API.html).
- The attention layouts match the saved C3 calls: time-axis Q/K `(410112,64,512,1)`, V `(1230336,64,1536,1)`, offset `1024`; frequency-axis V `(92160,64,1536,1)`.
- The previous critical-path counterexample now returns **200**, with union **200**. The quadratic implementation took approximately **1.5 seconds for 1,200 kernels × 50 passes** here.
- Perturbation filenames match the session’s filenames. The PDL counter arithmetic and both-instruction PTX check are correct.

33. **[BLOCKER] The manifest does not enforce its complete identity.**  
    [provenance.py:70](/home/muhit/ms/sm120-nulltest/fullmodel/provenance.py:70), [provenance.py:104](/home/muhit/ms/sm120-nulltest/fullmodel/provenance.py:104).

    The 41 hashed files omit root-level measuring dependencies, including `frontend_rms_fused.py`. `load_manifest()` compares only code hashes and HEAD; it does not compare registration/clip hashes, recorded provenance, or runtime pins. Changing an omitted dependency or registration under the same HEAD can pass. My mocked manifest with inconsistent registration, clips, environment and provenance was accepted.

    **Minimal fix:** hash the complete measuring dependency set and compare all registered identity fields and runtime invariants. Bind reference artifacts individually; the directory-wide marker at [timing.py:217](/home/muhit/ms/sm120-nulltest/fullmodel/timing.py:217) must not silently relabel existing references when rewritten.

34. **[BLOCKER] Headline SDR qualification is joined by arm/chunk alone.**  
    [collect_full.py:36](/home/muhit/ms/sm120-nulltest/fullmodel/collect_full.py:36), [collect_full.py:66](/home/muhit/ms/sm120-nulltest/fullmodel/collect_full.py:66), [run_full_arm.py:207](/home/muhit/ms/sm120-nulltest/fullmodel/run_full_arm.py:207).

    Collection accepts SDR from any tag without checking its manifest, effective variant, or embedded operating point. Required restarts need not share a manifest. I reproduced `HEADLINE` with mixed restart manifests and an unrelated SDR PASS.

    The runner also accepts ANVTX without `--nvtx-pass` as headline-eligible, and accepts **6×51** under the registered **5×50** statistic.

    **Minimal fix:** independently validate required records, exact protocol, attribution exclusions, complete variant identity—including stream count—and manifest/reference lineage. Join SDR from the registered qualification record for that same deployment.

35. **[BLOCKER] Matched acceptance still misses protocol and lifecycle failures.**  
    [matched_run.py:60](/home/muhit/ms/sm120-nulltest/fullmodel/matched_run.py:60), [matched_run.py:131](/home/muhit/ms/sm120-nulltest/fullmodel/matched_run.py:131), [matched_run.py:144](/home/muhit/ms/sm120-nulltest/fullmodel/matched_run.py:144), [collect_full.py:114](/home/muhit/ms/sm120-nulltest/fullmodel/collect_full.py:114).

    Joint settling can re-record the arm built first; its first-round changes precede the first snapshot, and initial per-arm graph gates are never refreshed. Validation changes are also discarded before the timing baseline. Joint/block snapshots are not actually preserved.

    Reproductions returned `pair_eligible=True` for:

    - **1 iteration × 2 repeats**;
    - a new zero-break-arm violation during the first joint round;
    - a new graph break during validation.

    Collection additionally accepts available matched restarts without requiring the registered three, and joins SDR without matching lineage/variant.

    **Minimal fix:** enforce 5×50 and the required matched restart set; requalify per-arm graph gates after joint settling; compare and preserve joint→validation→block snapshots; bind the SDR join.

36. **[BLOCKER] Stable counters still do not prove stable recordings.**  
    [timing.py:88](/home/muhit/ms/sm120-nulltest/fullmodel/timing.py:88), [timing.py:93](/home/muhit/ms/sm120-nulltest/fullmodel/timing.py:93), [run_full_arm.py:190](/home/muhit/ms/sm120-nulltest/fullmodel/run_full_arm.py:190).

    In pinned Torch 2.8, the recording counter increments by the **number of non-static inputs**. Recording a graph with zero such inputs adds zero. The tree snapshot records roots/functions, not all recorded nodes, and never participates in acceptance.

    **Minimal fix:** snapshot actual recorded graph identities/counts, including descendants, and compare them at every required phase/block boundary. A missing recording observer must remain unresolved.

37. **[BLOCKER] Graph verification still certifies incomplete or uncaptured execution.**  
    [graph_verify.py:26](/home/muhit/ms/sm120-nulltest/fullmodel/graph_verify.py:26), [graph_verify.py:68](/home/muhit/ms/sm120-nulltest/fullmodel/graph_verify.py:68), [graph_verify.py:158](/home/muhit/ms/sm120-nulltest/fullmodel/graph_verify.py:158).

    The current version returns `FULLY_GRAPHED` with a graph-launch API but **zero GPU kernels**. Generic `multi_tensor_apply`/`foreach` matching also excludes arithmetic: an eager `multi_tensor_apply_kernel<Add<float>>` passed my reproduction. Pass IDs are discarded; incorrect IDs can satisfy the count.

    **Minimal fix:** require complete node-level exports, valid expected pass IDs and positive captured computation; recognize the actual `Copy<…>` staging operation with ownership evidence.

    CorrId membership itself is sound for multiple launches. However, distinct CorrIds count **launch instances**, not distinct graph identities. Batched launches must retain launch-to-pass ownership; they cannot establish distinct-graph counts from CorrId alone.

38. **[BLOCKER] Ideal-cost “resolved” selection does not establish the executed program or denominator.**  
    [ideal_costs.py:63](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:63), [ideal_costs.py:76](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:76), [ideal_costs.py:151](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:151).

    Selection uses an unordered Triton multiset and greedy subset search. It cannot verify graph order, distinguish identical Triton inventories with different extern work, or account for extern-only segments. A synthetic two-segment program silently lost its extern-only segment while selection reported an exact match. Greedy selection also misses valid combinations.

    Expected pass count is never checked here. A **49-pass** fixture with `--passes 50` produced `variant_selection.status="ok"` and divided measurements by 50. Running `graph_verify` earlier does not gate this reduction.

    **Minimal fix:** establish a unique ordered executed variant sequence with complete call multiplicities; reject ambiguity, missing segments and pass-count mismatches. Reconcile timing totals against validated windows before emitting costs.

39. **[BLOCKER] Attention’s deployed-backend denominator remains misattributed.**  
    [attn_backend_bench.py:99](/home/muhit/ms/sm120-nulltest/fullmodel/attn_backend_bench.py:99), [attn_backend_bench.py:159](/home/muhit/ms/sm120-nulltest/fullmodel/attn_backend_bench.py:159), [session2.sh:83](/home/muhit/ms/sm120-nulltest/scripts/session2.sh:83).

    The detector counts Triton preprocessing names containing `flash` as flash-attention execution. A fixture containing only cuDNN attention plus a flash-named fused rotary kernel selected **flash**. One majority backend from the 352800 trace is then used for both axes and both operating points.

    **Minimal fix:** identify actual attention executions per `(chunk, axis)`, exclude preprocessing kernels, and leave ambiguous/missing deployments unresolved. Use each corresponding backend for its ratio.

40. **[BLOCKER] Boundary trace analysis bypasses the repaired qualification gates.**  
    [boundary_bench.py:165](/home/muhit/ms/sm120-nulltest/fullmodel/boundary_bench.py:165), [boundary_bench.py:198](/home/muhit/ms/sm120-nulltest/fullmodel/boundary_bench.py:198).

    Fits now reject failed proofs/PTX, but `analyze_trace()` reads neither qualification JSON nor expected launch counts. It emits a boundary estimate from failed or truncated cases. A nominal `N10` case containing only **two kernels**, instead of 20×10, was accepted as a serial boundary estimate.

    **Minimal fix:** join the traced benchmark’s case status, dependency/PTX verdicts and expected event counts before deriving calibration values.

41. **[BLOCKER] Perturbation publishes ratios from invalid runs and treats missing qualification as no caveat.**  
    [perturbation.py:33](/home/muhit/ms/sm120-nulltest/fullmodel/perturbation.py:33), [perturbation.py:55](/home/muhit/ms/sm120-nulltest/fullmodel/perturbation.py:55).

    Only block timing is read. INVALID runs with failed validation produced ordinary ratios and `caveat_required=False`; missing traces also produce `False`. Manifest, variant, settling, stability and matching block lengths are unchecked.

    **Minimal fix:** validate the three runs under a dedicated one-block perturbation protocol. Require matching deployment/lineage; emit `UNRESOLVED` when qualification or measurements are missing.

42. **[MAJOR] The L2 test still uses aggregate traffic.**  
    [ideal_costs.py:129](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:129), [ideal_costs.py:210](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:210).

    Although the caller checks `per_call_max`, `ideal_us()` rechecks summed per-pass bytes against capacity. Ten independent 20 MB calls consequently receive no L2 result for a 96 MiB cache. Read-plus-write traffic also differs from resident footprint, particularly for aliases/in-place accesses.

    **Minimal fix:** determine unique resident footprint separately and compute/sum per-call ideals under the registered residency assumption.

43. **[MAJOR] The three-hour supervisor remains incomplete.**  
    [pod_up.sh:55](/home/muhit/ms/sm120-nulltest/scripts/pod_up.sh:55), [session_lib.sh:22](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:22), [session_lib.sh:117](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:117), [session2.sh:60](/home/muhit/ms/sm120-nulltest/scripts/session2.sh:60).

    Setup precedes the deadline clock. Exports and CPU reductions run without deadline enforcement. Auxiliary failures leave marker files rather than terminal JSON, and no mandatory-cell audit produces the promised whole-session `INCOMPLETE` verdict.

    **Minimal fix:** carry an absolute pod/session deadline through setup and every stage, or explicitly defer CPU work off-pod; emit terminal records and session completeness mechanically.

44. **[MAJOR] Operator qualification’s `.json` is not JSON.**  
    [session1.sh:82](/home/muhit/ms/sm120-nulltest/scripts/session1.sh:82), [session_lib.sh:135](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:135).

    Redirecting the entire `run_aux` call captures its `--- name … cap=… ---` banner before the JSON. The subsequent grep accepts this malformed artifact.

    **Minimal fix:** send supervisor messages to stderr or have the qualification program write its own JSON file.

45. **[MINOR] H’s numerical explanation remains false.**  
    [padbatch.py:17](/home/muhit/ms/sm120-nulltest/fullmodel/padbatch.py:17) still claims the only difference is cuBLAS summation order.

    **Minimal fix:** use the registration’s algebraic-equivalence wording, which correctly allows changed reduction trees and compiler fusion.

The registration now explicitly states the headline rule, matched acceptance, stability lifecycle, provenance binding, narrowed SDR semantics, C1A once, C2 tracing at 4410 only, observed-order dependency model, and both attention layouts. **It still does not fully match enforcement:**

- **Headline/matched rules:** findings 34–36 remain violations. Accepting SDR from arbitrary tags—and allowing an arbitrary tag’s FAIL to veto the cell—is broader than the registered r1 qualification lifecycle.
- **Provenance:** complete measuring-file identity and mismatch rejection are promised but not enforced. Runtime pins are recorded rather than checked. The required pre-measurement push-time evidence is not part of an enforced preflight.
- **Graph/attribution:** expected 50-pass coverage is not enforced before ideal-cost/source-map reduction. “Ordered combination” is unsupported by the implemented multiset selector.
- **Attention:** pooled majority-backend selection across axes/operating points is unregistered. The registered `ASSUMED` fallback also conflicts with the plan’s rule that unattributable inputs remain `UNRESOLVED`.
- **Ideal costs:** class-level SDPA/GEMM substitutes and zero-FLOP fused pricing are not explicit amendments to the plan’s per-kernel FLOPs/bytes requirement. Their scope and unresolved portions need registration.
- **Budget/calibration:** the hard deadline and qualification requirements are stronger than the actual supervision and trace-calibration gates.
- **SDR wording:** [prereg_amendments.md:146](/home/muhit/ms/sm120-nulltest/prereg_amendments.md:146) still broadly rejects nonfinite paired values; lines 159–164 correctly narrow rejection to SDR-ref terms. Remove the contradictory earlier sentence.

C1A’s single attempt, C2’s 4410-only trace, the revised dependency bound, and the stated attention strides are implemented consistently. A successful GPU smoke would establish launcher viability; it would not close the demonstrated acceptance leaks.

EXECUTION-READY: no