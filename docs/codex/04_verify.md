Reviewed `602a52a` plus the current `setup.sh` diff. Verification used CPU counterexamples, syntax checks, and existing traces; no new GPU measurements.

46. **[BLOCKER] The selected SDR clip manifest can bypass provenance binding.**  
    [provenance.py:101](/home/muhit/ms/sm120-nulltest/fullmodel/provenance.py:101) hashes the repository’s fixed clip manifest, but [run_full_arm.py:70](/home/muhit/ms/sm120-nulltest/fullmodel/run_full_arm.py:70) accepts `--clips-manifest`, which [line 176](/home/muhit/ms/sm120-nulltest/fullmodel/run_full_arm.py:176) uses without comparing its hash to the session manifest. References and r1 SDR can therefore use another eight-clip pack while carrying the registered manifest ID and valid lineage sidecars.

    **Minimal fix:** verify the selected clip manifest against `clips_manifest_sha256` before generating references or evaluating SDR.

47. **[BLOCKER] Collection still trusts eligibility flags and omits the matched SDR variant join.**  
    [collect_full.py:81](/home/muhit/ms/sm120-nulltest/fullmodel/collect_full.py:81) and [line 140](/home/muhit/ms/sm120-nulltest/fullmodel/collect_full.py:140) trust stored eligibility booleans without independently checking protocol, validation, or reference lineage. CPU records marked `INVALID`, with validation `FAIL` but `headline_eligible=True`, produced `HEADLINE`.

    Separately, [line 149](/home/muhit/ms/sm120-nulltest/fullmodel/collect_full.py:149) compares matched SDR manifests but never compares each arm’s matched variant against its r1 variant. Three matched records with different variants from their SDR records produced a pair `HEADLINE`. [matched_run.py:206](/home/muhit/ms/sm120-nulltest/fullmodel/matched_run.py:206) also omits stream count and uses a different variant format from the single-arm runner.

    **Minimal fix:** share a versioned qualification validator and canonical variant representation; persist checked reference hashes; validate required records and join each matched arm directly to r1 by manifest, operating point, and complete variant.

48. **[MAJOR] The recording observer compares counts, not graph identities.**  
    [timing.py:119](/home/muhit/ms/sm120-nulltest/fullmodel/timing.py:119) collects node identities for traversal but discards them at [line 128](/home/muhit/ms/sm120-nulltest/fullmodel/timing.py:128). [recording_stable():140](/home/muhit/ms/sm120-nulltest/fullmodel/timing.py:140) compares only counters, root count, and node count. Replacing the nodes with different nodes of equal counts returned `True` in a CPU counterexample.

    **The observer does cover both arms:** Torch 2.8 uses the shared device container, and ordinary re-recording appends a node. An arm-A node addition during arm B’s block is detected by [matched_run.py:178](/home/muhit/ms/sm120-nulltest/fullmodel/matched_run.py:178). The remaining blind spot is identity replacement/reset with unchanged counts, not ordinary cross-arm node growth. [Pinned Torch implementation](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/_inductor/cudagraph_trees.py).

    **Minimal fix:** retain and compare manager identity and recorded node/graph identities.

49. **[BLOCKER] Missing export qualification still permits `FULLY_GRAPHED`.**  
    [graph_verify.py:168](/home/muhit/ms/sm120-nulltest/fullmodel/graph_verify.py:168) sets `exports_ok=None` when the export record is missing; [line 172](/home/muhit/ms/sm120-nulltest/fullmodel/graph_verify.py:172) accepts anything except explicit `False`. A 50-pass CPU fixture without `_exports.json` returned `FULLY_GRAPHED`. The expected pass count also defaults to unchecked.

    **Minimal fix:** require an affirmative node-export qualification and the registered pass count. Missing qualification must produce an unresolved verdict.

50. **[BLOCKER] “Interchangeable” compiled graphs can contain different executed programs.**  
    [ideal_costs.py:82](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:82) includes Triton names and pointer metadata in `graph_signature()`, but excludes kernel bodies, access sites, scalar iteration arguments, and extern calls. [line 114](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:114) consequently collapses distinct programs.

    A CPU counterexample with identical names/shapes/dtypes but different load counts, iteration spaces, and GEMM work was reported as a unique decomposition; the first graph supplied the pricing. Thus repeated C1 kernel names are **not soundly disambiguated**. Additionally, [perturbation.py:123](/home/muhit/ms/sm120-nulltest/fullmodel/perturbation.py:123) overwrites source mappings when selected graphs reuse a name.

    **Minimal fix:** distinguish graphs using complete pricing semantics and kernel-body hashes; reconcile complete call sequences. Preserve graph-qualified source mappings or mark collisions unresolved.

51. **[BLOCKER] Ideal-cost measurements are not reconciled to the validated pass windows.**  
    [ideal_costs.py:223](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:223) reads aggregate kernel totals, while variant selection independently examines windowed trace rows. Nothing reconciles their multiplicities or durations before [line 296](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:296) publishes per-pass values.

    A fixture with 50 windowed launches but 100 summary instances returned selection `ok`, pricing one call against two calls’ measured time. A 49-pass fixture correctly rejected variant selection but still emitted SDPA measured time divided by 50 and numerical SDPA ideals at [line 330](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:330). Eager arms bypass the pass check entirely.

    **Minimal fix:** derive measurements from validated windows, reconcile the aggregate export, and gate every derived class/row on successful coverage validation.

52. **[BLOCKER] Iteration-space parsing can price a real kernel at zero bytes.**  
    [ideal_costs.py:254](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:254) multiplies every numeric literal in the call text, including view dimensions, strides, offsets, and keyword expressions.

    A parsed call containing `reinterpret_tensor(..., 0)` and a 16-element fp16 copy produced selection `ok`, `bytes_per_pass=0`, and `recoverable_share_hbm_model=1.0`. The registered traffic for that copy is 64 bytes. The parser also truncates call text at [kernel_map.py:280](/home/muhit/ms/sm120-nulltest/fullmodel/attrib/kernel_map.py:280).

    **Minimal fix:** parse full calls structurally and bind iteration parameters to the kernel signature. Unknown call forms must remain unresolved.

53. **[MAJOR] The L2 footprint is still not the unique resident footprint.**  
    [ideal_costs.py:276](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:276) uses `max(b_read, b_write)`. Disjoint 80 MiB input and output buffers therefore qualify for a 96 MiB cache, despite requiring 160 MiB. This counterexample received an L2 ideal.

    **Minimal fix:** compute the union of resident storage regions, accounting for aliases. A lower bound on footprint cannot establish that the working set fits.

54. **[BLOCKER] Boundary calibration accepts missing qualification.**  
    [boundary_bench.py:210](/home/muhit/ms/sm120-nulltest/fullmodel/boundary_bench.py:210) rejects a failed case only **when a case exists**. Missing `--traced-json`, a missing file, or an absent case bypasses qualification. A 200-kernel `trivial/graph/N10` fixture without qualification returned `status="ok"` and a serial boundary estimate with `case_status=null`.

    **Minimal fix:** require the matching qualification case and its successful proof/PTX verdict before deriving any calibration value.

55. **[BLOCKER] Perturbation still accepts unresolved recording state and retains disqualified ratios.**  
    [_qualified():53](/home/muhit/ms/sm120-nulltest/fullmodel/perturbation.py:53) rejects only `counters_stable=False`; compiled `None` passes. It also does not require an effective variant or check the actual iteration-list length. A CPU fixture with an unavailable observer, absent variants, and one recorded iteration labelled as 50 produced ordinary ratios and `caveat_required=False`.

    Furthermore, an invalid unprofiled baseline adds a problem at [line 64](/home/muhit/ms/sm120-nulltest/fullmodel/perturbation.py:64), but [line 82](/home/muhit/ms/sm120-nulltest/fullmodel/perturbation.py:82) still computes ratios against it. The result becomes `UNRESOLVED` while retaining numerical ratios and inflation factor.

    **Minimal fix:** require affirmative compiled stability, complete block payloads, and explicit matching variants; clear all ratios when any prerequisite fails.

56. **[BLOCKER] The completeness audit does not match §5 and can certify missing or malformed evidence.**  
    [completeness.py:12](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:12), [line 49](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:49), and [line 59](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:59):

    - **Session 1:** the 21 matrix cells and three repeated pair families match. The once-only `(C1,C1A)` attempt is missing. Planted checks and C3 qualification are checked only for existence, without parsing outcomes or checking all four frame counts. The C2 `UNTESTED` exemption introduces a session-completeness allowance absent from §5.
    - **Session 2:** the eight sweep cells and 19 trace entries match. The audit does not require the 18 individual perturbation runs, graph-verification outputs, ideal-cost outputs, F/G timing views, or source-map coverage. Trace acceptance checks an export boolean without checking the traced run or its 50 passes. The four final benchmark/reduction files are checked only for existence.
    - Neither audit validates the session’s reference lineage, clock evidence, or auxiliary failure records.

    CPU fixtures returned `COMPLETE` for session 1 with no C1A attempt and malformed qualification files, and for session 2 without the individual measurements/reductions above.

    **Minimal fix:** enumerate the registered attempts and products explicitly, validate their contents and lineage, and distinguish permitted terminal unavailability from missing/failed work.

57. **[MAJOR] Failed postprocessing and unbounded stages can still end with successful-looking session output.**  
    Both scripts retain unconditional failure absorption: [session1.sh:137](/home/muhit/ms/sm120-nulltest/scripts/session1.sh:137), [session2.sh:121](/home/muhit/ms/sm120-nulltest/scripts/session2.sh:121). Collection and completeness can fail while execution reaches `SESSION… DONE`. Graph verification, timing views, and ideal-cost failures also continue without being required by the completeness audit.

    [session_lib.sh:117](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:117) accepts nonempty export files after an export failure, permitting stale exports to appear complete. Exports and reductions remain outside `cap_for`/`timeout`, including [session2.sh:60](/home/muhit/ms/sm120-nulltest/scripts/session2.sh:60) and [line 109](/home/muhit/ms/sm120-nulltest/scripts/session2.sh:109). Moving setup before the clock at [registration:334](/home/muhit/ms/sm120-nulltest/prereg_amendments.md:334) does not close that deadline gap.

    **Minimal fix:** supervise every on-pod stage, invalidate stale outputs before attempts, and finalize an explicit failed/incomplete session record when collection or audit fails. Alternatively, defer reductions off-pod explicitly.

58. **[MAJOR] Timing views count memcpy/memset operations as kernels.**  
    [timing_views.py:118](/home/muhit/ms/sm120-nulltest/fullmodel/timing_views.py:118) reads every GPU-trace row; [line 128](/home/muhit/ms/sm120-nulltest/fullmodel/timing_views.py:128) labels all of them kernels and includes them in summed duration and critical-path calculations.

    A fixture containing one 10 ns kernel and one 90 ns memcpy reported two kernels and 100 ns of summed kernel duration.

    **Minimal fix:** separate kernels from memops consistently with `graph_verify.py`; label any additional all-GPU-activity view explicitly.

59. **[MAJOR] The new pod driver can authorize termination after failed extraction.**  
    [pod_run.sh:47](/home/muhit/ms/sm120-nulltest/scripts/pod_run.sh:47) does not check the extraction pipeline before writing the success marker. [Line 53](/home/muhit/ms/sm120-nulltest/scripts/pod_run.sh:53) accepts any existing marker, without binding it to the current pod. Failed transfer/extraction or a marker from a previous pod can therefore satisfy the evidence-extraction prerequisite.

    **Minimal fix:** require successful transfer and archive validation; bind the extraction receipt to the exact pod ID and extracted evidence.

60. **[MINOR] The attribution parser’s vendoring certificate is false.**  
    [kernel_map.py:1](/home/muhit/ms/sm120-nulltest/fullmodel/attrib/kernel_map.py:1) still claims verbatim vendoring with hash `c1220608…`. Its current body hash after the two header lines is `d56caf5b…`.

    **Minimal fix:** label it as an adaptation and record the upstream provenance and current hash accurately.

For the explicit lineage question: with a valid generated manifest, the current single-arm runner checks reference sidecars at [timing.py:292](/home/muhit/ms/sm120-nulltest/fullmodel/timing.py:292) and [sdr_gate.py:148](/home/muhit/ms/sm120-nulltest/fullmodel/sdr_gate.py:148); missing/mismatched references invalidate the run. `--no-manifest` and reference-saving runs cannot qualify. The remaining publication bypass is collection’s unchecked flags. Mixed r1/r2/r3 manifest IDs were rejected in my counterexample.

For Nsight memops: the saved [C3 SQLite trace](/home/muhit/ms/axial-layout-scoping/evidence/extensions/c3/raw/phase1/trace_c3.sqlite), identified as [Nsight 2026.5.1](/home/muhit/ms/axial-layout-scoping/evidence/extensions/c3/raw/phase1/env.json:12), contains replayed memcpy and memset events with graph-node IDs whose CorrIds join to `cudaGraphLaunch`. This supports the classification for that exporter. Memop classification does **not** participate in the `FULLY_GRAPHED` predicate at [graph_verify.py:172](/home/muhit/ms/sm120-nulltest/fullmodel/graph_verify.py:172).

| Prior finding | Closure | Code evidence |
|---|---|---|
| 33 | **PARTIAL** | Runtime pins, expanded hashes, identity comparisons, and artifact checks implemented: [provenance.py:55](/home/muhit/ms/sm120-nulltest/fullmodel/provenance.py:55), [120](/home/muhit/ms/sm120-nulltest/fullmodel/provenance.py:120), [142](/home/muhit/ms/sm120-nulltest/fullmodel/provenance.py:142). Selected clip-manifest bypass remains—46. |
| 34 | **PARTIAL** | Exact 5×50 and ANVTX exclusion: [run_full_arm.py:211](/home/muhit/ms/sm120-nulltest/fullmodel/run_full_arm.py:211). Required restart/manifest/r1 joins: [collect_full.py:59](/home/muhit/ms/sm120-nulltest/fullmodel/collect_full.py:59). Independent qualification remains absent—47. |
| 35 | **PARTIAL** | Joint, validation, and block snapshots plus protocol checks: [matched_run.py:60](/home/muhit/ms/sm120-nulltest/fullmodel/matched_run.py:60), [117](/home/muhit/ms/sm120-nulltest/fullmodel/matched_run.py:117), [152](/home/muhit/ms/sm120-nulltest/fullmodel/matched_run.py:152). Matched SDR variant binding remains incomplete—47. |
| 36 | **PARTIAL** | Descendant traversal and missing-observer rejection implemented: [timing.py:95](/home/muhit/ms/sm120-nulltest/fullmodel/timing.py:95), [run_full_arm.py:189](/home/muhit/ms/sm120-nulltest/fullmodel/run_full_arm.py:189). Identity comparison remains missing—48. |
| 37 | **PARTIAL** | Copy-functor restriction, contiguous IDs, positive computation: [graph_verify.py:30](/home/muhit/ms/sm120-nulltest/fullmodel/graph_verify.py:30), [76](/home/muhit/ms/sm120-nulltest/fullmodel/graph_verify.py:76), [174](/home/muhit/ms/sm120-nulltest/fullmodel/graph_verify.py:174). Missing exports still pass—49. |
| 38 | **PARTIAL** | Ordered decomposition and pass-count rejection implemented: [ideal_costs.py:64](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:64), [100](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:100). Semantic equivalence, normalization, and byte extraction remain unsound—50–52. |
| 39 | **CLOSED** | Execution-family filtering, per-axis/per-chunk attribution, unresolved ratios: [attn_backend_bench.py:85](/home/muhit/ms/sm120-nulltest/fullmodel/attn_backend_bench.py:85), [100](/home/muhit/ms/sm120-nulltest/fullmodel/attn_backend_bench.py:100), [179](/home/muhit/ms/sm120-nulltest/fullmodel/attn_backend_bench.py:179). |
| 40 | **PARTIAL** | Present-case rejection and exact kernel counts implemented: [boundary_bench.py:207](/home/muhit/ms/sm120-nulltest/fullmodel/boundary_bench.py:207). Missing qualification still passes—54. |
| 41 | **PARTIAL** | Run/protocol/manifest checks added: [perturbation.py:39](/home/muhit/ms/sm120-nulltest/fullmodel/perturbation.py:39). Unresolved stability and retained disqualified ratios remain—55. |
| 42 | **PARTIAL** | Per-call ideals are summed: [ideal_costs.py:306](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:306). Resident footprint remains incorrect—53. |
| 43 | **PARTIAL** | Auxiliary terminal JSON and audit added: [session_lib.sh:133](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:133), [completeness.py:19](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:19). Coverage, failure propagation, and deadline gaps remain—56–57. |
| 44 | **CLOSED** | Qualification writes its own JSON; exact frame set checked: [session1.sh:82](/home/muhit/ms/sm120-nulltest/scripts/session1.sh:82), [91](/home/muhit/ms/sm120-nulltest/scripts/session1.sh:91). Supervisor banner goes to stderr: [session_lib.sh:143](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:143). |
| 45 | **CLOSED** | Algebraic-equivalence wording accurately allows reduction/fusion differences: [padbatch.py:17](/home/muhit/ms/sm120-nulltest/fullmodel/padbatch.py:17). |

EXECUTION-READY: no