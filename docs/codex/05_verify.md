Reviewed **`b039272`** against the registration and plan using code inspection, CPU counterexamples, and syntax checks. No new GPU measurements. Line numbers below refer to that commit. The checkout advanced to `f99c672` during review; that later change addresses the reference-set defect in finding 63.

61. **[BLOCKER] Independent qualification still trusts contradictory nested verdicts.**  
    [qualify.py:66](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:66), [qualify.py:80](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:80), [qualify.py:107](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:107). Validation checks `status`, not the required seeds, per-pass results, finiteness, or registered tolerance. SDR accepts `clips=[]` because `all([])` succeeds; it never recomputes the threshold. Recording stability and graph-gate verdicts are also accepted without checking snapshots/counts. CPU records with an empty SDR payload and mean ΔSDR **−99 dB** produced `HEADLINE`; contradictory failed validation and unavailable-observer payloads also passed.

    **Minimal fix:** validate required payloads and identities, derive SDR from eight finite paired rows, enforce the registered tolerance and graph-break rules, and recompute recording stability from complete snapshots.

62. **[BLOCKER] Collection resurrects matched pairs rejected during validation.**  
    [matched_run.py:167](/home/muhit/ms/sm120-nulltest/fullmodel/matched_run.py:167) records validation-induced recording changes in `ineligibility_reasons`, leaving `status="VALID"` but setting `pair_eligible=False`. [qualify.py:124](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:124) checks joint-settle and block flags but omits the validation snapshot comparison. [collect_full.py:147](/home/muhit/ms/sm120-nulltest/fullmodel/collect_full.py:147) consequently accepts the records. Three CPU records explicitly rejected for recording changes during validation produced a pair `HEADLINE`.

    **Minimal fix:** compare every required matched phase transition and reject stored-flag contradictions in both directions, as §5 requires.

63. **[MAJOR] The reference-set join rejects the registered r1-only SDR lifecycle.**  
    [qualify.py:39](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:39) combines validation-reference hashes with SDR-output hashes. [collect_full.py:62](/home/muhit/ms/sm120-nulltest/fullmodel/collect_full.py:62) requires identical combined sets across restarts, while [session1.sh:107](/home/muhit/ms/sm120-nulltest/scripts/session1.sh:107) runs SDR only in r1. A correct r1 therefore has eight additional hashes and the cell becomes `INCOMPLETE`. Reproduced directly from the `b039272` blobs.

    **Minimal fix:** compare validation-reference hashes across all three restarts and validate r1’s SDR references separately. This is the change subsequently made in `f99c672`.

64. **[BLOCKER] Published timing statistics are not verified against their samples.**  
    [qualify.py:22](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:22) checks block lengths but not finite positive samples or summary consistency. [collect_full.py:96](/home/muhit/ms/sm120-nulltest/fullmodel/collect_full.py:96) publishes stored minima; [collect_full.py:176](/home/muhit/ms/sm120-nulltest/fullmodel/collect_full.py:176) publishes stored matched deltas. A fixture containing five blocks of fifty **10 ms** samples published a **−123 ms HEADLINE** after changing the stored summary.

    **Minimal fix:** recompute block medians, restart statistics, throughput summaries, and matched deltas from validated payloads; reject inconsistent stored summaries.

65. **[BLOCKER] Completeness can still certify missing evidence.**  
    [completeness.py:90](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:90), [completeness.py:103](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:103), and [completeness.py:125](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:125) trust summary `HEADLINE` labels. The module imports `qualify` but never calls its validators. Reference auditing at [line 54](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:54) checks sidecar manifest IDs without checking artifact hashes; clock auditing at [line 67](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:67) accepts any nonempty file. Missing auxiliary records escape the glob-based audit.

    CPU fixtures returned **COMPLETE 33/33** for session 1 without matrix measurement records, and **COMPLETE 102/102** for session 2 with skeletal reductions and perturbation records lacking timing.

    **Minimal fix:** independently audit the actual records and required artifact inventory, bind them to this manifest, verify reference hashes and clock coverage, and enumerate mandatory auxiliary outcomes.

66. **[MAJOR] Valid eager graph-verification results prevent completeness.**  
    [graph_verify.py:190](/home/muhit/ms/sm120-nulltest/fullmodel/graph_verify.py:190) correctly emits `NO_REPLAY_EVIDENCE` for a qualified trace without graph launches. [completeness.py:148](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:148) accepts only `FULLY_GRAPHED` or `NOT_FULLY_GRAPHED`. Correct eager A/F traces can therefore make session 2 incomplete merely for showing no replay. A CPU fixture reproduced all four A/F rejections.

    **Minimal fix:** distinguish completed verification from successful graph deployment; accept qualified `NO_REPLAY_EVIDENCE` where the registration requires reporting rather than replay.

67. **[MAJOR] Observer identities survive JSON, but remain vulnerable to Python ID reuse.**  
    [timing.py:122](/home/muhit/ms/sm120-nulltest/fullmodel/timing.py:122), [timing.py:131](/home/muhit/ms/sm120-nulltest/fullmodel/timing.py:131), [timing.py:149](/home/muhit/ms/sm120-nulltest/fullmodel/timing.py:149). JSON round-tripping preserved the sorted identity lists correctly. However, snapshots retain addresses rather than lifetime-unique identities. Replacing a freed fixture node immediately reused its address; comparison against the JSON-round-tripped snapshot returned **`True`**. Python explicitly permits this reuse across object lifetimes. [Python documentation](https://docs.python.org/3/library/functions.html#id)

    **Minimal fix:** use a lifetime-safe manager generation and monotonically assigned node identities, such as a weak identity registry. Ordinary node growth remains detectable; serialization itself is not the defect.

68. **[BLOCKER] Graph signatures still equate programs with different pricing semantics.**  
    [ideal_costs.py:99](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:99) records argument views but omits base-allocation sizes and alias relationships used by L2 pricing. Two fixture graphs with identical pointer views but different allocation sizes had equal signatures; reversing their order changed the published L2 ideal while selection remained `ok`.

    Furthermore, [kernel_map.py:317](/home/muhit/ms/sm120-nulltest/fullmodel/attrib/kernel_map.py:317) does not preserve non-tensor arguments in extern/fallback signatures. Different `permute` axis lists produced equal signatures. [kernel_map.py:83](/home/muhit/ms/sm120-nulltest/fullmodel/attrib/kernel_map.py:83) also excludes custom-op namespaces from fallback parsing.

    **Minimal fix:** include allocation/alias semantics and normalized complete calls, including scalar/list/keyword arguments and custom operators; unresolved call parsing must prevent equivalence.

69. **[MAJOR] L2 residency still uses a lower bound on allocated storage.**  
    [ideal_costs.py:325](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:325) computes allocation size with `accessed_elems()`, which deliberately caps the result at logical element count: [kernel_map.py:416](/home/muhit/ms/sm120-nulltest/fullmodel/attrib/kernel_map.py:416). That is not full allocated storage for a strided allocation. A stride-2 fp16 fixture requiring approximately **128 MiB** was counted as **64 MiB** and received an L2 ideal for a **96 MiB** cache. Missing allocation metadata also falls back to the argument view.

    **Minimal fix:** compute storage extent separately from traffic, including strides, offsets, and allocation dtype; unknown full storage size must leave L2 eligibility unresolved.

70. **[MAJOR] The registered 50-pass requirement remains bypassable.**  
    [graph_verify.py:169](/home/muhit/ms/sm120-nulltest/fullmodel/graph_verify.py:169) accepts any supplied expected count: one pass with `--expected-passes 1` produced `FULLY_GRAPHED`. [ideal_costs.py:72](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:72) removes empty windows before checking the count: **51 explicit windows with one empty window** were accepted as 50.

    **Minimal fix:** enforce 50 for registered outputs and validate exactly 50 nonempty windows before filtering or aggregation. Nonregistered shakedowns need an explicit ineligible designation.

71. **[MINOR] Aggregate reconciliation can falsely report agreement.**  
    [ideal_costs.py:255](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:255) compares only instance counts for names present in both inventories. It ignores duration differences and missing/extra names. A fixture with a millionfold duration discrepancy and an extra aggregate kernel reported **“aggregate summary agrees.”**

    **Minimal fix:** compare complete name sets, counts, and durations, reporting outside-window differences explicitly. The published measurements themselves correctly remained window-derived.

72. **[MAJOR] Boundary qualification still trusts `status="ok"` over failed proof data.**  
    [boundary_bench.py:173](/home/muhit/ms/sm120-nulltest/fullmodel/boundary_bench.py:173) discards top-level PTX qualification, and [line 215](/home/muhit/ms/sm120-nulltest/fullmodel/boundary_bench.py:215) checks only the case status. A 200-kernel fixture with `dependency_proof=False` and `status="ok"` published a **0.9 µs boundary estimate**.

    **Minimal fix:** verify the applicable dependency counter/proof and PDL PTX result directly, rejecting contradictions or missing proof payloads.

73. **[MAJOR] Ambiguous source mappings can still receive 100% coverage.**  
    [perturbation.py:137](/home/muhit/ms/sm120-nulltest/fullmodel/perturbation.py:137) marks collisions ambiguous, but [line 148](/home/muhit/ms/sm120-nulltest/fullmodel/perturbation.py:148) credits any matching prefix. An ambiguous `triton_copy_10` was credited through unambiguous `triton_copy_1`, producing **coverage 1.0**. Entries with no resolved fused operations can also count.

    **Minimal fix:** preserve full kernel names and use exact or demonstrably unique matching with ordinal boundaries; ambiguous or unresolved source mappings must contribute zero coverage.

74. **[BLOCKER] An audit failure can preserve a previous `COMPLETE` verdict.**  
    [session_lib.sh:146](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:146) ignores the audit’s return status; [line 147](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:147) writes failure only when the status file is absent. With collection succeeding, an existing `COMPLETE`, and an audit timeout, the real shell finalizer printed **`SESSION VERDICT (session2): COMPLETE`**.

    Separately, [completeness.py:62](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:62) audits the previous `aux_completeness.json`; a previous failed audit can perpetuate failure on subsequent attempts.

    **Minimal fix:** invalidate the previous verdict, bind results to the current attempt, overwrite with `INCOMPLETE` on any audit failure, and handle the audit’s own terminal outcome in the supervisor.

75. **[MAJOR] Failed measurement attempts can retain stale successful artifacts.**  
    [session_lib.sh:112](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:112) preserves an existing run JSON after profiling fails. Export invalidation occurs only after profiling succeeds, at [line 119](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:119). Old complete exports and reductions can therefore survive a failed new attempt. The single-arm and matched supervisors similarly preserve existing JSON at [line 62](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:62) and [line 82](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:82).

    **Minimal fix:** archive/invalidate every affected artifact before attempting a stage, and require outputs to carry the current attempt identity before accepting them.

76. **[MAJOR] Boundary export and analysis remain outside deadline supervision.**  
    [session2.sh:98](/home/muhit/ms/sm120-nulltest/scripts/session2.sh:98) runs `nsys stats` directly, followed by unsupervised analysis at [line 101](/home/muhit/ms/sm120-nulltest/scripts/session2.sh:101). These stages can overrun the extraction reserve and leave no dedicated auxiliary failure record.

    **Minimal fix:** wrap each stage with its own bounded `run_aux`, stale-output invalidation, and terminal record. The tested ordinary `run_aux false` path **did continue under `set -u` with unchanged deadline origin**; no new nounset-specific abort was reproduced.

77. **[MAJOR] A receipt for older evidence still authorizes terminating the same pod.**  
    [pod_run.sh:61](/home/muhit/ms/sm120-nulltest/scripts/pod_run.sh:61) checks pod identity and the unchanged local archive, but never checks whether remote evidence has advanced. The sequence **extract session 1 → run session 2 → terminate** remains authorized by the session-1 receipt, losing newer evidence.

    **Minimal fix:** bind the receipt to the finalized remote evidence generation/content manifest and reject termination after subsequent measurement activity.

Closure of findings 46–60:

| Finding | Closure | Code evidence |
|---|---|---|
| 46 | **CLOSED** | Selected clip-manifest hash checked before SDR loading/output generation: [run_full_arm.py:176](/home/muhit/ms/sm120-nulltest/fullmodel/run_full_arm.py:176). |
| 47 | **PARTIAL** | Canonical matched/r1 variant join implemented: [collect_full.py:167](/home/muhit/ms/sm120-nulltest/fullmodel/collect_full.py:167). Qualification and publication bypasses remain: 61–65. |
| 48 | **PARTIAL** | Manager/node/function identities retained and compared: [timing.py:130](/home/muhit/ms/sm120-nulltest/fullmodel/timing.py:130). Lifetime reuse remains: 67. |
| 49 | **PARTIAL** | Missing exports or missing expected count now yield `UNRESOLVED`: [graph_verify.py:169](/home/muhit/ms/sm120-nulltest/fullmodel/graph_verify.py:169). Nonregistered counts still qualify: 70. |
| 50 | **PARTIAL** | Kernel hashes/access sites added: [ideal_costs.py:105](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:105). Pricing-equivalence and coverage defects remain: 68, 73. |
| 51 | **PARTIAL** | Measurements and decomposition consume the **same `split_passes` result**: [ideal_costs.py:71](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:71), [ideal_costs.py:254](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:254). Outside-window fixture kernels did not leak; 49-window fixtures emitted no rows. Remaining count/reconciliation defects: 70–71. |
| 52 | **CLOSED** | Positional numeric parsing and parameter-count checks: [kernel_map.py:283](/home/muhit/ms/sm120-nulltest/fullmodel/attrib/kernel_map.py:283), [ideal_costs.py:291](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:291). The zero-offset 16-element fp16 copy now prices at **64 bytes**. |
| 53 | **PARTIAL** | Distinct bases are unioned: [ideal_costs.py:320](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:320). Full allocation size remains incorrect: 69. |
| 54 | **PARTIAL** | Missing qualification case now rejects the segment: [boundary_bench.py:212](/home/muhit/ms/sm120-nulltest/fullmodel/boundary_bench.py:212). Contradictory proof/PTX qualification remains unchecked: 72. |
| 55 | **CLOSED** | Affirmative stability, recorded variant, and complete iteration list required: [perturbation.py:42](/home/muhit/ms/sm120-nulltest/fullmodel/perturbation.py:42). Failed prerequisites clear both ratios and inflation: [line 90](/home/muhit/ms/sm120-nulltest/fullmodel/perturbation.py:90); reproduced. |
| 56 | **PARTIAL** | Required-item coverage expanded: [completeness.py:105](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:105), [line 127](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:127). False certification and eager-result rejection remain: 65–66. |
| 57 | **PARTIAL** | Export exit checks and supervised finalization added: [session_lib.sh:119](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:119), [line 142](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:142). Remaining failures: 74–76. |
| 58 | **CLOSED** | Memops excluded from kernel sums and critical path: [timing_views.py:126](/home/muhit/ms/sm120-nulltest/fullmodel/timing_views.py:126). The 10 ns kernel + 90 ns memcpy fixture reports one kernel and 10 ns. |
| 59 | **CLOSED** | Transfer/archive/extraction/count checks and pod/hash-bound receipt implemented: [pod_run.sh:48](/home/muhit/ms/sm120-nulltest/scripts/pod_run.sh:48), [line 61](/home/muhit/ms/sm120-nulltest/scripts/pod_run.sh:61). Receipt freshness is a separate remaining defect: 77. |
| 60 | **CLOSED** | Header accurately labels adaptation and delegates current hashing to provenance: [kernel_map.py:1](/home/muhit/ms/sm120-nulltest/fullmodel/attrib/kernel_map.py:1). |

EXECUTION-READY: no