Reviewed **`845ba99`** directly. Used code inspection, in-memory CPU counterexamples, and Python/Bash syntax checks; no GPU measurements. Older smoke records were rejected as expected.

97. **BLOCKER — Completeness still certifies contradictory reductions.**  
    [completeness.py:298](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:298), [312](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:312), [319](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:319), [348](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:348). A CPU session-2 fixture returned **COMPLETE 151/151** with unreadable trace CSVs, `FULLY_GRAPHED` records whose passes had zero launches and unresolved kernels, impossible timing-view inequalities, negative ideal costs and perturbation ratios, and unsupported source-coverage claims. The audit checks selected fields without recomputing their supporting reductions. Additionally, [qualify.py:339](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:339) exempts ANVTX from numerical validation.

    **Minimal fix:** parse and reconcile required exports; independently derive reduction verdicts, summaries, ratios and coverage; validate ANVTX’s registered tight comparator.

98. **BLOCKER — Failed C3 qualification payloads can receive COMPLETE.**  
    [completeness.py:282](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:282) accepts any truthy `opcheck` and ignores capture/replay and isolated-compile results. A session-1 fixture with both opcheck tests, capture/replay, and fullgraph compilation explicitly marked failed returned **COMPLETE 36/36**.

    **Minimal fix:** require successful results for each registered qualification component and reject contradictions with the aggregate PASS label.

99. **BLOCKER — Session 2 can execute and certify C3/H without prerequisite qualification.**  
    [session2.sh:21](/home/muhit/ms/sm120-nulltest/scripts/session2.sh:21), [38](/home/muhit/ms/sm120-nulltest/scripts/session2.sh:38), [49](/home/muhit/ms/sm120-nulltest/scripts/session2.sh:49), [completeness.py:248](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:248). The fresh-session path creates a manifest and references, then runs C3/H without planted checks or operator qualification. Reusing session-1 evidence also does not require those prerequisites to have succeeded. The COMPLETE session-2 fixture contained neither prerequisite.

    **Minimal fix:** require validated, manifest-bound prerequisite records before session-2 measurements; regenerate missing prerequisites under supervision and audit them in both sessions.

100. **BLOCKER — Manifest “identity validation” checks presence, not identity.**  
     [completeness.py:84](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:84). Arbitrary commit, registration hash, clip hash, checkpoint pin, runtime, and eleven arbitrary code entries satisfy the predicate. Replacing these fields with incorrect values preserved **COMPLETE** in the CPU fixture.

     **Minimal fix:** verify pins, registered runtime, measuring-file inventory and hashes against the declared revision, then reconcile recorded run identities with that manifest.

101. **MAJOR — Clock coverage remains unsound.**  
     [completeness.py:165](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:165), [186](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:186), [193](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:193). Seven columns containing a timestamp followed by nonsense/NaNs passed. Required blocks preceding the first clock sample by more than 60 seconds disappear from the audit; later-session blocks remain included when auditing an earlier session. One nearby sample also satisfies coverage for an arbitrarily long block.

     **Minimal fix:** associate blocks with an explicit session interval; validate GPU identity and numeric fields; require sampling coverage throughout each block, including bounded gaps.

102. **MAJOR — Contradictory nested summaries still reach HEADLINE.**  
     [qualify.py:109](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:109), [424](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:424), [collect_full.py:164](/home/muhit/ms/sm120-nulltest/fullmodel/collect_full.py:164). CPU fixtures retained HEADLINE with a **NaN stored matched median**, incorrect matched extrema/percentages/per-block deltas, incorrect per-arm matched summaries, and single-arm `latency_medians_ms=[-123,…]`. `serialization_delta_pct` is also unchecked. Recomputing published headline values does not satisfy the registration’s requirement to reject contradictory records.

     **Minimal fix:** compare every persisted summary using finite-aware predicates, including arrays and all matched summaries.

103. **MAJOR — Matched qualification does not enforce block alternation or timestamps.**  
     [qualify.py:398](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:398), [424](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:424), [recomputed_matched_delta:445](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:445). Sorting blocks separately by arm/repeat loses the recorded execution order. All C1 blocks followed by all C3 blocks still produced a pair HEADLINE. Removing matched block timestamps also passed collection. Snapshot-count checks do not establish their association with the required alternating sequence.

     **Minimal fix:** validate the complete chronological `(repeat, arm)` sequence, finite ordered timestamps, and corresponding snapshot sequence before calculating deltas.

104. **MAJOR — “Complete” recording snapshots still admit malformed observers.**  
     [qualify.py:37](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:37), [60](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:60). A compiled fixture qualified with missing `unique_graphs`, an empty node list claiming 100 roots, and `function_ids` stored as a string. Equality of malformed snapshots establishes neither complete observation nor stability.

     **Minimal fix:** validate counter values and required keys, identity-list types and uniqueness, and observer count consistency before comparison.

105. **MAJOR — Validation statistics lack basic consistency checks.**  
     [qualify.py:184](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:184). A pass with negative `max_abs_diff` and `max_abs_ref`, plus `n_elements=1` for shape `[1,2,4410]`, passed validation. The new fields therefore do not yet provide a consistently checked numerical payload.

     **Minimal fix:** require nonnegative error/reference magnitudes, valid positive integer dimensions, and `n_elements == product(shape)`.

106. **MAJOR — Perturbation joins trust the stored variant string and omit run identities.**  
     [perturbation.py:65](/home/muhit/ms/sm120-nulltest/fullmodel/perturbation.py:65), [qualify.py:345](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:345). Changing G’s node run from whole-forward capture to `band_split_only` while retaining its old `effective_variant` string still published `status="ok"`. A node record with a different arm and operating point also produced a ratio.

     **Minimal fix:** join on checked arm/chunk, canonical `qualify.variant(record)`, manifest, and validation-reference hashes; reject inconsistent stored variant strings.

107. **MAJOR — Multiline and unrecognised calls still defeat graph-signature equivalence.**  
     [kernel_map.py:266](/home/muhit/ms/sm120-nulltest/fullmodel/attrib/kernel_map.py:266), [ideal_costs.py:118](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:118). Two valid wrappers containing multiline `torch.ops.custom.mutate_(arg0_1, 1)` versus `…, 2)` recorded only the identical opening line in `unparsed`. Their signatures were equal and selection succeeded. Tensor-method calls such as `arg0_1.add_(…)` were omitted entirely.

     **Minimal fix:** parse complete executable statements structurally, preserve their order, and prevent interchangeability when semantics remain unresolved.

108. **MAJOR — Reference timeouts still abort session 1 before finalization.**  
     [session1.sh:58](/home/muhit/ms/sm120-nulltest/scripts/session1.sh:58), [62](/home/muhit/ms/sm120-nulltest/scripts/session1.sh:62). Simulating timed-out reference attempts reached the missing-reference `exit 1`, bypassing finalization and remaining terminal records. Missing planted-check results have another unconditional exit at [line 68](/home/muhit/ms/sm120-nulltest/scripts/session1.sh:68).

     **Minimal fix:** distinguish validator failure from unavailable prerequisites; record dependent work as UNTESTED and guarantee finalization after measurement-stage failures.

109. **MAJOR — Extraction is checked, but termination still races new writers.**  
     [pod_run.sh:49](/home/muhit/ms/sm120-nulltest/scripts/pod_run.sh:49), [77](/home/muhit/ms/sm120-nulltest/scripts/pod_run.sh:77). Extraction now checks pre/post/archive content correctly. Termination, however, neither checks for a running session nor excludes a session starting after its final remote hash. Evidence written between that hash and termination remains unextracted.

     **Minimal fix:** coordinate session launch, extraction and termination through a shared remote exclusion mechanism held through the termination decision.

110. **MAJOR — Ideal-cost kernel inventories include GPU memops.**  
     [ideal_costs.py:84](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:84), [401](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:401). A `[CUDA memcpy DtoD]` fixture row entered the per-kernel table as `COPY`, including its launches and duration. It also creates a spurious discrepancy against the kernel-only aggregate summary.

     **Minimal fix:** separate memops from kernel inventories and class totals, consistently with graph verification and timing views.

111. **MINOR — Observed pass-count failures use the wrong registered status.**  
     [ideal_costs.py:279](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:279), [registration:470](/home/muhit/ms/sm120-nulltest/prereg_amendments.md:470). Requesting 50 with 49 observed windows emits `UNRESOLVED`, retains `passes=50`, and publishes no rows. The amended registration explicitly requires `UNREGISTERED_PROTOCOL` for an incorrect requested **or observed** count.

     **Minimal fix:** persist requested and observed counts separately and apply the registered status to count failures.

Closure of findings 78–96:

| Finding | Verdict | Code evidence |
|---|---|---|
| 78 | **CLOSED** | Initial eligibility is `None`; derivation occurs before publication: [matched_run.py:109](/home/muhit/ms/sm120-nulltest/fullmodel/matched_run.py:109), [214](/home/muhit/ms/sm120-nulltest/fullmodel/matched_run.py:214). Valid CPU control passes. |
| 79 | **CLOSED** | Main reductions and auxiliary records are invalidated only for tag `trace`: [session_lib.sh:98](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:98). Shell interception confirmed perturbation tags preserve them. |
| 80 | **PARTIAL** | Shapes, finiteness, tolerances, violations and tier overrides checked: [timing.py:259](/home/muhit/ms/sm120-nulltest/fullmodel/timing.py:259), [qualify.py:177](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:177). Statistical consistency remains incomplete: 105. |
| 81 | **CLOSED** | Exact indices/count, finite paired terms, subtraction, mean and status checked: [qualify.py:199](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:199). Original contradictory-score fixture rejects. |
| 82 | **PARTIAL** | Required phases and matched snapshot counts enforced: [qualify.py:229](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:229), [399](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:399). Schema/association gaps remain: 103–104. |
| 83 | **CLOSED** | Shared gate predicate checks settings and break/unimplemented counts: [qualify.py:252](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:252). Matched reconciliation uses each arm’s gate snapshot: [matched_run.py:52](/home/muhit/ms/sm120-nulltest/fullmodel/matched_run.py:52), [qualify.py:421](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:421). |
| 84 | **CLOSED** | Identities come solely from a monotonic counter: [timing.py:129](/home/muhit/ms/sm120-nulltest/fullmodel/timing.py:129). JSON and repeated node-replacement tests passed. |
| 85 | **PARTIAL** | Shared trace validation and auxiliary enumeration added: [completeness.py:302](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:302), [322](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:322). False COMPLETE paths remain: 97–99. |
| 86 | **PARTIAL** | Required references enumerated, rehashed and joined: [completeness.py:103](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:103), [134](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:134). Manifest identities remain unchecked: 100. |
| 87 | **PARTIAL** | Matched blocks and finite timestamps included: [completeness.py:181](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:181). Scope, sample contents and coverage remain defective: 101. |
| 88 | **PARTIAL** | Single-arm block extrema and selected summaries checked: [qualify.py:98](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:98). Remaining summary/protocol bypasses: 102–103. |
| 89 | **CLOSED** | Perturbation uses shared qualification and recomputed block medians: [perturbation.py:35](/home/muhit/ms/sm120-nulltest/fullmodel/perturbation.py:35). Original false-median fixture rejects. Separate identity defect: 106. |
| 90 | **CLOSED** | Requested counts other than 50 produce an explicit ineligible record without rows: [ideal_costs.py:260](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:260). One-pass counterexample now rejects; observed-count labelling is 111. |
| 91 | **PARTIAL** | Unassigned custom calls recognised: [kernel_map.py:84](/home/muhit/ms/sm120-nulltest/fullmodel/attrib/kernel_map.py:84). Incomplete statement capture still permits false equivalence: 107. |
| 92 | **CLOSED** | Numerical discrepancy always recorded: [ideal_costs.py:298](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:298). The 5,000/5,049 ns fixture reports `0.0098`. |
| 93 | **CLOSED** | Expected counter independently derived: [boundary_bench.py:226](/home/muhit/ms/sm120-nulltest/fullmodel/boundary_bench.py:226). Zero-counter fixture rejects with expected counter 270. |
| 94 | **CLOSED** | Run/report/export/reduction invalidation precedes deadline checks: [session_lib.sh:97](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:97), [104](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:104). Shell test confirmed execution order. |
| 95 | **PARTIAL** | Pre/post/archive hashes joined: [pod_run.sh:51](/home/muhit/ms/sm120-nulltest/scripts/pod_run.sh:51), [62](/home/muhit/ms/sm120-nulltest/scripts/pod_run.sh:62). Writer exclusion through termination remains absent: 109. |
| 96 | **CLOSED** | C3 qualification failure disables dependent work and continues: [session1.sh:83](/home/muhit/ms/sm120-nulltest/scripts/session1.sh:83), [96](/home/muhit/ms/sm120-nulltest/scripts/session1.sh:96). Timeout continuation reproduced; other abort paths remain in 108. |

For the explicit window question: **measurement and decomposition consume the same `split_passes` result**, at [ideal_costs.py:72](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:72). A kernel whose start lies outside every window cannot enter either inventory; the sentinel test confirmed this.

`node_ids` survives JSON comparison within one process, and registry shrinkage no longer causes identity reuse. Ordinary `run_aux false` under `set -u` returned failure without aborting or changing `T0`; the remaining session abort is explicit control flow, not nounset behaviour.

EXECUTION-READY: no