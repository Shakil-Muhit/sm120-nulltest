Reviewed **`e523ef9`** directly against the registration, plan, and gates 04–07. **Invalid records can still receive both HEADLINE and COMPLETE.**

Verification used code inspection, in-memory CPU counterexamples, and mocked shell stages. Full-audit fixtures stubbed live pin verification; qualification and audit logic were unchanged. Python/Bash syntax checks passed. Older local smoke records were rejected as expected.

112. **BLOCKER — Matched validation now checks only the last arm’s full payload.**  
     [qualify.py:435](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:435), [492](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:492). The full numerical-validation and block-stability checks were displaced outside the arm loop. A C1/C3 fixture with C1’s pass marked `ok=False`, `finite_out=False`, positive violations, and `max_abs_diff=99` returned `pair_problems=[]`, published a pair **HEADLINE**, and retained **COMPLETE 37/37**. C1 blocks explicitly marked unstable also passed.

     **Minimal fix:** move those checks inside the arm loop and require exactly the two declared arms.

113. **BLOCKER — Fresh session 2 permanently fails its prerequisites before creating references.**  
     [session2.sh:53](/home/muhit/ms/sm120-nulltest/scripts/session2.sh:53), [86](/home/muhit/ms/sm120-nulltest/scripts/session2.sh:86). Planted checks run before reference generation. On a fresh directory they fail, setting `PREREQ_OK=0`; creating references afterward never retries qualification or restores eligibility. Shell simulation reached the sweep with references available but every measurement still UNTESTED.

     **Minimal fix:** generate and verify references before planted checks, then derive prerequisite state from the completed checks.

114. **BLOCKER — Collection still accepts an arbitrary manifest token as binding.**  
     [qualify.py:319](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:319), [collect_full.py:81](/home/muhit/ms/sm120-nulltest/fullmodel/collect_full.py:81), [185](/home/muhit/ms/sm120-nulltest/fullmodel/collect_full.py:185). The collector never loads the manifest or verifies reference artifacts. Three otherwise qualified records with incorrect checkpoint/MSST pins, no manifest file, and no references still produced **HEADLINE**.

     **Minimal fix:** require verified manifest, run-provenance, reference-lineage, and prerequisite context before publishing a headline.

115. **BLOCKER — Completeness compares only subsets of manifest and run identity.**  
     [completeness.py:102](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:102), [109](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:109), [325](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:325). Manifest checks omit the verified `config` path and `msst_clean`; runtime GPU/driver fields are not reconciled with `env`. Run comparisons cover only three hashes, omitting runtime and GPU identity. Contradictory GPU identities, wrong run torch/triton versions, and `msst_clean=False` retained **COMPLETE 37/37**. The audit also omits the registered clean-tree check.

     **Minimal fix:** share a complete identity validator with provenance loading and compare every required run field against the verified manifest.

116. **BLOCKER — C3 qualification remains unbound and can omit registered tests.**  
     [completeness.py:267](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:267), [session2.sh:60](/home/muhit/ms/sm120-nulltest/scripts/session2.sh:60), [70](/home/muhit/ms/sm120-nulltest/scripts/session2.sh:70). `opcheck={"invented":"SUCCESS"}` passes completeness because required test names are never checked. Qualification is written without manifest binding, and neither session-2 preflight nor completeness verifies such binding. Both session fixtures remained COMPLETE with foreign qualification metadata and invented opchecks. Separately, the actual session-2 preflight accepted failed opchecks and missing output shapes.

     **Minimal fix:** require `test_schema` and `test_faketensor`, bind qualification to the manifest, and use one complete predicate in both preflight and auditing.

117. **BLOCKER — Export validation still permits unreadable trace evidence.**  
     [completeness.py:281](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:281), [389](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:389), [405](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:405). CSV validation checks header substrings and counts rows without interpreting them. Replacing C3’s GPU rows with `not,a,trace,row`, beneath the expected header, retained **COMPLETE 155/155**. Stored graph/timing reductions are not reconciled with those exports.

     **Minimal fix:** validate row contents and pass windows, then regenerate and compare the reductions from their actual exports.

118. **BLOCKER — Ideal-cost auditing still accepts impossible costs.**  
     [completeness.py:436](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:436), [453](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:453). `priced_ok` is unused for H/C1/C2. H rows with negative HBM/L2 ideals and recoverability `22` retained COMPLETE. C3 also passed with measured time `0.1 µs`, ideal `50 µs`, recoverability `0.9`, negative bytes, and `passes_requested=49`. Bounds alone do not establish the registered formula.

     **Minimal fix:** validate all optional numeric fields and both pass counts for every arm; recompute recoverability and cost/class reductions, including GEMM/SDPA products.

119. **BLOCKER — Completeness bypasses perturbation’s repaired joins.**  
     [completeness.py:456](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:456), [468](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:468), [perturbation.py:65](/home/muhit/ms/sm120-nulltest/fullmodel/perturbation.py:65). Changing G’s node record to an incompatible capture variant made the real perturbation reducer return **UNRESOLVED**, while completeness retained **COMPLETE 155/155**. The audit recomputes arithmetic without reproducing the joins. Both paths also accepted graph/node records with `nvtx_pass=False` and absent perturbation profiling artifacts.

     **Minimal fix:** share the complete perturbation predicate and require evidence establishing each profiling mode before accepting ratios.

120. **BLOCKER — Benchmark completeness still trusts skeletal summaries.**  
     [completeness.py:492](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:492), [498](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:498), [503](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:503). A fixture remained **COMPLETE 155/155** with attention cases marked FAIL, empty backend payloads, ratios `−999`, boundary service times `−99`, false dependency proofs, and no traced boundary record. `_qualification={"cases":1,"accepted_segments":1}` sufficed without any accepted segment.

     **Minimal fix:** validate benchmark case payloads, derive ratios from qualified measurements, and recheck boundary segments against their traced qualification and exports.

121. **MAJOR — Serialization percentages are never recomputed.**  
     [qualify.py:127](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:127), [timing.py:236](/home/muhit/ms/sm120-nulltest/fullmodel/timing.py:236). The new check enforces only list length and finiteness. Replacing the correct `100%` values with `−999%` retained `run_problems=[]` and **HEADLINE**.

     **Minimal fix:** compare each value with `(latency_median − throughput) / throughput × 100`.

122. **MAJOR — Complete snapshots still admit nonnumeric counters.**  
     [qualify.py:46](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:46). Counter containers are checked, but their values are not. Identical snapshots containing `inductor.cudagraph_recorded="garbage"` passed `snapshot_complete` and C3 qualification.

     **Minimal fix:** require appropriately typed, nonnegative integer counter values before stability or graph-gate comparisons.

123. **MAJOR — Clock samples can contain NaN and infinity.**  
     [completeness.py:214](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:214). Successful `float()` conversion is treated as numeric validity. Clock logs containing `NaN MHz`, `inf MHz`, `-inf W`, and `NaN %`, with an integer temperature, retained **COMPLETE 37/37**.

     **Minimal fix:** require finite parsed values before allowing a sample to provide coverage.

124. **MAJOR — Unparsed operations lose their position in graph signatures.**  
     [kernel_map.py:280](/home/muhit/ms/sm120-nulltest/fullmodel/attrib/kernel_map.py:280), [ideal_costs.py:125](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:125). Unparsed calls occupy a separate list appended after recognized calls. Moving `arg0_1.add_(1)` from before a Triton copy to after it produced identical signatures; selection reported one signature class and a unique decomposition for two different programs.

     **Minimal fix:** preserve one ordered statement sequence. Unresolved executable semantics must prevent interchangeability.

125. **MAJOR — The termination sentinel is not atomic exclusion.**  
     [pod_run.sh:31](/home/muhit/ms/sm120-nulltest/scripts/pod_run.sh:31), [33](/home/muhit/ms/sm120-nulltest/scripts/pod_run.sh:33), [74](/home/muhit/ms/sm120-nulltest/scripts/pod_run.sh:74), [81](/home/muhit/ms/sm120-nulltest/scripts/pod_run.sh:81). A launch can pass its checks, pause, then create tmux after termination creates the sentinel and hashes evidence. Launch redirection truncates the session log before `session_lib` rejects the sentinel; the trailing exit message writes again. Evidence can therefore change after the final hash.

     **Minimal fix:** use shared atomic exclusion covering launch and log creation, held by termination through hashing and destruction.

126. **MAJOR — Prerequisite skips preserve stale trace products.**  
     [session2.sh:81](/home/muhit/ms/sm120-nulltest/scripts/session2.sh:81), [161](/home/muhit/ms/sm120-nulltest/scripts/session2.sh:161), [session_lib.sh:98](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:98). `run_traced_q` returns before the existing invalidation machinery. Shell interception confirmed only an UNTESTED run record was written. Old reports, exports, reductions, and auxiliary successes survive; the later ideal-cost loop can consume those exports again.

     **Minimal fix:** invalidate the attempt’s artifacts before prerequisite checks and require a qualified current trace before downstream reduction.

127. **MINOR — Observed pass counts are still discarded.**  
     [ideal_costs.py:79](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:79), [287](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:287). A 49-window fixture now gets the correct `UNREGISTERED_PROTOCOL` label, but records `passes_observed=None`. Status selection searches the error string for `"windows"`, also conflating count failures with other window defects.

     **Minimal fix:** return structured window diagnostics containing the actual count and failure category.

Closure of findings 97–111:

| Finding | Verdict | Code evidence |
|---|---|---|
| 97 | **PARTIAL** | Verdict, inequality, coverage, and ANVTX checks improved: [completeness.py:405](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:405), [qualify.py:365](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:365). False COMPLETE paths remain: 117–120. |
| 98 | **PARTIAL** | Explicit failed components now reject at [completeness.py:273](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:273); required opcheck identities remain unchecked: 116. |
| 99 | **PARTIAL** | Both audits call prerequisites at [completeness.py:375](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:375). Fresh-session ordering and qualification binding remain defective: 113, 116. |
| 100 | **PARTIAL** | Commit/file hashes verified at [completeness.py:92](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:92); identity comparison remains incomplete: 114–115. |
| 101 | **PARTIAL** | Session attribution and bounded coverage implemented at [completeness.py:181](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:181), [242](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:242). Non-finite samples remain accepted: 123. |
| 102 | **PARTIAL** | Matched summaries now checked at [qualify.py:456](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:456), [483](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:483). Serialization percentages remain unchecked: 121. |
| 103 | **CLOSED** | Chronological alternation, finite timestamps, and non-overlap enforced at [qualify.py:467](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:467). Missing timestamps and sequential arm batches reject. |
| 104 | **PARTIAL** | Identity-list/count checks added at [qualify.py:48](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:48); malformed counter values still qualify: 122. |
| 105 | **CLOSED** | Magnitudes, positive integer dimensions, and element-count product checked at [qualify.py:205](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:205). Matched invocation regression is separately 112. |
| 106 | **CLOSED** | Canonical variants, identities, manifest, and reference joins enforced at [perturbation.py:68](/home/muhit/ms/sm120-nulltest/fullmodel/perturbation.py:68). Original mutations now clear ratios. Audit bypass is 119. |
| 107 | **PARTIAL** | Multiline constants now distinguish signatures at [kernel_map.py:233](/home/muhit/ms/sm120-nulltest/fullmodel/attrib/kernel_map.py:233); operation ordering remains unsound: 124. |
| 108 | **CLOSED** | Failure state and dependent UNTESTED records at [session1.sh:63](/home/muhit/ms/sm120-nulltest/scripts/session1.sh:63), [98](/home/muhit/ms/sm120-nulltest/scripts/session1.sh:98). Simulated reference/planted timeouts reached finalization. |
| 109 | **PARTIAL** | Sentinel and hash-under-sentinel added at [pod_run.sh:74](/home/muhit/ms/sm120-nulltest/scripts/pod_run.sh:74); launch exclusion remains racy: 125. |
| 110 | **CLOSED** | Memops removed before splitting and inventory at [ideal_costs.py:75](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:75). CPU inventory excluded all memcpy rows. |
| 111 | **PARTIAL** | Count-failure label corrected at [ideal_costs.py:287](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:287), but actual observed counts remain absent: 127. |

For the window question: **measurement and decomposition consume the same `split_passes` result**, at [ideal_costs.py:77](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:77). Sentinel kernels starting before, between, and after the windows did not enter either inventory. Membership is by kernel start.

`node_ids` survives JSON serialization correctly. Across **500 replacements**, including **483 reused Python addresses**, the observer assigned 500 distinct registry IDs and produced **zero false-stable comparisons**. The monotonic allocation is at [timing.py:129](/home/muhit/ms/sm120-nulltest/fullmodel/timing.py:129).

Ordinary failure, timeout, and deadline-skip tests of [run_aux](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:176) continued under `set -u` with unchanged `T0`. The remaining session defects are the ordering, exclusion, and invalidation paths described above.

EXECUTION-READY: no