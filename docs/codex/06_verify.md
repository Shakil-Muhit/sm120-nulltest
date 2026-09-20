Reviewed **`43c9e29`** directly against the registration, plan, and gates 04–05. Used code inspection, in-memory CPU counterexamples, and Python/Bash syntax checks; no new GPU measurements. Previous-observer smoke records were rejected as expected.

78. **BLOCKER — The matched runner disqualifies every otherwise eligible pair.**  
    [matched_run.py:108](/home/muhit/ms/sm120-nulltest/fullmodel/matched_run.py:108) initializes `pair_eligible=False`. Before replacing that value, it calls the validator at [line 212](/home/muhit/ms/sm120-nulltest/fullmodel/matched_run.py:212). [qualify.py:287](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:287) treats this initial value as a contradictory rejection. A valid CPU fixture changed from no problems to `stored pair flag says ineligible: []` solely by applying the runner’s initialization.

    **Minimal fix:** initialize unpublished eligibility to `None`; derive the verdict before checking contradictions in persisted records.

79. **BLOCKER — Perturbation tracing deletes required results from the preceding trace stage.**  
    [session_lib.sh:107](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:107) unconditionally deletes `graph_verify_<arm>_<chunk>.json` and `timing_views_<arm>_<chunk>.json`, regardless of trace tag. Consequently, the `pert_graph` and `pert_node` calls at [session2.sh:81](/home/muhit/ms/sm120-nulltest/scripts/session2.sh:81) erase the main-trace reductions for A/C3/G at both operating points. They are never regenerated. Reproduced by intercepting the real shell function’s deletion calls.

    **Minimal fix:** make reduction names and invalidation depend on their actual source trace; perturbation attempts must invalidate their own products.

80. **BLOCKER — Validation still accepts contradictory per-pass evidence.**  
    [qualify.py:145](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:145) checks `ok`, a finite difference, and a truthy hash, but ignores per-pass shapes, finiteness flags, and tolerances. A fixture with mismatched shapes, `finite_out=False`, and per-pass tolerances of `100` still produced `HEADLINE`. [Line 133](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:133) also permits an unplanted A record to override its registered tier with `forced_tier="loose"`.

    **Minimal fix:** check all per-pass invariants, forbid tier overrides for headline records, and persist sufficient error statistics to derive the tolerance verdict.

81. **BLOCKER — SDR qualification trusts stored deltas instead of the paired scores.**  
    [qualify.py:160](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:160) averages `delta_sdr_ref_db` without checking `sdr_ref_db`, `sdr_ref_A_db`, or their subtraction. Eight rows containing arm SDR **−99 dB**, A SDR **0 dB**, and stored delta **0** produced `HEADLINE`. An infinite paired term also passed. Distinct indices `100…107` qualify despite being outside the registered clip set.

    **Minimal fix:** require the registered clip identities, finite paired SDR terms, and recompute each delta before deriving the mean and status.

82. **BLOCKER — Missing recording snapshots can still qualify.**  
    [qualify.py:177](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:177) removes missing phases and accepts any three remaining snapshots. Removing both `after_timing` and `after_sdr` still produced `HEADLINE`. [Line 46](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:46) also accepts snapshots whose identity fields and counters are all absent, provided the tree claims `available=True` and `identity_mode="registry"`. The matched validator’s aggregate length check at [line 279](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:279) permits a missing `after_validation` snapshot.

    **Minimal fix:** require the exact phase schema applicable to the run, complete observer fields, and one correctly associated snapshot for every matched transition.

83. **BLOCKER — Graph-gate payloads are not fully requalified.**  
    [qualify.py:220](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:220) checks only the stored zero-break count for C3/H. It ignores contradictory snapshot counters, `unimplemented_count`, and `suppress_errors`. The matched path at [line 298](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:298) checks only `status="PASS"`: a C3 gate reporting **five graph breaks** reached a pair `HEADLINE`.

    **Minimal fix:** share the complete graph-gate predicate with the producer, check all required counters/settings, and reconcile them with the appropriate snapshots.

84. **MAJOR — Registry node identities are still reusable.**  
    [timing.py:127](/home/muhit/ms/sm120-nulltest/fullmodel/timing.py:127) assigns `len(_NODE_REGISTRY) + _REGISTRY_BASE[0]`. Registry length shrinks when weak references expire, so this expression is not monotonic. A CPU fixture freed a recorded node and another retained registry member; its replacement received the old identity **4**. JSON-round-tripped snapshots compared stable even though the replacement had a **different Python address**.

    **Minimal fix:** allocate identities solely from a monotonically increasing counter, independent of registry length.

85. **BLOCKER — Completeness still certifies skeletal or contradictory evidence.**  
    [completeness.py:193](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:193) does not apply shared payload qualification to traced runs. [Lines 210](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:210) and [230](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:230) accept reduction labels without verifying their supporting data. Mandatory auxiliary enumeration at [line 21](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:21) omits per-trace reductions.

    A CPU session-2 fixture returned **COMPLETE 109/109** with no trace CSVs or per-trace auxiliary records, negative trace samples, unstable recording state, perturbation records without timing or ratios, and NaN source coverage. Session 1 returned **COMPLETE 35/35** with planted checks containing one wrong seed marked `ok=True` and C3 qualification containing only PASS labels.

    **Minimal fix:** validate every mandatory product’s payload and dependencies, enumerate all required artifacts/outcomes, and recompute reductions through context-appropriate shared validators.

86. **BLOCKER — Reference auditing does not establish the required inventory or join it to runs.**  
    [completeness.py:62](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:62) audits whichever `.pt` files happen to exist. One unrelated file with a valid sidecar satisfies the check while every required validation and SDR reference is absent. Recorded reference hashes are never compared with the audited artifacts. [Line 130](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:130) accepts a manifest containing only `manifest_id`. Both complete-session counterexamples above used this arrangement.

    **Minimal fix:** validate the manifest’s required identity fields, enumerate references by operating point/seed/clip, and join each recorded hash to the corresponding rehashed artifact.

87. **MAJOR — Clock auditing checks the wrong scope and can omit entire measurements.**  
    [completeness.py:112](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:112) collects every `full_*` timing block across both sessions, then compares them only with the selected session’s clock log. Retaining an earlier session-1 run changed the session-2 fixture from **109/109 COMPLETE** to **108/109 INCOMPLETE**. Conversely, matched blocks are omitted, missing timestamps are discarded, and timestamp-only lines satisfy the clock check.

    **Minimal fix:** associate measurements with their session clock logs, include matched blocks, require finite timestamps for every block, and validate the required GPU/clock columns and coverage.

88. **MAJOR — Stored-summary contradictions remain unchecked.**  
    [qualify.py:227](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:227) checks three top-level summaries but ignores contradictory summaries inside `timing` and its blocks. A block median and nested minimum changed to **−123 ms** still qualified as `HEADLINE`. [collect_full.py:164](/home/muhit/ms/sm120-nulltest/fullmodel/collect_full.py:164) checks only the matched median, and its arithmetic comparison accepts NaN. Matched NaN throughput and duplicate repeat IDs also pass qualification.

    **Minimal fix:** validate the complete block/repeat protocol and cross-check every stored summary with finite recomputed values. Publication already recomputes the main headline numbers; rejection of contradictory records remains incomplete.

89. **MAJOR — Perturbation bypasses the new validator and sample recomputation.**  
    [perturbation.py:33](/home/muhit/ms/sm120-nulltest/fullmodel/perturbation.py:33) uses stored block medians; [line 42](/home/muhit/ms/sm120-nulltest/fullmodel/perturbation.py:42) retains a separate, weaker qualifier. With baseline samples of **10 ms**, node samples of **20 ms**, and a stored node median of **10 ms**, it published **ratio 1.0**, `status="ok"`, and `caveat_required=False`. Address-based observer snapshots also passed.

    **Minimal fix:** use shared qualification adapted for the one-block tracing protocol, and derive ratios from validated samples.

90. **MAJOR — Ideal costs still accept unregistered pass counts.**  
    [ideal_costs.py:255](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:255) forwards the caller’s count to window validation without requiring 50. A one-window fixture with `--passes 1` produced `measurement_status="ok"`, variant selection `ok`, and a priced row. Graph verification correctly returned `UNREGISTERED_PROTOCOL` for the equivalent case.

    **Minimal fix:** enforce 50 for registered ideal-cost output; otherwise publish an explicit ineligible result without rows.

91. **MAJOR — Unassigned custom-op calls remain invisible to graph signatures.**  
    [kernel_map.py:84](/home/muhit/ms/sm120-nulltest/fullmodel/attrib/kernel_map.py:84) requires an assignment before `torch.ops`; unmatched calls are silently skipped at [line 260](/home/muhit/ms/sm120-nulltest/fullmodel/attrib/kernel_map.py:260). Two valid wrappers differing in an unassigned mutating custom-op call produced identical signatures and selection `ok`. Recording complete text of recognized calls does not cover omitted calls.

    **Minimal fix:** parse executable calls structurally, including unassigned calls, and make unresolved call parsing prevent equivalence.

92. **MINOR — Reconciliation still reports agreement for differing durations.**  
    [ideal_costs.py:282](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:282) silently permits a 1% discrepancy. Window totals of **5,000 ns** and an aggregate total of **5,049 ns** reported “aggregate summary agrees.”

    **Minimal fix:** report the numerical discrepancy even when it falls within a documented tolerance.

93. **MAJOR — Boundary proof checks trust the stored expected counter.**  
    [boundary_bench.py:223](/home/muhit/ms/sm120-nulltest/fullmodel/boundary_bench.py:223) accepts `counter_delta == counter_expected` without deriving the expected count. A 200-kernel `trivial/graph/N10` fixture with both counters **0** and `dependency_proof=True` still published a **0.9 µs** boundary estimate; the registered complete execution requires counter delta **270**.

    **Minimal fix:** derive the expected counter from N, mode, warmup, and repeats; require both recorded counters to agree with it.

94. **MAJOR — Deadline skips preserve stale trace products.**  
    [session_lib.sh:94](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:94) returns before artifact invalidation at [line 107](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:107). A deadline-skipped trace overwrites the run JSON but leaves its old report, exports, and reductions. The shell counterexample confirmed that no invalidation ran.

    **Minimal fix:** invalidate the attempt’s correctly scoped artifacts before checking whether it can start.

95. **MAJOR — Extraction receipts can bind an old archive to newer remote evidence.**  
    [pod_run.sh:49](/home/muhit/ms/sm120-nulltest/scripts/pod_run.sh:49) transfers the archive before computing the remote content hash at [line 57](/home/muhit/ms/sm120-nulltest/scripts/pod_run.sh:57). If an existing evidence file changes between those operations, file counts still match, but the receipt contains the old archive hash and the new remote hash. [Line 70](/home/muhit/ms/sm120-nulltest/scripts/pod_run.sh:70) then permits termination without having extracted the newer contents. Neither operation requires writers to be stopped.

    **Minimal fix:** compare identical content manifests for the extracted archive and remote snapshot, and prevent evidence writes through the final termination check.

96. **MAJOR — Qualification timeout still aborts session 1 without finalization.**  
    [session1.sh:82](/home/muhit/ms/sm120-nulltest/scripts/session1.sh:82) follows `run_aux c3_operator_qualification` with `exit 1` on failure, including deadline skips and timeouts. This bypasses finalization and terminal records for remaining work, contradicting the continuation rule at [prereg_amendments.md:475](/home/muhit/ms/sm120-nulltest/prereg_amendments.md:475).

    **Minimal fix:** record qualification failure, mark dependent work unavailable, continue unaffected work, and ensure finalization writes `INCOMPLETE`.

Closure of findings 61–77:

| Finding | Closure | Code evidence |
|---|---|---|
| 61 | **PARTIAL** | Payload checks added at [qualify.py:124](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:124), but contradictory validation, SDR, snapshot, and gate evidence still qualifies: 80–83. |
| 62 | **PARTIAL** | Present validation changes and stored rejection flags are checked at [qualify.py:279](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:279); the original resurrection fixture now fails. Missing transitions remain accepted: 82. |
| 63 | **CLOSED** | Validation-only reference joining at [qualify.py:109](/home/muhit/ms/sm120-nulltest/fullmodel/qualify.py:109) and [collect_full.py:62](/home/muhit/ms/sm120-nulltest/fullmodel/collect_full.py:62). An r1-only SDR fixture qualifies. |
| 64 | **PARTIAL** | Headline statistics are recomputed at [collect_full.py:99](/home/muhit/ms/sm120-nulltest/fullmodel/collect_full.py:99); summary/protocol contradictions remain: 88. |
| 65 | **PARTIAL** | Cells/pairs are rederived at [completeness.py:135](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:135). Other content, inventory, lineage, and clock gaps still permit false certification: 85–87. |
| 66 | **CLOSED** | Qualified eager `NO_REPLAY_EVIDENCE` accepted at [completeness.py:209](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:209). |
| 67 | **PARTIAL** | Registry and address fallback implemented, but identity allocation at [timing.py:127](/home/muhit/ms/sm120-nulltest/fullmodel/timing.py:127) still permits reuse: 84. |
| 68 | **PARTIAL** | Allocation metadata and full recognized calls enter [ideal_costs.py:105](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:105). Omitted custom-op calls remain: 91. |
| 69 | **CLOSED** | Full allocation extent used at [ideal_costs.py:353](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:353); unknown footprint suppresses L2 pricing at [line 394](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:394). The stride-2 fixture measures approximately 128 MiB. |
| 70 | **PARTIAL** | Graph verification enforces 50 at [graph_verify.py:173](/home/muhit/ms/sm120-nulltest/fullmodel/graph_verify.py:173); 51 windows with one empty are rejected. Ideal costs still accept caller-selected counts: 90. |
| 71 | **PARTIAL** | Complete name-set/count/duration comparison exists at [ideal_costs.py:277](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:277), but silently tolerates duration discrepancies: 92. |
| 72 | **PARTIAL** | False dependency proof and failed PDL PTX now reject at [boundary_bench.py:222](/home/muhit/ms/sm120-nulltest/fullmodel/boundary_bench.py:222). Expected-counter bypass remains: 93. |
| 73 | **CLOSED** | Exact-name, resolved, unambiguous coverage at [perturbation.py:147](/home/muhit/ms/sm120-nulltest/fullmodel/perturbation.py:147). The collision/unresolved fixture receives zero coverage. |
| 74 | **CLOSED** | Previous verdict invalidation and failure overwrite at [session_lib.sh:152](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:152); audit excludes its own previous outcome at [completeness.py:90](/home/muhit/ms/sm120-nulltest/fullmodel/completeness.py:90). |
| 75 | **PARTIAL** | Started attempts invalidate outputs at [session_lib.sh:107](/home/muhit/ms/sm120-nulltest/scripts/session_lib.sh:107), but invalidation has incorrect scope and misses deadline skips: 79, 94. |
| 76 | **CLOSED** | Boundary export and analysis use bounded `run_aux` calls at [session2.sh:102](/home/muhit/ms/sm120-nulltest/scripts/session2.sh:102). |
| 77 | **PARTIAL** | Remote freshness checked at [pod_run.sh:70](/home/muhit/ms/sm120-nulltest/scripts/pod_run.sh:70); the receipt does not prove that those remote contents were extracted: 95. |

For the explicit window question: **measurement and decomposition use the same `split_passes` result**, at [ideal_costs.py:71](/home/muhit/ms/sm120-nulltest/fullmodel/ideal_costs.py:71). An outside-window fixture kernel did not enter published rows.

JSON serialization preserves `node_ids`; finding 84 defeats their lifetime guarantee. Ordinary `run_aux false` under `set -u` continued with unchanged `T0` and deadline accounting. Finding 96 is an explicit abort path.

EXECUTION-READY: no