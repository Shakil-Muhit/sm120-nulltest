You are the adversarial verifier for a pre-registered GPU measurement study (gate 05).
Repository: /home/muhit/ms/sm120-nulltest at commit b039272 (read it directly).
Registration: prereg_amendments.md. Plan: /home/muhit/ms/plan_ceiling_ledger.md.
Your gate-04 verdict is docs/codex/04_verify.md (findings 46–60, EXECUTION-READY: no).

Every finding 46–60 was worked on. Verify each closure against the CODE with the same
adversarial standard (CPU counterexamples welcome; the shakedown traces under
evidence_smoke/ may be used). For each of 46–60 output CLOSED / PARTIAL / OPEN with
file:line evidence. Then hunt for NEW defects. Standard: "could a hostile reviewer show that
a published number is not what the registration says it is?"

What changed (claimed):
- fullmodel/qualify.py (new, QUALIFY_VERSION 2): run_problems / sdr_problems / pair_problems
  re-derive eligibility from records; canonical variant() with stream count;
  reference_hashes(). Used by run_full_arm (stored flag = union), matched_run, collect_full,
  completeness.
- timing.validate / sdr_gate.run_gate persist ref_sha256 / a_output_sha256 for every
  lineage-checked reference; cudagraph_tree_state keeps manager_id, node_ids, function_ids;
  recording_stable compares them as sets.
- run_full_arm: --clips-manifest hash compared with the session manifest before SDR/refs;
  explicit planted_fault / nvtx_pass keys. matched_run: stream_count, iters/repeats recorded.
- collect_full v5: independent qualification; restarts must share one hashed reference set;
  matched arms joined to r1 by manifest, operating point and full canonical variant.
- graph_verify: qualified = registered pass count given and met AND _exports.json says
  complete with graph_trace_mode node; otherwise verdict UNRESOLVED.
- ideal_costs: per_pass_sequence returns windowed per-name instances/time; all rows and
  class totals come from the windows; the aggregate kernel summary is reconciled only; no
  windows → no rows. graph_signature covers kind/name/arg metas/kernel source hash/load+store
  sites/numeric literals for every call. Iteration space = positional integer literals bound
  one-to-one to the kernel's numeric params (kernel_map numeric_params + numeric_literals),
  else unresolved. L2 footprint = union of distinct base allocations at full size.
  kernel_map: full call lines, source_sha256 per kernel, header relabelled.
- boundary_bench.analyze_trace: a segment without a qualification case is UNRESOLVED;
  _qualification summary. perturbation: affirmative prerequisites (compile arms
  counters_stable is True, variant recorded, 50 recorded passes); all ratios cleared on any
  failure; source map keeps graph-qualified entries and marks name collisions AMBIGUOUS.
- timing_views: '[CUDA mem…]' rows are memops, excluded from kernel sums/critical path.
- completeness v2: content-validated registered items (see §5 of the registration).
- scripts: run_traced invalidates stale exports, bounds nsys stats, records export exit;
  graph_verify / timing_views / ideal_costs / perturbation run under run_aux; finalize_session
  supervises collection + audit and writes an explicit INCOMPLETE record on failure.
- scripts/pod_run.sh: extraction validates the archive and file count and writes a receipt
  bound to POD_ID + sha256; terminate requires that receipt for this pod and an unchanged archive.
- setup.sh: nsys package selection. prereg_amendments.md: §5 C2 allowance at session level,
  observer identity, independent qualification, completeness list, ideal-cost rules,
  graph-verify/timing-views rules, perturbation clearing.

Specific questions:
1. Is there any remaining path by which collect_full or completeness can publish a HEADLINE
   or COMPLETE from records that the registration would reject?
2. In ideal_costs, is the windowed measurement consistent with the variant decomposition's
   pass windows (same split_passes call), and can a kernel outside any window leak in?
3. Does the recording observer's node_ids list survive JSON serialisation and comparison
   across snapshots within one process without false positives (Python id reuse)?
4. Session scripts: with `set -u` semantics and the new run_aux wrappers, can a stage abort
   the whole session unexpectedly, or leave the deadline accounting wrong?
5. Anything in the registration that the code still contradicts.

Output: numbered findings continuing from 61 with severity (BLOCKER / MAJOR / MINOR),
file:line, defect, minimal fix; the closure table for 46–60; final line exactly
`EXECUTION-READY: yes` or `EXECUTION-READY: no`.
