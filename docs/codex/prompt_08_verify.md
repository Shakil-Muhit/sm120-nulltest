You are the adversarial verifier for a pre-registered GPU measurement study (gate 08).
Repository: /home/muhit/ms/sm120-nulltest at the current HEAD (the commit after 845ba99) (read it directly).
Registration: prereg_amendments.md. Plan: /home/muhit/ms/plan_ceiling_ledger.md.
Your gate-07 verdict is docs/codex/07_verify.md (findings 97–111, EXECUTION-READY: no); earlier gates: docs/codex/04–06_verify.md.

Every finding 97–111 was worked on. Verify each closure against the CODE with the same adversarial
standard (CPU counterexamples welcome). For each of 97–111 output CLOSED / PARTIAL / OPEN with file:line
evidence. Then hunt for NEW defects. The v5 shakedown on ml-beast (commit 845ba99) produced records that
the v5 validator accepts (run_problems / pair_problems return []) and the collector publishes HEADLINE
cells from; older records under evidence_smoke/ are expected to be rejected.

What changed (claimed):
- completeness v5 (fullmodel/completeness.py): audit_manifest re-derives repo_commit, registration hash,
  clip-manifest hash, code hashes, runtime pins and (where the pins are present) provenance.verify()
  and compares every field (100); every run/matched record's provenance pins must equal the manifest's;
  audit_prerequisites (planted per-pass checks + C3 qualification: status, native_bitwise_equal,
  opcheck all SUCCESS, capture/replay PASS, compiled_fullgraph PASS, output_shape) runs for BOTH
  sessions (98/99); exports parsed by header/row count; graph-verification verdict re-derived from
  per_pass and compared; timing views: per-pass inequalities cp<=union<=span<=window, sum>=union,
  medians == medians of per-pass; ideal costs: rows finite/non-negative, l2<=hbm, recoverable in [0,1],
  class totals == row sums, passes_observed == 50; perturbation ratios re-derived from the three run
  records' samples and compared incl. the caveat rule; source-map coverage re-derived from rows + kernel
  map (97); clock: session_blocks() attributes blocks by registered tag/OP, samples filtered to the
  manifest GPU uuid with numeric fields, coverage requires samples within 5 s of block start/end and
  gaps <= 5 s (101).
- qualify v5: snapshot_complete checks unique_graphs int, dict counters, node_ids unique ints with
  nodes == len and 0<=roots<=nodes, function_ids list of str, manager_id int (104);
  validation_problems: max_abs_diff/max_abs_ref >= 0, shape positive ints, n_elements == prod(shape)
  (105); recomputed_timing cross-checks latency_medians_ms and serialization_delta_pct (102);
  pair_problems: per-arm stored summaries vs samples, chronological (repeat, arm) alternation with
  finite non-overlapping timestamps, all matched_delta_ms fields vs recomputed (102/103); ANVTX
  validated (97).
- perturbation.py: joins on (arm, chunk), manifest, qualify.variant(record) == stored string for both
  records, equal validation reference hashes (106).
- kernel_map: physical lines joined into bracket-balanced logical statements; in-place tensor method
  calls (`x.add_(...)`) count as executable hints and are recorded unparsed (107).
- ideal_costs: '[CUDA mem…]' rows excluded before pass splitting and inventories (110); an observed
  window count != 50 → measurement_status UNREGISTERED_PROTOCOL with passes_requested/observed (111).
- session1.sh: missing references or a failed planted check set PREREQ_OK=0; every measurement is then
  recorded UNTESTED and finalisation still runs (108). session2.sh: planted checks and the C3
  qualification are verified against the manifest at preflight and regenerated under supervision when
  missing; failures leave dependent work UNTESTED (99).
- pod_run.sh terminate takes evidence/.terminating on the pod (refused if a session runs), re-hashes
  under the lock; session launch and session_lib refuse to start under the lock (109).

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

Output: numbered findings continuing from 112 with severity (BLOCKER / MAJOR / MINOR),
file:line, defect, minimal fix; the closure table for 97–111; final line exactly
`EXECUTION-READY: yes` or `EXECUTION-READY: no`.
