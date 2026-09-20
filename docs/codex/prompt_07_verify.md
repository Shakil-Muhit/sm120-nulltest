You are the adversarial verifier for a pre-registered GPU measurement study (gate 07).
Repository: /home/muhit/ms/sm120-nulltest at commit 845ba99 (read it directly).
Registration: prereg_amendments.md. Plan: /home/muhit/ms/plan_ceiling_ledger.md.
Your gate-06 verdict is docs/codex/06_verify.md (findings 78–96, EXECUTION-READY: no); earlier gates: docs/codex/04_verify.md, 05_verify.md.

Every finding 78–96 was worked on. Verify each closure against the CODE with the same adversarial
standard (CPU counterexamples welcome). For each of 78–96 output CLOSED / PARTIAL / OPEN with file:line
evidence. Then hunt for NEW defects. Note: evidence_smoke/ still holds records from older commits (they
lack the new per-pass fields and are expected to be rejected by the v4 validator).

What changed (claimed):
- matched_run: 'pair_eligible' initialised to None (78); per-arm gate_snapshot recorded right after
  each arm's gate and used for reconciliation.
- timing.compare_outputs persists violations / n_elements / max_abs_ref and forces ok False when
  allclose and the element count disagree (80); registry identities from a strictly monotonic counter (84).
- qualify v4: validation_problems checks shapes, finiteness flags, per-pass tolerances == registered tier,
  violations == 0, max_abs_diff <= atol + rtol*max_abs_ref, lineage hash, recomputed status; a tier
  override on a non-planted record disqualifies (80). sdr_recomputed requires clip indices exactly 0..7,
  n_clips, finite paired SDR_ref terms, stored delta == difference, recomputed mean/status (81).
  snapshot_complete + required_phases(rec) give the exact single-arm schema; the matched schema is
  rounds+2 joint-settle snapshots, before/after validation, 2×repeats block snapshots, all complete (82).
  gate_problems(arm, gate, snapshot) shared by single-arm and matched paths: status, suppress_errors,
  counters, zero-break rule for C3/H, reconciliation with the after-settle / per-arm gate snapshot (83).
  recomputed_timing cross-checks every nested summary (block median/min/max, timing summaries, repeat
  ids, finite timestamps) (88); recomputed_matched_delta is NaN-safe. trace_problems() qualifies
  one-block traced/perturbation runs (89).
- completeness v4: traced runs and perturbation runs via trace_problems; graph_verify / timing_views /
  ideal_costs checked by payload (qualified, registered protocol, 50 passes, finite medians, priced rows
  for C3); per-trace aux records enumerated; planted checks per pass (seeds, ok False, magnitude,
  violations>0); C3 qualification payload (PASS + native_bitwise_equal + opcheck); manifest identity
  fields; required reference inventory enumerated per OP/seed/clip, re-hashed, and every recorded hash
  joined to its artifact (86); clock coverage scoped to this session's log, matched blocks included,
  finite timestamps and 7-column samples required (87); exit 0 when the record is written.
- session_lib.run_traced: invalidates run JSON/report/exports for every tag before the deadline check;
  main-trace reductions and their aux records only for tag 'trace' (79, 94). run_one/run_matched
  invalidate before the deadline check. session1.sh: C3 qualification failure sets C3/H work UNTESTED
  and continues (96).
- ideal_costs: passes != 50 → explicit UNREGISTERED_PROTOCOL record without rows (90); reconciliation
  reports the max relative duration discrepancy and the count outside 1% (92); graph_signature includes
  the wrapper's unparsed executable lines. kernel_map: FALLBACK_RE matches unassigned calls in any
  torch.ops namespace; unrecognised executable lines recorded in graph['unparsed'] (91).
- boundary_bench.analyze_trace derives the expected counter (25N / 27N) and requires both recorded
  counters to equal it (93). pod_run.sh extract: refuses while a tmux session runs; content hash before
  and after the transfer must match; the extracted archive's own content manifest must equal it (95).
- prereg_amendments.md updated accordingly.

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

Output: numbered findings continuing from 97 with severity (BLOCKER / MAJOR / MINOR),
file:line, defect, minimal fix; the closure table for 78–96; final line exactly
`EXECUTION-READY: yes` or `EXECUTION-READY: no`.
