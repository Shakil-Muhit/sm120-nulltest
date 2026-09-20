You are the adversarial verifier for a pre-registered GPU measurement study (gate 06).
Repository: /home/muhit/ms/sm120-nulltest at commit 43c9e29 (read it directly).
Registration: prereg_amendments.md. Plan: /home/muhit/ms/plan_ceiling_ledger.md.
Your gate-05 verdict is docs/codex/05_verify.md (findings 61–77, EXECUTION-READY: no); gate 04 was docs/codex/04_verify.md.

Every finding 61–77 was worked on (63 was closed in f99c672). Verify each closure against the CODE with
the same adversarial standard (CPU counterexamples welcome; shakedown traces under evidence_smoke/ may be
used; the v3 smoke evidence there was produced by the PREVIOUS observer, so its C3 records lack
identity_mode and are expected to be rejected). For each of 61–77 output CLOSED / PARTIAL / OPEN with
file:line evidence. Then hunt for NEW defects.

What changed (claimed):
- fullmodel/qualify.py v3: validation_problems() (both seeds, registered tier + tolerances, finite
  per-pass diffs, ok flags, lineage hashes, recomputed status), sdr_recomputed() (exactly 8 distinct
  finite clip rows, threshold, stored mean/status must equal recomputed), stability_recomputed()
  from the stored phase snapshots, zero-break counter for C3/H, recomputed_timing() /
  recomputed_matched_delta() from samples with cross-checks of stored summaries, stored-flag
  contradictions in both directions; compile_stable()/recording_stable() moved here (timing delegates);
  pair_problems() recomputes every matched phase transition (joint settle, validation, blocks).
- timing.cudagraph_tree_state: weak-registry lifetime-safe node/manager identities
  (identity_mode 'registry'; 'address' fallback makes stability None).
- collect_full: statistics published only from recomputed samples; pair_summaries() shared.
- completeness v3: re-derives cells/pairs from run records via collect_full.cell()/pair_summaries()
  (never summary labels), re-hashes reference sidecars, mandatory aux outcomes enumerated per session,
  clock log must span every timing block, A/D/F accept qualified NO_REPLAY_EVIDENCE, exit 0 whenever the
  record is written.
- graph_verify: expected passes must be 50 for a verdict (else UNREGISTERED_PROTOCOL).
- ideal_costs: exactly 50 non-empty windows; graph_signature includes base allocation
  shape/stride/dtype + reinterpret flag + full normalised call text; storage_extent() for the L2
  footprint (unknown → no L2 ideal); reconciliation over name sets, counts and durations; kernel_full.
  kernel_map: FALLBACK_RE matches every torch.ops namespace.
- boundary_bench.analyze_trace: trivial chains need dependency_proof True with counter_delta ==
  counter_expected; PDL modes need pdl_ptx True. perturbation: exact-name coverage, ambiguous or
  unresolved entries contribute zero.
- scripts: run_one/run_matched/run_traced rm their outputs (and derived reductions) before the attempt;
  finalize_session removes the previous verdict/summary/aux records first and writes INCOMPLETE on any
  collection/audit failure or timeout; boundary export/analysis under run_aux with invalidation.
- pod_run.sh: receipt carries the remote evidence content hash; terminate recomputes it and refuses on change.
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

Output: numbered findings continuing from 78 with severity (BLOCKER / MAJOR / MINOR),
file:line, defect, minimal fix; the closure table for 61–77; final line exactly
`EXECUTION-READY: yes` or `EXECUTION-READY: no`.
