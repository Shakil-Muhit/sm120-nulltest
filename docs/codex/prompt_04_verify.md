You are the adversarial verifier for a pre-registered GPU measurement study (gate 04).
Repository: /home/muhit/ms/sm120-nulltest (read it directly). Registration: prereg_amendments.md.
Plan: /home/muhit/ms/plan_ceiling_ledger.md. Your previous verdict (gate 03) is in
docs/codex/03_verify.md: 9 blockers (33–41), majors 42–44, minor 45, EXECUTION-READY: no.

Every one of those findings was worked on since. Verify each closure against the CODE, not
against my description. For each finding 33–45 output: CLOSED / PARTIAL / OPEN with the
file:line evidence. Then hunt for NEW defects with the same standard as before:
"would a hostile reviewer be able to show that a published number is not what the
registration says it is?"

What changed (claimed):
- fullmodel/provenance.py: runtime pins (torch 2.8.0+cu128 / triton 3.4.0) enforced in verify();
  code_hashes() covers fullmodel/**, arms/*, scripts/*, configs/*, frontend.py,
  frontend_rms_fused.py, setup.sh, prereg_amendments.md; identity() bound and compared field by
  field in load_manifest(evidence_dir, paths, env) (clean tree required); bind_artifact /
  check_artifact lineage sidecars per reference file.
- fullmodel/timing.py: recording_state() = counters + cudagraph-trees node identity (roots and
  descendants), recording_stable() (None when the observer is missing); validate() binds/checks
  lineage per reference file.
- fullmodel/sdr_gate.py: lineage per eager-A clip output.
- fullmodel/run_full_arm.py: recording_state snapshots at every phase boundary; counters_stable
  None → ineligible; EXACT 5×50 protocol; ANVTX never headline; effective_variant string.
- fullmodel/matched_run.py: recording snapshots before/after joint settle, validation and every
  block; new graph breaks during the joint settle → ineligible; exact 5×50 required; per-arm
  validation with lineage; effective_variants.
- fullmodel/collect_full.py (v4): headline needs r1,r2,r3 all eligible, one manifest, one
  variant, SDR from the r1 record of the same manifest/variant; matched pair headline needs
  m1,m2,m3 + both arms' SDR. fullmodel/completeness.py: mandatory-item audit → session_status.json.
- fullmodel/graph_verify.py (v3): staging = multi_tensor_apply with Copy<…> / direct copy only;
  contiguous unique pass ids; positive captured computation; exports_complete; launch-instance
  wording; memops category.
- fullmodel/ideal_costs.py (v3): ordered unique variant decomposition with SDPA/GEMM
  reconciliation; pass-count check; per-call L2 footprint; ABI-signature dtypes; byte bound
  min(distinct footprint, sites × iteration space); bound_not_tight flag; recoverable clamp.
  fullmodel/attrib/kernel_map.py: signature dtypes, rand_strided input dtypes, own-name asserts,
  accessed_elems(), load/store site counts.
- fullmodel/attn_backend_bench.py (v3): deployed backend per (chunk, axis) from C3 node traces
  (12 SDPA executions per pass, Triton preprocessing excluded); UNRESOLVED otherwise, no ratio.
- fullmodel/boundary_bench.py: analyze_trace joins the traced run's own qualification JSON and
  requires N×REPEATS kernels per segment. fullmodel/perturbation.py: run qualification
  (VALID, PASS, 1×50, manifest, same variant) else UNRESOLVED incl. the caveat.
- scripts/session_lib.sh: run_aux banners to stderr + aux_<name>.json; scripts/session1.sh:
  C3 qualification writes its own JSON (validated for frames 11/51/201/801); both sessions end
  with completeness.py. scripts/session2.sh: attention bench takes the two C3 trace bases.
- prereg_amendments.md: headline/pair rule, recording observer, runtime pins + manifest scope,
  attention per-axis rule, ideal-cost pricing scope + byte model, boundary/perturbation
  qualification, completeness verdict, SDR non-finite sentence narrowed.
- fullmodel/padbatch.py: equivalence claim wording (algebraic, verified at loose tier).
- docs/ledger_spec.md: X6 uses recoverable_share_hbm_model; X6a (autocast weight re-casts)
  registered as a sub-row before session data.

Also check specifically:
1. Is there any path by which a run JSON can be headline_eligible without lineage-checked
   references? Any path by which collect_full can publish a HEADLINE from mixed manifests?
2. matched_run: does the recording observer actually observe BOTH arms' graphs (one global
   tree manager per device)? Is a re-recording of arm A during arm B's block detected?
3. ideal_costs: is the ordered decomposition sound for C1 (graph-broken, several graphs) —
   what if the same Triton kernel name appears in two graphs with different bodies?
4. graph_verify: with --cuda-graph-trace=node, do replayed memops carry the graph launch
   CorrId in nsys 2026.5 (we classify them as memops_graph)? If unsure, say what evidence
   would settle it and whether the FULLY_GRAPHED verdict depends on it.
5. completeness.py: does its mandatory list match §5 of the registration exactly
   (session 1: matrix, matched pairs, planted checks, C3 qualification; session 2: sweep,
   traces, benches, perturbation)? List any mismatch.
6. session scripts: any remaining place where a failed step is silently absorbed (|| true,
   missing set -e context) in a way that would let the session claim completeness?

Output format: numbered findings continuing from 46, each with severity (BLOCKER / MAJOR /
MINOR), file:line, the exact defect, and the minimal fix. End with the closure table for
33–45 and a final line exactly `EXECUTION-READY: yes` or `EXECUTION-READY: no`.

Shakedown evidence (engineering data from ml-beast GPU 4, never session evidence) is under
evidence_smoke/ (git-ignored): s1smoke/ and s2smoke/ are outputs of the PREVIOUS tree
(4b9f74e) run through scripts/session1.sh and session2.sh in SMOKE mode; ideal_C3_352800.json,
ideal_C3_4410.json and ideal_C2_4410.json are outputs of the NEW ideal_costs.py on those
traces (C2: 33 compiled graphs → UNRESOLVED by design; C3: unique decomposition). A re-smoke
of the new tree is running; its results will be reported separately.
