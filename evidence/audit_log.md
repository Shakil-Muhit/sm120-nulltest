# Audit sign-off — sm120-nulltest vs pre-registered spec
Auditor: Muhit (session owner). Date: 2026-09-01.
Method: conformance audit per audit map items 1–7; invariants derived from the
signed prereg ("what would violate this, where would it live").

## Items
1. Timing rig — PASS (3 checks from smoke evidence, 1 caveat documented).
   Pending question resolved via fix pass: kernels_per_pass floor (=100) for
   graph arms now enforced in collect.py + traces.sh, behaviorally tested
   (6 scenarios incl. absent-count and poisoned-busy rejection).
2. Validation integrity — PASS / scheduled.
   2(i): 2 replay sites (common.py:139 ReplayBandSplit, :179 step closure),
   2 copy-guarded. Mechanism verified: copy_ in (replay has no argument
   channel), clone out (replay recycles its output buffer), sync between
   (device-wide, error attribution), ref-before-step ordering (empty borrow
   window). 2(iii) planted fault: DEFERRED to pod opening. Registered
   prediction: FAIL both seeds, max_abs_diff large and identical across seeds.
3. Arm G composition — PASS. Fork event-guarded (N side streams), per-child
   join events, record_stream with capture guard, run() inside capture
   region, timed loop = copy+replay(+sync per 5a mode), throughput mode
   drains before timer stop. Corroborated by F4 busy>wall (0.1896/0.1101).
4a. Process isolation & run order — PASS. traces.sh: bash-level loop, one
   process per arm×chunk cell; --cuda-graph-trace=node gated at startup
   (refuses busy numbers without it); capture range = 5a latency block only;
   stale busy files removed before each pass; VALID-status refusal in
   extractor; run order chunk-major per header.
   NOTE: on-disk traces.sh found corrupted (self-paste at :131, mtime
   Aug 31, pre-session; bash -n failed — could never have produced numbers).
   Reconstructed by Code from embedded complete copy; verified by diff
   against the audited Sep 1 paste — sole deltas are the item-2 floor
   guard (verified again post-reconstruction in external review, 2026-09-01).
   Stamp re-anchored to reconstructed file.
4b. Validation tolerance — RULED TWO-TIER (was uninspected agent default
   1e-4/1e-5 for all arms). Tight (A/B/D/F/G + E per as-built): 1e-4/1e-5 —
   no compiler-regenerated kernels; B measured bitwise, G's Triton front-end
   measured ≤1e-4 in smoke despite reordered reductions (fp32 accumulation
   both sides). Loose (C1/C2): 1e-2/1e-3 — fp16 eps 9.77e-4, √K·ε over
   K≤512 reordered reductions; atol governs near-silent stem regions.
   Classification rule in code: tier = compiler-regenerated kernels present;
   explicit dict, no default entry (unlisted arm must KeyError).
   E reclassified tight per as-built (fused-Triton + replay, no Inductor);
   Code confirmation: applied, VAL_TOL shows E at (1e-4, 1e-5), loose
   tier = C1/C2 only. C1/C2/E NEVER YET VALIDATED — pod opening items.
5. Provenance — PASS. Model built only via pinned msst clone's own path
   (get_model_from_config → load_start_checkpoint, mirrors upstream
   inference.py:211-216); no model-defining files in repo; arms never import
   the model path (scope guard). Checkpoint: sha256 format-gated, mismatch →
   delete + one retry → delete + exit 1; no path leaves a corrupt file
   cached. use_amp + msst_commit recorded in run JSON (deviation 15).
   No sourcing fallbacks anywhere: refuse-and-crash before numbers exist.
6. setup.sh extend-don't-rewrite — PASS. diff original vs repo = pure
   insertion after §[4/4] (42a43,105); sections 1–4 byte-identical; tmux
   hint remains final output. Agent's byte-identity claim independently
   verified.
7. Transcription spot-check (nsys CSV re-read, cross-instrument, ~388
   kernel-count sanity) — SCHEDULED: pod-day prep; does not gate the push.

## Fix pass (applied by Code, diffs reviewed)
Floor guard; two-tier tolerance dict; NVTX per-submodule ranges in
p8_trace.py (boundary_mode records granularity); build_spec.md (borrow
discipline; capture-is-final); profiling_crib.md extended; loud stderr on
unresolved msst_commit. Two hardenings beyond spec ACCEPTED (absent-count
rejection; nonzero-exit git warning). build_spec.md at root ACCEPTED.

## Standing caveats
5a sync placement bias (documented); R<1 semantics under streams; smoke
outputs (evidence_smoke/) never citable; glob legacy-CSV finding recorded,
refuted as unreachable in clean sessions.

## Pod session opening block (instrument before readings)
(1) planted fault 2(iii) → (2) validate C1/C2/E at new gates → (3) traces.

## Pending
Amended prereg (Amendments block, P-B1–B4, five registered questions):
commit #2, required before pod session, not before first push.