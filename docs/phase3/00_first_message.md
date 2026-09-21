# Phase 3 — first message back (§8 of the 2026-09-21 handoff)

Files read: plan_ceiling_ledger.md; prereg_amendments.md (§0–§8); LEDGER.md / LEDGER.json; SESSION_RESULT.md;
docs/ledger_spec.md; docs/EXECUTION_REPORT_2026-09-20.md; docs/codex/08_verify.md; "The SM120 Study, Explained"
(rev 2026-09-21, sections 7–12); "Related work for the SM120 study" (verdicts, reading order, concrete edits,
wording, implied experiments, reporting practices); fullmodel/fullarms.py, loader.py, timing.py, the pod's
full_C3_c4410_r1.json and the C3 Inductor wrapper dump.

C3 compile mode: reduce-overhead (fullarms.py:88; pod record compile_mode = reduce-overhead). E2 applies.

Autocast cache condition (belief, to be confirmed on ml-beast, §2.5(viii)):
- compiled arms (X6a's measurement): Inductor traces the autocast weight cast as a graph op — the C3 wrapper at
  4 410 holds 420 `_to_copy` calls per pass whose only input is a graph parameter; the sixty 1536×1536
  mask-estimator weights dominate their time; the C++ autocast cache cannot apply inside a replayed graph.
- eager A: the timing loop enters one autocast context per 50-pass block, yet the trace shows the casts every
  pass (60 small cast kernels + 131 fp16 copy kernels per pass, 3.3 ms at 8 s), so the weight-cast cache is
  not hit; the cache requires fp32 leaf parameters with requires_grad=True and cache_enabled=True; belief:
  the public construction path yields requires_grad=False parameters.
- confirmation: print requires_grad after load_public_model; count cast kernels per pass in eager under the
  four (requires_grad × cache_enabled) combinations from nsys kernel summaries; cite the wrapper dump.

Rules not followable exactly as written:
- R1 GH Archive PushEvent: repository is private; GH Archive records public events only. Substitutes: local
  push log (client-side, labelled), OpenTimestamps proof of each registration commit (from the stamping day),
  GitHub Events API created_at if the owner supplies a token or runs the query (server-side, 90-day retention).
- Pod name: the handoff says "melband profiling" (space); will be used exactly unless the owner says otherwise.
- E7 SASS: requires cuobjdump on ml-beast; otherwise PTX-level griddepcontrol evidence only, stated as such.

## Addendum (2026-09-21, after the ml-beast probe; engineering data, shared node)
The eager-side belief above was wrong. `load_public_model` yields 672/674 parameters with `requires_grad=True`
(674 fp32 leaves, no views). Profiling eager A at 4 410 on ml-beast GPU 4 (torch.profiler, 2 passes after warm-up,
kernels named copy/nocast):
- requires_grad=True, cache_enabled=True (the harness's condition): 597 copy/cast-like kernels per pass, of which
  120 are `float16_copy_kernel` weight/activation copies;
- any of requires_grad=False or cache_enabled=False: 1173 per pass, `float16_copy_kernel` 696 per pass.
So the C++ autocast weight-cast cache IS hit in eager within each 50-pass block (the harness enters one autocast
context per block), and the per-pass cast cost measured as X6a arises only in the compiled arms: Inductor traces
`torch.autocast`'s parameter casts as `aten._to_copy` graph ops (420 parameter-only `_to_copy` calls per pass in
the C3 wrapper at 4 410; the sixty 1536×1536 mask-estimator weights dominate their time), and cudagraph-trees
replays them on every pass. E1(a) tests `torch._inductor.config.freezing=True`, which constant-folds exactly
these casts. Evidence: scratch probe `docs/phase3/probes/autocast_probe.py` (output recorded in
`docs/phase3/probes/autocast_probe.out`), wrapper `evidence/traces/dump_C3_4410_trace/.../model__1_inference_1.1/output_code.py`.
