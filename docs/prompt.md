You are building the benchmark session repo for a pre-registered GPU experiment. I will
audit every file before it runs, so: clean, well-documented code, no cleverness, and
every design decision traceable to the pre-registration document I'm attaching. Where
the prereg and your judgment conflict, THE PREREG WINS — flag the conflict in a comment,
do not silently resolve it.

CONTEXT (enough to build correctly — the science lives elsewhere)
I've measured a MelBandRoformer audio model's front-end as launch-bound on an RTX 5090:
~388 kernel launches/pass from a Python loop over 60 per-band RMSNorm+Linear ops with
RAGGED per-band widths (K_i ∈ ~[48, 800]), GPU idle 77–83% of wall time. The experiment
("null test") measures whether generic tooling — CUDA Graphs, torch.compile — deletes
that idle, or leaves structure only a custom grouped kernel can claim. Seven arms (A–G),
two chunk regimes, pre-registered decision rule. Runs on a rented RunPod 5090 pod
(Ubuntu 22.04 template, torch 2.8.0+cu128, triton 3.4.0, root shell, HW perf counters
LOCKED — nsys timelines and wall timing work; ncu does not). Pods are ephemeral: the
repo must go from `git clone` to measuring in minutes.

FILES I AM PROVIDING (bundle them; do not rewrite their substance)
1. prereg_null_test.md — THE SPEC. Arms table, decision rule, predictions P1–P8,
   methodology constraints, version matrix. Read it fully before writing any code.
2. frontend.py — my existing profiling harness: standalone MelBandRoformer front-end
   (public ops only), NVTX-staged, cudaProfilerApi capture range, warmup/measured
   iters, --chunk flag, sanity mode with bitwise oracle check. Extend around it;
   preserve its NVTX stage names and capture-range behavior. My fused RMSNorm Triton
   kernel is importable from it (arm D/E/G use it).
3. setup.sh — pod bootstrap (nsys install with fallbacks, counter check, tmux,
   workspace). Include as-is at repo root.

THE FULL MODEL (for the P8 full-forward trace — scripts/p8_full_trace.sh)
The full MelBandRoformer forward comes from PUBLIC sources only — this is a hard
provenance constraint, not a preference:
- Code: ZFTurbo's public Music-Source-Separation-Training repo (git clone in
  setup.sh, pinned to a commit hash recorded in the README).
- Weights + config: the public MelBandRoformer checkpoint I will name in
  configs/checkpoint.txt (URL + sha256). Add a fetch step: download once, verify
  hash, cache under a path that can live on a persistent volume; skip if present.
- p8_full_trace.sh loads the model via the public repo's own model-construction
  path, runs warmups, then ONE nsys pass of the full forward at the 8 s chunk,
  inside a cudaProfilerApi capture range consistent with frontend.py's pattern.
  Extract from the trace: (1) front-end share of GPU time (use the NVTX stage
  boundary if the public path exposes one, else the kernel-sequence boundary —
  document which), (2) the attention stack's kernel count and gap structure.
  Both land in evidence/ as the P8 result.
- SCOPE GUARD: the full model is used ONLY for this trace and (if trivially
  reusable) as the eager reference for validation. Arms A–G remain front-end-only
  — do not wire the full forward into the arm harness.

REPO LAYOUT TO PRODUCE
sm120-nulltest/
├── README.md            # session runbook: clone → setup → tmux → run order → extract
├── setup.sh             # provided
├── frontend.py          # provided
├── prereg_null_test.md  # provided — the spec, versioned with the code that obeys it
├── configs/
│   └── checkpoint.txt   # public checkpoint URL + sha256 (I fill this in myself)
├── arms/
│   ├── common.py        # shared: model build, static I/O buffers, timing loop,
│   │                    #   validation, result schema
│   ├── run_arm.py       # CLI: --arm {A..G} --chunk {352800,4410} --iters N ...
│   └── streams.py       # multi-stream band-loop machinery for F/G
├── analysis/
│   ├── collect.py       # merge per-run JSON → the arms×regimes table (median, min,
│   │                    #   max, per-iteration list preserved)
│   └── traces.sh        # the two nsys timeline captures + kernel-count extraction
├── scripts/
│   ├── run_all.sh       # full session in prereg order: P8 trace FIRST, then
│   │                    #   output_code reads, then arms (separate process per
│   │                    #   regime), then traces, then version-matrix venv check
│   └── p8_full_trace.sh # session opener: nsys of the FULL model forward, 8 s chunk
└── evidence/            # gitignored except .gitkeep — outputs land here, get
                         #   extracted off-pod, never committed from the pod

ARM IMPLEMENTATIONS (spec: prereg arms table — key mechanics per arm)
A  eager baseline, stock F.normalize chain.
B  A + manual whole-forward torch.cuda.CUDAGraph capture. Expect failure at the
   complex STFT (prereg P4): implement graceful fallback to capturing the band_split
   stage only, and RECORD which scope actually captured in the result JSON.
C1 torch.compile(mode="reduce-overhead", dynamic=False).
C2 torch.compile(mode="max-autotune", dynamic=False).
D  my fused RMSNorm kernel (from frontend.py), eager elsewhere.
E  D + capture (same fallback protocol as B).
F  A + multi-stream band loop: fork/join with events, N streams configurable
   (default 4), correctness unaffected by stream count.
G  D + multi-stream + capture — stream-ordered work captured into one graph.
   This is the honest-counterfactual arm; its wall/busy ratio at the 4410 chunk
   feeds the pre-registered decision rule. Get this one most carefully right.

NON-NEGOTIABLE METHODOLOGY (prereg constraints — encode, don't paraphrase)
- One process per (arm × regime): run_all.sh launches fresh python each time.
- Timed region NEVER includes: model build, compile/autotune settling, graph capture,
  warmup. Compile arms: run until recompilation stops (log it) before timing starts.
- Static I/O buffers + .copy_() for all capture arms; capture on a side stream after
  warmup, per torch CUDA Graphs docs.
- ≥50 timed iters, torch.cuda.synchronize() discipline, PER-ITERATION times printed
  and saved (never only aggregates); median/min/max in the JSON.
- Validation: EVERY arm, ≥2 passes with DIFFERENT input data, outputs checked against
  the eager reference within stated tolerance — this catches stale-buffer aliasing
  under capture. Validation runs outside the timed region. A failed validation marks
  the arm's result INVALID in the JSON; it does not crash the session.
- TORCH_LOGS captured to evidence/ for every compile arm (perf hints + recompiles).
- output_code dump for C2 at both chunk sizes saved to evidence/ (torch's
  debug/output_code mechanism) — this is the kernel-inventory evidence.
- GPU-busy per arm: extract from the nsys kernel summary of that arm's trace where a
  trace exists; otherwise from a dedicated short nsys pass — document which.
- Result schema (one JSON per run): arm, chunk, torch/triton/driver versions, GPU
  name, timestamp, iters, per-iter ms list, median/min/max ms, validation status,
  capture scope (B/E/G), stream count (F/G), notes.

README REQUIREMENTS
Runbook a tired person can follow at 2am: exact command sequence including tmux,
expected wall-clock per phase, what "done" looks like per phase, the extraction step
(everything in evidence/ leaves the pod BEFORE terminate), and a troubleshooting
section for the three likely failures (nsys install, capture failure at STFT,
recompile storms).

WHAT YOU MUST NOT DO
- No benchmark tuning, no optimization of my kernel or the model, no refactoring of
  frontend.py's internals, no analysis/plots beyond the collection table.
- No pip installs beyond stdlib + what frontend.py already imports (torch, triton,
  numpy). nsys comes from setup.sh.
- Do not invent arms, metrics, or thresholds not in the prereg. If something in the
  prereg is ambiguous, implement the conservative reading and flag it in README
  under "Deviations & ambiguities" — I audit that section first.

MY AUDIT MAP (structure the code so these are easy to find — I will check them
line-by-line before anything runs)
1. Timing-region boundaries in common.py — that capture/compile/warmup are excluded.
2. The validation function — that pass 2 genuinely uses different data, and that
   capture arms re-copy inputs into static buffers via .copy_() before replay.
3. Arm G's stream+capture composition — fork/join events, single graph replay.
4. run_all.sh ordering — P8 trace first, one process per run, version-matrix venv
   check last.
5. p8_full_trace.sh — that it builds the model through the PUBLIC repo's own path,
   pinned commit, hash-verified checkpoint; no other model source anywhere.
Produce the repo, then print a file-by-file summary with line counts and, for each
audit-map item, the file:line where it lives.
