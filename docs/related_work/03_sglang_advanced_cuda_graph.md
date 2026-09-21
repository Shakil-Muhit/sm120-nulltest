# SGLang team, "Advanced CUDA Graph Techniques" (LMSYS blog, 2026-08-17)

**Reading-order item 3.** Tag: F2 THREAT/OVERLAP, F5 ADJACENT.

## Research question
For LLM decode serving, how much do three ways of graphing a forward differ: one full CUDA graph, a graph split at boundaries with copies between segments, and compiler-driven piecewise graphs?

## Main contribution (authors' claims, as read)
Measured speedups over eager for the three variants on one serving configuration: full CUDA graph 1.93×, breakable CUDA graph (segments with copies) 1.70×, torch.compile piecewise CUDA graphs 1.45×, on gpt-oss-120b with tensor parallel 4 on four GB300 GPUs.

## Evidence and scope
One model, one GPU type (GB300, data-center Blackwell), decode-phase serving, SGLang runtime. Baselines: eager SGLang. Validation: throughput/latency measurement as reported; no CIs given in the post as read.

## Exact version and date read
Blog post dated 2026-08-17 at lmsys.org; read 2026-09-20.

## Relevant source locations
The results table with the three variants; the section describing the breakable-graph mechanism.

## Relationship to this study
- Overlap: the ordering full > segmented-with-copies > compiler-piecewise is the same ordering this study measures between C3 (one graph, 28.87 ms at 8 s) and C1 (96 graphs with staging copies, 35.61 ms) (`SESSION_RESULT_v2.md`, arms table; `LEDGER.md` X1).
- Difference: different workload family (LLM decode vs a band-split audio model), different GPU class (GB300 vs GeForce RTX 5090, SM120), and SGLang reports totals where this study decomposes the segmented case into copies, lost fusion, idle and selection.
- Scientifically meaningful: the ordering is corroborating evidence, not a claim of this study; the decomposition is.

## Consequence for this work
Cite for the full-vs-piecewise ordering; do not present the ordering as a finding.

## Unresolved
Whether the post's "piecewise" variant used Inductor's graph partition or vLLM-style splitting; whether copies were counted separately.
