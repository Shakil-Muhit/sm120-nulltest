# MPK — Jia et al., "Mirage Persistent Kernel: compiling LLM inference into a megakernel" (arXiv 2512.22219v1, Dec 2025); with the accompanying blog example

**Reading-order item 7.** Tag: F5 OVERLAP, F6 THREAT.

## Research question
Can a compiler turn an LLM inference graph into one persistent kernel with fine-grained task scheduling, and how much of the gain comes from pipelining across tasks versus from removing kernel boundaries?

## Main contribution (authors' claims, as read)
A compiler and runtime that partitions the computation into tasks executed by a persistent kernel with an on-GPU scheduler. Ablation: cross-task software pipelining "reduces task runtime by 1.2–1.3×"; overlap of compute with communication adds about 1.1×. End-to-end gains over vLLM/SGLang of 1.0–1.7×, larger at batch size 1 (1.4–1.6×) and smaller at batch 16 (1.0–1.3×), on Qwen3-8B on a B200 (as read in the v1 HTML). The blog example (Medium, Z. Jia): Qwen3-8B on an A100-40GB from 14.5 ms to 12.5 ms per step against a 10 ms floor computed as 16 GB / 1.6 TB/s.

## Evidence and scope
LLM decode/prefill on B200 (paper) and A100 (blog); baselines vLLM and SGLang; ablations by mechanism. No per-launch µs figure was found in the v1 text read (the sweep attributes "3.8/0.8 µs" to MPK; that attribution could not be located and is left unresolved here).

## Exact version and date read
arXiv 2512.22219v1 HTML, read 2026-09-20; blog post read 2026-09-20.

## Relevant source locations
Ablation section (pipelining and overlap factors); end-to-end results by batch size; blog example arithmetic.

## Relationship to this study
- Overlap: MPK's ablation sizes the mechanism that X12 excludes; the study's X12 note lists "MPK pipelining 1.2–1.3× per task" as an excluded mechanism (`LEDGER.md` X12).
- Contradiction of F6 as originally worded: a weight-floored model (blog example) still gains 1.16×.
- Difference: MPK is a compiler for LLM graphs; the study's persistent-execution row is a figure computed from a boundary microbenchmark slope and the H arm's kernel count (`LEDGER.md` X12 evidence field), with no persistent kernel built (kernel work is the owner's lane, rule R5).
- Scientifically meaningful: yes for the framing of X12; the study makes no megakernel claim.

## Consequence for this work
X12 carries the excluded-mechanism list with MPK's factor. "Megakernels would not pay" is never written.

## Unresolved
The per-launch µs numbers the sweep attributes to MPK; the exact batch sizes behind the 1.0–1.7× range.
