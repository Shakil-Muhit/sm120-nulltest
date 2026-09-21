# Hazy Research, "Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B" (blog, 2025-05-27)

**Reading-order item 8.** Tag: F5, F6 OVERLAP; F7 CITE.

## Research question
For Llama-3.2-1B decode at batch 1, how close can a hand-written megakernel get to the weight-bandwidth floor, and what limits it?

## Main contribution (authors' claims, as read)
A megakernel for Llama-1B decode on H100 and B200. Launch costs quoted: about 2.1 µs per stream launch and about 1.3 µs per graph-launched kernel on H100. Floor arithmetic: 2.48 GB of weights at 3.35 TB/s gives about 1,350 forward passes per second, against about 770 predicted for 112 kernels at about 5 µs each, so launch removal pays. B200 breakdown of a roughly 600 µs pass: about 30 µs waiting on weights, about 250 µs in stores and activation loads, and about 120 µs of the megakernel's own synchronization and setup.

## Evidence and scope
One small model, two data-center GPUs, decode at batch 1; the authors' own kernel; numbers from their instrumentation; no CIs.

## Exact version and date read
Blog post dated 2025-05-27; read 2026-09-20.

## Relevant source locations
The launch-cost paragraph; the floor arithmetic; the B200 time breakdown.

## Relationship to this study
- Overlap: the study's per-node costs are the same kind of number on a different GPU: 4.883 µs/node for stream launches, 0.732 µs/node for graph replay, 0.579 µs/node for graph replay with programmatic dependent launch, from the slope of a dependent chain of trivial Triton kernels, plus 0.288 µs and 0.160 µs median serial gaps from the node-level trace (`SESSION_RESULT_v2.md`, boundary section; `evidence/boundary_bench.json`). Hazy's 1.3 µs per graph kernel on H100 and the study's 0.73 µs on the RTX 5090 are both device-side replay costs; they are not the same measurement (Hazy's method is not spelled out in the post as read).
- Prior art for F6's arithmetic: Hazy's floor-versus-kernel-count comparison is the argument F6 makes; the study's version at 100 ms has a weight floor and a boundary term of the same order (`SESSION_RESULT_v2.md`, boundary section), so the two terms are not separated.
- F7 analogue: Hazy's 120 µs of in-megakernel sync and setup is a fixed per-pass cost inside a single kernel, the counterweight to any fixed-cost removal claim.
- Scientifically meaningful: the SM120 numbers were not found in the sources reviewed; the argument form is Hazy's.

## Consequence for this work
F5b is stated as "per-node stream, graph and PDL-graph costs on the RTX 5090, not found in the sources reviewed", with Hazy's H100 numbers beside them. F6 uses the boundary-removal wording. X12's excluded-mechanism list cites the 120 µs as the counterweight.

## Unresolved
Hazy's measurement method for the 2.1/1.3 µs figures.
