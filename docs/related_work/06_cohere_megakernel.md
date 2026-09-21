# Cohere, "Megakernels" (engineering blog, 2026-09-08)

**Reading-order item 6.** Tag: F6 THREAT, F5 OVERLAP.

## Research question
How far can one persistent megakernel move a weight-bound MoE decode step toward its bandwidth speed-of-light, and where do the gains come from?

## Main contribution (authors' claims, as read)
A single-kernel decode step for a 30B-parameter, 3.3B-active MoE on an H100, batch size 1: throughput from vLLM's 185 tok/s to 292 tok/s, a 1.58× gain, stated against a 470 tok/s speed-of-light (6.6 GB of weights per step); memory-bandwidth utilization ("MBU") 39 % → 62 %. The post names three sources of gain: wave quantization (no kernel boundary to round the last wave up to), false dependencies between operations that a kernel boundary imposes, and weight prefetch that starts before the activation dependency is satisfied. The launch gap itself is not listed as one of the three.

## Evidence and scope
One model, one GPU (H100), decode at batch 1, vLLM baseline. Measurements are throughput numbers in the post; no CIs, no per-mechanism breakdown with numbers as read.

## Exact version and date read
Blog post dated 2026-09-08; read 2026-09-20.

## Relevant source locations
The throughput comparison; the "where the gains come from" list.

## Relationship to this study
- Contradiction of F6 as originally worded: a model whose weight traffic dominates still gains 1.58× from sub-kernel execution, through achieved-bandwidth mechanisms rather than boundary removal.
- Overlap with X12: the study's X12 is a boundary-removal figure only (0.2019 ms, 8.63 % at 100 ms; `LEDGER.md` X12), and the three Cohere mechanisms are listed there as excluded.
- Applies to the band structure: the 60 per-band MLPs of the mask estimator run as many small GEMMs (per-band shapes, M = frames per band, K from 24 to 520; `docs/phase3/probes/band_shapes_probe.json`), which is where wave-quantization effects would live in this model. This is an inference; no measurement of wave quantization exists in the evidence.
- Scientifically meaningful: yes; the study's own terms (weight floor and boundary cost at 100 ms) are of the same order rather than separated (`SESSION_RESULT_v2.md`, boundary section).

## Consequence for this work
F6 is worded as: boundary-removal gains pay only when weight_bytes/bandwidth is small next to N × boundary cost; achieved-bandwidth improvement is a separate lever. X12 stays a figure, not a bound, until its removable costs are justified.

## Unresolved
How much of Cohere's 1.58× each of the three mechanisms contributes (not broken out in the post).
