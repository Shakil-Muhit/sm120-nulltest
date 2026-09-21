# NVIDIA, "CUDA Graph Best Practice for PyTorch" — Troubleshooting: Performance Issues, and Torch Integration

**Reading-order item 2.** Tag: F2, F7 THREAT/OVERLAP.

## Research question
Not a study. Vendor guidance: what makes CUDA-graph deployment of PyTorch models slow, and how the torch integration (manual capture, `torch.compile` cudagraph trees) behaves.

## Main contribution (authors' claims, as read)
Named failure modes with prescribed remedies. The pages read name, in their own headings and sentences: "Too Many Small CUDA Graphs" (fragmentation of one forward into many graphs, each paying launch and serialization cost); input-copy overhead at every replay when inputs are not written into the static buffers directly; channel serialization when graphs are launched back to back; and, for autocast, that a "cast operation is part of the captured graph" and the advice to "Always Disable Cache with CUDA Graphs" (the autocast weight-cast cache is unsafe with capture).

## Evidence and scope
Prose and small code examples; no measurements, no models, no GPU named for a number. Execution mode: manual `torch.cuda.CUDAGraph` capture and `torch.compile` with cudagraph trees.

## Exact version and date read
docs.nvidia.com/dl-cuda-graph, "latest" (the pages carry no version string or date in the body read). Read on 2026-09-20 (`troubleshooting/performance-issues.html`, `torch-cuda-graph/torch-integration.html`).

## Relevant source locations
Performance-issues page: sections on graph fragmentation, input copies, serialization. Torch-integration page: the autocast section.

## Relationship to this study
- Overlap: the mechanisms the pages describe are exactly the ones the C1 trace shows (many graphs per pass, staging copies per replay, idle between graphs) and the one X6a prices (casts captured inside the graph and replayed).
- Difference: the pages describe; they attach no number. This study attaches per-pass numbers under one model: 96 graph launches, 63 staging copies, 0.735 ms idle and 2.502 ms of copies per 8 s pass for C1 (`docs/phase3/x1/x1_decomposition_352800.json`); weight re-casts 0.9198 ms at 8 s and 0.8574 ms at 100 ms for C3 (`LEDGER.md` X6a).
- On the cast condition: the pages say the cache must be disabled under capture; the study's determination is that the compiled arms carry the casts as `aten._to_copy` graph operations on parameters that have `requires_grad=True` (672 of 674), replayed by cudagraph trees, and that eager hits the C++ weight-cast cache within a block (`docs/phase3/00_first_message.md` addendum; `docs/phase3/probes/autocast_probe.out`). That is consistent with the pages and adds the condition.
- Scientifically meaningful difference: the pricing and the operating-point dependence (3.19 % of C3 at 8 s versus 36.66 % at 100 ms) are not in the vendor text.

## Consequence for this work
F2 and F7 cite these pages for the mechanism and claim only the attribution. The word "negligible" for re-casts is never used without noting the operating point at which it stops being true.

## Unresolved
Page version and date (no version string found); whether the pages have changed since 2026-09-20.
