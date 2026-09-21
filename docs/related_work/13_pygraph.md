# PyGraph — "PyGraph: Robust Compiler Support for CUDA Graphs in PyTorch" (arXiv 2503.19779v2, 17 Dec 2025)

**Additional full note requested by the Phase-3 handoff.** Tag: F2 THREAT/OVERLAP; F7 reconcile.

## Research question
Why does PyTorch 2's CUDA-graph support (cudagraph trees under `torch.compile`) leave performance on the table, and which compiler changes recover it?

## Main contribution (authors' claims, as read)
Three techniques on top of PyTorch 2's CUDA graphs: (1) CUDA-graph-aware data placement/copy reduction ("CGCT" in the paper's naming as read), which avoids the per-replay copies of inputs into static buffers by changing how parameters and inputs are placed; (2) parameter indirection ("PI"), passing pointers rather than copying tensors into graph-owned memory; (3) selective CUDA graphs ("SCG"), a cost model that declines to graph regions where graphing slows things down. Numbers as read: input copies account for 24 % and 17 % of end-to-end time in DALLE2 and DeepRecommender under PyTorch 2's CUDA graphs; PyGraph gives a 29 % geomean improvement over PyTorch 2 with CUDA graphs across the benchmark set; about a quarter of the graphs the selective policy examines would degrade performance if graphed. Hardware: H100 NVL.

## Evidence and scope
TorchBench-style model set on one data-center GPU; execution mode `torch.compile` with cudagraph trees; baselines eager, PyTorch 2 default, PyTorch 2 with CUDA graphs; validation by end-to-end timing.

## Exact version and date read
arXiv 2503.19779v2 (17 Dec 2025) HTML; read 2026-09-20.

## Relevant source locations
§2 motivation (copy overhead figures); §4 the three techniques; §6 evaluation tables.

## Relationship to this study
- Overlap: PyGraph identifies per-replay input copies as a first-order cost of cudagraph trees and fixes it in the compiler. The study measures the same cost class in the segmented C1 arm: staging copies of 2.502 ms per 8 s pass, 63 kernels, 25.8 MB (`docs/phase3/x1/x1_decomposition_352800.json`), and in C3 finds the copies gone once the forward is one graph.
- Reconciliation with F7: PyGraph's copies are input/parameter staging; the study's X6a re-casts are `aten._to_copy` operations on fp32 parameters captured inside the graph (`docs/phase3/00_first_message.md` addendum). They are different copies with different fixes (placement/indirection versus freezing or pre-casting, E1).
- Difference: PyGraph changes the compiler; the study uses the public path unchanged (rule R2, external wrappers only) and attributes cost by mechanism.
- Scientifically meaningful: yes for the attribution on a non-LLM, non-TorchBench model; no for the existence of copy overhead.

## Consequence for this work
F2 cites PyGraph for the compiler-side copy analysis; F7 states the re-casts are not PyGraph's copies. E1 (freezing or pre-cast) is the study's fix for X6a and is compared against PyGraph's indirection in words only.

## Unresolved
The exact technique names and section numbers were recorded from one reading and should be re-checked against the PDF before the paper cites them; whether PyGraph's selective policy interacts with graph breaks of the C1 kind.
