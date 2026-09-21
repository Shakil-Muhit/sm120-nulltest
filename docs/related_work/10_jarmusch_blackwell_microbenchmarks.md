# Jarmusch, Graddon & Chandrasekaran, "Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks" (arXiv 2507.10789v2, 21 Jul 2025)

**Reading-order item 10.** Tag: SM120 THREAT.

## Research question
How do the GB203 (GeForce RTX 5080, consumer Blackwell) and GH100 (H100 PCIe) differ at the microarchitectural level: memory hierarchy, SM pipelines, and fifth-generation tensor cores with FP4/FP6/FP8?

## Main contribution (authors' claims, from the PDF)
PTX and CUDA C++ microbenchmarks under CUDA 12.8 / PTX 8.7. Tensor cores: a comparison table (Table IV area, p. 4) lists supported MMA instructions as "mma, wmma, tcgen05" for Blackwell against "mma, wmma, wgmma" for Hopper; the text on the same page states "Even though the wgmma instruction isn't supported in GB203 and the tcgen05 instruction hasn't been implemented yet for GB203, NVIDIA has implementation of the mma ... instruction for the tensor cores to be analyzed." Low-precision MMA: GB203 sustains higher throughput than GH100 "across all low-precision formats, peaking at over 11 TFLOP/s with 6 ILP and 32 active warps" (a single-SM-scale microbenchmark figure, not a device peak). Completion latency 1.21 cycles (GB203) vs 1.66 (GH100) for all low-precision formats (as printed). Memory: L2 65 MB on GB203 vs 50 MB on GH100; global-memory access begins beyond 71 MB (GB203) and 55 MB (GH100) with latencies about 876.7 and 658.7 cycles; sustained-transfer "peak read bandwidth" 15.8 TB/s (GH100) vs 8.2 TB/s (GB203), write 2.2 vs 1.6 TB/s (§VI.D, Figure 10; these exceed DRAM bandwidth and are read here as cache-level aggregate figures, an inference). Case study §VII.A: FP8 dense GEMM through cuBLASLt (`nv_fp8_e4m3` inputs, bf16 C, FP8 D, 32 MB workspace, 100 runs averaged); Table VII gives "TFLOP/s" of 0.887 (Hopper) vs 0.233 (Blackwell) at 8192³, with the 8192³ Blackwell runtime given as 4.710 ms in the Figure 11 caption. §VII.B: TensorRT GPT-NeoX inference power (Table VIII: FP16 57.64 W Hopper vs 47.78 W Blackwell; FP8 57.69 vs 45.14).

## Inference (not the authors' claim)
2 × 8192³ FLOP in 4.710 ms is 233 TFLOP/s, so Table VII's unit is mis-scaled by 10³ and the RTX 5080 achieved about 233 TFLOP/s FP8 through cuBLASLt at 8192³, against the H100's about 887. The paper's tcgen05 attribution to "Blackwell" in the table conflicts with its own text for GB203; the text is the operative statement for SM120: no wgmma, no tcgen05, `mma.sync` only. This matches the study's grouped-GEMM gate reading (`_grouped_mm` gated to compute capability 9.0 in torch 2.8.0 `Blas.cpp`) and the SM120 register-MMA-only picture.

## Evidence and scope
Two GPUs (RTX 5080 GB203, H100 PCIe GH100); microbenchmarks plus two case studies; nvidia-smi power; no fp16-accumulate versus fp32-accumulate throughput split was found in the PDF text (the tables are power and latency, not TFLOPS by accumulator).

## Exact version and date read
arXiv 2507.10789v2, "[cs.DC] 21 Jul 2025"; PDF saved and extracted with `pdftotext` on 2026-09-21 (`scratchpad/jarmusch.txt`).

## Relevant source locations
§IV tensor cores (Tables IV–VI, Figures 4–5); §VI memory (Figures 6–10); §VII case studies (Table VII, Figure 11, Table VIII).

## Relationship to this study
- Overlap: an academic microbenchmark of consumer Blackwell (GB203, SM120) already exists; the "no wgmma / no tcgen05 on GB203" statement is prior.
- Difference: different die (GB203 vs the RTX 5090's GB202), instruction-level and cache-level microbenchmarks versus this study's framework-level rates (cuBLAS/Triton GEMM classes under PyTorch 2.8/Triton 3.4, SDPA backend behaviour, `_grouped_mm` gating, launch/PDL slopes). No fp16/fp32-accumulator split was measured by Jarmusch et al.; the study's X9 row uses the RTX Blackwell whitepaper's 419/209.5 TFLOPS with per-site roofline bounds (`LEDGER.md` X9, f_eff 0.346 at 8 s).
- Scientifically meaningful: the joint characterization under a framework is not found in the sources reviewed; the microarchitecture is.

## Consequence for this work
The SM120 claim is worded: a characterization of the RTX 5090 (GB202, SM120) that jointly reports realized tensor-core rates by precision and accumulator under PyTorch 2.8/Triton 3.4, SDPA backend behaviour, framework feature gating and launch/PDL costs, not found in the sources reviewed; Jarmusch et al. are cited as the prior SM120 microbenchmark, with the GB203/GB202 difference stated.

## Unresolved
The Table VII unit (inference above); whether v2 is the latest version; whether the authors' B200 suite (arXiv 2512.02189) has been run on a 5090 by anyone.
