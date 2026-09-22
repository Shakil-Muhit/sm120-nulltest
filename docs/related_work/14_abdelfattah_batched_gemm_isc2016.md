# Abdelfattah, Haidar, Tomov & Dongarra, "Performance, Design, and Autotuning of Batched GEMM for GPUs" (ISC High Performance 2016)

**Contribution-test source (closest prior on batched/padded versus per-problem GEMM execution as a function of size).** Read from the ICL PDF (`icl-utk-871-2016.pdf`, `pdftotext` extraction `scratchpad/abdelfattah16.txt`) on 2026-09-22.

## Research question
How should a GEMM kernel for batches of small matrices, of fixed or variable size, be designed and tuned on a GPU, and when does it beat the alternatives?

## Main contribution (authors' claims)
Batched GEMM kernels for fixed-size and variable-size ("vbatched") batches in MAGMA, built from templated "ETM" kernels with an autotuning process over tile shapes, plus a dynamic-parallelism variant that launches one GEMM per matrix from the GPU side. Abstract: "We address batched problems with both fixed and variable sizes, and show that specialized GEMM designs and a comprehensive autotuning process are needed to handle problems of small sizes."

## Evidence and scope (authors' claims)
Tesla K40c (745 MHz, ECC on), double precision, §4 "Performance Results and Analysis". Competitors: cuBLAS batched GEMM, "the cuBLAS classic GEMM kernel offloaded to concurrent streams" (the per-problem alternative), and MKL. Size-dependent winners are stated, not modelled: for variable sizes, "The two MAGMA kernels outperform other competitors for Maximum m up to 300. The streamed GEMM dominates the midrange, and then gets nearly matched or slightly outperformed by the MAGMA kernel based on dynamic parallelism. For the case of square matrices (Figures 11(f) and 12(f)), the streamed GEMM achieves the best performance unless matrices are too small, where the ETM-based MAGMA kernel is the best choice." For fixed sizes: "Otherwise, the streamed cuBLAS kernel mostly achieves the best performance, expect for the midrange in Figure 9(e), where MAGMA takes over." Sub-warp sizes: "Performance there is memory bound and can be modeled."

## Exact version and date read
ISC 2016 paper PDF as hosted by ICL (icl-utk-871-2016), read 2026-09-22.

## Relevant source locations
§3 (design, vbatched variants, dynamic parallelism), §4 (Figures 9, 11, 12 and the quoted winner-by-size statements), §5.

## Relationship to this study
- Overlap: the same question as E9 in a different setting: when does batching many small (variable-size) GEMMs beat running them one by one, as a function of size. Their variable dimension is the matrix size within a batch; this study's is the chunk length (frames), with the 60 band GEMMs ragged along K.
- Difference: their answer is empirical per GPU and per shape class (tile autotuning, no cost model with fixed per-call terms); their crossover is between a batched kernel and streamed individual cuBLAS calls at the kernel level, inside a numerical library, on a K40c. This study asks the same kind of question at the application level (pad-and-batch versus per-band GEMMs inside a compiled, graphed forward), registers an attributed cost model with fixed and length-scaling terms, and tests it at held-out lengths.
- Scientifically meaningful difference: yes in method (registered model, held-out test) and in level (whole forward under CUDA graphs); no in the existence of a size-dependent crossover, which they establish for batched versus streamed GEMM.

## Consequence for this work
The contribution test cites this paper for "batched versus per-problem GEMM crossovers exist and are size-dependent"; the study's claim is restricted to the attributed model, its held-out failure, and the direction and bracket of the crossover for this workload class.

## Unresolved
Whether the authors' later vbatched paper (ICS 2017, "Novel HPC techniques to batch execution of many variable size BLAS computations on GPUs") adds a cost-model rule; its full text was not opened (ACM DL).
