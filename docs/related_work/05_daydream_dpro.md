# Daydream — Zhu et al., "Daydream: Accurately Estimating the Efficacy of Optimizations for DNN Training" (USENIX ATC 2020); with dPRO — Hu et al., "dPRO: A Generic Performance Diagnosis and Optimization Toolkit for Expediting Distributed DNN Training" (MLSys 2022)

**Reading-order item 5.** Tag: ledger THREAT.

## Research question
Given a profile of one training iteration, can a what-if simulation over a kernel-level dependency graph predict the iteration-time effect of an optimization before it is implemented?

## Main contribution (authors' claims)
Daydream: builds a dependency graph of CPU and GPU tasks from CUPTI traces, supports graph transformations that model an optimization (fusion, mixed precision, different hardware, distributed settings), and simulates the new iteration time. Quoted (abstract): mixed precision "will improve the iteration time of training BERTLARGE model by 17.2% (with <3% error), while the kernel fusion technique can improve it by 38.7% (with <7% error)". dPRO (abstract only, read 2026-09-21): a global data-flow graph across devices with fine-grained communication operations; what-if simulation from computation, communication and memory perspectives; "predicts the performance of distributed training in various settings with < 5% errors in most cases"; "up to 3.48x speed-up over the baselines"; TensorFlow and MXNet, AllReduce and Parameter Server.

## Evidence and scope
Daydream: training iterations of BERT and other models on data-center GPUs (V100-class), CUPTI traces, validation of predicted against measured iteration time per optimization. Two caveats stated in the paper: profiling with CUPTI "usually serialize[s] GPU kernel execution" (text line ~780 of the extraction), and the dependency graph, not device time alone, determines wall time. dPRO: distributed training; abstract only was read.

## Exact version and date read
Daydream ATC 2020 PDF (`scratchpad/daydream.txt`, read 2026-09-20). dPRO arXiv 2205.02473 abstract page, read 2026-09-21 (revised 18 May 2022, MLSys 2022).

## Relevant source locations
Daydream abstract; §3 dependency graph; §5 evaluation tables. dPRO abstract.

## Relationship to this study
- Overlap: the ledger's rows are what-if estimates from traces; Daydream's bar (predict, then measure, then report error) is the bar the ledger must meet for its validated rows.
- Difference: the ledger predicts a ceiling (an upper bound), not a point estimate; the objection that recovered device time may not appear in wall time applies with force at 100 ms, where C3's wall is 2.514 ms of kernels inside a 3.261 ms span (node-level trace; `SESSION_RESULT_v2.md`, "Three timing views", C3 4410 row) and the host launch of the single graph takes 222 µs (`docs/phase3/x1/x1_decomposition_352800.json`, host launch field). The ledger's Z at 100 ms is therefore a device-time fraction until an intervention shows it in wall time.
- Scientifically meaningful: yes; the study should never write "we predict" for a ledger row.

## Consequence for this work
The E1–E4a rows are reported as estimated ceiling versus realized wall-time gain with error, at both operating points, and the device-time/wall-time reconciliation is reported alongside (the three timing views are already in `SESSION_RESULT_v2.md`). The CUPTI serialization caveat is why the primary timings are unprofiled (rule R3) and why the perturbation ratios are reported separately.

## Unresolved
dPRO's body (validation setup and the exact error table) was not read.
