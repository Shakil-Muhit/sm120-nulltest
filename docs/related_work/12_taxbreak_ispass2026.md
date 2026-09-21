# TaxBreak — "TaxBreak: Attributing the Orchestration Tax of LLM Inference" (arXiv 2603.12465v1; ISPASS 2026)

**Reading-order item 12.** Tag: ledger/genre OVERLAP.

## Research question
How much of an eager-mode LLM inference step is host-side orchestration rather than device execution, and to which framework components does it belong?

## Main contribution (authors' claims, as read)
A per-kernel decomposition of host time into three components (framework time ΔFT, CUDA-runtime/driver time ΔCT, kernel-launch time ΔKT) with device-active time kept as an aggregate, and an index HDBI = T_dev / (T_dev + T_orch). Workloads: Llama-3.2-1B and 3B, GPT-2, OLMoE, Qwen1.5-MoE on H100 and H200, eager mode. A null-kernel launch floor of about 4.7 µs (H100) and 4.5 µs (H200) is reported.

## Evidence and scope
Five models, two data-center GPUs, eager execution only (no CUDA graphs, no torch.compile), profiler-based attribution; recommendations follow from the decomposition; the recommendations are not validated by implementing them in the paper as read.

## Exact version and date read
arXiv 2603.12465v1 HTML; read 2026-09-20.

## Relevant source locations
Decomposition definitions (§3); HDBI; per-model tables; null-kernel floor.

## Relationship to this study
- Same venue and same "decompose, then recommend" structure. TaxBreak decomposes the host side in eager mode; this study decomposes the device side under CUDA-graph replay and prices host-side fixed costs only as slopes and intercepts (`SESSION_RESULT_v2.md`, boundary section: stream intercept 6.466 µs, graph intercept 2.398 µs).
- The study's per-node stream cost, 4.883 µs on the RTX 5090, is the same kind of number as TaxBreak's null-kernel floor of 4.7 µs on H100 (different method: chain slope versus null-kernel launch).
- What the genre accepts: attribution without implementing the recommendation. What it does not validate: the recommendation's realized effect. The ledger's realized/ceiling column is the step beyond that.
- Scientifically meaningful: complementary, not overlapping, on the measured quantity; overlapping on the genre.

## Consequence for this work
Cite TaxBreak as the ISPASS-genre neighbour; state the difference in execution mode (eager versus graphed) and side (host versus device); never write that the study's costs are "invisible to TaxBreak" or that graphs "deleted the host side".

## Unresolved
Whether TaxBreak's artifact includes a CUDA-graph mode (not seen in v1).
