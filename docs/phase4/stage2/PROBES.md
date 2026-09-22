# Stage-2 probe cards (pre-registered before each run; ml-beast GPU 4, engineering evidence — never final-session evidence)

Host: ml-beast, RTX 5090 index 4 (GPU-eb6e4a8a…), shared machine (other users' jobs on GPUs 0–3; an idle API server holds 714 MiB on GPU 4), pinned venv (torch 2.8.0+cu128, triton 3.4.0), MSST at the pinned commit, same checkpoint. Timings here are directional/mechanistic; no number from this file is a headline and none is compared with pod numbers as a paired result.

## Probe A1 — intervention on the matched-runner artefact (candidate A)

- **Question.** Is the 15 ms slow-down of the second-built C1-family arm caused by dynamo's recompile limit being exhausted on the shared `Attention.forward` frame (one code object for both attention axes), so that raising the limit removes it?
- **Design.** Matched A/A process `matched_run.py --pair C1 C1 --chunk 352800 --repeats 2 --no-manifest` twice: (i) default `torch._dynamo.config.recompile_limit = 8`; (ii) `recompile_limit = 64` set before any compile (wrapper `probe_a1_recompile_limit.py`). Read from the torch logs: presence of `[7/8] … hit config.recompile_limit` in the second build, graph-break counts of both builds, recorded cudagraphs, block medians of both arms.
- **Discrimination.** Recompile-limit mechanism → (ii) shows no `[7/8]` warning, equal break counts (8/8) and block medians within ≈ 0.5 ms; alternative (in-process memory/allocator/cache-pressure effect independent of the limit) → the ≈ 15 ms gap persists in (ii). A partial reduction would indicate two contributions.
- **Cost.** Two 8 s matched processes ≈ 12–15 min on GPU 4; no pod.

## Probe B1 — GEMM operand locality after cast removal (candidate B)

- **Question.** Do the model's small-M GEMMs at the 100 ms operating point run slower after pre-casting/freezing because the removed cast kernel used to leave the fp16 weight L2-resident (producer-side warming), rather than because a different GEMM kernel is selected?
- **Design.** Capture the (input, weight) shapes of every `nn.Linear` call in one eager forward at chunk 4,410 (hooks). For each distinct shape (weighted by call count), time the GEMM with CUDA events over 200 iterations under four conditions, each preceded by an L2 flush (write 256 MB) so every iteration starts cold: (a) fp32 weight, `.half()` cast kernel immediately followed by the GEMM (the C3 pattern); (b) pre-cast fp16 weight, GEMM alone (the C3P/C3G pattern); (c) pre-cast fp16 weight, a read-only "touch" kernel over the weight immediately before the GEMM (warms L2 without producing the operand); (d) pre-cast fp16 weight, a cast of a *different* fp32 tensor of the same size immediately before the GEMM (same extra work, no warming of the operand). GEMM kernel names and durations from `torch.profiler` (CUDA activities) on a separate short run.
- **Discrimination.** Locality mechanism → GEMM time (a) ≈ (c) < (b) ≈ (d), same kernel name in all four; selection/layout mechanism → kernel names differ between (a) and (b) or (c) ≈ (b); if (b) ≈ (a) at all shapes the in-model effect is not reproducible in isolation (context-dependent) and the direction is weakened. Expected size if locality: the in-model excess is ≈ 1.6 µs per launch of the dominant 16×16 wmma kernel (243 µs / 155 launches); a per-call excess of that order under (b)/(d) but not (a)/(c) supports the mechanism.
- **Cost.** ≈ 5 min on GPU 4; no pod.

## Probe C0 — pad-and-batch waste from the band layout (candidate C; analysis only, CPU)

- **Question.** What fraction of the pad-and-batch GEMM work is padding for this band layout (60 mel bands over 1,025 bins), and is it constant in frames (so it cannot by itself produce the 6 s → 8 s ranking flip)?
- **Design.** Instantiate the model class on CPU from the pinned config, read the per-band bin counts, compute Σ(max − w_i)/Σw_i and the per-length GEMM shapes of the batched path. No timing.
- **Discrimination.** A frame-independent waste fraction means the flip needs a regime change on one side (e.g., the batched GEMM reaching a compute-bound regime at 8 s, or the per-band path's launch overhead saturating); a frame-dependent waste would be a direct explanation.
- **Cost.** CPU seconds.

## Deviations from the cards (recorded after the runs)
- B1 ran 100 iterations per condition (the card said 200; `PROBE_B1_ITERS` default) — the launch-weighted contrasts are far larger than the p10–p90 spread, so the shorter run was kept.
- A1 as carded (A/A C1 process) did not reproduce the artefact; the discriminating C1 vs C1P form (A1b) was added and run with the same design (limits 8 and 64). Both are reported.
- The first probe attempt failed on script errors (band-width tensor shape, profiler field, missing `--ref-dir`); no GPU result from it is used (`results/run_probes.log` is the successful attempt).
