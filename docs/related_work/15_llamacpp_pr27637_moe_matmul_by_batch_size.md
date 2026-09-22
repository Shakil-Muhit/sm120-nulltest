# llama.cpp PR #27637, "opencl: choose the MoE expert matmul by batch size for speculative decoding/MTP" (merged 2026-09-15)

**Contribution-test source (a production decision rule that selects the implementation by input size).** Read from the GitHub PR page on 2026-09-22.

## Research question
Not a study. An engineering change: which of two MoE expert-matmul kernels should the OpenCL backend run when a speculative-decoding verify step routes only a handful of tokens?

## Main contribution (authors' claims, as read)
The kernel choice is made per call from the number of expert routings the batch produces: "The choice is now made on how many expert routings the batch actually produces"; below a full batch the dp4a kernel wins, at a full batch the prebuilt vendor kernel is selected. Result quoted: "Gemma-4-26B-A4B: Recorded as 33.91 → 41.33 t/s (+21.9%)" on an Adreno X2E GPU. A secondary change removes zero-filling of padded activation slots.

## Evidence and scope
One backend (OpenCL, Adreno), one model, throughput before/after; the threshold and how it was chosen are not documented in the PR text read.

## Exact version and date read
PR #27637 as merged on 2026-09-15 (author @wanghqc, merged by @lhez); read 2026-09-22.

## Relevant source locations
PR description (overview, results).

## Relationship to this study
- Overlap: a size-dependent switch between a padded/batched implementation and a per-problem one, applied in an inference engine, is exactly the decision E9's model is meant to inform (which implementation at which chunk length).
- Difference: the PR's rule is a measured threshold on a mobile GPU with no stated cost model, no held-out validation and no attribution; this study registers the model, states the crossover before measuring, and reports that the registered model missed the held-out points by 0.69–0.76 ms (`SESSION3_RESULT.md` E9).
- Scientifically meaningful: the PR shows practitioners already make this choice by size; what was not found in the sources reviewed is an attributed, pre-registered cost model for it and a test of whether such a model transfers.

## Consequence for this work
The contribution test cites this PR as evidence that size-dependent implementation selection is practised; the study's contribution is the tested (and falsified) attribution, not the switch.

## Unresolved
The exact threshold and the hardware sweep behind it.
