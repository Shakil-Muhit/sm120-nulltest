# Plan 0.5 — researcher outreach (owner sends; agent drafted 2026-09-20)

Status: DRAFT — not sent. Record here, per recipient: date sent, what was
attached, and the reply (or "no reply by <date>"). This file is the audit
record the plan requires.

## Recipients (owner to finalize)
1. Pekhimenko group (Daydream / Habitat lineage — what-if analysis).
2. Authors of the closest prior work on graph-launch overhead / persistent
   kernels on consumer GPUs (Hazy Research "megakernel"/Mirage authors).
3. One systems researcher who has published on torch.compile / CUDA Graphs
   characterization (PyGraph authors or equivalent).

## Email text (edit freely)
Subject: request for a 30-minute critique of one GPU trace (MelBandRoformer on RTX 5090)

I am running a pre-registered measurement study of compiled inference for an
axial-transformer audio model (MelBandRoformer, 228M params) on a consumer
Blackwell GPU (RTX 5090), and I would value a critical reading of ONE trace
before I build anything on it. The question the study asks is a what-if
analysis in the Daydream/Habitat spirit: for each candidate optimization we
compute a registered ceiling from traces and simple mechanism models, then
score interventions against those ceilings.

Attached: the pre-registration and amendments (what was predicted before the
run), the audit log, one node-level Nsight Systems trace of the compiled
forward at the deployed 8-second operating point with its kernel table, and
the one-page summary of what the trace appears to say. I am specifically
asking: (1) is the attribution method (SDPA-anchored ordinal segmentation of
generated Inductor code, not NVTX hooks) defensible; (2) which of the ceiling
mechanisms (roofline per kernel, boundary-cost model for the streaming
regime, backend ratio for attention) you would not trust, and why; (3) what
would make this uninteresting to a systems venue.

I will credit any comments in the write-up, and I am happy to share the
full evidence archive (hashes included).

## Attachments checklist
- prereg_null_test.md.pdf + prereg_amendments.md (this repo)
- docs/audit_log.md, evidence/audit_log.md
- axial-layout-scoping RESULT.md (C1/C3) + one .nsys-rep + kernel CSV
- one-page summary (owner writes after session 1)
