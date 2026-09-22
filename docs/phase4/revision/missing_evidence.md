# Missing or omitted evidence (Phase 4, 2026-09-22)

| item | where referenced | present on the owner's machine? | present in the handover ZIPs? | what it resolves |
|---|---|---|---|---|
| Raw Nsight reports and SQLite exports (`traces/*.nsys-rep`, `*.sqlite`), per-event CSVs (`*_cuda_gpu_trace.csv`, `*_cuda_api_trace.csv`, `*_nvtx_gpu_proj_trace.csv`) for sessions 1–3 | `docs/EXECUTION_REPORT_*.md`, `docs/release/zenodo_subset.md` | yes (`evidence_pod/*/evidence/traces`) | no (size) | event-level alignment for any standalone traced run |
| Compile dumps (`traces/dump_<arm>_<chunk>_trace/` with `output_code.py`) | `LEDGER.md` X6a rule, `inventory_diff_session3.json` | yes (session 3: 21 dump directories) | no | source provenance of fused ops (used in this revision) |
| A trace of the MATCHED C1/C1P process | never collected in Phase 3 | **yes since session 4** (`traces/trace_matched_C1_C1P_352800_p1.*`, `matched_trace_C1_C1P_352800_p1.json`) | derived JSON yes; raw report/CSVs no (size) | PB: resolved — the 15 ms is GPU-busy time of the second-built arm |
| Standalone unprofiled C1P timing | never collected in Phase 3 | **yes since session 4** (`full_C1P_c352800_r{1,2,3}.json`) | yes | PB: resolved — no regression outside the matched context |
| Same-session matched C3 vs H at 8 s and 1 s | never collected before session 4 | **yes since session 4** (`matched_C3_H_c{44100,352800}_m{1,2,3}.json`, with 4 s / 6 s and A/A on the same GPU) | yes | PA: measured (8 s: C3 faster by the registered rule, thin margin over 0.30 ms) |
| Reference tensors (`ref/*.pt`) and the eight SDR clips | `qualify` lineage hashes | tensors yes (`evidence_pod/*/evidence/ref`); clips yes (`~/ms/data/sdr_clips`, never committed) | no | independent re-run of validation and SDR |
| GitHub Events `created_at` for the registration pushes | `docs/provenance/push_evidence.md` | no (needs a token or a public repository) | no | server-side registration time |

## Added after session 4 (2026-09-22)

| item | where referenced | present on the owner's machine? | present in the handover ZIP? | what it resolves |
|---|---|---|---|---|
| Session-4 raw Nsight reports, SQLite and per-event CSVs (`traces/*.nsys-rep`, `*.sqlite`, `*_cuda_gpu_trace.csv`, `*_cuda_api_trace.csv`, `*_nvtx_*_trace.csv`) and compile dumps (`traces/dump_*`) | `EXPERIMENT_REPORT_session4.md` §8–9 | yes (`evidence_pod/nsfwv548ifydon_20260922_2034/evidence/traces`, 465 MB in total; archive sha256 in `pod.env.extracted`) | no (size); derived JSON yes | event-level re-analysis; source provenance of fused ops |
| Standalone unprofiled C1 in session 4 | `SESSION4_RESULT.md` PB | no (not in the PB design; C1 measured standalone in sessions 1–3 and traced in session 4) | — | a same-session unprofiled C1 baseline for the PB table (the traced C1 block median 37.01 ms stands in, labelled) |
| Session-4 traces of H and C3M | `SESSION4_RESULT.md` PQ | no (not in the plan; historical traces on other GPUs: C3M from session 3, H at 8 s / 100 ms from the Sep-20 session 2, plus H 100 ms variant traces in session 3) | — | same-session inventory diffs for H → HG and C3M → C3MG (currently cross-session, descriptive) |
| Intervention test of the recompile-limit mechanism (second build with a raised `torch._dynamo.config.recompile_limit`) | report §6 | no | — | causal confirmation of the matched-runner artefact |
| RunPod invoice lines for the three pods | report §13 | no (only `costPerHr` and created/deleted timestamps from the API) | — | actual cost instead of the price × hours estimate |
