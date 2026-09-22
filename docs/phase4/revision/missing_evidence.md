# Missing or omitted evidence (Phase 4, 2026-09-22)

| item | where referenced | present on the owner's machine? | present in the handover ZIPs? | what it resolves |
|---|---|---|---|---|
| Raw Nsight reports and SQLite exports (`traces/*.nsys-rep`, `*.sqlite`), per-event CSVs (`*_cuda_gpu_trace.csv`, `*_cuda_api_trace.csv`, `*_nvtx_gpu_proj_trace.csv`) for sessions 1–3 | `docs/EXECUTION_REPORT_*.md`, `docs/release/zenodo_subset.md` | yes (`evidence_pod/*/evidence/traces`) | no (size) | event-level alignment for any standalone traced run |
| Compile dumps (`traces/dump_<arm>_<chunk>_trace/` with `output_code.py`) | `LEDGER.md` X6a rule, `inventory_diff_session3.json` | yes (session 3: 21 dump directories) | no | source provenance of fused ops (used in this revision) |
| A trace of the MATCHED C1/C1P process | never collected in Phase 3 | no | no | PB: whether the matched-context regression shows in GPU intervals, launches, copies or idle within aligned pass windows (session 4 collects it) |
| Standalone unprofiled C1P timing | never collected in Phase 3 | no | no | PB: whether the regression exists outside the matched context (session 4 collects it) |
| Same-session matched C3 vs H at 8 s and 1 s | never collected (Sep 20 pair at 8 s is on another card; session 3 had only 4 s / 6 s pairs) | no | no | PA (session 4 collects it) |
| Reference tensors (`ref/*.pt`) and the eight SDR clips | `qualify` lineage hashes | tensors yes (`evidence_pod/*/evidence/ref`); clips yes (`~/ms/data/sdr_clips`, never committed) | no | independent re-run of validation and SDR |
| GitHub Events `created_at` for the registration pushes | `docs/provenance/push_evidence.md` | no (needs a token or a public repository) | no | server-side registration time |
