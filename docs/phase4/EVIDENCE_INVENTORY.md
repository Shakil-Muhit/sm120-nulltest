# Evidence inventory (2026-09-22, end of Stage 2)

Owner's machine paths; nothing below is public. Hashes establish content identity only. "In ZIP" refers to `~/ms/sm120_phase4_handover_20260922.zip` (rebuilt at the end of Stage 2; its `MANIFEST.json` lists every packaged file with sha256).

| evidence set | location (owner's disk) | files / size | receipt (archive sha256, remote content hash) | in ZIP | grade |
|---|---|---|---|---|---|
| Sessions 1–2, pod 9tblar2zwo97ln (Sep 20, EU-RO-1, GPU-6bf4cb32) | `~/ms/evidence_pod/9tblar2zwo97ln_20260920_1855/evidence` (final extraction; earlier partial extractions `_1755`, `_1756`, `_1851` kept) | 1,877 / 1.2 GB | `97117d63…` / `488d653b…` (`pod.env.extracted`) | core subset (Phase-3 ZIP; `original/` in the reviewer bundle) | pod, Phase 1–2 record on `main` |
| Session 3, pod d5n5vfr3z1tbp6 (Sep 22, EU-RO-1, GPU-7d47d5c2) | `~/ms/evidence_pod/d5n5vfr3z1tbp6_20260922_1031/evidence` | 1,699 / 746 MB | `0691998a…` / `15a33bd1…` | core subset (Phase-3 ZIP) | pod, Phase-3 record on `main` (audits 198/198 on-pod, 195/199 corrected) |
| Session 4, pod nsfwv548ifydon (Sep 22, EU-CZ-1, GPU-01ffe7fc) | `~/ms/evidence_pod/nsfwv548ifydon_20260922_2034/evidence` (+ `session_status_onpod_original.json`, `session_status_reaudit_gate12.json` added off-pod) | 1,351 / 465 MB | `98d96557…` / `dfbb780a…` | `evidence_session4_core/` (records, logs, derived JSON; 27.9 MB uncompressed) | pod, Phase-4 record (audit 169/170 on-pod, 170/170 re-audit) |
| On-pod smoke before session 4 | `scratchpad/smoke_p4_pod` copy → ZIP `smoke_engineering_not_evidence/` | 9 files | — | yes | engineering, never evidence |
| Aborted first launch (preflight only) | ZIP `aborted_first_launch_not_evidence/` | 2 logs | — | yes | no measurement |
| Stage-2 pilot probes (ml-beast GPU 4, GPU-eb6e4a8a, driver 595.84) | `docs/phase4/stage2/results/` (repo) and `ml-beast:~/axial/stage2/` | 6 JSON + 2 logs | — | yes (repo + `revision_outputs/stage2/`) | engineering, exploratory; never paired with pod numbers |
| ml-beast shakedowns (Phase 3) | `ml-beast:~/axial/fullmodel_smoke/` (2.8 GB) | — | — | no | engineering |
| Audio clips (SDR gate) | `~/ms/data/sdr_clips` (16 files; manifest sha256 `37d6d2ea…` in every session manifest) | — | — | never | input data, never committed |
| Checkpoint / MSST | pod `/root/work` (deleted with the pod), ml-beast `~/axial/msst`, `~/axial/ckpt`; sha256 `87201f4d…` / commit `ea7eb9c2…` recorded in every manifest | — | — | no (pinned by hash; `scripts/fetch_checkpoint.sh`) | inputs |

Omitted from the ZIP for size, all present under the paths above with the receipts' hashes: raw Nsight Systems reports (`traces/*.nsys-rep`), SQLite exports, per-event CSVs (`*_cuda_gpu_trace.csv`, `*_cuda_api_trace.csv`, `*_nvtx_*_trace.csv`, kernel/NVTX summaries), compile dumps (`traces/dump_*`), reference tensors (`ref/*.pt`). Nothing omitted is claimed to be in the ZIP.

Repository: `git@github-shakil:Shakil-Muhit/sm120-nulltest.git`, branch `phase4-followup` (Stage 1 commit 1dca907; Stage 2 commit recorded in the ZIP README), `main` at 9298a6a unchanged.
