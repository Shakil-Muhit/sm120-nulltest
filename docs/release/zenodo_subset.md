# Evidence bundle: manifest and proposed Zenodo subset (Phase 3 §2.8)

Full manifest: `docs/release/evidence_manifest.csv` (path, bytes, sha256) over the extracted pod evidence directory `evidence_pod/9tblar2zwo97ln_20260920_1855/evidence` (content manifest 9bd6639d; `session_status_session1.json` COMPLETE 100/100, `session_status_session2.json` COMPLETE 172/172).

| kind | files | MB | in Zenodo subset | why |
|---|---|---|---|---|
| .csv (nsys exports: node-level kernel/API tables, sweeps) | 162 | 646.4 | yes, xz-compressed per file | the numbers in SESSION_RESULT and the ledger are computed from these |
| .sqlite (nsys export databases) | 26 | 281.7 | no | regenerable from the .nsys-rep with `nsys export --type sqlite`; largest and fully derived |
| .nsys-rep (raw Nsight Systems reports) | 32 | 99.1 | yes, as-is | primary trace record; already compressed internally |
| .pt (reference outputs for tier validation) | 40 | 97.9 | yes | needed to re-run the tight/loose tier check without the pod; model outputs only (no audio) |
| .txt / .log (console, nvidia-smi clock logs, pip freeze, provenance) | 489 | 49.8 | yes | provenance and the per-block clock evidence |
| .py (harness copy frozen at the session commit) | 562 | 37.4 | yes | binds the evidence to the code that produced it |
| .json (summaries, matched pairs, graph verification, ideal costs, session status) | 561 | 24.5 | yes | the headline inputs |
| .md, other | 4 | 0.3 | yes | — |

Never included, by rule: audio (the eight MUSDB SDR clips are fetched by `scripts/prepare_sdr_clips.py` and are not in the evidence tree); RunPod credentials (`pod.env` is git-ignored and not in the bundle).

Estimated deposit size: about 300 MB after compressing the CSVs (to be measured at packaging; Zenodo's per-record limit is 50 GB).

Deposit metadata draft: `.zenodo.json` at the repository root (owner to confirm creators, affiliation and license before use).
