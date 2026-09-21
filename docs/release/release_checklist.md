# Release checklist (Phase 3 §2.8) — state on 2026-09-21

- [x] Pre-commit scanner in the tree: `.githooks/pre-commit` → `scripts/precommit.sh`; `git config core.hooksPath` is `.githooks`; ran clean on the current tree (exit 0).
- [x] Name scan run (terms: organisation names, private hostnames, `production-*` pod names, tailnet addresses). Result: no organisation name, no production pod name, no tailnet address in tracked files. Two private hostnames appear: the shakedown workstation name (24 files: `prereg_amendments.md`, `fullmodel/loader.py`, `fullmodel/capture.py`, `fullmodel/completeness.py`, `SESSION_RESULT_v2.md`, `docs/fullmodel_runbook.md`, `docs/codex/*`, `docs/phase3/*`, `docs/EXECUTION_REPORT_2026-09-20.md`) and the driver workstation name (10 files: `scripts/pod_run.sh`, `scripts/pod_up.sh`, `docs/fullmodel_runbook.md`, `docs/codex/*.log`). **Owner decision:** replace with "shakedown host (GPU 4)" and "driver workstation" in live code and docs before deposit, or accept them; the codex logs are verbatim records and are best left unedited (exclude from the deposit if the names must not ship).
- [x] Evidence manifest built: `docs/release/evidence_manifest.csv` (1,876 files, 1.237 GB).
- [x] Zenodo subset proposed: `docs/release/zenodo_subset.md`; metadata draft `.zenodo.json`.
- [x] README quickstart written (README.md, "Quickstart: one arm in ten minutes").
- [x] IEEE AI-use disclosure drafted: `docs/release/ai_use_disclosure.md`.
- [x] Codex gate log summary: `docs/release/codex_gate_summary.md`.
- [x] Push-time evidence: `docs/provenance/push_evidence.md` (client-side timestamps; repository private so no GH Archive event exists; OpenTimestamps proof `registration_hashes.txt.ots` submitted to three calendars on 2026-09-21, upgrade with `ots upgrade` after ~1 day for the Bitcoin attestation).
- [ ] Owner: obtain the GitHub Events `created_at` for the registration pushes with a token (command in `push_evidence.md`), or make the repository public so GH Archive records future pushes.
- [ ] Owner: confirm `.zenodo.json` creators/affiliation/license; deposit after the Phase-3 session.
- [ ] Owner: decide the hostname replacement above.
