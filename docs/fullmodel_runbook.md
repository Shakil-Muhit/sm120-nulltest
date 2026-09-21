# Full-model sessions runbook (ceiling-ledger Phase 1)

Registration: `prereg_amendments.md` (pushed before session 1). Plan: `../plan_ceiling_ledger.md`.

## Off-pod, once
```bash
python3 -m venv ~/ms/.venv-clips && ~/ms/.venv-clips/bin/pip install pyarrow soundfile numpy
# MUSDB18-HQ test shards from the HF mirror (revision pinned in configs/sdr_clips.json)
python scripts/prepare_sdr_clips.py --shards test-00000.parquet test-00001.parquet \
    --revision 32fd4e850b45ea0b635be067b921661984b53029 --out-dir ~/ms/data/sdr_clips \
    --manifest configs/sdr_clips.json          # regenerates the pack; sha256s must match
```

## Pod (RunPod secure cloud, RTX 5090, name `melband_profiling`)
From muhit-pc, with the tree COMMITTED (the session binds to the commit):
```bash
bash scripts/pod_up.sh                  # creates the pod (EU datacenters in order), ships the
                                        # committed tree as a git bundle + the clip pack, runs setup.sh
. pod.env; ssh -i ~/.runpod/ssh/RunPod-Key-Go -p $POD_PORT root@$POD_HOST
# on the pod
cd /root/sm120-nulltest && tmux new -s run
bash scripts/session1.sh                # ≤ 3 h (deadline-aware; see session_lib.sh)
# extract, terminate; session 2 on a fresh pod repeats pod_up.sh, then:
bash scripts/session2.sh
tar czf /root/evidence_sN_$(date +%Y%m%d_%H%M).tgz -C /root/sm120-nulltest evidence   # scp OFF-POD
```
Knobs (`scripts/session_lib.sh`): `ITERS` 50, `REPEATS` 5, `RESTARTS` 3, `CLIPS_DIR`,
`T_COMPILE` 1500 s, `T_OTHER` 900 s, `T_MIN` 240 s, `SESSION_BUDGET_S` 10800,
`EXTRACT_RESERVE_S` 900, `C2_BUDGET_S` 2400; session 1 also `ARMS`, `PAIRS`, `SKIP_C2`,
`SMOKE`. Every launch is one fresh process bounded by min(cap, remaining usable time) and
leaves a terminal JSON; the preflight refuses an uncommitted tree or wrong pins.

## Outputs (evidence/, git-ignored)
`full_<arm>_c<chunk>_<tag>.json` per run (timing blocks, validation, SDR gate, capture
scope/adapter, graph gate, settle, notes); `ref/` (eager-A references); `generated/`
(external forwards, C3 verification); `traces/` (session 2: `.nsys-rep`, exports,
`graph_verify_*.json`, `timing_views_*.json`, `ideal_costs_*.{json,csv}`, dumps);
`attn_backend_bench.json`, `boundary_bench.json`; `clocks_session*.csv`;
`summary_full.json` (collector). `SESSION_RESULT.md` cites these paths.

- `session_status.json` — mechanical session verdict from `fullmodel/completeness.py`
  (COMPLETE / INCOMPLETE with per-item reasons over the registered mandatory items).
- `aux_<name>.json` — terminal record of every bounded auxiliary step (OK / FAILED /
  UNTESTED); supervisor banners go to stderr.
- `ref/*.lineage.json` — per-artifact lineage sidecars (manifest id + sha256) written
  with every reference output and checked before any use.
- `traces/ideal_costs_<arm>_<chunk>.{json,csv}` — per-kernel ideal costs; rows carry
  `recoverable_share_hbm_model` and, where the byte bound over-counts, `bound_not_tight`.

## Shakedown (engineering only, never evidence)
ml-beast GPU 4: `rsync -a --exclude .git --exclude evidence sm120-nulltest/ ml-beast:~/axial/sm120-nulltest/`,
env in `~/axial/fullmodel_smoke/env.sh`, outputs in `~/axial/fullmodel_smoke/`.


### Phase 3 (session 3) shakedown and pod flow
Shakedown host (GPU 4), engineering data only (never evidence):
```bash
# driver workstation: commit, then sync the committed tree WITH .git (the manifest needs a clean committed tree)
rsync -a --delete --exclude '/evidence/' --exclude '/evidence_smoke/' --exclude '__pycache__' --exclude 'pod.env' sm120-nulltest/ ml-beast:~/axial/sm120-nulltest/
# shakedown host, inside tmux: user-space nsys (2026.5.1) is at $(cat ~/axial/nsys_path.txt)
source ~/axial/fullmodel_smoke/env.sh && source ~/axial/venv/bin/activate
export PATH=$(dirname $(cat ~/axial/nsys_path.txt)):$PATH
export EVID=~/axial/fullmodel_smoke/session3_shakedown_$(date +%Y%m%d_%H%M) SHAKEDOWN=1 SESSION_BUDGET_S=25200 EXTRACT_RESERVE_S=600 E5_BUDGET_S=5400
bash scripts/session3.sh        # E9 held-out lengths are recorded UNTESTED by rule; audit verdict carries shakedown=true
```
Pod (owner go-ahead required; name "melband profiling"): `bash scripts/pod_up.sh` → `bash scripts/pod_run.sh ship` →
`SESSION_ENV="SESSION_BUDGET_S=18000 EXTRACT_RESERVE_S=900 E5_BUDGET_S=3600" bash scripts/pod_run.sh session3` (exports SESSION_POD=1, so the E9
held-out lengths run) → `pod_run.sh tail session3` → `pod_run.sh extract` → `pod_run.sh terminate`. Off-pod: `completeness.py --session session3`
re-audit, `e_tools.py e9_check`, then the ledger realized/ceiling fill.

## Phase 2 (off-pod, no GPU)
```bash
python fullmodel/session_result.py --evidence-dir <extracted evidence> --out SESSION_RESULT.md
python fullmodel/ledger.py         --evidence-dir <extracted evidence> --out LEDGER.md   # + LEDGER.json
```
Both generators compute every number from the cited evidence files (docs/ledger_spec.md); missing
inputs print UNRESOLVED and are never proxied.
