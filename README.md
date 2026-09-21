# sm120-nulltest — session runbook

Pre-registered null test: can generic launch tooling (CUDA Graphs,
torch.compile) delete the ~80% GPU-idle of the MelBandRoformer front-end on an
RTX 5090, or is there structure only a custom grouped ragged-K kernel can
claim? **`prereg_null_test.md.pdf` is the spec — where anything here seems to
disagree with it, the prereg wins and the disagreement is a bug.**

Auditors: read **Deviations & ambiguities** first, then the **Audit map**.

---

## 0. Before the pod (once, off-pod)

The two provenance pins ship **filled** (session owner, 2026-08-16) and were
verified at build time; if you change either, re-verify and re-commit.

**Config file formats (resolved design decision, 2026-08-16 — not a
deviation).** In both files, lines starting with `#` are comments — purely
documentation, no exceptions.

- `configs/msst_commit.txt`: ONE data line — the full 40-char commit hash of
  [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training).
  Currently `ea7eb9c2…` (release v1.0.21). `setup.sh` [5/6] FAILS LOUDLY if
  missing, malformed, or not 40 hex chars.
- `configs/checkpoint.txt`: EXACTLY THREE data lines, in order:
  1. checkpoint URL, 2. its sha256, 3. model config path **relative to the
  pinned clone's root**. Configs are never URL-fetched: a config inside the
  pinned clone is transitively hash-pinned by the commit SHA. Escape hatch if
  a config is not in the clone: vendor the yaml into THIS repo's `configs/`
  and use a path relative to this repo. `fetch_checkpoint.sh` FAILS LOUDLY on
  a wrong line count, a malformed URL/sha256, or a line-3 path that exists in
  neither location after checkout. `MODEL_TYPE` is the constant
  `mel_band_roformer` (env-overridable).

Verified at build time against the pinned commit `ea7eb9c2` (2026-08-16):
the sha256 on line 2 matches HuggingFace's LFS hash for the file; the line-3
config exists at the pin; the front-end harness is bitwise-exact against the
pinned model code on CPU; the full construction path
(`get_model_from_config` on the line-3 config) builds a 228.2M-param
MelBandRoformer with the module structure the P8 hooks assume.

Before the session: commit the repo (the runbook's `git clone` needs it).
After any clone, enable the integrity gate (blocks commits containing
unparseable .sh/.py or terminal-paste artifacts — see the etiology header
in `scripts/precommit.sh`):

```bash
git config core.hooksPath .githooks
```

## Quickstart: one arm in ten minutes on a rented RTX 5090

On a RunPod pod from the template `runpod/pytorch:1.1.0-rc.154-cu1281-torch280-ubuntu2204` (torch 2.8.0+cu128, Triton 3.4.0; the image the Sep-20 sessions used), as root:

```bash
git clone git@github.com:Shakil-Muhit/sm120-nulltest.git && cd sm120-nulltest
bash setup.sh                                   # pins the MSST clone at ea7eb9c2, fetches and hash-checks the checkpoint
python fullmodel/run_full_arm.py --arm A  --chunk 4410 --tag ref --save-reference --evidence-dir evidence_quick
python fullmodel/run_full_arm.py --arm C3 --chunk 4410 --tag r1 --evidence-dir evidence_quick
python - <<'PY'
import json; d = json.load(open('evidence_quick/full_C3_c4410_r1.json')); print(d['status'], d['tier'], round(d['min_of_medians_ms'], 3), 'ms')
PY
```

The first command runs the eager baseline once and saves the reference outputs the tier validator needs; the second runs the repaired compiled arm C3 (compile plus 5 × 50 timed passes) and validates against that reference. On the Sep-20 pod, arm A at 4,410 took 16 s and arm C3 at 4,410 took 21 s after setup (timestamps in `evidence/session1_console.log`, lines 383–411); setup itself is dominated by the checkpoint download. Expected result: `status VALID`, tier `loose`, min-of-medians about 3.3 ms for C3 versus about 9.5 ms for A at this operating point (`SESSION_RESULT_v2.md`, arms table). This is a reproduction of one cell, not a session: headline statistics need the three-restart protocol in `scripts/session1.sh`.

## 1. Session command sequence (2am version)

```bash
# on the pod (RunPod Ubuntu 22.04 PyTorch template, root shell)
git clone <this-repo> && cd sm120-nulltest
bash setup.sh                      # [1/4]..[6/6]; idempotent, safe to re-run
tmux new -s run                    # NON-OPTIONAL: dropped tab must not kill arms
bash scripts/run_all.sh            # the whole session, prereg order
# ... wait (see wall-clock table). Inside tmux: Ctrl-b d detaches, tmux attach resumes.
tar czf /root/evidence_$(date +%Y%m%d_%H%M).tgz -C ~/sm120-nulltest evidence
# scp / runpodctl the tarball OFF THE POD, verify it landed, only then terminate
```

If the clone lands elsewhere, adjust the tar path — every script resolves its
own location; nothing assumes a fixed repo path. Knobs (env vars):
`RESTARTS` (default 3, min 3 — prereg), `ITERS` (default 50, min 50 — prereg),
`SKIP_P8`, `CKPT_DIR`, `MSST_DIR`, `P8_WARMUP`/`P8_ITERS` (3/3), `VENV`.

Optional sanity check of the standalone front-end against repo code (off the
profiled path, any time after setup):

```bash
PYTHONPATH=~/work/msst python frontend.py --sanity     # expects "bitwise exact"
```

## 1a. Pod smoke test (shakedown — run once before the real session)

```bash
bash setup.sh && bash scripts/smoke.sh     # ~15-30 min incl. setup
```

Proves the CUDA-only behavior nothing off-GPU can: GPU bitwise sanity vs the
pinned repo; arms A/B/G at 4410 with validation PASS (first real launch of
the fused Triton kernel, and the capture/fallback protocol on real hardware);
kernels inside a graph replay VISIBLE to nsys under `--cuda-graph-trace=node`
(what R depends on); one full P8 trace. Prints a per-phase PASS/FAIL verdict.

Hygiene (pre-registration rules): smoke outputs go to `evidence_smoke/`
(gitignored, never session evidence, never cited); the real session then runs
with a clean `evidence/` and ONE uninterrupted `run_all.sh` — a fresh pod is
equivalent. Mechanical fixes prompted by the smoke are committed before the
session so the audited hash is what runs; any change to arms, thresholds, or
predictions is a prereg amendment to document explicitly, never a silent edit.

## 2. Phases: expected wall-clock, and what "done" looks like

| phase | what | expect | done looks like |
|---|---|---|---|
| setup.sh | nsys install, msst clone @ pin, deps, ckpt fetch | 3–10 min (ckpt download dominates) | `manifest written: .../manifest.env`, then the tmux hint |
| [1] P8 trace | full-model nsys pass, 8 s chunk, session opener | 2–5 min | `P8 RESULT` block printed; `evidence/p8_result.json` + `trace_p8_full.nsys-rep` exist |
| [2] output_code | C2 kernel inventory, both chunks | 5–25 min (max-autotune compiles, cold cache) | `evidence/output_code_C2_352800.log` and `..._4410.log` non-empty (contain generated Triton source) |
| [3] arms | 8 arms × 2 regimes × 3 restarts, fresh process each | 45–90 min (C2 settles dominate) | 48 `evidence/arm_*_r?.json`; each run printed per-iter times + a `status=VALID` line |
| [4] traces | 16 nsys passes + busy extraction | 30–60 min | 16 `evidence/busy_*.json`; `[traces] done.` names the two structural timelines |
| [5] collect | arms × regimes table | seconds | table + `DECISION RULE ... R = ...` printed; `evidence/summary.json` |
| [6] version matrix | venv + latest-stable torch + grouped_mm probe | 5–10 min (torch wheel download) | `P7 ...` verdict line; `evidence/version_matrix_sm120_latest.{json,txt}` |

A validation FAIL or a crashed arm marks that run INVALID in its JSON and the
session **continues** (prereg). INVALID cells show up in the [5] table.

## 3. Extraction (before terminate — pods are ephemeral)

Everything of value is in `evidence/` (gitignored; never committed from the
pod): per-run JSONs, per-iteration times, `.nsys-rep` timelines, kernel-summary
CSVs, TORCH_LOGS captures, output_code dumps, P8 result, version-matrix
verdict, `session_console.log`. Tar it, move it off-pod, **verify the tarball
arrived**, then terminate. The two files the decision rule cannot live
without: `evidence/arm_G_c4410_trace.json` and `evidence/busy_G_4410.json`.

## 4. Troubleshooting the three likely failures

**nsys install fails (setup.sh [2/4]).**
`apt-cache search nsight-systems` — if empty, the NVIDIA devtools repo add
failed; check network/DNS, re-run `bash setup.sh` (idempotent). Still stuck:
download the nsight-systems `.deb` from NVIDIA's devtools page and
`dpkg -i` it. `run_all.sh` refuses to start without nsys (preflight), because
GPU-busy, R, and both structural timelines would be impossible; individual
arms *can* be run manually via `arms/run_arm.py` for wall numbers only, but
that is an incomplete session — fix nsys instead. Note: `analysis/traces.sh`
additionally requires an nsys that supports `--cuda-graph-trace=node`
(anything setup.sh installs does) — without node-level graph tracing,
kernels replayed inside CUDA graphs (arms B/E/G, C1/C2) would be invisible to
the kernel summary and busy numbers would be silently wrong; the script
hard-fails rather than record them.

**Capture failure at the STFT (arms B/E/G).**
Predicted by prereg P4 and pre-authorized: the harness automatically falls
back to capturing the band_split stage only. This is NOT an error. Check the
console line `capture scope: whole_forward|band_split_only` and the
`capture_scope` field in the JSON; the whole-forward traceback is preserved in
`notes`. Only investigate if the FALLBACK also failed (run status INVALID,
error in JSON).

**Recompile storms (arms C1/C2).**
Symptom: `settle_capped: true` in the JSON, long [3] phase, recompile lines in
`evidence/torchlogs_C*_*.log`. Shapes are fixed and `dynamic=False`, so storms
are unexpected; the log states each recompile's reason. A capped settle is
recorded in `notes` and the run still times (auditable afterwards). To give it
more room: `python arms/run_arm.py --arm C2 ... --settle-max 200`.

Also worth knowing: checkpoint sha256 mismatch hard-fails by design (fix
`configs/checkpoint.txt`, never bypass); if `python -m venv` fails in [6],
run that cell on a separate micro-pod (prereg allows it).

**P8 says "NVTX ranges not found".**
The extraction found no `p8_*` ranges in the kernel summary. Check
`evidence/p8_run.json:boundary_mode` (did the hooks install?), inspect the
raw `evidence/p8*kern*sum*.csv` and `trace_p8_full.nsys-rep` in the nsys GUI.
If the model forward crashed the console shows it; if the CSV has ranges
under other names, the nsys report format changed — extract manually from the
CSV (front-end share = frontend range GPU time / total) and record how.

## 5. Deviations & ambiguities (audit this first)

1. **Prereg file format**: provided and bundled as `prereg_null_test.md.pdf`
   (PDF), not `.md`. Content untouched.
2. **setup.sh extended, not rewritten** (session-owner resolution 2026-08-16):
   sections [1/4]–[4/4] byte-identical to the provided file; new [5/6] msst
   clone @ pin, [6/6] checkpoint fetch; the closing ">>> Next: tmux" hint
   (originally the last two lines) relocated below [6/6] so it stays the final
   output. Section numbering jump ([4/4] → [5/6]) is as specified.
3. **P8 vs output_code ordering**: prereg calls the output_code reads "first
   action of the session, before any timing" and P8 the "session opener ...
   before any arms". Session-owner order encoded: P8 trace, then output_code,
   then arms — both constraints hold under the reading that "timing" means arm
   timing (`scripts/run_all.sh`).
4. **Validation tolerance** (amended 2026-08-18; E corrected 2026-09-01):
   two-tier, per the classification rule in `arms/common.py::VAL_TOL` —
   tier = compiler-regenerated kernels present or not. Replay class
   A/B/D/E/F/G: `rtol 1e-4, atol 1e-5` (single fp32 divergence source; B
   measured bitwise, G within 1e-4 in smoke; repo-E is fused+replay with no
   Inductor on its path, so the rule places it tight). Recompiled class
   C1/C2: `rtol 1e-2, atol 1e-3` (per-kernel reordering compounding with
   depth — full rationale in the code comment). TF32 stays at torch defaults
   (off). C1/C2/E (and D/F) have never been validated on hardware — first
   validation is a pod-session opening item.
5. **Headline across ≥3 restarts**: the cross-restart statistic is not named
   in the prereg. Used: median of per-restart latency medians; every per-iter
   list preserved in `evidence/summary.json` (`analysis/collect.py`).
6. **GPU-busy source**: prereg allows "own trace where it exists, else a
   dedicated short pass". Implemented: every (arm × regime) gets its own nsys
   pass, so busy ALWAYS comes from that arm's own trace and pairs with that
   same run's wall (`analysis/traces.sh`); the "otherwise" branch never runs.
   All busy passes use `--cuda-graph-trace=node` so kernels replayed inside
   CUDA graphs (B/E/G and inductor cudagraphs in C1/C2) are counted
   individually — required for the prereg's kernel-sum busy and the
   ~60-GEMMs structural criterion; `evidence/nsys_version.txt` records the
   profiler version. Busy-per-pass divides by the passes the trace run
   actually completed, and R is only computed from a VALID, validation-PASS
   trace run.
7. **Python deps**: beyond frontend.py's imports (torch/triton/numpy/librosa/
   einops), setup.sh installs the minimal set the public repo's
   mel_band_roformer construction path imports **at the pinned commit
   ea7eb9c2** (pyyaml, omegaconf, ml_collections, soundfile, wandb, beartype,
   rotary_embedding_torch, tqdm, packaging, loralib) — NOT the repo's full
   requirements.txt. Required by the P8 provenance constraint. wandb and
   loralib are top-level imports at the pin (not at upstream HEAD).
8. **P8 front-end boundary**: the public path exposes no NVTX stages, so the
   kernel-sequence boundary is made explicit via EXTERNAL forward hooks
   (public torch API, repo code unmodified); `p8_run.json:boundary_mode`
   records whether the 3-range split (frontend/attention/tail) or the 2-range
   fallback (frontend/rest) ran. The attention-end hook assumes the last
   module of the last `model.layers` block executes last (true at the pin).
9. **`--chunk` restricted** to the two prereg regimes (352800, 4410) in
   `arms/run_arm.py` — drift prevention.
10. **"Best generic arm"** (for the structural timeline) := lowest headline
    latency median at 4410 among A/B/C1/C2/F with validation PASS
    (`analysis/collect.py`). The prereg does not define it more precisely.
11. **output_code runs use C2** (max-autotune) at both chunks per
    session-owner instruction. The inductor cache they warm may shorten later
    C2 settles; the fresh-process-per-run rule still holds (prereg mandates
    process isolation, not cold caches).
12. **Timing protocol** is prereg 5a (registered 2026-08-16, pre-data):
    latency primary (defines "wall"), free-running throughput secondary,
    <2% expected bias, >5% escalation clause — encoded verbatim in
    `arms/common.py::timed_run` and surfaced by `analysis/collect.py`.
13. **Fused kernel source**: `rms_fused` is imported from
    `frontend_rms_fused.py` (supplied by the session owner 2026-08-16), not
    from `frontend.py` as the original instruction assumed. `frontend.py`
    stays the canonical model for ALL arms; the fused file contributes only
    the kernel.
14. **Version-matrix probe shapes**: the prereg pins patterns and dtypes, not
    shapes; band-flavored shapes used and recorded in the probe JSON.
15. **P8 precision**: the P8 full-forward trace runs under the public repo's
    OWN inference precision — `torch.cuda.amp.autocast(enabled=
    config.training.use_amp)`, default True, mirroring `utils/model_utils.py`
    demix() lines 137-139 at the pinned commit. Plain fp32 aborts ("No
    available kernel"): the public config sets `flash_attn: true` and
    attend.py forces the flash SDPA backend, which has no fp32 kernels
    (2026-08-17 smoke finding). `use_amp` is recorded in `p8_run.json`.
    Arms A-G remain fp32 (front-end-only, per the prereg front-end config).

## 6. Audit map (grep the marker; exact anchors)

| # | claim | where |
|---|---|---|
| 1 | timed region excludes build/compile/capture/warmup | `arms/common.py` — `timed_run()` (marker `AUDIT-MAP[1]`); phase order in `arms/run_arm.py::main` |
| 2 | validation: pass 2 uses different data; capture arms re-`copy_()` before replay | `arms/common.py` — `validate()` (marker `AUDIT-MAP[2]`); `step` closure in `_capture_with_fallback()`; `ReplayBandSplit.forward` |
| 3 | arm G: fork/join events, single-graph replay of stream-ordered work | `arms/common.py` — `build_arm_step()` arm-G branch (marker `AUDIT-MAP[3]`); `arms/streams.py::MultiStreamBandLoop.run` |
| 4 | run order: P8 first, one process per run, version matrix last | `scripts/run_all.sh` header (marker `AUDIT-MAP[4]`) |
| 5 | P8: public repo path, pinned commit, hash-verified ckpt, no other model source | `scripts/p8_trace.py` docstring + `main()` (marker `AUDIT-MAP[5]`); guards in `scripts/p8_full_trace.sh`; `scripts/fetch_checkpoint.sh` |

Scope guard: the full model is touched ONLY by `scripts/p8_trace.py`. Arms A–G
import `frontend.py` (+ `rms_fused` from `frontend_rms_fused.py`) and nothing
else; `frontend.py` internals are never modified — arm variants swap the
`band_split` submodule from outside.

## 7. Repo layout

```
README.md                 this runbook
build_spec.md             standing rules: borrow discipline, capture-is-final
scripts/precommit.sh      pre-commit integrity gate (parse checks + paste-artifact
                          scan; wired via .githooks/, see section 0)
docs/profiling_crib.md    every nsys/NVTX/profiler-API call: flags, expected output,
                          failure detection (reviewer reproduction reference)
docs/prompt.md            the original build brief (owner-archived)
setup.sh                  provided [1/4]-[4/4] + appended [5/6] clone, [6/6] ckpt fetch
frontend.py               provided harness — canonical front-end, NVTX stages, capture range
frontend_rms_fused.py     provided — session owner's fused RMSNorm Triton kernel (rms_fused)
prereg_null_test.md.pdf   THE SPEC
configs/                  msst_commit.txt + checkpoint.txt (owner-filled pins)
arms/common.py            build, capture, settle, validate, 5a timing, JSON schema
arms/run_arm.py           CLI: one (arm x regime) per fresh process
arms/streams.py           fork/join multi-stream band loop (F/G)
analysis/collect.py       arms x regimes table, R, 5a escalation surfacing
analysis/traces.sh        per-arm nsys passes, busy/kernel-count extraction
scripts/run_all.sh        the session, prereg order (audit-map 4)
scripts/p8_full_trace.sh  P8 opener: nsys of the full public model, extraction
scripts/p8_trace.py       P8 driver: public construction path + NVTX hooks
scripts/fetch_checkpoint.sh  hash-verified, cached checkpoint fetch
scripts/version_matrix_check.sh  isolated venv, latest stable torch, probe
scripts/grouped_mm_probe.py      P7 probe (both patterns, all dtypes)
evidence/                 all outputs; gitignored except .gitkeep
```

## 8. Ceiling-ledger full-model sessions (2026-09-20 →)

The plan `../plan_ceiling_ledger.md` reuses this harness for a FULL-MODEL arm
matrix. Registration: `prereg_amendments.md` (pushed before session 1).
Runbook: `docs/fullmodel_runbook.md`. Code: `fullmodel/` (arm builder,
5a timing, capture ladder, pad-and-batch arm H, SDR gate, graph
verification, microbenchmarks, collectors); sessions: `scripts/session1.sh`,
`scripts/session2.sh`; clip manifest `configs/sdr_clips.json` (audio never
committed). The original front-end-only arms (`arms/run_arm.py`,
`scripts/run_all.sh`) are unchanged.
