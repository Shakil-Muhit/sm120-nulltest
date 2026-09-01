# Profiling crib — every profiler invocation in this repo

Reproduction reference for reviewers. One entry per nsys / NVTX / profiler-API call
site: the exact invocation, what each flag does, what correct output looks like, and
how to detect that it silently went wrong.

Every "observed" figure below is real output, not an illustration: 5090 shakedown of
2026-08-17 (`evidence_smoke/`), nsys **2026.4.1**, torch 2.8.0+cu128, driver 610.43.02.
Reproduce with `bash setup.sh && bash scripts/smoke.sh`.

**Environment facts that shape everything here**

- HW performance counters are **locked** on RunPod host types (`ERR_NVGPUCTRPERM`).
  Wall timing and nsys timelines are unaffected; `ncu` sessions are not possible.
- nsys is **absent** from the stock image; `setup.sh [2/4]` installs it.
- nsys **≥ 2023.4 is mandatory** — see §4.1. Older builds cannot see inside CUDA graphs.

---

## 1. In-process instrumentation

### 1.1 Capture-range control (`torch.cuda.profiler`)

**Where** [`arms/common.py:399,406`](../arms/common.py#L399) (arm timing) ·
[`scripts/p8_trace.py:228,234`](../scripts/p8_trace.py#L228) (P8) ·
[`frontend.py:249,255`](../frontend.py#L249) (standalone harness)

```python
torch.cuda.synchronize()
torch.cuda.profiler.start()     # == cudaProfilerStart(); opens the nsys capture range
for _ in range(iters):
    t0 = time.perf_counter(); step(x); torch.cuda.synchronize()
    latency_ms.append((time.perf_counter() - t0) * 1e3)
torch.cuda.profiler.stop()      # == cudaProfilerStop(); closes it
```

| call | effect |
|---|---|
| `profiler.start()` | begins collection when nsys runs with `--capture-range=cudaProfilerApi`; a no-op otherwise |
| `profiler.stop()` | ends collection; with `--capture-range-end=stop` the process keeps running and is not killed |

**What it deliberately excludes:** model build, `torch.compile` settling, CUDA-graph
capture, warmup, validation, and the free-running throughput block all execute outside
this range. A trace therefore contains **only** the latency-block passes, which is what
makes GPU-busy comparable to that same run's wall time.

**Correct output** — kernel instances in the report divide evenly by the iteration
count. Arm G @4410, 50 iters: 6 700 instances → exactly 134.0/pass.

**Failure detection**

| symptom | cause |
|---|---|
| `No reports were generated`, `.nsys-rep` < ~50 KB | `start()` never reached (exception during build, or CPU fallback path) |
| kernels/pass non-integral or far above expectation | range opened too early — warmup or capture inside it |
| full-process trace, no capture range honored | `--capture-range` missing from the nsys command line |

### 1.2 Front-end stage ranges (NVTX, provided harness)

**Where** [`frontend.py:122-157`](../frontend.py#L122) — six push/pop pairs.

```python
_nvtx_push('1_window'); ...; _nvtx_pop()      # then 2_stft, 3_view_as_real_merge_stereo,
                                              # 4_band_gather, 5_fold_complex_into_freq,
                                              # 6_band_split_norm_linear
```

Stage names are part of the provided harness and **must not be renamed** — arm
timelines are read against them.

**Correct output** — in the nsys GUI, six ranges per pass, in order, nested under no
parent. `nsys stats -r nvtx_pptrace` lists them if you want them textually.

**Failure detection** — a missing `_nvtx_pop()` makes subsequent ranges nest inside the
previous one; the GUI shows a staircase instead of six siblings, and NVTX-keyed reports
attribute later stages to earlier ranges. On CPU the push/pop are lambdas returning
`None` and no ranges appear at all — expected.

### 1.3 Full-model stage ranges (NVTX via forward hooks)

**Where** [`scripts/p8_trace.py`](../scripts/p8_trace.py) → `install_nvtx_hooks()`

The public model exposes no NVTX stages, so boundaries are imposed from outside with
`register_forward_pre_hook` / `register_forward_hook` — the upstream repo is never
edited:

```
model pre-hook              → push 'p8_frontend'
model.band_split post-hook  → pop, push 'p8_attention'
model.layers[-1][-1] post   → pop, push 'p8_tail'
model post-hook             → pop
```

**Hooks must return `None`.** A non-`None` return from a forward hook *replaces the
module's output*, and `nvtx.range_push/pop` return ints — so these are named functions,
never bare lambdas. This was a real defect caught pre-run.

`boundary_mode` in `p8_run.json` records which split was realized:
`frontend/attention/tail`, or `frontend/rest` if the last transformer submodule could
not be resolved.

**Caveat to state in any writeup:** the attention range closes when the *last submodule*
of the last block returns. Work performed by the enclosing loop after that call is
attributed to `p8_tail`, so attention is a slight under-count.

**Failure detection** — model output becomes garbage or the forward raises (hook
returned non-`None`); `boundary_mode` starting `frontend/rest` when three ranges were
expected; range set in the CSV ≠ `{p8_frontend, p8_attention, p8_tail}`.

**Sub-ranges (fix-pass 2026-08-18).** Nested inside `p8_attention`, per transformer
block *i*: `p8_L<i>_time_attn`, `p8_L<i>_rearrange` (the unpack→rearrange→pack gap
between the time and freq calls, bracketed by time's post-hook and freq's pre-hook),
`p8_L<i>_freq_attn`, and `p8_L<i>_linear_attn` when a block has three modules. Each
block's pre-time pack and post-freq unpack stay in the parent range and are reported
by extraction as `attention_other_ms_per_pass`. `boundary_mode` records the realized
granularity (`...;sub=per_layer(time_attn,rearrange,freq_attn)x6`, or `;sub=none`
when the block structure was not recognized — coarse ranges still work). Correct
output: nested rows in `nvtx_kern_sum` whose Range path contains both the parent and
the sub-name; parent totals are unchanged by nesting because each kernel row carries
its full path and is counted once per aggregation key. Failure signature:
`sub=none` in `boundary_mode`, or sub-range kinds missing from
`p8_result.json:attention_substages`; an unbalanced push/pop (a bug, not an nsys
condition) shows as sub-ranges bleeding across layer indices in the GUI.

---

## 2. `nsys profile` invocations

### 2.1 Per-arm trace (the busy/kernel-count source)

**Where** [`analysis/traces.sh:76`](../analysis/traces.sh#L76) — runs for all 8 arms × 2 regimes.

```bash
TORCH_LOGS="$logs_env" nsys profile \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --cuda-graph-trace=node \
    --force-overwrite=true -o "$base" \
    python arms/run_arm.py --arm "$arm" --chunk "$chunk" --iters "$ITERS" --tag trace
```

| flag | one-line effect |
|---|---|
| `--capture-range=cudaProfilerApi` | collect only between `cudaProfilerStart/Stop` — i.e. the latency block |
| `--capture-range-end=stop` | end collection at `cudaProfilerStop()` and let the process finish normally |
| `--cuda-graph-trace=node` | record each kernel **inside** a CUDA graph replay individually instead of one opaque graph activity |
| `--force-overwrite=true` | replace an existing `<base>.nsys-rep` instead of erroring |
| `-o <base>` | output base name; produces `<base>.nsys-rep` |
| `TORCH_LOGS=recompiles,perf_hints` | compile arms only; stderr is redirected to `evidence/torchlogs_<arm>_<chunk>_trace.log` |

**Correct output** — `<base>.nsys-rep`, ~240 KB for a 50-iteration front-end arm
(observed: `trace_G_4410.nsys-rep` = 242 183 B).

**Failure detection**

- pass exits non-zero → `[traces] WARN: nsys pass failed for <arm>/<chunk> — skipping`,
  and no `busy_*.json` is written for that cell (`rm -f` runs before every pass, so a
  stale file cannot survive to be mispaired).
- **`--cuda-graph-trace` silently defaulting to `graph`** is the dangerous one — see §4.1.

### 2.2 P8 full-model trace

**Where** [`scripts/p8_full_trace.sh:43`](../scripts/p8_full_trace.sh#L43) — one pass, 8 s chunk, session opener.

```bash
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
    --force-overwrite=true -o "$EVID/trace_p8_full" \
    python scripts/p8_trace.py --msst-dir ... --ckpt ... --config ... \
        --model-type mel_band_roformer --chunk 352800 --warmup 3 --iters 3 \
        --json "$EVID/p8_run.json"
```

Same flags as §2.1 minus `--cuda-graph-trace` — the full model runs eager under
autocast, with no CUDA graphs to look inside. Guarded upstream by a pinned-commit check
and sha256 checkpoint verification; the forward runs under
`torch.cuda.amp.autocast(enabled=config.training.use_amp)` to match the public repo's
own inference (fp32 aborts — see §6).

**Correct output** — `trace_p8_full.nsys-rep` ≈ 350 KB (observed 349 741 B) plus
`p8_run.json` containing 3 wall times ≈ 67 ms/pass and `boundary_mode`.

**Failure detection** — `FATAL: P8 trace run failed` with the child traceback in the
console log; the most likely causes are precision (§6) and a stale msst pin.

### 2.3 Shakedown pass

**Where** [`scripts/smoke.sh:76`](../scripts/smoke.sh#L76) — arm G @4410, identical flags to §2.1,
output redirected to `evidence_smoke/`. Exists to prove graph-node visibility on the
actual pod before the session commits to it.

---

## 3. `nsys stats` extraction

Both call sites try the modern report name first and fall back to the legacy alias, so
the repo works on old and new nsys.

### 3.1 Kernel summary → GPU-busy and kernels/pass

**Where** [`analysis/traces.sh:86`](../analysis/traces.sh#L86) (and `smoke.sh:80`)

```bash
nsys stats --force-export=true --force-overwrite=true \
    -r cuda_gpu_kern_sum --format csv --output "$base" "$base.nsys-rep"
# fallback: -r gpukernsum   (nsys < ~2023)
```

| flag | one-line effect |
|---|---|
| `--force-export=true` | re-export the `.sqlite` from the `.nsys-rep` even if one exists |
| `--force-overwrite=true` | overwrite existing report CSVs — **required for any re-run** |
| `-r cuda_gpu_kern_sum` | CUDA GPU Kernel Summary (legacy alias `gpukernsum`) |
| `--format csv` | machine-readable output |
| `--output <base>` | writes `<base>_<report>.csv` |

**Correct output** — `<base>_cuda_gpu_kern_sum.csv` plus a ~1 MB `.sqlite`. Header is:

```
Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
```

Observed for arm G @4410 (50 iters, whole-forward capture, 4 streams): **14 kernel
rows**, Σ Instances 6 700, Σ Total Time 9 479 137 ns. Derived:

```
busy_ms_per_pass = 9_479_137 / 1e6 / 50 = 0.1896
kernels_per_pass = 6_700 / 50           = 134.0
```

The top row is `gemmSN_TN_kernel<...>` with **3 000 instances = 60.0 per pass** — the 60
per-band GEMMs, i.e. the prereg's structural criterion readable straight off the CSV.

Extraction lives in the heredoc at the end of `traces.sh` and writes
`busy_<arm>_<chunk>.json`; it divides by `len(latency_ms_per_iter)` (passes actually
completed) and **refuses to write anything if the run JSON is not `VALID`**.

**Failure detection**

| symptom | cause |
|---|---|
| stats exits non-zero on a re-run | `--force-overwrite` omitted (hand-run); the CSV already exists |
| `StopIteration` inside the extractor | header dialect changed — check `head -1` against the header above |
| `[traces] WARN: busy extraction failed for <arm>/<chunk>` | no CSV matched, run not VALID, or zero completed iterations |
| kernels/pass ≈ 0–1 on captured arms | graph tracing not at node granularity — §4.1 |

### 3.2 NVTX summary → per-stage GPU time

**Where** [`scripts/p8_full_trace.sh:53`](../scripts/p8_full_trace.sh#L53)

```bash
nsys stats --force-export=true --force-overwrite=true \
    -r nvtx_kern_sum --format csv --output "$EVID/p8" "$EVID/trace_p8_full.nsys-rep"
# fallback: -r nvtxkernsum
```

**Correct output** — `p8_nvtx_kern_sum.csv`. **Header differs from §3.1** and this is the
single most common parsing trap:

```
NVTX Range,Style,PID,TID,NVTX Inst,Kern Inst,Total Time (ns),Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Kernel Name
```

Note `Kern Inst` / `NVTX Inst` here versus plain `Instances` in the kernel summary; the
extractor resolves columns by exact name first, with a guarded substring fallback that
excludes `Range`/`Proj`/`NVTX` matches. Range values carry a **leading colon**
(`:p8_frontend`), so matching is by substring, not equality.

Observed for P8 (3 iters, 8 s chunk, autocast): **83 rows**, three ranges present, and:

| range | ms/pass | kernels/pass |
|---|---|---|
| `p8_frontend` | 0.699 | 434 |
| `p8_attention` | 55.017 | 732 |
| `p8_tail` | 2.931 | 470 |

→ front-end share 1.19 %, written to `p8_result.json`.

**Failure detection** — the extractor exits with
`P8 NVTX ranges not found in the kernel summary` if no `p8_*` range matched; the script
runs under `set -o pipefail`, so that failure is fatal rather than swallowed by `tee`.

---

## 4. Gates and version checks

### 4.1 Graph-trace capability gate — **the load-bearing one**

**Where** [`analysis/traces.sh:49`](../analysis/traces.sh#L49) (and `smoke.sh:73`)

```bash
if ! nsys profile --help 2>/dev/null | grep -q 'cuda-graph-trace'; then
    echo "FATAL: this nsys does not support --cuda-graph-trace=node; busy"
    ...
    exit 1
fi
```

**Why it exists.** Since nsys 2023.4, with driver ≥ CUDA 12.3, `--cuda-graph-trace`
defaults to `graph`: a whole graph replay is reported as **one activity** and its
kernels never appear in `cuda_gpu_kern_sum`. Arms B/E/G replay a manually captured
graph, and C1/C2 replay Inductor cudagraphs — so under the default, GPU-busy for every
captured arm collapses toward zero, `R = wall / busy` inflates, and the decision rule is
mechanically pushed toward "measured headroom stands". The ~60-GEMM structural criterion
also becomes uninspectable. The gate refuses to produce numbers rather than produce
wrong ones.

**Correct output** — silent pass, then `nsys --version` is echoed and recorded to
`evidence/nsys_version.txt`.

**Failure detection, by hand** — compare an eager arm with a captured one at the same
chunk:

```bash
python - <<'EOF'
import csv,glob
for a in ('A','G'):
    p=sorted(glob.glob(f'evidence/trace_{a}_4410*kern*sum*.csv'))[-1]
    rows=list(csv.reader(open(p))); i=rows[0].index('Instances')
    print(a, sum(int(float(r[i])) for r in rows[1:] if r[i].strip())/50, 'kernels/pass')
EOF
```

Healthy: A ≫ 100, G ≈ 134. Broken: G ≈ 0–1 while A looks normal.

### 4.2 nsys install verification

**Where** [`setup.sh:29`](../setup.sh#L29)

```bash
command -v nsys >/dev/null 2>&1 && nsys --version | head -1 \
    || { echo "FATAL: nsys install failed. Inspect: apt-cache search nsight-systems"; exit 1; }
```

**Correct output** — one line, e.g. `NVIDIA Nsight Systems version 2026.4.1.191-...`.

**Known failure, and its fix.** On images with Ubuntu *universe* enabled, the fallback
package search `'^nsight-systems'` also matches Ubuntu's CLI-less
`nsight-systems-target` (2021.3), which sorts last and provides no `nsys`. Symptom:
apt reports success, then this check FATALs. Install a versioned package explicitly:

```bash
apt-cache search '^nsight-systems-[0-9]' | awk '{print $1}' | sort -V | tail -1
apt-get install -y nsight-systems-<version>
```

---

## 5. `ncu` — informational only

**Where** [`setup.sh:34`](../setup.sh#L34)

```bash
ncu --set basic -c 1 python -c "import torch; (torch.randn(256,256,device='cuda')@torch.randn(256,256,device='cuda')).sum().item()"
```

**Expected output on the target host type: failure**, with `ERR_NVGPUCTRPERM`. HW
counters are locked; this call documents that fact rather than depending on it. Nothing
in the experiment consumes ncu output. `ncu not on PATH — fine for null test` is equally
acceptable.

---

## 6. Reproduction gotchas

**Profiling inflates wall time.** Observed on arm G @4410: median **0.0791 ms**
untraced versus **0.1101 ms** under nsys — **+39 %**. `R` is computed from the traced
run's own wall (the prereg's "same run" requirement), so `R` carries that overhead:
0.581 traced versus 0.417 if the clean wall were paired with the traced busy. Never mix
traced and untraced walls in one comparison, and state which convention a reported `R`
uses.

**P8 must run under autocast.** The public config sets `flash_attn: true` and the
upstream `attend.py` restricts SDPA to the flash backend, which has no fp32 kernels.
Plain fp32 fails with `RuntimeError: No available kernel. Aborting execution.`
`p8_trace.py` mirrors the repo's own `demix()` context; `use_amp` is recorded in
`p8_run.json`. Arms A–G remain fp32 by design.

**One process per measurement.** Every trace is a fresh `python arms/run_arm.py`; no
report is ever derived from a process that also ran another arm.

## 7. Symptom → cause triage

| symptom | first check |
|---|---|
| `busy_ms_per_pass` ≈ 0, `R` huge, only on B/E/G/C1/C2 | `--cuda-graph-trace=node` present? (§4.1) |
| `No reports were generated` | did `profiler.start()` execute? (§1.1) |
| `StopIteration` in an extractor | CSV header dialect (§3.1 vs §3.2) |
| `nsys stats` fails on second run | `--force-overwrite=true` (§3.1) |
| P8 ranges missing | hook install / `boundary_mode` (§1.3) |
| P8 `No available kernel` | autocast (§6) |
| setup FATALs after "successful" apt | `nsight-systems-target` trap (§4.2) |
| `R` disagrees with a previously reported value | traced vs untraced wall (§6) |
| kernels/pass not an integer | capture range too wide (§1.1) |
