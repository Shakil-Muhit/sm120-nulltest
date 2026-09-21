"""SESSION_RESULT_v2.md — the Sep 20 session result with the Phase-3 §2.5 corrections and §2.1 statistics.

Regenerates the v1 tables (session_result.py) and adds: precision regime; per-timing-block SM/memory clocks
(manifest GPU, median/min/max per block, CSV with every block); class totals labelled as measured durations;
Vol. 1 reconciliation; the 2× line definitions; boundary-cost definitions; the ANVTX sublayer table; the
autocast cast condition; block-level bootstrap intervals (docs/phase3/stats); the X1 / pad-and-batch / X2
decompositions. Every number cites its file."""
import argparse, bisect, csv, datetime as _dt, glob, json, os, statistics, subprocess, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None

def med(xs):
    return statistics.median(xs) if xs else None

def fmt(v, spec='.4f'):
    return format(v, spec) if isinstance(v, (int, float)) and not isinstance(v, bool) else 'UNRESOLVED'

def clock_samples(ev, gpu_uuid):
    out = []
    for f in sorted(glob.glob(os.path.join(ev, 'clocks_session*.csv'))):
        for line in open(f):
            parts = [x.strip() for x in line.split(',')]
            if len(parts) != 7 or (gpu_uuid and parts[1] != gpu_uuid):
                continue
            try:
                # nvidia-smi logs the POD's local time; the pod ran UTC (session logs carry +00:00) — CLOCK_TZ_OFFSET_S overrides
                ts = _dt.datetime.strptime(parts[0], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=_dt.timezone.utc).timestamp() - float(os.environ.get('CLOCK_TZ_OFFSET_S', '0'))
                out.append((ts, float(parts[2].replace('MHz', '')), float(parts[3].replace('MHz', '')), float(parts[4].replace('W', '')), int(parts[5])))
            except ValueError:
                continue
    out.sort()
    return out

def per_block_clocks(ev, gpu_uuid):
    S = clock_samples(ev, gpu_uuid)
    ts = [s[0] for s in S]
    rows = []
    for path in sorted(glob.glob(os.path.join(ev, 'full_*_r[123].json'))) + sorted(glob.glob(os.path.join(ev, 'matched_*_m[123].json'))):
        r = load(path) or {}
        blocks = ((r.get('timing') or {}).get('blocks') or []) if 'timing' in r else (r.get('blocks') or [])
        ident = f"{r.get('arm', '+'.join(r.get('pair') or []))}@{r.get('chunk')}/{r.get('tag')}"
        for b in blocks:
            a, e = b.get('unix_start'), b.get('unix_end')
            if not (isinstance(a, (int, float)) and isinstance(e, (int, float))):
                continue
            i, j = bisect.bisect_left(ts, a), bisect.bisect_right(ts, e)
            seg = S[i:j]
            sm = [s[1] for s in seg]; mem = [s[2] for s in seg]; pw = [s[3] for s in seg]; tp = [s[4] for s in seg]
            rows.append({'run': ident, 'arm_in_block': b.get('arm', r.get('arm')), 'repeat': b.get('repeat'), 'samples': len(seg),
                         'sm_median_mhz': med(sm), 'sm_min_mhz': min(sm) if sm else None, 'sm_max_mhz': max(sm) if sm else None,
                         'mem_median_mhz': med(mem), 'power_median_w': med(pw), 'temp_median_c': med(tp), 'block_ms': b.get('latency_median_ms')})
    return rows

def anvtx_table(ev, O):
    p = os.path.join(ev, 'traces', f'trace_ANVTX_{O}_trace_nvtx_gpu_proj_trace.csv')
    if not os.path.exists(p):
        return None
    d = {}
    with open(p, newline='') as f:
        for r in csv.DictReader(f):
            name = r['Name'].split(':')[-1].strip().strip('"')
            if name.startswith('pass_'):
                continue
            try:
                d.setdefault(name, []).append(float(r.get('Duration (ns)') or r.get('Projected Duration (ns)') or 0))
            except ValueError:
                continue
    return {k: (med(v) / 1e3, len(v)) for k, v in d.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence-dir', required=True)
    ap.add_argument('--out', default='SESSION_RESULT_v2.md')
    ap.add_argument('--registration-commit', default='e523ef9')
    args = ap.parse_args()
    ev = args.evidence_dir
    tmp = os.path.join(_ROOT, 'docs', 'phase3', '_session_result_v1_regenerated.md')
    subprocess.run([sys.executable, os.path.join(_HERE, 'session_result.py'), '--evidence-dir', ev, '--out', tmp,
                    '--registration-commit', args.registration_commit, '--pushed-utc', '2026-09-20 ~11:00 UTC (local push log; client-side timestamp, see docs/provenance/push_evidence.md)'], check=True, capture_output=True)
    v1 = open(tmp).read()
    man = load(os.path.join(ev, 'session_manifest.json')) or {}
    env = man.get('env') or {}
    runs = {}
    for path in glob.glob(os.path.join(ev, 'full_*_r1.json')):
        r = load(path)
        if r: runs[(r['arm'], r['chunk'])] = r
    amp = sorted({str(r.get('use_amp')) for r in runs.values()})
    # (i) precision regime beside the headline table
    v1 = v1.replace('### Arms × operating points (headline = median over r1–r3 of per-process min-of-medians; unprofiled)',
                    '### Arms × operating points (headline = median over r1–r3 of per-process min-of-medians; unprofiled)\n'
                    f'**Precision regime:** every arm runs the public path under `torch.autocast(\'cuda\', enabled=use_amp)` with `use_amp = {"/".join(amp)}` as recorded in each run record (fp16 autocast, fp32 master weights; Deviation 15). Compiled arms carry the autocast casts as graph ops (see §2.5(viii) below).')
    # (iii) class totals label
    v1 = v1.replace('### Per-kernel ideal costs (class totals µs/pass; GEMM/SDPA ideal vs measured)',
                    '### Per-kernel ideal costs — the six class columns are MEASURED device durations by kernel class (µs/pass, node-level trace, perturbation caveat applies); the two ideal columns are roofline ideals')
    # (ii) per-block clocks: replace the session-median clock lines
    rows = per_block_clocks(ev, env.get('gpu_uuid'))
    os.makedirs(os.path.join(_ROOT, 'docs', 'phase3', 'stats'), exist_ok=True)
    cpath = os.path.join(_ROOT, 'docs', 'phase3', 'stats', 'clocks_per_block.csv')
    with open(cpath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    lines = [l for l in v1.splitlines() if not l.startswith('clocks session')]
    v1 = '\n'.join(lines)
    by_run = {}
    for r in rows:
        by_run.setdefault(r['run'], []).append(r)
    clk = ['## Per-timing-block clocks (manifest GPU, 1 Hz nvidia-smi samples; CSV: docs/phase3/stats/clocks_per_block.csv)',
           '| run | blocks | SM median MHz (median over blocks) | SM min MHz (any block) | SM max MHz | memory MHz | power W (median) | max block-to-block SM drift |',
           '|---|---|---|---|---|---|---|---|']
    for k, rs in sorted(by_run.items()):
        sm = [x['sm_median_mhz'] for x in rs if x['sm_median_mhz'] is not None]
        if not sm:
            clk.append(f'| {k} | {len(rs)} | UNRESOLVED (no samples in block windows) | | | | | |'); continue
        drift = (max(sm) - min(sm)) / max(sm)
        clk.append(f"| {k} | {len(rs)} | {fmt(med(sm), '.0f')} | {fmt(min(x['sm_min_mhz'] for x in rs if x['sm_min_mhz'] is not None), '.0f')} | {fmt(max(x['sm_max_mhz'] for x in rs if x['sm_max_mhz'] is not None), '.0f')} | {fmt(med([x['mem_median_mhz'] for x in rs if x['mem_median_mhz'] is not None]), '.0f')} | {fmt(med([x['power_median_w'] for x in rs if x['power_median_w'] is not None]), '.0f')} | {drift*100:.1f} % |")
    # statistics (2.1)
    st = [ '## Statistics (Phase 3 §2.1): estimands, units, intervals',
           '- **Estimands.** The headline table reports, per (arm, operating point), the *favourable attainable latency*: the median over three process restarts of each restart\'s min-of-medians (minimum over five 50-pass blocks of the block median). The "typical" column below reports the *typical latency*: the median over all 15 blocks of the block medians. Both are stated; the headline is the registered quantity (§5 of the registration); the typical latency is what a deployment sees on average.',
           '- **Experimental units.** Process restarts are the independent units (fresh process, fresh compile/capture, fresh CUDA context); blocks are nested in restarts and passes in blocks. The matched pairs are paired at the block level: five alternating (a, b) blocks per restart in one process, so the pair difference is a within-process quantity.',
           '- **Precision needed.** Claims of the form "arm X is 2× faster" or "C1 pays 19 % for graph breaks" need far less precision than rankings within 1 % (for example C3 vs H at 8 s, −1.7 %). The intervals below say which is which.',
           '- **Method.** Non-parametric bootstrap, 10,000 resamples, seed 20260921: for headlines, resampling passes within blocks and blocks within each restart (the three restart groups are kept), giving **block-level** intervals; for matched pairs, resampling the five alternating blocks of each restart. These are block-level intervals: restart-level inference needs more restarts (E6, five per headline arm). No bootstrap over the three restart summaries is presented as restart-level uncertainty.',
           '- **A/A block.** The Sep 20 evidence contains no A/A block (same arm alternated against itself); E6 adds one at both operating points. An A/A block detects spurious differences under the ABAB design; it validates no causal explanation.',
           '',
           '| arm | O | headline (min-of-medians) ms | block-level 95 % CI | typical (median of block medians) ms | 95 % CI | n restarts / blocks / passes |',
           '|---|---|---|---|---|---|---|']
    with open(os.path.join(_ROOT, 'docs', 'phase3', 'stats', 'bootstrap_headlines.csv')) as f:
        for r in csv.DictReader(f):
            st.append(f"| {r['arm']} | {r['chunk']} | {float(r['headline_min_of_medians_ms']):.4f} | [{float(r['ci95_lo']):.4f}, {float(r['ci95_hi']):.4f}] | {float(r['typical_median_ms']):.4f} | [{float(r['typical_ci95_lo']):.4f}, {float(r['typical_ci95_hi']):.4f}] | {r['n_restarts']} / {r['n_blocks']} / {r['passes_per_block']} |")
    st += ['', '| pair | restart | difference a − b (ms) | 95 % CI | ratio a / b | 95 % CI | blocks | interval crosses zero? |', '|---|---|---|---|---|---|---|---|']
    flags = []
    with open(os.path.join(_ROOT, 'docs', 'phase3', 'stats', 'bootstrap_pairs.csv')) as f:
        for r in csv.DictReader(f):
            lo, hi = float(r['diff_ci95_lo']), float(r['diff_ci95_hi'])
            cross = lo <= 0 <= hi
            if cross: flags.append(f"{r['pair']} {r['tag']}")
            st.append(f"| {r['pair']} | {r['tag']} | {float(r['diff_median_ms']):+.4f} | [{lo:+.4f}, {hi:+.4f}] | {float(r['ratio_median']):.4f} | [{float(r['ratio_ci95_lo']):.4f}, {float(r['ratio_ci95_hi']):.4f}] | {r['n_blocks']} | {'YES' if cross else 'no'} |")
    st += ['', f"**Flags.** Pair intervals crossing zero: {', '.join(flags) if flags else 'none'}. Large-and-consistent (robust to n = 3 restarts): C1 − C3 at 8 s (+6.7 ms, every block interval far from zero), the 4 410 ordering H < C3 < G/E/B < C1/C2 < D/A/F (non-overlapping block-level intervals), the 8 s ordering C3 < H < C2 < C1 ≈ C1D < eager arms. Fine rankings not yet supported at the restart level: C3 vs H at 8 s (−0.5 ms, 1.7 %; block-level intervals exclude zero in every restart but n = 3), C1 vs C1D (−0.05 ms; one restart's interval crosses zero), C2 vs C3 at 4 410 order vs C1. Restart-level intervals require E6.",
           'Files: docs/phase3/stats/bootstrap_headlines.csv, docs/phase3/stats/bootstrap_pairs.csv, generated from evidence/full_*_r*.json and evidence/matched_*_m*.json.']
    # (v) the 2x line
    S = load(os.path.join(ev, 'summary_full.json')) or {}
    c1, c3, h = S['cells']['C1_4410']['headline_ms'], S['cells']['C3_4410']['headline_ms'], S['cells']['H_4410']['headline_ms']
    twox = ['## The "2× line" at 4 410 (§2.5(v))',
            f'The plan (Phase 3.2) reads: "Streaming 4410 (probe): Y ≥ 2× on wall over the best arm — reported, but pursued only if the owner reopens the streaming lane." Two readings exist. Reading A, a 2× speed-up: (wall + Y) / wall ≥ 2, i.e. Y ≥ T_wall. Reading B, half the wall: Z = Y / T_wall ≥ 50 %. For X1 at 4 410 (Y = C1 − C3 = {c1 - c3:.4f} ms, best-arm wall = H = {h:.4f} ms): reading A gives (wall + Y)/wall = {(h + (c1 - c3)) / h:.3f}×, below 2×; reading B gives Z = {100 * (c1 - c3) / h:.1f} %, above 50 %. The Sep 20 LEDGER.md and execution report applied reading B ("meets 2× on wall") — LEDGER.md coded the threshold as Y ≥ T_wall × (1 − 1/2), which is reading B. **Which reading the registration meant is an owner decision (§5 of the handoff); until it is made, both readings are reported side by side and no row is described as meeting the streaming line.** Registration text: plan_ceiling_ledger.md Phase 3.2; code: fullmodel/ledger.py THRESH.']
    # (vi) boundary definitions
    bb = load(os.path.join(ev, 'boundary_bench.json')) or {}
    fits = bb.get('fits') or {}
    ta = load(os.path.join(ev, 'boundary_bench_trace_analysis.json')) or {}
    c3r = load(os.path.join(ev, 'traces', 'graph_verify_C3_352800.json')) or {}
    bd = ['## What the boundary numbers are (§2.5(vi))',
          f"- **0.73 µs/node (graph)** and **4.88 µs/node (stream)** are the SLOPES of the unprofiled chain wall time T(N) against N for a dependent chain of trivial int32 Triton kernels (N ∈ {{10, 100, 1000, 2000}}, 20 timed repeats after 5 warm-ups; evidence/boundary_bench.json `fits`: trivial/graph slope {fmt(fits.get('trivial/graph', {}).get('slope_us_per_node'), '.3f')} µs, intercept {fmt(fits.get('trivial/graph', {}).get('intercept_us'), '.3f')} µs; trivial/stream slope {fmt(fits.get('trivial/stream', {}).get('slope_us_per_node'), '.3f')} µs). They are DEVICE-SIDE per-node costs as seen by the host clock across the whole chain: execution of the trivial kernel plus its inter-node dependency resolution, per node. They are not host launch costs.",
          f"- **0.58 µs/node (PDL graph)** is the same slope for the chain launched with programmatic dependent launch (Triton 3.4 `launch_pdl=True`, which sets `cudaLaunchAttributeProgrammaticStreamSerialization`; the kernel executes `griddepcontrol.wait` before its dependent load and `griddepcontrol.launch_dependents` after its store) and captured into a graph: evidence/boundary_bench.json `fits['trivial/pdl_graph']` slope {fmt(fits.get('trivial/pdl_graph', {}).get('slope_us_per_node'), '.3f')} µs. `griddepcontrol.wait` / `.launch_dependents` are confirmed in the PTX of the compiled kernel (`pdl_ptx = {bb.get('pdl_ptx')}`); the SASS was not inspected on Sep 20 (E7). Whether any overlap actually occurs on SM120 is NOT established by a slope: E7's node-level trace is the test, and the programming guide states the overlap is opportunistic.",
          f"- **0.29 µs / 0.16 µs (graph / PDL-graph serial gap)** are the median idle gaps between consecutive kernels of the N = 1000 trivial chain in the node-level trace of a separate traced run (evidence/boundary_bench_trace_analysis.json `trivial/graph/N1000` gap {fmt((ta.get('trivial/graph/N1000') or {}).get('gap_us_median'), '.3f')} µs; `trivial/pdl_graph/N1000` gap {fmt((ta.get('trivial/pdl_graph/N1000') or {}).get('gap_us_median'), '.3f')} µs): the boundary cost with kernel execution excluded. These are the numbers the ledger's X10/X12 use.",
          f"- **Host-side launch cost** (NVIDIA's constant-time-launch distinction: CPU time inside `cudaGraphLaunch`) was not a registered quantity; the C3 trace's API rows give it as context: median `cudaGraphLaunch` duration for the 1142-node C3 graph at 8 s = 222 µs (evidence/traces/trace_C3_352800_trace_cuda_api_trace.csv, i.e. ≈0.19 µs per node), against NVIDIA's CUDA 12.6 figure of 2.5 µs + ~1 ns per node for a straight-line graph on an RTX 3060 — the C3 graph is not straight-line (two streams, {c3r.get('medians', {}).get('captured_model_kernels', 'n/a')} captured kernels), so the comparison is indicative only.",
          "- **The NVIDIA device-side comparator**: 1,025-node straight-line graph executing in 567 µs on an RTX 3060 under CUDA 12.6 (≈0.55 µs/node) is the device-execution analogue of the 0.73 µs/node slope, not of the host launch cost."]
    # (vii) ANVTX table
    an = ['## ANVTX per-sublayer table (§2.5(vii); node-level trace of the attribution arm, projected NVTX ranges, median over 50 passes)',
          '| sublayer range | 8 s µs | share of pass | 4 410 µs | share of pass |', '|---|---|---|---|---|']
    t8, t4 = anvtx_table(ev, 352800), anvtx_table(ev, 4410)
    if t8 and t4:
        pass8 = load(os.path.join(ev, 'traces', 'timing_views_ANVTX_352800.json')) or {}
        pass4 = load(os.path.join(ev, 'traces', 'timing_views_ANVTX_4410.json')) or {}
        w8 = (pass8.get('median_us') or {}).get('window_ns'); w4 = (pass4.get('median_us') or {}).get('window_ns')
        for k in sorted(set(t8) | set(t4), key=lambda s: (s.split('_')[1] if s.count('_') > 1 else '', s)):
            a, b = t8.get(k), t4.get(k)
            an.append(f"| {k} | {fmt(a[0], '.1f') if a else '—'} | {fmt(100 * a[0] / w8, '.1f') + ' %' if (a and w8) else '—'} | {fmt(b[0], '.1f') if b else '—'} | {fmt(100 * b[0] / w4, '.1f') + ' %' if (b and w4) else '—'} |")
        an.append('Files: evidence/traces/trace_ANVTX_*_trace_nvtx_gpu_proj_trace.csv; the pass window is the ANVTX timing view (evidence/traces/timing_views_ANVTX_*.json). Ranges are the p8 hooks of the attribution arm (eager); compiled arms are attributed from generated code, never from hooks.')
    else:
        an.append('| UNRESOLVED: ANVTX NVTX exports missing | | | | |')
    # (viii) cast condition; (iv) Vol.1 reconciliation; decompositions
    extra = ['## Which condition produces the weight re-casts (§2.5(viii))',
             'Determined from the harness and the traces, not from memory (docs/phase3/00_first_message.md addendum; probe docs/phase3/probes/autocast_probe.py, output autocast_probe.out; engineering run on ml-beast GPU 4). The parameters leave `load_public_model` with `requires_grad=True` (672 of 674) and the timing loop enters one `torch.autocast` context per 50-pass block with the cache enabled, so in eager the C++ weight-cast cache is hit: per-pass fp16 copy kernels fall from ~700 (cache disabled or `requires_grad=False`) to 120. The per-pass re-casts measured as X6a exist only in the compiled arms, where Inductor traces the autocast parameter casts as `aten._to_copy` graph ops (420 parameter-only `_to_copy` calls per pass in the C3 wrapper at 4 410; the sixty 1536×1536 mask-estimator weights dominate the time) and cudagraph-trees replays them every pass. NVIDIA\'s CUDA-graph guide states the same mechanism ("the cast operation is part of the captured graph"). `torch._inductor.config.freezing=True` constant-folds exactly these casts (E1a).',
             '', '## Vol. 1 eager figure versus the pod (§2.5(iv))',
             'The Vol. 1 archive ("Melband Systems ML.md", 16 Sep 2026 export) is not in this repository, so its 55 ms eager figure cannot be cited here by file; the pod measured eager A at 62.275 ms (evidence/full_A_c352800_r*.json; median of three restart minima, block-level 95 % CI [62.256, 62.333]). Known differences that can move an eager wall by 10 %: host CPU and its single-thread speed (eager A launches 1,353 kernels per pass, so per-launch host time is part of its wall — the pod\'s vCPUs vs the Vol. 1 host), driver and image (580.178.04 / torch 2.8.0+cu128 on the pod), SM clocks (pod: 2805 MHz median in session 1) and whether Vol. 1 timed the full model under the same 5a protocol. **UNRESOLVED** until the Vol. 1 host, clocks and image are read from the archive by the owner; nothing here changes the pod\'s numbers.',
             '', '## Decompositions (Phase 3 §2.2–2.4; trace-derived, MODELLED)']
    x1 = load(os.path.join(_ROOT, 'docs', 'phase3', 'x1', 'x1_decomposition_352800.json')) or {}
    if x1:
        p = x1['parts_ms']
        extra += ['**X1 at 8 s (docs/phase3/x1/x1_decomposition_352800.{json,csv}; C1 and C3 node traces, per-pass medians).** '
                  f"Span difference C1 − C3 = {x1['trace_span_delta_ms']:.3f} ms = input-staging copies {p['input_staging_copies_ms']:.3f} ms (63 kernels, 25.8 MB memcpy per pass) + lost fusion {p['lost_fusion_ms']:.3f} ms (904 fused + eager elementwise kernels vs 791 fused) + GPU idle gaps {p['gpu_idle_gaps_ms']:.3f} ms (96 `cudaGraphLaunch` calls per pass, host median 3.2 µs each) + GEMM/SDPA kernel-selection difference {p['gemm_sdpa_selection_delta_ms']:+.3f} ms; the parts sum to the span difference within 0.002 ms. The unprofiled matched pair is {x1['matched_delta_ms']:.3f} ms: residual {x1['residual_vs_matched_ms']:.3f} ms is the profiled-vs-unprofiled difference (C3 perturbation ×1.013; C1\'s own perturbation was not measured)."]
    pb = load(os.path.join(_ROOT, 'docs', 'phase3', 'padbatch', 'decomposition_table.json')) or []
    for r in pb:
        extra += [f"**Pad-and-batch, C3 vs H at {r['O']} (docs/phase3/padbatch/decomposition_table.{{json,csv}}, region_times.csv, regions_and_arithmetic.json).** "
                  f"Net wall difference C3 − H = {r['wall_delta_C3_minus_H_ms (unprofiled)']:+.3f} ms (unprofiled; trace span {r['trace_span_delta_ms']:+.3f} ms). Attribution: batching benefit (GPU idle removed, {r['C3_kernels']} → {r['H_kernels']} kernels) {r['batching_benefit_ms (gap/launch floor removed)']:+.3f} ms; padding cost (extra {r['extra_padded_GFLOP']:.1f} GFLOP at H's measured {r['H_fronttail_GEMM_rate_TFLOPS']:.1f} TFLOPS) {-r['padding_cost_ms (extra FLOPs at H rate)']:+.3f} ms; kernel-efficiency change (true work at H's {r['H_fronttail_GEMM_rate_TFLOPS']:.1f} vs C3's {r['C3_fronttail_GEMM_rate_TFLOPS']:.1f} TFLOPS) {-r['kernel_efficiency_change_ms (true work at H rate vs C3 rate; + = H slower)']:+.3f} ms; non-GEMM front/tail kernels {-r['non_GEMM_front_tail_delta_ms (H - C3)']:+.3f} ms; transformer region {-r['transformer_middle_delta_ms (H - C3)']:+.3f} ms; residual vs the traced span {r['residual_vs_trace_span_ms']:+.3f} ms, vs the unprofiled wall {r['residual_vs_wall_ms']:+.3f} ms. Band split and the mask last layer are padded to one bucket (K_max 520, N_max 1040; 74.6 % of their padded FLOPs are waste at both operating points)."]
    x2 = load(os.path.join(_ROOT, 'docs', 'phase3', 'x2', 'x2_split.json')) or {}
    for O, rec in x2.items():
        extra += [f"**X2 split at {O} (docs/phase3/x2/x2_split.{{json,csv}}; ideal-cost rows with read/write bytes, docs/phase3/ideal_costs/).** "
                  + '; '.join(f"{k}: {v['time_ms_per_pass']:.3f} ms over {v['n_rows']} kernel rows, f = write/(read+write) = {v['f_modelled']:.3f}, Y = {v['Y_ms']:.3f} ms (upper bound {v['upper_bound_ms']:.3f})" for k, v in rec.items())]
    extra += ['', '## Corrections applied in v2 relative to the Sep 20 SESSION_RESULT.md',
              '(i) precision regime stated at the headline table; (ii) per-block clocks replace session medians; (iii) class totals labelled as measured durations; (iv) Vol. 1 reconciliation stated as UNRESOLVED with the reasons; (v) both readings of the 2× line reported; (vi) the boundary numbers defined; (vii) ANVTX table generated; (viii) the cast condition determined from code and traces; §2.1 block-level intervals added. The Sep 20 file is kept unchanged as the registered Phase-1 deliverable.']
    text = v1.replace('## Conventions', '\n'.join(clk) + '\n\n' + '\n'.join(st) + '\n\n' + '\n'.join(twox) + '\n\n' + '\n'.join(bd) + '\n\n' + '\n'.join(an) + '\n\n' + '\n'.join(extra) + '\n\n## Conventions', 1)
    text = text.replace('# Ceiling-ledger Phase 1 — SESSION_RESULT (filled mechanically from evidence/)', '# Ceiling-ledger — SESSION_RESULT_v2 (Sep 20 evidence; Phase 3 §2.1 statistics and §2.5 corrections; generated by fullmodel/session_result_v2.py)')
    with open(args.out, 'w') as f:
        f.write(text)
    print(f'-> {args.out} ({len(text.splitlines())} lines; per-block clock rows {len(rows)})')

if __name__ == '__main__':
    main()
