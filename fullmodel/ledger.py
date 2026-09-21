"""LEDGER.md generator — plan Phase 2 (no GPU), per docs/ledger_spec.md v0.1.

Every row is Y(X, O) = T_attributable(X, O) × f_recoverable(X) [ms] and
Z = Y / T_wall(best arm, O), computed ONLY from evidence files the row cites.
An input that cannot be attributed to a file leaves the row UNRESOLVED (never
a proxy). Trace-derived attributable times divided by an unprofiled wall are
labelled MODELLED; the node-level perturbation factor of the trace is carried
as a caveat when > 5 %.

Usage: python fullmodel/ledger.py --evidence-dir <dir with session 1+2 evidence> --out LEDGER.md
"""

import argparse
import glob
import json
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, 'attrib'))
from classify import classify  # noqa: E402
from kernel_map import parse_output_code, read_gpu_trace, nvtx_pass_windows, split_passes  # noqa: E402

OPS = (352800, 4410)
FRAMES = {352800: 801, 4410: 11, 22050: 51, 88200: 201}
EPILOGUE_OPS = ('gelu', 'add', 'layer_norm', 'linalg_vector_norm', 'rsqrt', 'silu', 'sigmoid')
PURE_COPY_OPS = ('aten.clone', 'aten._to_copy', 'aten.copy', 'aten.copy_', 'aten.contiguous', 'prims.convert_element_type')
FFT_OPS = ('unfold', 'view_as_real', 'view_as_complex', 'constant_pad', 'stft', 'hann', 'fft')
THRESH = {'offline_AB': 5.0, 'offline_C': 10.0, 'streaming_factor': 2.0}
# RTX Blackwell (GB202) whitepaper dense rates used for the X8/X9 per-shape bounds (registered 2026-09-21):
RATE_FP16_ACC_FP16_TFLOPS = 419.0   # fp16 inputs, fp16 accumulate (GeForce full rate)
RATE_FP16_ACC_FP32_TFLOPS = 209.5   # fp16 inputs, fp32 accumulate (the deployed path)
RATE_FP8_TFLOPS = 419.0             # FP8 via cuBLASLt requires CUBLAS_COMPUTE_32F on GeForce: shares the 419 ceiling
BW_GBPS = 1792.0
# f is the fraction of T_attributable REMOVED: a class speed-up s gives f = 1 - 1/s (s = 2 -> 0.5; s = 1.6 -> 0.375)


def f_from_speedup(s):
    return max(0.0, 1.0 - 1.0 / s) if s and s > 0 else None


def gemm_precision_bounds(sites, gemm_class_ms):
    """Per-shape roofline bounds for the GEMM class (X8/X9). Each extern site (M, N, K, fp16) gets
    t32 = max(2MNK / 209.5 TFLOPS, bytes / BW) and t16 = max(2MNK / 419 TFLOPS, bytes / BW); the class time
    is apportioned to sites by their t32 share; f_site = 1 - t16/t32. FP8: compute term at 419 (COMPUTE_32F
    ceiling) and traffic halved (weights + activations in 1 byte) -> f_fp8_site = 1 - t8/t32."""
    tot32 = sum(max(2 * g['M'] * g['N'] * g['K'] * g.get('batch', 1) / (RATE_FP16_ACC_FP32_TFLOPS * 1e12), g['bytes'] / (BW_GBPS * 1e9)) for g in sites)
    if not tot32 or not gemm_class_ms:
        return None
    y16 = y8 = 0.0
    shapes = {}
    for g in sites:
        fl = 2 * g['M'] * g['N'] * g['K'] * g.get('batch', 1)
        t32 = max(fl / (RATE_FP16_ACC_FP32_TFLOPS * 1e12), g['bytes'] / (BW_GBPS * 1e9))
        t16 = max(fl / (RATE_FP16_ACC_FP16_TFLOPS * 1e12), g['bytes'] / (BW_GBPS * 1e9))
        t8 = max(fl / (RATE_FP8_TFLOPS * 1e12), 0.5 * g['bytes'] / (BW_GBPS * 1e9))
        share = t32 / tot32
        y16 += gemm_class_ms * share * (1 - t16 / t32)
        y8 += gemm_class_ms * share * (1 - t8 / t32)
        key = (g['M'], g['N'], g['K'])
        d = shapes.setdefault(key, {'sites': 0, 'f16': 1 - t16 / t32, 'f8': 1 - t8 / t32, 'compute_bound_fp32acc': fl / (RATE_FP16_ACC_FP32_TFLOPS * 1e12) >= g['bytes'] / (BW_GBPS * 1e9)})
        d['sites'] += 1
    return {'Y16_ms': y16, 'f16_eff': y16 / gemm_class_ms, 'Y8_ms': y8, 'f8_eff': y8 / gemm_class_ms, 'shapes': shapes}


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float('inf')


class Evidence:
    def __init__(self, ev):
        self.ev = ev
        self.summary = load(os.path.join(ev, 'summary_full.json')) or {}
        self.cells = self.summary.get('cells') or {}
        self.pairs = self.summary.get('matched_pairs') or {}
        self.pert = load(os.path.join(ev, 'perturbation_and_source_map.json')) or {}
        self.attn = load(os.path.join(ev, 'attn_backend_bench.json')) or {}
        self.boundary = load(os.path.join(ev, 'boundary_bench.json')) or {}
        self.boundary_trace = load(os.path.join(ev, 'boundary_bench_trace_analysis.json')) or {}
        self.manifest = load(os.path.join(ev, 'session_manifest.json')) or {}
        self.status = {s: load(os.path.join(ev, f'session_status_{s}.json')) or load(os.path.join(ev, 'session_status.json'))
                       for s in ('session1', 'session2')}
        self._ideal, self._graphs, self._views, self._gv = {}, {}, {}, {}

    # ---- headline walls ------------------------------------------------
    def headline(self, arm, O):
        c = self.cells.get(f'{arm}_{O}') or {}
        return c.get('headline_ms') if c.get('headline_status') == 'HEADLINE' else None

    def throughput(self, arm, O):
        c = self.cells.get(f'{arm}_{O}') or {}
        return c.get('throughput_best_ms') if c.get('headline_status') == 'HEADLINE' else None

    def best_arm(self, O):
        best = None
        for key, c in self.cells.items():
            arm, chunk = key.rsplit('_', 1)
            if int(chunk) != O or arm == 'ANVTX' or c.get('headline_status') != 'HEADLINE':
                continue
            if best is None or c['headline_ms'] < best[1]:
                best = (arm, c['headline_ms'])
        return best

    def pair(self, a, b, O):
        p = self.pairs.get(f'{a}_vs_{b}_c{O}') or {}
        return p.get('median_delta_ms') if p.get('pair_headline') == 'HEADLINE' else None

    # ---- traces --------------------------------------------------------
    def ideal(self, arm, O):
        k = (arm, O)
        if k not in self._ideal:
            self._ideal[k] = load(os.path.join(self.ev, 'traces', f'ideal_costs_{arm}_{O}.json')) or {}
        return self._ideal[k]

    def views(self, arm, O):
        k = (arm, O)
        if k not in self._views:
            self._views[k] = load(os.path.join(self.ev, 'traces', f'timing_views_{arm}_{O}.json')) or {}
        return self._views[k]

    def graph_verify(self, arm, O):
        k = (arm, O)
        if k not in self._gv:
            self._gv[k] = load(os.path.join(self.ev, 'traces', f'graph_verify_{arm}_{O}.json')) or {}
        return self._gv[k]

    def pert_factor(self, arm, O):
        rec = (self.pert.get('perturbation') or {}).get(f'{arm}_{O}') or {}
        return rec.get('node_over_unprofiled') if rec.get('status') == 'ok' else None

    def wrapper(self, O):
        """The C3 wrapper calls of the selected variant, in execution order."""
        if O in self._graphs:
            return self._graphs[O]
        ic = self.ideal('C3', O)
        sel = (ic.get('variant_selection') or {})
        dump = os.path.join(self.ev, 'traces', f'dump_C3_{O}_trace')
        calls = None
        if sel.get('status') == 'ok' and os.path.isdir(dump):
            wanted = set(sel.get('selected') or [])
            for g in parse_output_code(dump):
                if os.path.relpath(g['path'], dump) in wanted:
                    calls = (calls or []) + [(c, g) for c in g['calls']]
        self._graphs[O] = calls
        return calls

    def row_time(self, O, name):
        for r in self.ideal('C3', O).get('rows') or []:
            if (r.get('kernel_full') or r.get('kernel')) == name:
                return r
        return None

    def sdpa_axis_time(self, O):
        """SDPA device time per pass split by axis from the C3 node trace: the
        12 attention executions per pass alternate (time, freq) × 6."""
        base = os.path.join(self.ev, 'traces', f'trace_C3_{O}_trace')
        try:
            rows = read_gpu_trace(base + '_cuda_gpu_trace.csv')
            windows, err = nvtx_pass_windows(base + '_nvtx_gpu_proj_trace.csv')
            if err or len(windows) != 50:
                return None
            passes, _ = split_passes(rows, windows, classify)
        except Exception:
            return None
        t_time, t_freq = [], []
        for p in passes:
            sd = [r for r in p if classify(r['name']) == 'SDPA' and not r['name'].startswith('triton')]
            if len(sd) != 12:
                return None
            t_time.append(sum(r['dur'] for r in sd[0::2]) / 1e6)
            t_freq.append(sum(r['dur'] for r in sd[1::2]) / 1e6)
        return {'time': statistics.median(t_time), 'freq': statistics.median(t_freq)}


def unresolved(X, O, why, **kw):
    return {'X': X, 'O': O, 'status': 'UNRESOLVED', 'why': why, **kw}


def row(X, O, T_ms, f, conf, effort, risk, group, evidence, lower_if, wall, modelled=True, caveats=(), note=''):
    if T_ms is None or f is None or wall is None:
        return unresolved(X, O, 'attributable time, recoverable fraction, or headline wall missing',
                          evidence=evidence, group=group, conf=conf)
    Y = T_ms * f
    return {'X': X, 'O': O, 'status': 'ok', 'T_attributable_ms': T_ms, 'f_recoverable': f, 'Y_ms': Y,
            'Z_pct': 100.0 * Y / wall, 'T_wall_ms': wall, 'conf': conf, 'effort': effort, 'quality_risk': risk,
            'group': group, 'evidence': evidence, 'would_be_lower_if': lower_if,
            'label': 'MODELLED' if modelled else 'MEASURED', 'caveats': list(caveats), 'note': note}


def build(ev):
    E = Evidence(ev)
    rows = []
    for O in OPS:
        best = E.best_arm(O)
        wall = best[1] if best else None
        best_arm = best[0] if best else None
        pc3 = E.pert_factor('C3', O)
        cav = [f'C3 trace perturbation ×{pc3:.3f} (node-level)'] if fin(pc3) and pc3 > 1.05 else []
        ic = E.ideal('C3', O)
        ideal_ok = ic.get('measurement_status') == 'ok' and (ic.get('variant_selection') or {}).get('status') == 'ok'
        irows = ic.get('rows') or [] if ideal_ok else []
        calls = E.wrapper(O) if ideal_ok else None
        smap = ((E.pert.get('source_map') or {}).get(f'C3_{O}') or {}).get('kernels') or {}

        # ---- X1 graph-break removal (measured)
        if O == 352800:
            d = E.pair('C1', 'C3', O)
            rows.append(row('X1 graph-break removal', O, d, 1.0, 'A', 'wrapper', 'none (validated)', '—',
                            f'matched_C1_C3_c{O}_m1..m3.json (median over alternating blocks)',
                            'C1 recovered part of its breaks without the vendored repair', wall, modelled=False))
        else:
            h1, h3 = E.headline('C1', O), E.headline('C3', O)
            rows.append(row('X1 graph-break removal', O, (h1 - h3) if (h1 is not None and h3 is not None) else None, 1.0,
                            'A', 'wrapper', 'none (validated)', '—', 'summary_full.json: headline C1 − headline C3',
                            'C1 recovered part of its breaks without the vendored repair', wall, modelled=False))

        # ---- kernel sets from the wrapper order
        sets = {'X2': {}, 'X4': {}, 'X6a': {}, 'X13_fused': {}, 'X13_gemm_idx': [], 'X7': {}, 'front': {}, 'tail': {}}
        if calls:
            names = [c['name'] for c, g in calls]
            kinds = [c['kind'] for c, g in calls]
            sdpa_idx = [i for i, (c, g) in enumerate(calls) if c['kind'] == 'fallback' and 'scaled_dot_product' in c['name']]
            first_sdpa, last_sdpa = (sdpa_idx[0], sdpa_idx[-1]) if sdpa_idx else (None, None)
            gemm_pos = 0
            for i, (c, g) in enumerate(calls):
                if c['kind'] == 'triton':
                    ops = (g['kernels'].get(c['name']) or {}).get('ops') or []
                    r = E.row_time(O, c['name'])
                    t = (r or {}).get('time_us_per_pass')
                    if t is None:
                        continue
                    # each kernel row counted once per set (a name's row is the per-pass total)
                    prev_extern = i > 0 and kinds[i - 1] == 'extern' and any(x in names[i - 1] for x in ('mm', 'bmm', 'addmm'))
                    prev_sdpa = i > 0 and (i - 1) in sdpa_idx
                    # epilogue = the fused kernel READS the GEMM's output buffer (its out= arg or
                    # assigned name) and applies bias/activation/residual/norm ops
                    gemm_out = set()
                    if prev_extern:
                        pc = calls[i - 1][0]
                        gemm_out = {a.get('buf') for a in pc.get('args') or [] if a.get('write')} | ({pc.get('assigned')} if pc.get('assigned') else set())
                    reads_gemm = any(a.get('read') and a.get('buf') in gemm_out for a in c.get('args') or [])
                    if prev_extern and reads_gemm and any(any(e in o for e in EPILOGUE_OPS) for o in ops):
                        sets['X2'][c['name']] = r
                    if prev_sdpa and ops and all(o in PURE_COPY_OPS for o in ops):
                        sets['X4'][c['name']] = r
                    if ('_to_copy' in c['name'] and all(a.get('buf', '').startswith('arg') for a in c['args'] if a.get('read'))
                            and any(a.get('read') for a in c['args'])):
                        sets['X6a'][c['name']] = r
                    if any(any(fo in o for fo in FFT_OPS) for o in ops):
                        sets['X7'][c['name']] = r
                    if last_sdpa is not None and i > last_sdpa:
                        sets['X13_fused'][c['name']] = r
                        sets['tail'][c['name']] = r
                    if first_sdpa is not None and i < first_sdpa:
                        sets['front'][c['name']] = r
                elif c['kind'] == 'extern' and any(x in c['name'] for x in ('mm', 'bmm', 'addmm')):
                    if last_sdpa is not None and i > last_sdpa:
                        sets['X13_gemm_idx'].append(gemm_pos)
                    gemm_pos += 1

        def tsum(d):
            return sum((r.get('time_us_per_pass') or 0.0) for r in d.values()) / 1e3

        # ---- X2 epilogue fusion: the Phase-3 split (E4a norm/residual, E4b RoPE, gating) when available
        x2s = (load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'phase3', 'x2', 'x2_split.json')) or {}).get(str(O))
        if x2s:
            for key, label, effort, lane in (('norm_residual', 'X2a norm-scale + residual folding into the following GEMM epilogues (E4a)', 'wrapper', 'agent'),
                                             ('rope_qkv', 'X2b RoPE folded into the QKV GEMM epilogue (E4b)', 'kernel', 'owner lane'),
                                             ('other_epilogue', 'X2c GLU gating epilogue (not scheduled)', 'config→kernel', '—')):
                v = x2s.get(key) or {}
                rows.append(row(label, O, v.get('time_ms_per_pass'), v.get('f_modelled'), 'B', effort,
                                'numerical: rounding points move; task-level SDR gate required per stem', 'G1',
                                f"docs/phase3/x2/x2_split.json ({v.get('n_rows')} kernel rows, {v.get('launches_per_pass', 0):.0f} launches/pass, read {v.get('bytes_read_per_pass')} B, write {v.get('bytes_write_per_pass')} B); ideal_costs v2 rows",
                                'NonGEMM Bench prior: non-GEMM operators remain 15–48 % of latency after fusion; the epilogue-fused GEMM may run slower than the bare GEMM; ' + lane, wall,
                                caveats=cav, note=f"f = write/(read+write): the GEMM-output round trip disappears; upper bound (whole time) = {v.get('upper_bound_ms', 0):.4f} ms"))
        elif sets['X2']:
            T = tsum(sets['X2'])
            rw = [(r.get('bytes_write_per_pass'), r.get('bytes_per_pass')) for r in sets['X2'].values()]
            if all(fin(w) and fin(b) and b > 0 for w, b in rw):
                f = sum(w for w, b in rw) / sum(b for w, b in rw)
                conf, note = 'B', 'f = written bytes / (read+written): the GEMM-output round trip disappears'
            else:
                f, conf, note = 1.0, 'C', 'upper bound (whole time): rows carry no read/write split'
            rows.append(row('X2 epilogue fusion into GEMMs', O, T, f, conf, 'config→kernel', 'numerical', 'G1',
                            f'traces/ideal_costs_C3_{O}.json rows + dump wrapper order: {len(sets["X2"])} kernels',
                            'the epilogue kernels are already partly overlapped or the fused GEMM slows down', wall,
                            caveats=cav, note=note))
        elif not x2s:
            rows.append(unresolved('X2 epilogue fusion into GEMMs', O, 'no C3 unique variant / wrapper order available', group='G1'))

        # ---- X3 attention backend per axis
        ax = E.sdpa_axis_time(O)
        for axis in ('time', 'freq'):
            case = (E.attn.get('cases') or {}).get(f'c{O}_{axis}_model') or {}
            ratio = case.get('ratio_best_over_model')
            if ax and fin(ratio):
                rows.append(row(f'X3 attention backend ({axis} axis → {case.get("best_backend")})', O, ax[axis],
                                max(0.0, 1.0 - ratio), 'B', 'config', 'numerical (backend rounding)', '—',
                                f'trace_C3_{O}_trace SDPA per axis + attn_backend_bench.json c{O}_{axis}_model',
                                'the model layout needs an extra copy for the faster backend', wall, caveats=cav))
            else:
                rows.append(unresolved(f'X3 attention backend ({axis} axis)', O, 'per-axis SDPA time or bench ratio missing'))

        # ---- X4 head-merge copies
        if sets['X4']:
            rows.append(row('X4 head-merge copy elimination', O, tsum(sets['X4']), 1.0, 'B', 'wrapper', 'none', '—',
                            f'dump wrapper order: {len(sets["X4"])} pure-copy kernels after SDPA; ideal_costs rows',
                            'the copies are needed by the faster attention backend layout', wall, caveats=cav))
        else:
            rows.append(unresolved('X4 head-merge copy elimination', O, 'no pure-copy kernels after SDPA identified'))

        # ---- X5 band-split + mask-estimator restructuring (H measured)
        if O == 352800:
            d = E.pair('C3', 'H', O)
        else:
            h3, hh = E.headline('C3', O), E.headline('H', O)
            d = (h3 - hh) if (h3 is not None and hh is not None) else None
        pbt = {r_['O']: r_ for r_ in (load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'phase3', 'padbatch', 'decomposition_table.json')) or [])}
        pb = pbt.get(O) or {}
        attr = (f"attribution (docs/phase3/padbatch/decomposition_table.json): batching benefit {pb.get('batching_benefit_ms (gap/launch floor removed)', float('nan')):+.3f} ms, "
                f"padding cost {-pb.get('padding_cost_ms (extra FLOPs at H rate)', float('nan')):+.3f} ms, kernel-efficiency change {-pb.get('kernel_efficiency_change_ms (true work at H rate vs C3 rate; + = H slower)', float('nan')):+.3f} ms, "
                f"non-GEMM front/tail {-pb.get('non_GEMM_front_tail_delta_ms (H - C3)', float('nan')):+.3f} ms, transformer {-pb.get('transformer_middle_delta_ms (H - C3)', float('nan')):+.3f} ms, residual vs wall {pb.get('residual_vs_wall_ms', float('nan')):+.3f} ms") if pb else 'attribution pending 2.4'
        rows.append(row('X5 band-split + mask-estimator restructuring (H): NET difference C3 − H', O, d, 1.0, 'A', 'wrapper', 'none (validated)', '—',
                        'matched_C3_H / headline C3 − H (net difference, never "padding waste")',
                        'varying-K grouped-GEMM kernels exist (CUTLASS grouped GEMM, cuBLAS grouped API, Triton grouped-GEMM tutorial, torch._grouped_mm 2D×2D); any ragged-K row rests on a measured comparison in this shape regime, not on their absence', wall, modelled=False,
                        note=attr))
        if sets['front'] or sets['tail']:
            rows.append({'X': 'X5 residual pool (front + tail fused kernels)', 'O': O, 'status': 'ok', 'T_attributable_ms': tsum(sets['front']) + tsum(sets['tail']),
                         'f_recoverable': None, 'Y_ms': None, 'Z_pct': None, 'T_wall_ms': wall, 'conf': 'C', 'effort': 'kernel',
                         'quality_risk': 'none', 'group': '—', 'evidence': 'dump wrapper order ranges; ideal_costs rows',
                         'would_be_lower_if': 'n/a', 'label': 'MODELLED', 'caveats': cav,
                         'note': 'padded-FLOP residual model not derived (UNRESOLVED f); pool reported only'})

        # ---- X6 elementwise traffic (not X2/X13/X6a)
        excl = set(sets['X2']) | set(sets['X13_fused']) | set(sets['X6a'])
        x6 = [r for r in irows if r.get('class') == 'INDUCTOR_FUSED' and (r.get('kernel_full') or r.get('kernel')) not in excl]
        if x6:
            T = sum(r.get('time_us_per_pass') or 0.0 for r in x6) / 1e3
            Y = sum((r.get('time_us_per_pass') or 0.0) * (r.get('recoverable_share_hbm_model') or 0.0)
                    for r in x6 if not r.get('bound_not_tight')) / 1e3
            l2 = sum(max(0.0, (r.get('time_us_per_pass') or 0.0) - (r.get('ideal_us_l2') or r.get('time_us_per_pass') or 0.0))
                     for r in x6 if r.get('ideal_us_l2') is not None and not r.get('bound_not_tight')) / 1e3
            rows.append(row('X6 elementwise/norm/RoPE/gating traffic', O, T, (Y / T) if T else None, 'B', 'kernel', 'numerical', 'G1',
                            f'traces/ideal_costs_C3_{O}.json: {len(x6)} INDUCTOR_FUSED rows (HBM model; bound_not_tight rows = 0)',
                            'kernels already hit L2 (the L2-rate variant gives Y = %.3f ms, reported, not summed)' % l2, wall,
                            caveats=cav))
        else:
            rows.append(unresolved('X6 elementwise/norm/RoPE/gating traffic', O, 'no priced INDUCTOR_FUSED rows', group='G1'))
        # ---- X6a weight re-casts
        if sets['X6a']:
            rows.append(row('X6a autocast weight re-casts', O, tsum(sets['X6a']), 1.0, 'B', 'wrapper', 'numerical (identical fp16 weights)', 'G1',
                            f'dump wrapper order: {len(sets["X6a"])} to_copy kernels reading only graph parameters',
                            'some casts are of activations mis-identified as parameters', wall, caveats=cav))
        else:
            rows.append(unresolved('X6a autocast weight re-casts', O, 'no parameter-only to_copy kernels identified', group='G1'))

        # ---- X7 STFT/iSTFT
        fft_rows = [r for r in irows if r.get('class') == 'FFT']
        T7 = (sum(r.get('time_us_per_pass') or 0.0 for r in fft_rows) / 1e3) + tsum(sets['X7'])
        bw = (ic.get('constants') or {}).get('BW_hbm_tbps')
        if T7 > 0 and fin(bw):
            bytes_ = 2 * 2 * 2 * 1025 * FRAMES[O] * 8
            roof_ms = bytes_ / (bw * 1e12) * 1e3
            rows.append(row('X7 STFT / iSTFT', O, T7, max(0.0, 1.0 - roof_ms / T7), 'C', 'kernel', 'numerical', '—',
                            f'ideal_costs_C3_{O}.json FFT rows + framing/windowing fused rows ({len(sets["X7"])}); roofline bytes {bytes_}',
                            'the FFTs are compute-bound at this size (assumption: memory-bound)', wall, caveats=cav))
        else:
            rows.append(unresolved('X7 STFT / iSTFT', O, 'FFT rows or bandwidth constant missing'))

        # ---- X8 / X9 GEMM precision (G2 alternatives)
        gemm_t = ((ic.get('class_totals_us_per_pass') or {}).get('GEMM') or 0.0) / 1e3 if ideal_ok else None
        gb = gemm_precision_bounds([g for g in (ic.get('gemm_sites') or []) if g.get('M')], gemm_t) if gemm_t else None
        if gb:
            sh = '; '.join(f"({m},{n},{k})×{v['sites']}: f16 {v['f16']:.3f}, f8 {v['f8']:.3f}, {'compute' if v['compute_bound_fp32acc'] else 'traffic'}-bound at fp32-acc" for (m, n, k), v in sorted(gb['shapes'].items(), key=lambda kv: -kv[1]['sites'])[:6])
            rows.append(row('X9 fp16-accumulate GEMMs (per-shape roofline bound)', O, gemm_t, gb['f16_eff'], 'C', 'kernel/config', 'TASK-LEVEL (SDR pre-check required; fp16 range at K = 1536)', 'G2',
                            f'ideal_costs_C3_{O}.json class GEMM time apportioned to {sum(v["sites"] for v in gb["shapes"].values())} extern sites by fp32-acc roofline share; rates 419 / 209.5 TFLOPS, 1792 GB/s (RTX Blackwell whitepaper); f_site = 1 − t16/t32 (s → f = 1 − 1/s); {sh}',
                            'the fp16-accumulate MMA rate is not reached at these tile shapes (Luo et al.: ratios vary by shape and generation); traffic-bound sites gain nothing; exists only because the path runs fp16 autocast (BF16 has no fp16-accumulate MMA)', wall, caveats=cav))
            rows.append(row('X8 FP8 GEMMs — memory-traffic lever, quality-gated (downgraded)', O, gemm_t, gb['f8_eff'], 'C', 'kernel/config', 'TASK-LEVEL (SDR pre-check required; per-tensor scaling)', 'G2',
                            f'same sites; FP8 shares the 419 TFLOPS ceiling on GeForce (cuBLASLt needs CUBLAS_COMPUTE_32F), traffic halved; cannot compound with X9', 
                            'scaling/quantisation kernels are added (vLLM measured FP8 17 % slower than BF16 on a 5090 with torch 2.8, issue #28234); only traffic-bound sites gain', wall, caveats=cav))
        else:
            for X in ('X9 fp16-accumulate GEMMs (per-shape roofline bound)', 'X8 FP8 GEMMs — memory-traffic lever, quality-gated (downgraded)'):
                rows.append(unresolved(X, O, 'GEMM class time or extern site shapes unavailable', group='G2'))

        # ---- X10 / X11 / X12 (G3, 4410 regime)
        gap_key = 'trivial/graph/N1000'
        gap = (E.boundary_trace.get(gap_key) or {})
        c_boundary = gap.get('gap_us_median') if gap.get('status') == 'ok' else None
        if O == 4410 and best_arm:
            gv = E.graph_verify(best_arm, O)
            pp = gv.get('per_pass') or []
            if pp and gv.get('qualified'):
                n_cur = statistics.median(p.get('captured_model_kernels', 0) + p.get('uncaptured_model_kernels', 0) + p.get('input_staging_kernels', 0) for p in pp)
            else:
                n_cur = None
            n_floor, floor_src = None, None
            best_calls = None
            icb = E.ideal(best_arm, O)
            selb = icb.get('variant_selection') or {}
            dumpb = os.path.join(E.ev, 'traces', f'dump_{best_arm}_{O}_trace')
            if selb.get('status') == 'ok' and os.path.isdir(dumpb):
                wanted = set(selb.get('selected') or [])
                for g in parse_output_code(dumpb):
                    if os.path.relpath(g['path'], dumpb) in wanted:
                        best_calls = (best_calls or []) + [(c, g) for c in g['calls']]
                floor_src = f'{best_arm} wrapper'
            floor_calls = best_calls or (calls if best_arm == 'C3' else None)
            if floor_calls:
                groups, prev = 0, None
                for c, g in floor_calls:
                    k = c['kind']
                    if k == 'triton' and prev != 'triton':
                        groups += 1
                    prev = k
                n_floor = groups + sum(1 for c, g in floor_calls if c['kind'] != 'triton')
            if fin(n_cur) and fin(n_floor) and fin(c_boundary):
                rows.append(row('X10 kernel-count reduction', O, c_boundary * max(0.0, n_cur - n_floor) / 1e3, 1.0, 'C', 'kernel', 'numerical', 'G3',
                                f'graph_verify_{best_arm}_{O}.json (N_current {n_cur:.0f}), {floor_src} (N_floor {n_floor}), boundary_bench_trace_analysis {gap_key} gap {c_boundary:.3f} µs',
                                'boundary gaps overlap with execution (gap is a serial upper bound)', wall))
            else:
                rows.append(unresolved('X10 kernel-count reduction', O, 'kernel counts or the boundary gap are unresolved', group='G3'))
            for arm in ('G', 'F'):
                v = E.views(arm, O)
                m = v.get('median_us') or {}
                if v.get('passes') == 50 and fin(m.get('sum_ns')) and fin(m.get('critical_path_observed_order_ns')):
                    T = m['sum_ns'] / 1e3
                    rows.append(row(f'X11 concurrency ({arm} trace)', O, T, max(0.0, 1.0 - m['critical_path_observed_order_ns'] / m['sum_ns']), 'C',
                                    'wrapper', 'none', 'G3', f'timing_views_{arm}_{O}.json (observed-order critical path vs summed durations)',
                                    'true dependencies are longer than the observed-order chain', wall,
                                    caveats=[f'{arm} trace perturbation ×{E.pert_factor(arm, O):.3f}'] if fin(E.pert_factor(arm, O)) and E.pert_factor(arm, O) > 1.05 else []))
                else:
                    rows.append(unresolved(f'X11 concurrency ({arm} trace)', O, 'timing views missing', group='G3'))
            vb = E.views(best_arm, O).get('median_us') or {}
            cp = vb.get('critical_path_observed_order_ns')
            if fin(cp) and fin(c_boundary) and wall:
                rows.append(row('X12 boundary-removal figure (persistent execution; NOT a bound until removable costs are justified and the microbenchmark coefficient is shown to transfer)', O,
                                max(0.0, wall - cp / 1e3 - c_boundary / 1e3), 1.0, 'C', 'architecture', 'numerical', 'G3',
                                f'wall(best={best_arm}) − observed-order critical path (timing_views_{best_arm}_{O}) − one graph-mode gap ({c_boundary:.3f} µs, boundary_bench_trace_analysis)',
                                'excluded mechanisms, sized from the literature: cross-task pipelining 1.2–1.3× per task (MPK, Qwen3-8B on B200); achieved-bandwidth lift 39 % → 62 % of speed-of-light (Cohere, weight-floored decode on H100); a megakernel\'s own ~120 µs sync + setup (Hazy, Llama-3.2-1B on B200) as the counterweight', wall))
            else:
                rows.append(unresolved('X12 boundary-removal figure', O, 'critical path of the best arm or the boundary gap missing', group='G3'))

        # ---- X13 mask-estimator chain
        gsites = ic.get('gemm_sites') or []
        if sets['X13_gemm_idx'] and gsites and gemm_t:
            tail_sites = [gsites[i] for i in sets['X13_gemm_idx'] if i < len(gsites)]
            all_flops = sum(g.get('flops', 0) for g in gsites) or None
            tail_flops = sum(g.get('flops', 0) for g in tail_sites)
            T_gemm = gemm_t * (tail_flops / all_flops) if all_flops else None
            ideal_gemm = sum(g.get('ideal_us_hbm', 0) for g in tail_sites) / 1e3
            T_f = tsum(sets['X13_fused'])
            ideal_f = sum((r.get('ideal_us_hbm') or r.get('time_us_per_pass') or 0.0) for r in sets['X13_fused'].values()) / 1e3
            if T_gemm is not None:
                T = T_gemm + T_f
                f = max(0.0, 1.0 - (ideal_gemm + ideal_f) / T) if T else None
                rows.append(row('X13 mask-estimator MLP chain fusion', O, T, f, 'B', 'kernel', 'numerical', 'G1',
                                f'ideal_costs_C3_{O}.json gemm_sites (tail: {len(tail_sites)} sites, GEMM time apportioned by FLOPs) + {len(sets["X13_fused"])} tail fused rows',
                                'the tail GEMM time share differs from its FLOP share', wall, caveats=cav,
                                note='GEMM class time apportioned by FLOP share (cuBLAS names carry no site)'))
            else:
                rows.append(unresolved('X13 mask-estimator MLP chain fusion', O, 'GEMM site FLOPs unavailable', group='G1'))
        else:
            rows.append(unresolved('X13 mask-estimator MLP chain fusion', O, 'tail GEMM sites not identified', group='G1'))

        # ---- X14 cross-chunk batching (throughput)
        fit = (E.boundary.get('fits') or {}).get('trivial/graph') or {}
        t_fixed = fit.get('intercept_us')
        thr = E.throughput(best_arm, O) if best_arm else None
        if fin(t_fixed) and fin(thr) and thr > 0:
            tf = max(0.0, t_fixed) / 1e3
            rows.append(row('X14 cross-chunk batching (THROUGHPUT, reported separately)', O, thr, tf / thr, 'C', 'wrapper', 'none', '—',
                            f'boundary_bench.json fits trivial/graph intercept {t_fixed:.3f} µs; throughput_ms_per_pass_best of {best_arm}',
                            'fixed per-pass costs are larger than the graph-launch floor', wall, modelled=False,
                            note='throughput headroom, not latency; Z here is relative to the throughput wall'))
        else:
            rows.append(unresolved('X14 cross-chunk batching (throughput)', O, 'graph-mode intercept or throughput wall missing'))
        # G1 union for the stacking note
        g1 = set(sets['X2']) | set(sets['X13_fused']) | set(sets['X6a']) | {(r.get('kernel_full') or r.get('kernel')) for r in x6}
        g1_union = sum((E.row_time(O, n) or {}).get('time_us_per_pass') or 0.0 for n in g1) / 1e3
        rows.append({'X': 'G1 union of attributable time (X2 ∪ X6 ∪ X6a ∪ X13 fused)', 'O': O, 'status': 'ok', 'T_attributable_ms': g1_union,
                     'f_recoverable': None, 'Y_ms': None, 'Z_pct': None, 'T_wall_ms': wall, 'conf': '—', 'effort': '—', 'quality_risk': '—',
                     'group': 'G1', 'evidence': 'ideal_costs rows (distinct kernels)', 'would_be_lower_if': 'n/a', 'label': 'MODELLED',
                     'caveats': cav, 'note': 'stacking bound: G1 rows may sum to at most this'})
    return E, rows


def fmt_r(v):
    return f'{v:.4f}' if fin(v) else ''


def render(E, rows, out):
    L = []
    m = E.manifest
    L.append('# LEDGER.md — MelBandRoformer ceiling ledger (plan Phase 2)')
    L.append(f"evidence: manifest {m.get('manifest_id')} · repo {str(m.get('repo_commit'))[:12]} · msst {str((m.get('provenance') or {}).get('msst_commit'))[:12]} "
             f"· ckpt {str((m.get('provenance') or {}).get('ckpt_sha256'))[:12]} · GPU {(m.get('env') or {}).get('gpu')} {(m.get('env') or {}).get('gpu_uuid')}")
    L.append('')
    L.append('`Y(X,O) = T_attributable × f_recoverable` [ms]; `Z = Y / T_wall(best arm, O)`. MODELLED = trace-derived time over an unprofiled wall. '
             'UNRESOLVED rows cite the missing input; they are never proxied.')
    L.append('')
    for O in OPS:
        best = E.best_arm(O)
        st = {s: (E.status.get(s) or {}).get('verdict') for s in ('session1', 'session2')}
        p3s = getattr(E, 'phase3_status', None)
        L.append(f"## O = {O} ({'8 s' if O == 352800 else '0.1 s streaming'}) — best arm {best[0] if best else 'UNRESOLVED'}: "
                 f"T_wall = {best[1]:.4f} ms" if best else f"## O = {O} — best arm UNRESOLVED")
        L.append(f"session verdicts: session1 {st['session1']}, session2 {st['session2']}"
                 + (f", session3 {p3s.get('verdict')} (shakedown={p3s.get('shakedown')})" if p3s else '')
                 + " (an INCOMPLETE session keeps every row that cites a missing item UNRESOLVED)")
        L.append('')
        L.append('| X | T_attr (ms) | f | Y (ms) | Z (%) | conf | effort | quality risk | group | label | evidence | would be lower if | caveats | realized (ms) | realized / ceiling |')
        L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for r in rows:
            if r['O'] != O:
                continue
            if r['status'] != 'ok':
                L.append(f"| {r['X']} | UNRESOLVED | — | — | — | {r.get('conf', '—')} | — | — | {r.get('group', '—')} | — | {r.get('evidence', '—')} | {r['why']} | — | | |")
                continue
            f = f"{r['f_recoverable']:.3f}" if fin(r.get('f_recoverable')) else '—'
            Y = f"{r['Y_ms']:.4f}" if fin(r.get('Y_ms')) else '—'
            Z = f"{r['Z_pct']:.2f}" if fin(r.get('Z_pct')) else '—'
            rz = r.get('realized_ms')
            rc = f"{rz / r['Y_ms']:.2f}" if (fin(rz) and fin(r.get('Y_ms')) and r['Y_ms']) else ''
            if fin(rz) and r.get('realized_ci95_ms'):
                rc += f" [{r['realized_ci95_ms'][0] / r['Y_ms']:.2f}, {r['realized_ci95_ms'][1] / r['Y_ms']:.2f}]" if r['Y_ms'] else ''
            L.append(f"| {r['X']} | {r['T_attributable_ms']:.4f} | {f} | {Y} | {Z} | {r['conf']} | {r['effort']} | {r['quality_risk']} | {r['group']} | {r['label']} | "
                     f"{r['evidence']} | {r['would_be_lower_if']} | {'; '.join(r['caveats']) or '—'}{(' · ' + r['note']) if r.get('note') else ''} | {fmt_r(rz)} | {rc} |")
        L.append('')
        # thresholds
        L.append('**Thresholds (plan 3.2, mechanical):**')
        for r in rows:
            if r['O'] != O or r['status'] != 'ok' or not fin(r.get('Z_pct')) or r['X'].startswith('G1') or r['X'].startswith('X14'):
                continue
            if O == 352800:
                need = THRESH['offline_AB'] if r['conf'] in ('A', 'B') else THRESH['offline_C']
                verdict = 'PURSUE' if r['Z_pct'] >= need else 'below threshold'
                L.append(f"- {r['X']}: Z = {r['Z_pct']:.2f} % vs {need} % ({r['conf']}) → {verdict}")
            else:
                need_ms = r['T_wall_ms'] * (1 - 1 / THRESH['streaming_factor'])
                verdict = 'meets 2× on wall' if r['Y_ms'] >= need_ms else 'below 2× on wall (reported only)'
                L.append(f"- {r['X']}: Y = {r['Y_ms']:.4f} ms vs {need_ms:.4f} ms needed for 2× → {verdict}")
        L.append('')
    L.append('## Coverage check and tightness (Phase 3 §2.6 iv)')
    real = [r for r in rows if r['status'] == 'ok' and fin(r.get('realized_ms'))]
    over = [r for r in real if r['realized_ms'] > r['Y_ms'] + 1e-9]
    L.append(f"- realized values filled: {len(real)}; rows whose realized value EXCEEDS its ceiling (falsified attribution): {[(r['X'], r['O']) for r in over] or 'none'}")
    for g in ('A', 'B', 'C'):
        rs = [r for r in rows if r['status'] == 'ok' and r['conf'] == g and fin(r.get('Y_ms'))]
        scored = [r for r in rs if fin(r.get('realized_ms'))]
        tight = [r['realized_ms'] / r['Y_ms'] for r in scored if r['Y_ms'] > 0]
        L.append(f"- confidence {g}: {len(rs)} rows, {len(scored)} scored; realized/ceiling median {statistics.median(tight):.2f} (min {min(tight):.2f}, max {max(tight):.2f})" if tight else f"- confidence {g}: {len(rs)} rows, none scored yet")
    for n in getattr(E, 'phase3_notes', []):
        L.append(f'- {n}')
    for r in real:
        L.append(f"- realized source {r['X']} @{r['O']}: {r.get('realized_source')} (CI95 {r.get('realized_ci95_ms')})")
    L.append('- notation: f_recoverable is the fraction of T_attributable removed; a class speed-up s gives f = 1 − 1/s; the remaining-time fraction is never written in the f column.')
    L.append('')
    L.append('## Stacking note (mandatory)')
    L.append('- G1 (X2, X6, X6a, X13) share kernels: sum at most the "G1 union of attributable time" row of each O, never the arithmetic sum.')
    L.append('- G2 (X8, X9) are alternatives: never summed; both require an SDR pre-check before any kernel work.')
    L.append('- G3 (X10, X11, X12) model the same launch-bound regime three ways: report the max, not the sum.')
    L.append('- X1 and X5 are measured deltas already inside the C3 wall; they may be summed only with rows attributed from the C3 trace.')
    L.append('- X14 is throughput headroom and is never added to latency rows.')
    L.append('')
    L.append('## Conventions')
    L.append('Headline walls are unprofiled (summary_full.json). Trace-derived numbers are MODELLED and carry the node-level perturbation factor of their trace as a caveat when > 5 %. '
             'Every number above is computed by fullmodel/ledger.py from the cited files; nothing is transcribed by hand.')
    with open(out, 'w') as f:
        f.write('\n'.join(L) + '\n')
    with open(out.replace('.md', '.json'), 'w') as f:
        json.dump(rows, f, indent=1)


def _load_json(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def _boot_ci(vals, n=10000, seed=20260921):
    import random
    rnd = random.Random(seed)
    out = []
    k = len(vals)
    for _ in range(n):
        out.append(statistics.median([vals[rnd.randrange(k)] for _ in range(k)]))
    out.sort()
    return [out[int(0.025 * n)], out[int(0.975 * n) - 1]]


def matched_realized(p3, a, b, O):
    """median over ELIGIBLE restarts of the matched block-median difference a − b
    (ms), with a block-level bootstrap interval over all eligible blocks."""
    import glob, os
    meds, blocks, files = [], [], []
    for f in sorted(glob.glob(os.path.join(p3, f'matched_{a}_{b}_c{O}_m*.json'))):
        m = _load_json(f)
        if m and m.get('pair_eligible') and m.get('matched_delta_ms'):
            meds.append(m['matched_delta_ms']['median'])
            blocks += m['matched_delta_ms']['a_minus_b_per_block']
            files.append(os.path.basename(f))
    if not meds:
        return None
    return {'realized_ms': statistics.median(meds), 'ci95_ms': _boot_ci(blocks), 'restarts': len(meds), 'files': files}


def fill_realized(E, rows, p3):
    """Phase 3 §6: realized values from the session-3 matched pairs (prereg v0.2).
    Rows filled: X6a (E1: C3 − C3F, or C3 − C3P when the registered branch ran),
    X2a (E4a: control − C3N). Aggregate comparisons that map to no single row
    (E2 → X2a+X2b+X2c, E3 → X1, E9 → X5 crossover, E7 → X12 coefficient) are
    recorded as notes. Every value cites its files."""
    import os
    E.phase3_status = (_load_json(os.path.join(p3, 'session_status.json')) or {})
    E.phase3_notes = []
    branch = _load_json(os.path.join(p3, 'e1_branch.json')) or {}
    dec = _load_json(os.path.join(p3, 'e4a_decision.json')) or {}
    ctrl = dec.get('control') or 'C3'
    for r in rows:
        if r['status'] != 'ok':
            continue
        O = r['O']
        if r['X'].startswith('X6a'):
            rec = matched_realized(p3, 'C3', 'C3F', O)
            src = 'E1 C3 − C3F'
            if branch.get('run_C3P'):
                rec2 = matched_realized(p3, 'C3', 'C3P', O)
                if rec2 and (not rec or rec2['realized_ms'] > rec['realized_ms']):
                    rec, src = rec2, 'E1(b) C3 − C3P'
            if rec:
                r['realized_ms'], r['realized_ci95_ms'], r['realized_source'] = rec['realized_ms'], rec['ci95_ms'], f"{src}: {rec['files']}"
        if r['X'].startswith('X2a'):
            rec = matched_realized(p3, ctrl, 'C3N', O)
            if rec:
                r['realized_ms'], r['realized_ci95_ms'], r['realized_source'] = rec['realized_ms'], rec['ci95_ms'], f"E4a {ctrl} − C3N ({dec.get('mode')}): {rec['files']}"
    for O in OPS:
        x2 = [r for r in rows if r['O'] == O and r['status'] == 'ok' and r['X'][:3] in ('X2a', 'X2b', 'X2c')]
        rec = matched_realized(p3, 'C3', 'C3M', O)
        if rec and x2:
            ceil = sum(r['Y_ms'] for r in x2)
            E.phase3_notes.append(f"E2 @{O}: C3 − C3M realized {rec['realized_ms']:+.4f} ms {rec['ci95_ms']} ({rec['restarts']} restarts) vs X2a+X2b+X2c ceiling {ceil:.4f} ms "
                                  f"→ {'EXCEEDS ceiling (attribution falsified or other mechanisms in play)' if rec['realized_ms'] > ceil + 1e-9 else f'{rec['realized_ms'] / ceil:.2f} of ceiling'}; files {rec['files']}")
        if O == 352800:
            rec = matched_realized(p3, 'C1', 'C1P', O)
            x1 = next((r for r in rows if r['O'] == O and r['X'].startswith('X1 ') and r['status'] == 'ok'), None)
            if rec and x1:
                E.phase3_notes.append(f"E3 @{O}: C1 − C1P realized {rec['realized_ms']:+.4f} ms {rec['ci95_ms']} vs X1 (C1 − C3) {x1['Y_ms']:.4f} ms → partition recovers {rec['realized_ms'] / x1['Y_ms']:.2f} of the break tax; files {rec['files']}")
    e9 = _load_json(os.path.join(p3, 'e9_check.json'))
    if e9:
        E.phase3_notes.append(f"E9 (X5 crossover): verdict {e9.get('verdict')}; L* predicted {e9.get('crossover_L_star_pred')}; " + '; '.join(
            f"L={L}: pred {v.get('predicted_C3_minus_H_ms'):+.3f} obs {v.get('observed_C3_minus_H_ms (median over restarts of block medians)')} err {v.get('error_ms')}" for L, v in (e9.get('held_out') or {}).items()) + ' (e9_check.json)')
    bb = _load_json(os.path.join(p3, 'boundary_bench.json'))
    if bb and bb.get('e7_pdl_saving'):
        E.phase3_notes.append('E7 (X12 coefficient): per-node PDL saving ' + ', '.join(f"{k}: {v['pdl_saving_us_per_node']:+.3f} µs" for k, v in bb['e7_pdl_saving'].items())
                              + f"; SASS mnemonics {(bb.get('pdl_sass') or {}).get('griddep_mnemonics_found')} (boundary_bench.json)")
        ta = _load_json(os.path.join(p3, 'boundary_bench_trace_analysis.json')) or {}
        ov = {k: v.get('overlap_observed') for k, v in ta.items() if '/pdl_' in k and isinstance(v, dict) and v.get('status') == 'ok'}
        if ov:
            E.phase3_notes.append(f"E7 trace: overlap observed per PDL segment {ov} (boundary_bench_trace_analysis.json)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence-dir', required=True)
    ap.add_argument('--phase3-dir', default=None, help='session-3 evidence directory (fills realized/ceiling)')
    ap.add_argument('--out', default='LEDGER.md')
    args = ap.parse_args()
    E, rows = build(args.evidence_dir)
    if args.phase3_dir:
        fill_realized(E, rows, args.phase3_dir)
    render(E, rows, args.out)
    n_ok = sum(1 for r in rows if r['status'] == 'ok')
    print(f'[ledger] {n_ok} resolved rows, {len(rows) - n_ok} UNRESOLVED -> {args.out}')
    for r in rows:
        if r['status'] != 'ok':
            print(f"   UNRESOLVED {r['X']} @ {r['O']}: {r['why']}")


if __name__ == '__main__':
    main()
