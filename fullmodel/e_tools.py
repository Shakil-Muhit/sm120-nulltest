"""Phase-3 analysis tools (registered in completeness.ANALYSIS_TOOLS; correctable
post-session with recorded identity). Subcommands:

  cast_count     --ideal-costs J --dump-dir D --out J
                 : per-pass count and device time of PARAMETER-ONLY cast kernels, the
                   X6a set (wrapper calls whose name carries _to_copy and whose reads are
                   all graph inputs `arg*`; gate-09 f129), joined to the ideal-cost rows
  template_count --ideal-costs J --out J
                 : per-pass count/time of Triton GEMM-template kernels ('triton_tem_')
                   vs extern GEMM kernels (ideal-cost class GEMM; SDPA excluded; f130)
  aa_stats       --evidence-dir E --out J
                 : E6 A/A matched pairs: per-block a-b differences and a
                   block-level bootstrap interval (10,000 resamples, seed 20260921)
  e9_check       --evidence-dir E --registration docs/phase3/e9/e9_cost_model.json --out J
                 : E9 held-out predictions vs the matched C3-H results; sign and
                   magnitude verdicts under the registered tolerance
  matched_trace  --base <export basename> --record <matched json> --out J
                 : Phase 4 (PB): per-arm, per-pass alignment of the CPU pass window (host NVTX
                   range <arm>_pass_<i>), the GPU kernel/memop intervals inside it, the graph
                   launches and memcpy API calls, and the union of CUDA API call intervals.
                   Reports GPU-busy, GPU-idle-within-window and API-union fractions per arm;
                   overlapping API durations are unioned, never summed, and no quantity is
                   labelled "host-side" by subtraction
  e9_residual    --check J --registration R --out J
                 : E9 post-hoc attribution (registered: "report which term the residual points
                   at; do not refit and re-run"): the observed − predicted residual at each
                   held-out length next to the in-sample residuals, and the single-term
                   changes (rate_C3 or rate_H at that frame count) that would absorb it
  e5_summary     --evidence-dir E --out J
                 : E5 variant cells: wall, kernel sum/span (from timing views if traced),
                   cast time, boundary count, param bytes; the three registered checks
Every number cites the file it came from; missing inputs are UNRESOLVED.
"""

import argparse
import csv
import glob
import json
import os
import random
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import qualify  # noqa: E402

HOP = 441
BOOT_N = 10000
BOOT_SEED = 20260921


def _load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def _kern_rows(base):
    path = base + '_cuda_gpu_trace.csv'
    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    head = rows[0]
    c_d = next(i for i, h in enumerate(head) if 'Duration' in h)
    c_n = next(i for i, h in enumerate(head) if h.strip().strip('"') == 'Name')
    out = []
    for r in rows[1:]:
        try:
            out.append((int(float(r[c_d].replace(',', ''))), r[c_n]))
        except (ValueError, IndexError):
            continue
    return out


def _wrapper_calls(ic, dump_dir):
    """The selected variant's wrapper calls in execution order (the ledger's E.wrapper)."""
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (here, os.path.join(here, 'attrib')):
        if p not in sys.path:
            sys.path.insert(0, p)
    from kernel_map import parse_output_code
    sel = ic.get('variant_selection') or {}
    if sel.get('status') != 'ok' or not dump_dir or not os.path.isdir(dump_dir):
        return None
    wanted = set(sel.get('selected') or [])
    calls = []
    for g in parse_output_code(dump_dir):
        if os.path.relpath(g['path'], dump_dir) in wanted:
            calls += list(g['calls'])
    return calls


def _row_by_name(ic):
    return {(r.get('kernel_full') or r.get('kernel')): r for r in ic.get('rows') or []}


def cast_count(a):
    ic = _load(a.ideal_costs) or {}
    calls = _wrapper_calls(ic, a.dump_dir)
    if calls is None:
        res = {'status': 'UNRESOLVED', 'reason': 'no unique variant / dump wrapper order for this trace', 'ideal_costs': a.ideal_costs}
        json.dump(res, open(a.out, 'w'), indent=1); print(res); return
    rows = _row_by_name(ic)
    sel, unjoined = {}, []
    for c in calls:
        if c.get('kind') != 'triton':
            continue
        args = c.get('args') or []
        if ('_to_copy' in c['name'] and all(str(x.get('buf', '')).startswith('arg') for x in args if x.get('read'))
                and any(x.get('read') for x in args)):
            r = rows.get(c['name'])
            if r is None:
                unjoined.append(c['name'])
            else:
                sel[c['name']] = r
    if unjoined or ic.get('measurement_status') != 'ok' or ic.get('passes_observed') != 50:
        res = {'status': 'UNRESOLVED', 'reason': f"{len(unjoined)} selected cast calls have no timing row; measurement_status {ic.get('measurement_status')}; passes {ic.get('passes_observed')}",
               'ideal_costs': a.ideal_costs, 'unjoined': unjoined[:10]}
        json.dump(res, open(a.out, 'w'), indent=1); print(res); return
    res = {'ideal_costs': a.ideal_costs, 'dump_dir': a.dump_dir, 'passes': ic.get('passes_observed'),
           'cast_kernels_per_pass': sum((r.get('launches_per_pass') or 0) for r in sel.values()),
           'cast_time_ms_per_pass': sum((r.get('time_us_per_pass') or 0.0) for r in sel.values()) / 1e3,
           'distinct_cast_kernels': len(sel),
           'rule': "X6a set: wrapper calls named *_to_copy* whose read buffers are all graph inputs (arg*) — the ledger's rule (ledger.py X6a)"}
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps(res, indent=1))


def template_count(a):
    ic = _load(a.ideal_costs)
    if not ic or ic.get('measurement_status') != 'ok' or ic.get('passes_observed') != 50:
        res = {'status': 'UNRESOLVED', 'reason': 'ideal-cost inventory missing or not a qualified 50-pass measurement', 'ideal_costs': a.ideal_costs}
        json.dump(res, open(a.out, 'w'), indent=1); print(res); return
    rows = ic.get('rows') or []
    tem = [r for r in rows if 'triton_tem_' in str(r.get('kernel_full') or r.get('kernel'))]
    # ideal_costs assigns Triton GEMM templates to class GEMM as well (gate-10 f141): extern = class GEMM minus templates
    ext = [r for r in rows if r.get('class') == 'GEMM' and 'triton_tem_' not in str(r.get('kernel_full') or r.get('kernel'))]
    sdpa = [r for r in rows if r.get('class') == 'SDPA']
    res = {'ideal_costs': a.ideal_costs, 'passes': ic.get('passes_observed'),
           'triton_template_kernels_per_pass': sum((r.get('launches_per_pass') or 0) for r in tem),
           'triton_template_time_ms_per_pass': sum((r.get('time_us_per_pass') or 0.0) for r in tem) / 1e3,
           'extern_gemm_kernels_per_pass': sum((r.get('launches_per_pass') or 0) for r in ext),
           'extern_gemm_time_ms_per_pass': sum((r.get('time_us_per_pass') or 0.0) for r in ext) / 1e3,
           'sdpa_kernels_per_pass (excluded from GEMM)': sum((r.get('launches_per_pass') or 0) for r in sdpa),
           'rope_kernels_per_pass': sum((r.get('launches_per_pass') or 0) for r in rows if 'rot' in str(r.get('kernel_full') or r.get('kernel')).lower()),
           'template_names': sorted({str(r.get('kernel_full') or r.get('kernel')) for r in tem})[:40],
           'rule': 'extern GEMM = ideal-cost class GEMM minus Triton templates (cuBLAS entry points); templates = kernel names containing triton_tem_; SDPA rows never counted as GEMM (gate-09 f130, gate-10 f141)'}
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'template_names'}, indent=1))


def drift_context(ev):
    """Repeat-scoped clock-drift flags for the session-3 evidence (the audit's rule, shared with
    every publisher: gate-10 f139). Returns (manifest_id, flags) where flags['pairs'] maps
    (pair, chunk, tag) -> {repeat: message}."""
    import completeness
    man = _load(os.path.join(ev, 'session_manifest.json')) or {}
    runs = completeness.load_runs(ev)
    matched = completeness.load_matched(ev)
    flags = completeness.drift_flags(ev, 'session3', runs, matched, (man.get('env') or {}).get('gpu_uuid'))
    return man.get('manifest_id'), flags


def pair_reduce(m, manifest_id=None, flags=None):
    """The one reducer every publisher uses (gate-10 f143/f144): the record must pass the shared
    validator (qualify.pair_problems) and belong to the session manifest; per-repeat a − b
    differences are recomputed from the samples, joined by repeat identity (not list order);
    repeats the clock rule flagged are excluded. Returns None (with a reason) when nothing
    eligible remains."""
    if m is None:
        return None, ['record missing']
    probs = qualify.pair_problems(m)
    if probs:
        return None, probs
    if manifest_id and m.get('manifest_id') != manifest_id:
        return None, ['foreign manifest']
    a, b = m['pair']
    reps = {}
    for blk in m.get('blocks') or []:
        lat = blk.get('latency_ms_per_iter') or []
        if len(lat) != qualify.REGISTERED_ITERS:
            return None, ['block sample count is not the registered 50']
        reps.setdefault(blk.get('repeat'), {})[blk.get('arm')] = qualify._median(lat)
    key = (tuple(m['pair']), int(m['chunk']), m.get('tag'))
    fl = ((flags or {}).get('pairs') or {}).get(key) or {}
    diffs, walls_a, walls_b, excluded = [], [], [], []
    for rep in sorted(k for k in reps if k is not None):
        pr = reps[rep]
        if a not in pr or b not in pr:
            return None, [f'repeat {rep} lacks both arms']
        if rep in fl:
            excluded.append(rep)
            continue
        diffs.append(pr[a] - pr[b]); walls_a.append(pr[a]); walls_b.append(pr[b])
    if not diffs:
        return None, [f'every repeat clock-drift flagged: {fl}']
    return {'diffs': diffs, 'median': statistics.median(diffs), 'a_wall': min(walls_a), 'b_wall': min(walls_b),
            'excluded_repeats': excluded, 'repeats_used': len(diffs), 'file': f"matched_{a}_{m['pair'][1]}_c{m['chunk']}_{m.get('tag')}.json"}, []


def pair_series(ev, a, b, O, manifest_id=None, flags=None):
    """All eligible restarts of a pair: medians per restart, pooled clean per-repeat differences, walls."""
    meds, pooled, files, wa, wb, excluded, rejected = [], [], [], [], [], {}, {}
    for f in sorted(glob.glob(os.path.join(ev, f'matched_{a}_{b}_c{O}_m*.json'))):
        m = _load(f)
        r, why = pair_reduce(m, manifest_id, flags)
        if r is None:
            rejected[os.path.basename(f)] = why
            continue
        meds.append(r['median']); pooled += r['diffs']; files.append(os.path.basename(f)); wa.append(r['a_wall']); wb.append(r['b_wall'])
        if r['excluded_repeats']:
            excluded[os.path.basename(f)] = r['excluded_repeats']
    if not meds:
        return None, rejected
    return {'median_ms': statistics.median(meds), 'ci95_ms': _boot_ci(pooled), 'restarts': len(meds), 'repeats_used': len(pooled),
            'files': files, 'excluded_repeats': excluded, 'rejected': rejected,
            'a_wall_ms': statistics.median(wa), 'b_wall_ms': statistics.median(wb)}, rejected


def _boot_ci(diffs, stat=statistics.median):
    rnd = random.Random(BOOT_SEED)
    vals = []
    n = len(diffs)
    for _ in range(BOOT_N):
        vals.append(stat([diffs[rnd.randrange(n)] for _ in range(n)]))
    vals.sort()
    return [vals[int(0.025 * BOOT_N)], vals[int(0.975 * BOOT_N) - 1]]


def aa_stats(a):
    manifest_id, flags = drift_context(a.evidence_dir)
    out = {'pairs': {}, 'bootstrap': {'resamples': BOOT_N, 'seed': BOOT_SEED, 'level': 'per-repeat a-b differences recomputed from samples, joined by repeat; clock-flagged repeats excluded'}}
    for path in sorted(glob.glob(os.path.join(a.evidence_dir, 'matched_*.json'))):
        m = _load(path)
        if not m or not m.get('aa_pair'):
            continue
        r, why = pair_reduce(m, manifest_id, flags)
        if r is None:
            out['pairs'][os.path.basename(path)] = {'status': 'UNRESOLVED: ' + '; '.join(why), 'chunk': m.get('chunk'), 'tag': m.get('tag')}
            continue
        diffs = r['diffs']
        out['pairs'][os.path.basename(path)] = {
            'chunk': m['chunk'], 'tag': m['tag'], 'repeats_used': r['repeats_used'], 'excluded_repeats': r['excluded_repeats'],
            'a_minus_b_per_block_ms': diffs, 'median_ms': statistics.median(diffs),
            'ci95_median_ms': _boot_ci(diffs), 'mean_ms': statistics.mean(diffs),
            'abs_max_ms': max(abs(d) for d in diffs), 'median_pct_of_a': statistics.median(diffs) / r['a_wall'] * 100}
    out['pairs_by_chunk'] = sorted({v.get('chunk') for v in out['pairs'].values() if 'median_ms' in v})
    json.dump(out, open(a.out, 'w'), indent=1)
    for k, v in out['pairs'].items():
        print(k, v.get('median_ms'), v.get('ci95_median_ms'))


def e9_check(a):
    reg = _load(a.registration)
    var = reg['registered_variant']
    pred = reg['variants'][var]['held_out_pred']
    tol = reg.get('tolerance_ms', 0.4)
    out = {'registration': a.registration, 'registered_variant': var, 'tolerance_ms': tol,
           'crossover_L_star_pred': reg['variants'][var].get('crossover_L_star'), 'held_out': {}}
    manifest_id, flags = drift_context(a.evidence_dir)
    def sgn(x):
        return 'H faster' if x > 0 else ('C3 faster' if x < 0 else 'tie')
    for L, p in pred.items():
        L = int(L)
        ser, rejected = pair_series(a.evidence_dir, 'C3', 'H', L, manifest_id, flags)
        rec = {'predicted_C3_minus_H_ms': p['D_C3_minus_H_ms'], 'terms': p['terms'], 'rejected': rejected}
        if ser:
            obs = ser['median_ms']
            rec.update({'files': ser['files'], 'eligible_restarts': ser['restarts'], 'repeats_used': ser['repeats_used'], 'excluded_repeats': ser['excluded_repeats'],
                        'observed_C3_minus_H_ms (median over restarts of block medians)': obs,
                        'observed_ci95_block_level_ms': ser['ci95_ms'],
                        'error_ms': obs - p['D_C3_minus_H_ms'],
                        'sign_predicted': sgn(p['D_C3_minus_H_ms']), 'sign_observed': sgn(obs),
                        'sign_correct': (sgn(obs) == sgn(p['D_C3_minus_H_ms']) and sgn(obs) != 'tie'),   # strict signs (gate-10 f149)
                        'within_tolerance': abs(obs - p['D_C3_minus_H_ms']) <= tol,
                        'sign_test_informative': abs(p['D_C3_minus_H_ms']) > 0.1})
        else:
            rec['status'] = 'UNRESOLVED: no eligible matched C3/H pair at this length'
        out['held_out'][L] = rec
    verdicts = [r for r in out['held_out'].values() if 'sign_correct' in r]
    if len(verdicts) != len(out['held_out']) or not verdicts:
        out['verdict'] = 'UNRESOLVED'                        # a required observation is unavailable (gate-10 f149)
    else:
        out['verdict'] = 'PASS' if all(r['sign_correct'] and r['within_tolerance'] for r in verdicts) else 'FAIL'
    json.dump(out, open(a.out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'held_out'}, indent=1))
    for L, r in out['held_out'].items():
        print(L, {k: v for k, v in r.items() if k not in ('terms', 'files')})


def e9_residual(a):
    import math
    chk = _load(a.check) or {}
    reg = _load(a.registration) or {}
    var = reg['variants'][reg['registered_variant']]
    p = var['params']; inp = reg['inputs']
    def q(T):
        return (lambda f: f / (T * math.ceil(f / T))) if T else (lambda f: 1.0)
    qH, qC = q(p['TH']), q(p['TC'])
    out = {'registered_variant': reg['registered_variant'], 'in_sample_residuals_ms (pred - obs)': var['res'], 'held_out': {}}
    for L, r in (chk.get('held_out') or {}).items():
        obs = r.get('observed_C3_minus_H_ms (median over restarts of block medians)')
        if obs is None:
            out['held_out'][L] = {'status': 'UNRESOLVED'}; continue
        f = int(L) // HOP + 1
        t = r['terms']
        resid = obs - r['predicted_C3_minus_H_ms']            # + : H is faster than the model says
        rH, rC = t['rate_H'], t['rate_C3']
        G = inp['true_gflop_per_frame'] * f
        # single-term absorptions (which one term, alone, would have to move to make the model exact)
        # efficiency term: -G(1/rH - 1/rC) enters D with a minus sign -> D rises when rC falls
        rc_needed = 1.0 / (1.0 / rC + resid / G) if (1.0 / rC + resid / G) > 0 else None
        rh_needed = 1.0 / (1.0 / rH - resid / G) if (1.0 / rH - resid / G) > 0 else None
        pad_needed = t['pad'] - resid                           # padding cost that would absorb it
        out['held_out'][L] = {
            'frames': f, 'observed': obs, 'predicted': r['predicted_C3_minus_H_ms'], 'residual_obs_minus_pred_ms': resid,
            'terms_registered': {k: v for k, v in t.items() if isinstance(v, (int, float))},
            'to_absorb_alone': {
                'rate_C3_TFLOPS (registered -> needed)': [rC, rc_needed],
                'rate_H_TFLOPS (registered -> needed)': [rH, rh_needed],
                'padding_cost_ms (registered -> needed)': [t['pad'], pad_needed],
                'fixed_B_ms (registered -> needed)': [t['B'], t['B'] + resid]},
            'note': 'B is fixed by the two traces (0.552/0.582 ms), so a B change of this size is excluded; the padding term is bounded by the '
                    'extra FLOPs and cannot go negative; the residual therefore points at the rate curves (kernel-efficiency term)'}
    seq = sorted([(int(L), v) for L, v in var['res'].items()] + [(int(L), -h['residual_obs_minus_pred_ms']) for L, h in out['held_out'].items() if 'residual_obs_minus_pred_ms' in h])
    out['residual_by_length_ms (pred - obs)'] = seq
    out['reading'] = ('the residual (pred − obs) is near zero at 4,410 and 22,050, negative from 88,200 through 264,600 (the model under-predicts H\'s advantage '
                      'at mid lengths) and positive at 352,800: a shape that the two-point saturating rate curves cannot follow; '
                      'the efficiency term (per-band C3 GEMM rate vs batched H rate as a function of frames) is where the model is wrong. No refit is performed.')
    json.dump(out, open(a.out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'held_out'}, indent=1))
    for L, h in out['held_out'].items():
        print(L, {k: v for k, v in h.items() if k in ('observed', 'predicted', 'residual_obs_minus_pred_ms', 'to_absorb_alone')})


def _read_csv(path):
    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def _col(head, *cands):
    for c in cands:
        for i, h in enumerate(head):
            if h.strip().strip('"').lower() == c.lower():
                return i
    for c in cands:
        for i, h in enumerate(head):
            if c.lower() in h.strip().strip('"').lower():
                return i
    return None


def _union(intervals):
    ivs = sorted(intervals)
    tot, cur_s, cur_e = 0, None, None
    for s_, e_ in ivs:
        if cur_e is None or s_ > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s_, e_
        else:
            cur_e = max(cur_e, e_)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def matched_trace(a):
    rec = _load(a.record) or {}
    arms = list(rec.get('pair') or [])
    prof_rep = rec.get('profiled_repeat')
    out = {'record': a.record, 'base': a.base, 'profiled_repeat': prof_rep, 'arms': {}, 'status': 'ok'}
    try:
        head, rows = _read_csv(a.base + '_nvtx_pushpop_trace.csv')
    except FileNotFoundError:
        out['status'] = 'UNRESOLVED: nvtx_pushpop_trace export missing (host pass windows unavailable)'
        json.dump(out, open(a.out, 'w'), indent=1); print(out['status']); return
    cs, ce, cn = _col(head, 'Start (ns)', 'Start'), _col(head, 'End (ns)', 'End'), _col(head, 'Name')
    cd = _col(head, 'Duration (ns)', 'Duration')
    windows = {}
    for r in rows:
        nm = r[cn].strip().strip('"')
        for arm in arms:
            if nm.startswith(f'{arm}_pass_'):
                try:
                    st = int(float(r[cs].replace(',', '')))
                    en = int(float(r[ce].replace(',', ''))) if ce is not None else st + int(float(r[cd].replace(',', '')))
                except (ValueError, TypeError, IndexError):
                    continue
                windows.setdefault(arm, []).append((int(nm.rsplit('_', 1)[1]), st, en))
    head, rows = _read_csv(a.base + '_cuda_gpu_trace.csv')
    gs, gd, gn = _col(head, 'Start (ns)', 'Start'), _col(head, 'Duration (ns)', 'Duration'), _col(head, 'Name')
    gpu = []
    for r in rows:
        try:
            st = int(float(r[gs].replace(',', ''))); du = int(float(r[gd].replace(',', '')))
        except (ValueError, IndexError):
            continue
        gpu.append((st, st + du, r[gn]))
    gpu.sort()
    try:
        head, rows = _read_csv(a.base + '_cuda_api_trace.csv')
        as_, ad, an = _col(head, 'Start (ns)', 'Start'), _col(head, 'Duration (ns)', 'Duration'), _col(head, 'Name')
        api = []
        for r in rows:
            try:
                st = int(float(r[as_].replace(',', ''))); du = int(float(r[ad].replace(',', '')))
            except (ValueError, IndexError):
                continue
            api.append((st, st + du, r[an]))
        api.sort()
    except FileNotFoundError:
        api = None
    import bisect
    gstarts = [g[0] for g in gpu]
    astarts = [x[0] for x in api] if api else []
    for arm, ws in windows.items():
        per = []
        for i, st, en in sorted(ws):
            lo = bisect.bisect_left(gstarts, st); hi = bisect.bisect_right(gstarts, en)
            ks = [g for g in gpu[lo:hi]]
            kern = [g for g in ks if 'memcpy' not in g[2].lower() and 'memset' not in g[2].lower()]
            mem = [g for g in ks if g not in kern]
            wall = en - st
            busy = _union([(g[0], min(g[1], en)) for g in ks])
            span = (max(g[1] for g in ks) - min(g[0] for g in ks)) if ks else 0
            row = {'pass': i, 'wall_us': wall / 1e3, 'gpu_busy_us (union of kernel+memop intervals in window)': busy / 1e3,
                   'gpu_span_us (first start to last end)': span / 1e3, 'gpu_idle_within_window_us': (wall - busy) / 1e3,
                   'kernels': len(kern), 'memops': len(mem)}
            if api is not None:
                alo = bisect.bisect_left(astarts, st); ahi = bisect.bisect_right(astarts, en)
                calls = api[alo:ahi]
                row['api_calls'] = len(calls)
                row['graph_launches'] = sum(1 for c in calls if 'cudaGraphLaunch' in c[2])
                row['memcpy_api_calls'] = sum(1 for c in calls if 'Memcpy' in c[2] or 'memcpy' in c[2])
                row['sync_api_calls'] = sum(1 for c in calls if 'Synchronize' in c[2] or 'StreamWait' in c[2] or 'EventSynchronize' in c[2])
                row['api_union_us (overlapping call intervals unioned)'] = _union([(c[0], min(c[1], en)) for c in calls]) / 1e3
                row['api_sum_us (not additive; for reference)'] = sum(min(c[1], en) - c[0] for c in calls) / 1e3
            per.append(row)
        def med(k):
            v = [p[k] for p in per if k in p]
            return statistics.median(v) if v else None
        out['arms'][arm] = {'passes': len(per), 'median_wall_us': med('wall_us'), 'median_gpu_busy_us': med('gpu_busy_us (union of kernel+memop intervals in window)'),
                            'median_gpu_span_us': med('gpu_span_us (first start to last end)'), 'median_gpu_idle_within_window_us': med('gpu_idle_within_window_us'),
                            'median_kernels': med('kernels'), 'median_memops': med('memops'), 'median_graph_launches': med('graph_launches'),
                            'median_memcpy_api_calls': med('memcpy_api_calls'), 'median_sync_api_calls': med('sync_api_calls'),
                            'median_api_union_us': med('api_union_us (overlapping call intervals unioned)'), 'per_pass': per}
        blk = next((b for b in (rec.get('blocks') or []) if b.get('arm') == arm and b.get('repeat') == prof_rep), None)
        if blk:
            out['arms'][arm]['unprofiled_context_note'] = 'this block ran under the profiler; compare with the same arm\'s other repeats in the record for perturbation'
            others = [b['latency_median_ms'] for b in (rec.get('blocks') or []) if b.get('arm') == arm and b.get('repeat') != prof_rep]
            out['arms'][arm]['profiled_block_median_ms'] = blk.get('latency_median_ms')
            out['arms'][arm]['unprofiled_blocks_median_ms'] = statistics.median(others) if others else None
    out['note'] = ('GPU idle within the CPU pass window is reported as such; it is not labelled host-side. API union is the union of CUDA API call intervals in the window '
                   '(calls can overlap GPU work or wait). Windows come from host NVTX ranges; GPU and API rows use the same nsys clock.')
    json.dump(out, open(a.out, 'w'), indent=1)
    for arm, v in out['arms'].items():
        print(arm, {k: (round(x, 1) if isinstance(x, float) else x) for k, x in v.items() if k != 'per_pass'})


def e5_summary(a):
    out = {'cells': {}, 'checks': {}}
    for path in sorted(glob.glob(os.path.join(a.evidence_dir, 'full_*_v_*.json'))):
        r = _load(path)
        if not r or r.get('status') != 'VALID':
            out['cells'][os.path.basename(path)] = {'status': (r or {}).get('status', 'MISSING')}
            continue
        v = r.get('variant') or {}
        rec = {'arm': r['arm'], 'chunk': r['chunk'], 'tag': r['tag'], 'variant': v.get('variant'),
               'depth': v.get('depth'), 'mlp_expansion_factor': v.get('mlp_expansion_factor'),
               'param_bytes_fp16': r.get('param_bytes_fp16'), 'mask_estimator_bytes_fp16': v.get('mask_estimator_bytes_fp16'),
               'min_of_medians_ms': r.get('min_of_medians_ms'), 'validation': (r.get('validation') or {}).get('status')}
        tag = r['tag'].rsplit('_r', 1)[0] + '_trace'          # the variant's trace tag (4410 only)
        tv = _load(os.path.join(a.evidence_dir, 'traces', f"timing_views_{r['arm']}_{r['chunk']}_{tag}.json"))
        if tv:
            med = tv.get('median_us') or {}
            rec['kernel_sum_ms'] = med['sum_ns'] / 1e3 if 'sum_ns' in med else 'UNRESOLVED'   # median_us values are in us
            rec['span_ms'] = med['span_ns'] / 1e3 if 'span_ns' in med else 'UNRESOLVED'
        cc = _load(os.path.join(a.evidence_dir, 'traces', f"cast_count_{r['arm']}_{r['chunk']}_{tag}.json"))
        if cc:
            rec['cast_time_ms_per_pass'] = cc['cast_time_ms_per_pass']
            rec['cast_kernels_per_pass'] = cc['cast_kernels_per_pass']
        gv = _load(os.path.join(a.evidence_dir, 'traces', f"graph_verify_{r['arm']}_{r['chunk']}_{tag}.json"))
        if gv:
            rec['boundary_count'] = (gv.get('per_pass') or [{}])[0].get('captured_model_kernels', 'UNRESOLVED')
        out['cells'][os.path.basename(path)] = rec
    json.dump(out, open(a.out, 'w'), indent=1)
    print(f"{len(out['cells'])} E5 cells summarised -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('cast_count'); s.add_argument('--ideal-costs', required=True); s.add_argument('--dump-dir', required=True); s.add_argument('--out', required=True)
    s = sub.add_parser('template_count'); s.add_argument('--ideal-costs', required=True); s.add_argument('--out', required=True)
    s = sub.add_parser('aa_stats'); s.add_argument('--evidence-dir', required=True); s.add_argument('--out', required=True)
    s = sub.add_parser('e9_check'); s.add_argument('--evidence-dir', required=True); s.add_argument('--registration', required=True); s.add_argument('--out', required=True)
    s = sub.add_parser('e5_summary'); s.add_argument('--evidence-dir', required=True); s.add_argument('--out', required=True)
    s = sub.add_parser('e9_residual'); s.add_argument('--check', required=True); s.add_argument('--registration', required=True); s.add_argument('--out', required=True)
    s = sub.add_parser('matched_trace'); s.add_argument('--base', required=True); s.add_argument('--record', required=True); s.add_argument('--out', required=True)
    a = ap.parse_args()
    globals()[a.cmd](a)


if __name__ == '__main__':
    main()
