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
    seen, sel = set(), {}
    for c in calls:
        if c.get('kind') != 'triton':
            continue
        args = c.get('args') or []
        if ('_to_copy' in c['name'] and all(str(x.get('buf', '')).startswith('arg') for x in args if x.get('read'))
                and any(x.get('read') for x in args)):
            r = rows.get(c['name'])
            if r is not None:
                sel[c['name']] = r
    res = {'ideal_costs': a.ideal_costs, 'dump_dir': a.dump_dir, 'passes': ic.get('passes_observed'),
           'cast_kernels_per_pass': sum((r.get('launches_per_pass') or 0) for r in sel.values()),
           'cast_time_ms_per_pass': sum((r.get('time_us_per_pass') or 0.0) for r in sel.values()) / 1e3,
           'distinct_cast_kernels': len(sel),
           'rule': "X6a set: wrapper calls named *_to_copy* whose read buffers are all graph inputs (arg*) — the ledger's rule (ledger.py X6a)"}
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps(res, indent=1))


def template_count(a):
    ic = _load(a.ideal_costs) or {}
    rows = ic.get('rows') or []
    tem = [r for r in rows if 'triton_tem_' in str(r.get('kernel_full') or r.get('kernel'))]
    ext = [r for r in rows if r.get('class') == 'GEMM']
    sdpa = [r for r in rows if r.get('class') == 'SDPA']
    res = {'ideal_costs': a.ideal_costs, 'passes': ic.get('passes_observed'),
           'triton_template_kernels_per_pass': sum((r.get('launches_per_pass') or 0) for r in tem),
           'triton_template_time_ms_per_pass': sum((r.get('time_us_per_pass') or 0.0) for r in tem) / 1e3,
           'extern_gemm_kernels_per_pass': sum((r.get('launches_per_pass') or 0) for r in ext),
           'extern_gemm_time_ms_per_pass': sum((r.get('time_us_per_pass') or 0.0) for r in ext) / 1e3,
           'sdpa_kernels_per_pass (excluded from GEMM)': sum((r.get('launches_per_pass') or 0) for r in sdpa),
           'rope_kernels_per_pass': sum((r.get('launches_per_pass') or 0) for r in rows if 'rot' in str(r.get('kernel_full') or r.get('kernel')).lower()),
           'template_names': sorted({str(r.get('kernel_full') or r.get('kernel')) for r in tem})[:40],
           'rule': 'extern GEMM = ideal-cost class GEMM (cuBLAS entry points); templates = Inductor kernel names containing triton_tem_; SDPA rows never counted as GEMM (gate-09 f130)'}
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'template_names'}, indent=1))


def _boot_ci(diffs, stat=statistics.median):
    rnd = random.Random(BOOT_SEED)
    vals = []
    n = len(diffs)
    for _ in range(BOOT_N):
        vals.append(stat([diffs[rnd.randrange(n)] for _ in range(n)]))
    vals.sort()
    return [vals[int(0.025 * BOOT_N)], vals[int(0.975 * BOOT_N) - 1]]


def aa_stats(a):
    out = {'pairs': {}, 'bootstrap': {'resamples': BOOT_N, 'seed': BOOT_SEED, 'level': 'per-block a-b differences'}}
    for path in sorted(glob.glob(os.path.join(a.evidence_dir, 'matched_*.json'))):
        m = _load(path)
        if not m or not m.get('aa_pair'):
            continue
        blocks = m.get('blocks') or []
        pa, pb = m['pair']
        da = [b['latency_median_ms'] for b in blocks if b['arm'] == pa]
        db = [b['latency_median_ms'] for b in blocks if b['arm'] == pb]
        if not da or len(da) != len(db):
            out['pairs'][os.path.basename(path)] = {'status': 'UNRESOLVED: blocks missing'}
            continue
        diffs = [x - y for x, y in zip(da, db)]
        out['pairs'][os.path.basename(path)] = {
            'chunk': m['chunk'], 'tag': m['tag'], 'pair_eligible': m.get('pair_eligible'),
            'a_minus_b_per_block_ms': diffs, 'median_ms': statistics.median(diffs),
            'ci95_median_ms': _boot_ci(diffs), 'mean_ms': statistics.mean(diffs),
            'abs_max_ms': max(abs(d) for d in diffs), 'median_pct_of_a': statistics.median(diffs) / statistics.median(da) * 100}
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
    for L, p in pred.items():
        L = int(L)
        files = sorted(glob.glob(os.path.join(a.evidence_dir, f'matched_C3_H_c{L}_m*.json')))
        meds = []
        allblocks = []
        for f in files:
            m = _load(f)
            if m and m.get('pair_eligible') and m.get('matched_delta_ms'):
                meds.append(m['matched_delta_ms']['median'])
                allblocks += m['matched_delta_ms']['a_minus_b_per_block']
        rec = {'predicted_C3_minus_H_ms': p['D_C3_minus_H_ms'], 'terms': p['terms'],
               'files': [os.path.basename(f) for f in files], 'eligible_restarts': len(meds)}
        if meds:
            obs = statistics.median(meds)
            rec.update({'observed_C3_minus_H_ms (median over restarts of block medians)': obs,
                        'observed_ci95_block_level_ms': _boot_ci(allblocks),
                        'error_ms': obs - p['D_C3_minus_H_ms'],
                        'sign_predicted': ('H faster' if p['D_C3_minus_H_ms'] > 0 else 'C3 faster'),
                        'sign_observed': ('H faster' if obs > 0 else 'C3 faster'),
                        'sign_correct': (obs > 0) == (p['D_C3_minus_H_ms'] > 0),
                        'within_tolerance': abs(obs - p['D_C3_minus_H_ms']) <= tol,
                        'sign_test_informative': abs(p['D_C3_minus_H_ms']) > 0.1})
        else:
            rec['status'] = 'UNRESOLVED: no eligible matched C3/H pair at this length'
        out['held_out'][L] = rec
    verdicts = [r for r in out['held_out'].values() if 'sign_correct' in r]
    out['verdict'] = ('PASS' if verdicts and all(r['sign_correct'] and r['within_tolerance'] for r in verdicts)
                      and len(verdicts) == len(out['held_out']) else
                      ('FAIL' if verdicts else 'UNRESOLVED'))
    json.dump(out, open(a.out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'held_out'}, indent=1))
    for L, r in out['held_out'].items():
        print(L, {k: v for k, v in r.items() if k not in ('terms', 'files')})


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
    a = ap.parse_args()
    globals()[a.cmd](a)


if __name__ == '__main__':
    main()
