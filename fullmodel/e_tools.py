"""Phase-3 analysis tools (registered in completeness.ANALYSIS_TOOLS; correctable
post-session with recorded identity). Subcommands:

  cast_count     --base <trace export basename> --passes N --out J
                 : per-pass count and device time of parameter-cast kernels
                   (Inductor names containing '_to_copy'; E1 evidence)
  template_count --base <trace export basename> --passes N --out J
                 : per-pass count of Triton GEMM-template kernels ('triton_tem_')
                   vs extern GEMM kernels (E2 evidence: template selection)
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


def cast_count(a):
    rows = _kern_rows(a.base)
    casts = [(d, n) for d, n in rows if '_to_copy' in n or 'to_copy' in n]
    res = {'source': a.base + '_cuda_gpu_trace.csv', 'passes': a.passes,
           'cast_kernels_per_pass': len(casts) / a.passes,
           'cast_time_ms_per_pass': sum(d for d, _ in casts) / 1e6 / a.passes,
           'all_kernels_per_pass': len(rows) / a.passes,
           'note': 'kernel-name match on _to_copy (Inductor cast kernels); parameter-only vs activation casts are '
                   'not separated here (the dump-wrapper order in ideal_costs does that)'}
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps(res, indent=1))


def template_count(a):
    rows = _kern_rows(a.base)
    tem = [(d, n) for d, n in rows if 'triton_tem_' in n]
    ext = [(d, n) for d, n in rows if any(k in n.lower() for k in ('nvjet', 'gemm', 'cutlass', 'sm90', 'sm120', 'ampere_', 'xmma'))
           and 'triton_' not in n]
    res = {'source': a.base + '_cuda_gpu_trace.csv', 'passes': a.passes,
           'triton_template_kernels_per_pass': len(tem) / a.passes,
           'triton_template_time_ms_per_pass': sum(d for d, _ in tem) / 1e6 / a.passes,
           'extern_gemm_kernels_per_pass': len(ext) / a.passes,
           'extern_gemm_time_ms_per_pass': sum(d for d, _ in ext) / 1e6 / a.passes,
           'template_names': sorted({n for _, n in tem})[:40]}
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
    for name in ('cast_count', 'template_count'):
        s = sub.add_parser(name); s.add_argument('--base', required=True); s.add_argument('--passes', type=int, required=True); s.add_argument('--out', required=True)
    s = sub.add_parser('aa_stats'); s.add_argument('--evidence-dir', required=True); s.add_argument('--out', required=True)
    s = sub.add_parser('e9_check'); s.add_argument('--evidence-dir', required=True); s.add_argument('--registration', required=True); s.add_argument('--out', required=True)
    s = sub.add_parser('e5_summary'); s.add_argument('--evidence-dir', required=True); s.add_argument('--out', required=True)
    a = ap.parse_args()
    globals()[a.cmd](a)


if __name__ == '__main__':
    main()
