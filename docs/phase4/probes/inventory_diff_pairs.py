"""Phase 4 (session 4): kernel-inventory differences between two traced arms at one operating point,
from the session's compile dumps and ideal-cost rows (raw data on the owner's disk; omitted from the ZIP).
Descriptive only — class totals are node-level traced device time per pass (perturbed); their deltas
need not equal the unprofiled matched difference and no causal split is asserted.
Usage: python inventory_diff_pairs.py <EVID> <OUT.json> A:B:O [A:B:O ...]      (delta = A − B)
"""
import ast, collections, json, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
for p in (os.path.join(ROOT, 'fullmodel'), os.path.join(ROOT, 'fullmodel', 'attrib')):
    sys.path.insert(0, p)
from kernel_map import parse_output_code  # noqa: E402
EV, OUT = sys.argv[1], sys.argv[2]
PAIRS = [tuple(x.split(':')) for x in sys.argv[3:]]


def inventory(arm, O):
    ic = json.load(open(os.path.join(EV, 'traces', f'ideal_costs_{arm}_{O}.json')))
    rows = {(r.get('kernel_full') or r.get('kernel')): r for r in ic['rows']}
    dump = os.path.join(EV, 'traces', f'dump_{arm}_{O}_trace')
    sel = set((ic.get('variant_selection') or {}).get('selected') or [])
    calls, kmeta = [], {}
    for g in parse_output_code(dump):
        if os.path.relpath(g['path'], dump) in sel:
            calls += g['calls']
            kmeta.update(g.get('kernels') or {})
    for c in calls:
        km = kmeta.get(c['name']) or {}
        ops = km.get('ops')
        if isinstance(ops, str):
            try:
                ops = ast.literal_eval(ops)
            except Exception:
                ops = []
        c['ops'] = list(ops or [])
        c['ops_source'] = km.get('ops_source')
    return ic, rows, calls


def class_totals(ic):
    return {k: round(v / 1e3, 4) for k, v in (ic.get('class_totals_us_per_pass') or {}).items()}


def signature(c):
    ops = tuple(sorted(set(c.get('ops') or [])))
    args = c.get('args') or []
    reads_params = all(str(x.get('buf', '')).startswith('arg') for x in args if x.get('read')) and any(x.get('read') for x in args)
    return (c['kind'], ops, 'param-only' if ('_to_copy' in c['name'] and reads_params) else '')


def sig_table(calls, rows):
    d = collections.defaultdict(lambda: {'kernels': 0, 'launches': 0, 'time_ms': 0.0})
    seen = set()
    for c in calls:
        if c['kind'] != 'triton' or c['name'] in seen:
            continue
        seen.add(c['name'])
        r = rows.get(c['name']) or {}
        s = signature(c)
        k = f"{s[0]}|{','.join(s[1])}|{s[2]}"
        d[k]['kernels'] += 1
        d[k]['launches'] += r.get('launches_per_pass') or 0
        d[k]['time_ms'] += (r.get('time_us_per_pass') or 0) / 1e3
    return d


def rnd(v):
    return {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}


out = {}
for a, b, O in PAIRS:
    try:
        icA, rowsA, callsA = inventory(a, O)
        icB, rowsB, callsB = inventory(b, O)
    except FileNotFoundError as e:
        out[f'{a}_vs_{b}_{O}'] = {'status': 'UNRESOLVED', 'reason': f'missing input: {e}'}
        continue
    sA, sB = sig_table(callsA, rowsA), sig_table(callsB, rowsB)
    cA, cB = class_totals(icA), class_totals(icB)
    rec = {
        'status': 'ok',
        'measurement_status': {a: icA.get('measurement_status'), b: icB.get('measurement_status')},
        'class_totals_ms_per_pass': {a: cA, b: cB},
        f'class_deltas_ms ({a} - {b})': {k: round(cA.get(k, 0) - cB.get(k, 0), 4) for k in set(cA) | set(cB)},
        'kernel_launches_per_pass': {a: sum((r.get('launches_per_pass') or 0) for r in rowsA.values()),
                                     b: sum((r.get('launches_per_pass') or 0) for r in rowsB.values())},
        'param_only_casts': {a: {k: rnd(v) for k, v in sA.items() if k.endswith('|param-only')},
                             b: {k: rnd(v) for k, v in sB.items() if k.endswith('|param-only')}},
        f'signatures_gone_in_{b} (non-cast)': {k: rnd(v) for k, v in sorted(sA.items(), key=lambda kv: -kv[1]['time_ms']) if k not in sB and not k.endswith('|param-only')},
        f'signatures_new_in_{b}': {k: rnd(v) for k, v in sorted(sB.items(), key=lambda kv: -kv[1]['time_ms']) if k not in sA},
        'signatures_changed_time_ms (present in both, |delta| >= 0.02 ms)': {k: round(sA[k]['time_ms'] - sB[k]['time_ms'], 4) for k in sA if k in sB and abs(sA[k]['time_ms'] - sB[k]['time_ms']) >= 0.02},
        'templates': {a: sum(1 for c in callsA if c['kind'] == 'triton' and 'triton_tem_' in c['name']),
                      b: sum(1 for c in callsB if c['kind'] == 'triton' and 'triton_tem_' in c['name'])},
    }
    rec['sum_of_class_deltas_ms'] = round(sum(rec[f'class_deltas_ms ({a} - {b})'].values()), 4)
    rec['cast_time_ms'] = {a: round(sum(x['time_ms'] for x in rec['param_only_casts'][a].values()), 4),
                           b: round(sum(x['time_ms'] for x in rec['param_only_casts'][b].values()), 4)}
    rec['note'] = 'node-level traced device time per pass (caches disabled, profiler on); descriptive, not a causal split of the unprofiled matched difference'
    out[f'{a}_vs_{b}_{O}'] = rec
json.dump(out, open(OUT, 'w'), indent=1)
for k, v in out.items():
    print('==', k, v.get('status'))
    if v.get('status') != 'ok':
        print('  ', v.get('reason')); continue
    a, b = k.split('_vs_')[0], k.split('_vs_')[1].rsplit('_', 1)[0]
    print('  class deltas:', v[f'class_deltas_ms ({a} - {b})'], 'sum', v['sum_of_class_deltas_ms'])
    print('  casts:', v['cast_time_ms'], 'launches', v['kernel_launches_per_pass'], 'templates', v['templates'])
    print('  gone (top):', list(v[f'signatures_gone_in_{b} (non-cast)'].items())[:5])
    print('  new (top):', list(v[f'signatures_new_in_{b}'].items())[:4])
    print('  changed:', list(v['signatures_changed_time_ms (present in both, |delta| >= 0.02 ms)'].items())[:6])
