"""Phase 4 §4: kernel-inventory attribution from the session-3 compile dumps and ideal-cost
rows (raw data on the owner's disk; omitted from the handover ZIP).
  (a) E2: for C3 vs C3M at 8 s, class totals (descriptive), and for every Triton template kernel
      the fused ops it carries (from the wrapper/kernel source parse), so that 'operations moved
      into templates' is listed from source provenance, not inferred from kernel names.
  (b) E1: C3 vs C3G inventories at 8 s and 100 ms: kernels that disappear, appear, or change time,
      by fused-op signature, with the classified parameter casts separated.
Descriptive only: no causal split of the unprofiled gains is asserted."""
import json, os, sys, collections
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
for p in (os.path.join(ROOT, 'fullmodel'), os.path.join(ROOT, 'fullmodel', 'attrib')):
    sys.path.insert(0, p)
from kernel_map import parse_output_code  # noqa: E402
EV = sys.argv[1]

def inventory(arm, O):
    ic = json.load(open(os.path.join(EV, 'traces', f'ideal_costs_{arm}_{O}.json')))
    rows = {(r.get('kernel_full') or r.get('kernel')): r for r in ic['rows']}
    dump = os.path.join(EV, 'traces', f'dump_{arm}_{O}_trace')
    sel = set((ic.get('variant_selection') or {}).get('selected') or [])
    calls = []
    for g in parse_output_code(dump):
        if os.path.relpath(g['path'], dump) in sel:
            calls += g['calls']
    return ic, rows, calls

def class_totals(ic):
    return {k: round(v / 1e3, 4) for k, v in (ic.get('class_totals_us_per_pass') or {}).items()}

def signature(c):
    ops = tuple(sorted(set(c.get('ops') or [])))
    reads_params = all(str(x.get('buf', '')).startswith('arg') for x in (c.get('args') or []) if x.get('read')) and any(x.get('read') for x in (c.get('args') or []))
    return (c['kind'], ops, 'param-only' if ('_to_copy' in c['name'] and reads_params) else '')

out = {}
# ---- (a) E2 at 8 s
icC3, rowsC3, callsC3 = inventory('C3', 352800)
icM, rowsM, callsM = inventory('C3M', 352800)
e2 = {'class_totals_ms_per_pass': {'C3': class_totals(icC3), 'C3M': class_totals(icM)}}
e2['class_deltas_ms (C3 - C3M)'] = {k: round(class_totals(icC3).get(k, 0) - class_totals(icM).get(k, 0), 4) for k in set(class_totals(icC3)) | set(class_totals(icM))}
e2['sum_of_class_deltas_ms'] = round(sum(e2['class_deltas_ms (C3 - C3M)'].values()), 4)
tem = []
for c in callsM:
    if c['kind'] == 'triton' and 'triton_tem_' in c['name']:
        r = rowsM.get(c['name']) or {}
        tem.append({'kernel': c['name'], 'fused_ops': sorted(set(c.get('ops') or [])), 'time_us_per_pass': r.get('time_us_per_pass'), 'launches_per_pass': r.get('launches_per_pass')})
by_ops = collections.defaultdict(lambda: {'kernels': 0, 'launches': 0, 'time_ms': 0.0})
for t in tem:
    k = ','.join(t['fused_ops']); by_ops[k]['kernels'] += 1; by_ops[k]['launches'] += t['launches_per_pass'] or 0; by_ops[k]['time_ms'] += (t['time_us_per_pass'] or 0) / 1e3
e2['templates_by_fused_ops'] = {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in sorted(by_ops.items(), key=lambda kv: -kv[1]['time_ms'])}
e2['template_count'] = len(tem)
# which fused-op signatures present in C3 (triton kernels) are absent in C3M, and vice versa (moved into templates or elsewhere)
def sig_table(calls, rows):
    d = collections.defaultdict(lambda: {'kernels': 0, 'launches': 0, 'time_ms': 0.0})
    for c in calls:
        if c['kind'] != 'triton':
            continue
        r = rows.get(c['name']) or {}
        s = signature(c); k = f"{s[0]}|{','.join(s[1])}|{s[2]}"
        d[k]['kernels'] += 1; d[k]['launches'] += r.get('launches_per_pass') or 0; d[k]['time_ms'] += (r.get('time_us_per_pass') or 0) / 1e3
    return d
sC3, sM = sig_table(callsC3, rowsC3), sig_table(callsM, rowsM)
e2['triton_signatures_gone_in_C3M'] = {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in sorted(sC3.items(), key=lambda kv: -kv[1]['time_ms']) if k not in sM}
e2['triton_signatures_new_in_C3M'] = {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in sorted(sM.items(), key=lambda kv: -kv[1]['time_ms']) if k not in sC3}
e2['note'] = 'class totals are node-level traced device time per pass (perturbed); their deltas are descriptive and need not sum to the unprofiled 4.80 ms; templates fuse the listed ops'
out['E2_C3_vs_C3M_352800'] = e2
# ---- (b) C3 vs C3G at both O
for O in (352800, 4410):
    icA, rowsA, callsA = inventory('C3', O); icG, rowsG, callsG = inventory('C3G', O)
    sA, sG = sig_table(callsA, rowsA), sig_table(callsG, rowsG)
    rec = {'class_totals_ms_per_pass': {'C3': class_totals(icA), 'C3G': class_totals(icG)},
           'class_deltas_ms (C3 - C3G)': {k: round(class_totals(icA).get(k, 0) - class_totals(icG).get(k, 0), 4) for k in set(class_totals(icA)) | set(class_totals(icG))},
           'kernel_launches_per_pass': {'C3': sum((r.get('launches_per_pass') or 0) for r in rowsA.values()), 'C3G': sum((r.get('launches_per_pass') or 0) for r in rowsG.values())},
           'param_only_casts': {'C3': {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in sA.items() if k.endswith('|param-only')},
                                'C3G': {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in sG.items() if k.endswith('|param-only')}},
           'signatures_gone_in_C3G (non-cast)': {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in sorted(sA.items(), key=lambda kv: -kv[1]['time_ms']) if k not in sG and not k.endswith('|param-only')},
           'signatures_new_in_C3G': {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in sorted(sG.items(), key=lambda kv: -kv[1]['time_ms']) if k not in sA},
           'signatures_changed_time_ms (present in both, |delta| >= 0.02 ms)': {k: round(sA[k]['time_ms'] - sG[k]['time_ms'], 4) for k in sA if k in sG and abs(sA[k]['time_ms'] - sG[k]['time_ms']) >= 0.02}}
    rec['sum_of_class_deltas_ms'] = round(sum(rec['class_deltas_ms (C3 - C3G)'].values()), 4)
    out[f'E1_C3_vs_C3G_{O}'] = rec
json.dump(out, open(os.path.join(ROOT, 'docs/phase4/inventory_diff_session3.json'), 'w'), indent=1)
for k, v in out.items():
    print('==', k)
    print(' class deltas:', v.get('class_deltas_ms (C3 - C3M)') or v.get('class_deltas_ms (C3 - C3G)'), 'sum', v.get('sum_of_class_deltas_ms'))
    if 'templates_by_fused_ops' in v:
        for kk, vv in list(v['templates_by_fused_ops'].items())[:6]: print('  template ops:', kk[:120], vv)
        print('  gone in C3M (top):', list(v['triton_signatures_gone_in_C3M'].items())[:5])
    else:
        print('  casts:', {a: sum(x['time_ms'] for x in d.values()) for a, d in v['param_only_casts'].items()}, 'launches', v['kernel_launches_per_pass'])
        print('  gone non-cast (top):', list(v['signatures_gone_in_C3G (non-cast)'].items())[:6])
        print('  new (top):', list(v['signatures_new_in_C3G'].items())[:4])
        print('  changed:', list(v['signatures_changed_time_ms (present in both, |delta| >= 0.02 ms)'].items())[:6])
