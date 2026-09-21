"""2.2 — X2 split into (i) norm-scale + residual folding into the following GEMM epilogues (E4a) and (ii) RoPE folded
into the QKV GEMM epilogue (E4b), with attributable time and bytes per mechanism from the C3 wrapper order and the
v2 ideal-cost rows (read/write split). Modelled fraction: f = bytes_write / (bytes_read + bytes_write) — the
GEMM-output round trip disappears, other inputs and the final write remain."""
import json, os, sys, csv
sys.path.insert(0, 'fullmodel'); sys.path.insert(0, 'fullmodel/attrib')
from kernel_map import parse_output_code
EV = '/home/muhit/ms/evidence_pod/9tblar2zwo97ln_20260920_1855/evidence'
out = {}
for O in (352800, 4410):
    ic = json.load(open(f'docs/phase3/ideal_costs/ideal_costs_C3_{O}_v2.json'))
    rows = {(r.get('kernel_full') or r['kernel']): r for r in ic['rows']}
    d = f'{EV}/traces/dump_C3_{O}_trace'
    sel = set(ic['variant_selection']['selected'])
    g = [x for x in parse_output_code(d) if os.path.relpath(x['path'], d) in sel][0]
    calls = g['calls']
    sets = {'norm_residual': {}, 'rope_qkv': {}, 'other_epilogue': {}}
    for i, c in enumerate(calls):
        if c['kind'] != 'triton':
            continue
        ops = (g['kernels'].get(c['name']) or {}).get('ops') or []
        r = rows.get(c['name'])
        if not r or r.get('bytes_per_pass') is None:
            continue
        prev = calls[i - 1] if i else None
        prev_gemm = prev is not None and prev['kind'] == 'extern' and any(x in prev['name'] for x in ('mm', 'bmm', 'addmm'))
        gemm_out = set()
        if prev_gemm:
            gemm_out = {a.get('buf') for a in prev.get('args') or [] if a.get('write')} | ({prev.get('assigned')} if prev.get('assigned') else set())
        reads_gemm = any(a.get('read') and a.get('buf') in gemm_out for a in c.get('args') or [])
        opset = ' '.join(ops)
        if 'scaled_dot_product' in c['name'] and ('cos' in opset or 'sin' in opset or 'cos' in c['name']):
            sets['rope_qkv'][c['name']] = r
        elif prev_gemm and reads_gemm and ('linalg_vector_norm' in opset or 'rsqrt' in opset or 'layer_norm' in opset or ('add' in opset and 'mul' in opset)):
            sets['norm_residual'][c['name']] = r
        elif prev_gemm and reads_gemm:
            sets['other_epilogue'][c['name']] = r
    rec = {}
    for k, s in sets.items():
        t = sum(r['time_us_per_pass'] for r in s.values()) / 1e3
        br = sum(r.get('bytes_read_per_pass') or 0 for r in s.values()); bw = sum(r.get('bytes_write_per_pass') or 0 for r in s.values())
        f = (bw / (br + bw)) if (br + bw) else None
        rec[k] = {'kernels': sorted(s), 'n_rows': len(s), 'launches_per_pass': sum(r['launches_per_pass'] for r in s.values()),
                  'time_ms_per_pass': t, 'bytes_read_per_pass': br, 'bytes_write_per_pass': bw, 'f_modelled': f,
                  'Y_ms': (t * f) if f is not None else None, 'upper_bound_ms': t}
    out[str(O)] = rec
    print(O, {k: (round(v['time_ms_per_pass'], 4), v['n_rows'], round(v['f_modelled'] or 0, 3), round(v['Y_ms'] or 0, 4)) for k, v in rec.items()})
os.makedirs('docs/phase3/x2', exist_ok=True)
json.dump(out, open('docs/phase3/x2/x2_split.json', 'w'), indent=1)
with open('docs/phase3/x2/x2_split.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['O', 'mechanism', 'kernel rows', 'launches/pass', 'time ms/pass', 'bytes read/pass', 'bytes write/pass', 'f = write/(read+write)', 'Y ms', 'upper bound ms'])
    for O, rec in out.items():
        for k, v in rec.items():
            w.writerow([O, k, v['n_rows'], round(v['launches_per_pass'], 1), round(v['time_ms_per_pass'], 4), v['bytes_read_per_pass'], v['bytes_write_per_pass'], round(v['f_modelled'] or 0, 4), round(v['Y_ms'] or 0, 4), round(v['upper_bound_ms'], 4)])
