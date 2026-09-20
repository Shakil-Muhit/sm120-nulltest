"""Post-processing (gate-02 f32): profiling-perturbation ratios (plan 1.6),
boundary-bench perturbation, and the kernel -> source-op map with coverage
(plan 1.3). Pure JSON/CSV; runs on the pod at the end of session 2 or off-pod.

Perturbation: for each (arm, chunk) with pert_unprofiled / pert_graph /
pert_node runs (each ONE 5a block), ratio = traced block median / unprofiled
block median; caveat_required = node ratio > 1.05 (registered 5 %).
Boundary: traced vs unprofiled per-node service time per (chain, mode).
Source map: from the accepted steady variants of each compiled arm's dump
(ideal_costs variant_selection), every Triton kernel name -> fused ATen ops
+ source file; coverage = share of INDUCTOR_FUSED device time whose kernel
name is mapped.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'attrib'))
from classify import classify                      # noqa: E402
from kernel_map import parse_output_code           # noqa: E402


def j(path):
    with open(path) as f:
        return json.load(f)


def block_median(run):
    t = run.get('timing') or {}
    blocks = t.get('blocks') or []
    return blocks[0]['latency_median_ms'] if blocks else None


COMPILE_ARMS = ('C1', 'C2', 'C3', 'C1A', 'C1D', 'H')


def _qualified(run, iters=50):
    """One-block perturbation protocol: VALID, validation PASS, exactly one
    block holding exactly `iters` recorded passes, manifest-bound, effective
    variant recorded, compile arms affirmatively recording-stable."""
    if not run or run.get('status') != 'VALID':
        return 'run missing or not VALID'
    if (run.get('validation') or {}).get('status') != 'PASS':
        return 'validation not PASS'
    t = run.get('timing') or {}
    blocks = t.get('blocks') or []
    if len(blocks) != 1 or t.get('iters_per_block') != iters:
        return f'not a one-block x {iters} run'
    if len(blocks[0].get('latency_ms_per_iter') or []) != iters:
        return 'the block does not hold the registered pass count'
    if run.get('manifest_id') is None:
        return 'not manifest-bound'
    if not run.get('effective_variant'):
        return 'effective variant not recorded'
    if run.get('settle') and run['settle'].get('settle_capped'):
        return 'settle capped'
    if run.get('arm') in COMPILE_ARMS and run.get('counters_stable') is not True:
        return 'recording stability not affirmatively established'
    return None


def perturbation(evidence, iters=50):
    out = {}
    for path in glob.glob(os.path.join(evidence, 'full_*_pert_unprofiled.json')):
        base = j(path)
        arm, chunk = base['arm'], base['chunk']
        rec = {'unprofiled_block_ms': block_median(base), 'status': 'ok', 'problems': []}
        q = _qualified(base, iters)
        if q:
            rec['problems'].append(f'unprofiled: {q}')
        for mode in ('graph', 'node'):
            p = os.path.join(evidence, f'full_{arm}_c{chunk}_pert_{mode}.json')
            r = j(p) if os.path.exists(p) else None
            q = _qualified(r, iters)
            rec[f'{mode}_block_ms'] = block_median(r) if r else None
            rec[f'{mode}_over_unprofiled'] = None
            if q:
                rec['problems'].append(f'{mode}: {q}')
                continue
            if r.get('manifest_id') != base.get('manifest_id') or r.get('effective_variant') != base.get('effective_variant'):
                rec['problems'].append(f'{mode}: manifest/variant differs from the unprofiled run')
                continue
            m = block_median(r)
            rec[f'{mode}_over_unprofiled'] = m / rec['unprofiled_block_ms'] if (m and rec['unprofiled_block_ms']) else None
        if rec['problems'] or rec.get('node_over_unprofiled') is None or rec.get('graph_over_unprofiled') is None:
            # every prerequisite failure clears ALL derived ratios (gate-04 f55)
            rec['status'] = 'UNRESOLVED'
            rec['caveat_required'] = 'UNRESOLVED'
            rec['graph_over_unprofiled'] = None
            rec['node_over_unprofiled'] = None
            rec['inflation_factor_node'] = None
        else:
            rec['caveat_required'] = rec['node_over_unprofiled'] > 1.05
            rec['inflation_factor_node'] = rec['node_over_unprofiled']
        out[f'{arm}_{chunk}'] = rec
    return out


def boundary_perturbation(evidence):
    a, b = (os.path.join(evidence, 'boundary_bench.json'), os.path.join(evidence, 'boundary_bench_traced.json'))
    if not (os.path.exists(a) and os.path.exists(b)):
        return {'status': 'UNRESOLVED: unprofiled or traced boundary run missing'}
    ua, tb = j(a), j(b)
    out = {}
    for key, rec in ua.get('chains', {}).items():
        t = tb.get('chains', {}).get(key, {})
        if rec.get('status') == 'ok' and t.get('status') == 'ok':
            out[key] = {'unprofiled_per_launch_us': rec['per_launch_us'],
                        'traced_per_launch_us': t['per_launch_us'],
                        'traced_over_unprofiled': t['per_launch_us'] / rec['per_launch_us']}
    return out


def source_map(evidence, iters):
    out = {}
    for path in glob.glob(os.path.join(evidence, 'traces', 'ideal_costs_*_*.json')):
        ic = j(path)
        arm, chunk = ic['arm'], ic['chunk']
        sel = ic.get('variant_selection') or {}
        dump = os.path.join(evidence, 'traces', f'dump_{arm}_{chunk}_trace')
        rec = {'variant_selection': sel.get('status'), 'kernels': {}, 'coverage': None}
        if sel.get('status') == 'ok' and os.path.isdir(dump):
            wanted = set(sel.get('selected') or [])
            for g in parse_output_code(dump):
                rel = os.path.relpath(g['path'], dump)
                if rel not in wanted:
                    continue
                for name, k in g['kernels'].items():
                    entry = {'fused_ops': k.get('ops') or [], 'ops_source': k.get('ops_source'),
                             'source_file': rel, 'source_sha256': k.get('source_sha256')}
                    prev = rec['kernels'].get(name)
                    if prev is not None and prev.get('source_sha256') != entry['source_sha256']:
                        # the same kernel name denotes different programs in two selected
                        # graphs (gate-04 f50): the mapping is ambiguous, not overwritten
                        rec['kernels'][name] = {'fused_ops': [], 'ops_source': 'AMBIGUOUS',
                                                'source_file': f"{prev['source_file']};{rel}",
                                                'source_sha256': None}
                    elif prev is None:
                        rec['kernels'][name] = entry
            fused_rows = [r for r in ic['rows'] if r['class'] == 'INDUCTOR_FUSED']
            tot = sum(r['time_us_per_pass'] for r in fused_rows)
            mapped = sum(r['time_us_per_pass'] for r in fused_rows
                         if any(r['kernel'].startswith(n) and k.get('ops_source') != 'AMBIGUOUS'
                                for n, k in rec['kernels'].items()))
            rec['coverage'] = {'fused_time_us_per_pass': tot, 'mapped_time_us_per_pass': mapped,
                               'share': (mapped / tot) if tot else None}
        out[f'{arm}_{chunk}'] = rec
        csv_path = os.path.join(evidence, 'traces', f'kernel_source_map_{arm}_{chunk}.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['kernel', 'fused_ops', 'ops_source', 'source_file', 'source_sha256'])
            for name, k in sorted(rec['kernels'].items()):
                w.writerow([name, ';'.join(k['fused_ops']), k['ops_source'], k['source_file'], k.get('source_sha256')])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence-dir', required=True)
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    res = {'perturbation': perturbation(args.evidence_dir, args.iters),
           'boundary_perturbation': boundary_perturbation(args.evidence_dir),
           'source_map': source_map(args.evidence_dir, args.iters)}
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=1)
    for k, v in res['perturbation'].items():
        print(f"[perturbation] {k}: {v['status']} graph x{v.get('graph_over_unprofiled') or 0:.3f} node x"
              f"{v.get('node_over_unprofiled') or 0:.3f} caveat={v['caveat_required']} {v['problems']}")
    for k, v in res['source_map'].items():
        cov = v.get('coverage') or {}
        print(f"[source_map] {k}: variants={v['variant_selection']} coverage={cov.get('share')}")
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
