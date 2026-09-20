"""Per-kernel ideal costs (plan 1.4), v2 after gate 02 (finding 25).

Steady-graph selection: a TORCH_COMPILE_DEBUG dump holds every compiled
variant (cold-cache and steady graphs). Only the variant(s) whose Triton
call multiset matches the per-pass trace inventory EXACTLY are accepted
(the axial study's select_c3_graph rule); with a single steady graph this is
one output_code.py, with graph breaks it is an ordered combination that
reproduces the per-pass inventory. No match -> every wrapper-derived row is
UNRESOLVED (no numbers), never a sum over superseded variants.

Rows (per kernel name, per pass): class, launches, device time; and, only
when resolved from the accepted variants:
  GEMM   : per call site (M, N, K, dtype) from extern mm/addmm/bmm args;
           FLOPs = 2MNK per call; bytes = A + B + C at their dtypes; the
           rate is the fp16 rate for fp16 operands, the fp32 rate for fp32
  SDPA   : CLASS-level (12 calls/pass: 6 time-axis + 6 freq-axis at the
           model's shapes), not per kernel row
  fused  : bytes = read + write bytes from the parser's access analysis;
           any argument with unresolved dtype/shape/access -> UNRESOLVED
Ideal = max(FLOPs / P, bytes / BW): BW_HBM = 1.79 TB/s; BW_L2 (measured by
boundary_bench's L2-resident chain, passed in) applies when the CALL's own
working set fits the device L2. Unresolved metadata never receives default
pricing.
"""

import argparse
import collections
import csv
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'attrib'))
from classify import classify                      # noqa: E402
from kernel_map import (parse_output_code, arg_meta, refined_access, nbytes,   # noqa: E402
                        read_gpu_trace, nvtx_pass_windows, split_passes)

P_FP16_TFLOPS = 209.5
P_FP32_TFLOPS = 104.8
BW_HBM_TBPS = 1.79
MODEL = {'heads': 8, 'dim_head': 64, 'bands': 60}
FRAMES = {352800: 801, 4410: 11, 22050: 51, 88200: 201}


def read_kern_sum(path):
    with open(path, newline='') as f:
        rd = csv.reader(f)
        head = next(rd)
        c_t = next(i for i, h in enumerate(head) if 'Total Time' in h)
        c_i = next(i for i, h in enumerate(head) if h.strip().strip('"') in ('Instances', 'Kern Inst') or 'Instances' in h)
        c_n = next(i for i, h in enumerate(head) if h.strip().strip('"') == 'Name')
        out = {}
        for r in rd:
            try:
                out[r[c_n]] = {'total_ns': int(float(r[c_t].replace(',', ''))),
                               'instances': int(float(r[c_i].replace(',', '')))}
            except (ValueError, IndexError):
                continue
    return out


def per_pass_inventory(base):
    rows = read_gpu_trace(base + '_cuda_gpu_trace.csv')
    windows, err = nvtx_pass_windows(base + '_nvtx_gpu_proj_trace.csv')
    if err or not windows:
        return None, f'nvtx windows unavailable: {err}'
    passes, _ = split_passes(rows, windows, classify)
    invs = [collections.Counter(r['name'] for r in p if classify(r['name']) == 'INDUCTOR_FUSED')
            for p in passes if p]
    if not invs or any(c != invs[0] for c in invs):
        return None, 'per-pass Triton inventories differ across passes'
    return invs[0], None


def select_variants(graphs, observed):
    """Single graph with an exact match, else an ordered subset whose
    multiset union matches exactly; else None."""
    inv = [collections.Counter(c['name'] for c in g['calls'] if c['kind'] == 'triton') for g in graphs]
    for g, c in zip(graphs, inv):
        if c == observed:
            return [g], 'single-graph exact match'
    # ordered combination (graph-broken arms): greedy over graphs in dump order
    total = collections.Counter()
    chosen = []
    for g, c in zip(graphs, inv):
        if not c:
            continue
        if all(total[k] + c[k] <= observed[k] for k in c):
            total += c
            chosen.append(g)
    if chosen and total == observed:
        return chosen, f'{len(chosen)}-graph combination exact match'
    return None, 'no variant set reproduces the per-pass Triton inventory'


def sdpa_class_costs(chunk):
    T = FRAMES[chunk]
    H, D, F = MODEL['heads'], MODEL['dim_head'], MODEL['bands']
    fl = by = 0.0
    for B, N in ((F, T), (T, F)):
        fl += 6 * 4.0 * B * H * N * N * D
        by += 6 * 4 * B * H * N * D * 2
    return fl, by


def gemm_site(call, gr):
    reads = [arg_meta(gr, a) for a in call['args'] if a.get('read')]
    writes = [arg_meta(gr, a) for a in call['args'] if a.get('write')]
    mats = [m for m in reads if len(m['shape']) in (2, 3)]
    if len(mats) < 2 or not writes:
        return None
    a, b = mats[-2], mats[-1]
    if any(m['dtype'] is None for m in mats + writes):
        return None
    if len(a['shape']) == 2 and len(b['shape']) == 2:
        M, K = a['shape']; K2, N = b['shape']; batch = 1
    elif len(a['shape']) == 3 and len(b['shape']) == 3:
        batch, M, K = a['shape']; _, K2, N = b['shape']
    else:
        return None
    if K != K2:
        return None
    by = sum(nbytes(m['shape'], m['dtype']) for m in mats) + sum(nbytes(m['shape'], m['dtype']) for m in writes)
    return {'M': M, 'N': N, 'K': K, 'batch': batch, 'dtype': a['dtype'],
            'flops': 2.0 * batch * M * N * K, 'bytes': by}


def ideal_us(flops, by, dtype, bw_l2, l2_bytes):
    p = (P_FP32_TFLOPS if dtype in ('torch.float32', 'torch.float64') else P_FP16_TFLOPS) * 1e12
    t_flop = flops / p * 1e6
    t_hbm = by / (BW_HBM_TBPS * 1e12) * 1e6
    rec = {'ideal_us_hbm': max(t_flop, t_hbm)}
    if bw_l2 and l2_bytes and by <= l2_bytes:
        rec['ideal_us_l2'] = max(t_flop, by / (bw_l2 * 1e12) * 1e6)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--chunk', type=int, required=True)
    ap.add_argument('--arm', default='')
    ap.add_argument('--dump-dir', default=None)
    ap.add_argument('--passes', type=int, default=50)
    ap.add_argument('--l2-bytes', type=int, default=None)
    ap.add_argument('--bw-l2-tbps', type=float, default=None)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    ks = read_kern_sum(args.base + '_cuda_gpu_kern_sum.csv')
    n_pass = max(1, args.passes)
    out = {'arm': args.arm, 'chunk': args.chunk, 'passes': n_pass,
           'constants': {'P_fp16_tflops': P_FP16_TFLOPS, 'P_fp32_tflops': P_FP32_TFLOPS,
                         'BW_hbm_tbps': BW_HBM_TBPS, 'BW_l2_tbps': args.bw_l2_tbps, 'l2_bytes': args.l2_bytes},
           'variant_selection': None, 'rows': [], 'gemm_sites': [], 'unresolved': []}

    fused_bytes, gemm_sites = {}, []
    if args.dump_dir and os.path.isdir(args.dump_dir):
        graphs = parse_output_code(args.dump_dir)
        observed, err = per_pass_inventory(args.base)
        if observed is None:
            out['variant_selection'] = {'status': 'UNRESOLVED', 'reason': err}
        else:
            chosen, how = select_variants(graphs, observed)
            if chosen is None:
                out['variant_selection'] = {'status': 'UNRESOLVED', 'reason': how,
                                            'candidates': [os.path.relpath(g['path'], args.dump_dir) for g in graphs]}
            else:
                out['variant_selection'] = {'status': 'ok', 'rule': how,
                                            'selected': [os.path.relpath(g['path'], args.dump_dir) for g in chosen]}
                for g in chosen:
                    for c in g['calls']:
                        if c['kind'] == 'triton':
                            k = g['kernels'].get(c['name'], {})
                            b_read = b_write = 0
                            unresolved = False
                            for a in c['args']:
                                r, w, unres = refined_access(k, a.get('role', ''))
                                m = arg_meta(g, a)
                                if unres or m['dtype'] is None or not m['shape']:
                                    unresolved = True
                                    continue
                                nb = nbytes(m['shape'], m['dtype'])
                                b_read += nb if r else 0
                                b_write += nb if w else 0
                            fused_bytes.setdefault(c['name'], []).append(None if unresolved else b_read + b_write)
                        elif c['kind'] == 'extern' and any(t in c['name'] for t in ('mm', 'bmm', 'addmm')):
                            gs = gemm_site(c, g)
                            if gs is None:
                                out['unresolved'].append({'call': c['name'], 'line': c['line'][:160],
                                                          'reason': 'GEMM shapes/dtypes unresolved'})
                            else:
                                gs['name'] = c['name']
                                gs.update(ideal_us(gs['flops'], gs['bytes'], gs['dtype'], args.bw_l2_tbps, args.l2_bytes))
                                gemm_sites.append(gs)
    else:
        out['variant_selection'] = {'status': 'n/a', 'reason': 'eager arm: no generated code'}

    rows = []
    for name, v in ks.items():
        cls = classify(name)
        row = {'kernel': name[:120], 'class': cls, 'launches_per_pass': v['instances'] / n_pass,
               'time_us_per_pass': v['total_ns'] / n_pass / 1e3, 'flops_per_pass': None,
               'bytes_per_pass': None, 'ideal_us_hbm': None, 'ideal_us_l2': None, 'resolution': 'UNRESOLVED'}
        if cls == 'INDUCTOR_FUSED' and name in fused_bytes:
            vals = fused_bytes[name]
            if all(x is not None for x in vals):
                by = sum(vals)
                row.update(flops_per_pass=0.0, bytes_per_pass=by, resolution='wrapper access bytes (all call sites)')
                per_call_max = max(vals)
                r = ideal_us(0.0, by, 'torch.float16', args.bw_l2_tbps,
                             args.l2_bytes if per_call_max <= (args.l2_bytes or 0) else None)
                row.update(r)
            else:
                row['resolution'] = 'UNRESOLVED: an access/dtype/shape could not be resolved'
        elif cls == 'SDPA':
            row['resolution'] = 'class-level (see sdpa_class)'
        elif cls == 'GEMM':
            row['resolution'] = 'class-level (cuBLAS names carry no shapes; see gemm_sites)'
        rows.append(row)
    out['rows'] = rows
    out['gemm_sites'] = gemm_sites
    out['class_totals_us_per_pass'] = {c: sum(r['time_us_per_pass'] for r in rows if r['class'] == c)
                                       for c in ('GEMM', 'SDPA', 'INDUCTOR_FUSED', 'COPY', 'FFT', 'OTHER')}
    fl, by = sdpa_class_costs(args.chunk)
    out['sdpa_class'] = {'calls_per_pass': 12, 'flops_per_pass': fl, 'bytes_per_pass': by,
                         **ideal_us(fl, by, 'torch.float16', None, None),
                         'measured_us_per_pass': out['class_totals_us_per_pass']['SDPA'],
                         'note': 'model shapes: 6 time-axis (60 x 8 x T x 64) + 6 freq-axis (T x 8 x 60 x 64)'}
    if gemm_sites:
        out['gemm_class'] = {'sites_per_pass': len(gemm_sites),
                             'flops_per_pass': sum(g['flops'] for g in gemm_sites),
                             'bytes_per_pass': sum(g['bytes'] for g in gemm_sites),
                             'ideal_us_hbm_sum': sum(g['ideal_us_hbm'] for g in gemm_sites),
                             'measured_us_per_pass': out['class_totals_us_per_pass']['GEMM']}
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    with open(args.out.replace('.json', '.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['kernel'])
        w.writeheader()
        for r in sorted(rows, key=lambda r: -r['time_us_per_pass']):
            w.writerow(r)
    sel = out['variant_selection']
    print(f"[ideal] {args.arm}@{args.chunk}: variants={sel.get('status')} classes="
          f"{json.dumps({k: round(v, 1) for k, v in out['class_totals_us_per_pass'].items()})} "
          f"gemm_sites={len(gemm_sites)} unresolved={len(out['unresolved'])}")


if __name__ == '__main__':
    main()
