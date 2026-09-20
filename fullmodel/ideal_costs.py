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
import re
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'attrib'))
from classify import classify                      # noqa: E402
from kernel_map import (TRITON_ITEMSIZE, accessed_elems, parse_output_code, arg_meta, refined_access, nbytes,   # noqa: E402
                        read_gpu_trace, nvtx_pass_windows, split_passes)

REGISTERED_PASSES = 50
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


def per_pass_sequence(base, expected_passes):
    """ORDERED per-pass Triton kernel sequence + SDPA/GEMM class counts; every
    pass must agree and the pass count must equal the expected one."""
    rows = read_gpu_trace(base + '_cuda_gpu_trace.csv')
    windows, err = nvtx_pass_windows(base + '_nvtx_gpu_proj_trace.csv')
    if err or not windows:
        return None, f'nvtx windows unavailable: {err}'
    passes, _ = split_passes(rows, windows, classify)
    # exactly the expected number of windows, every one non-empty (gate-05 f70)
    if len(windows) != expected_passes or any(not p for p in passes):
        return None, (f'{len(windows)} windows ({sum(1 for p in passes if not p)} empty) '
                      f'!= expected {expected_passes} non-empty passes')
    seqs = [[r['name'] for r in p if classify(r['name']) == 'INDUCTOR_FUSED'] for p in passes]
    counts = [collections.Counter(classify(r['name']) for r in p) for p in passes]
    if any(sq != seqs[0] for sq in seqs) or any(c != counts[0] for c in counts):
        return None, 'per-pass kernel sequences / class counts differ across passes'
    # measurements from the VALIDATED windows only (gate-04 f51): per-name
    # launch counts and device time summed over the accepted passes
    by_name = {}
    for p in passes:
        for r in p:
            d = by_name.setdefault(r['name'], {'instances': 0, 'total_ns': 0})
            d['instances'] += 1
            d['total_ns'] += r['dur']
    return {'triton_seq': seqs[0], 'class_counts': counts[0], 'by_name': by_name,
            'passes': len(passes)}, None


def graph_signature(g):
    """What pricing sees of a graph: the ordered Triton calls with each
    argument's (role, shape, stride, element type). Two graphs with equal
    signatures are interchangeable for the byte model."""
    sig = []
    for c in g['calls']:
        k = g['kernels'].get(c['name'], {}) if c['kind'] == 'triton' else {}
        args = []
        for a in c['args']:
            m = arg_meta(g, a)
            base = m.get('base')
            alloc = g['allocs'].get(base, {})
            # allocation / alias semantics used by pricing (gate-05 f68)
            args.append((a.get('role', ''), tuple(m['shape']), tuple(m['stride']),
                         k.get('ptr_dtypes', {}).get(a.get('role', ''), m['dtype']),
                         base, tuple(alloc.get('shape') or ()), tuple(alloc.get('stride') or ()),
                         alloc.get('dtype'), bool(a.get('reinterpret'))))
        # complete pricing semantics: kernel body hash, access sites, iteration
        # literals, and the FULL normalised call text (scalar/list/keyword args
        # of extern, fallback and custom-op calls included)
        sig.append((c['kind'], c['name'], tuple(args), k.get('source_sha256'),
                    tuple(sorted((k.get('load_sites') or {}).items())),
                    tuple(sorted((k.get('store_sites') or {}).items())),
                    tuple(c.get('numeric_literals') or ()),
                    ' '.join((c.get('line') or '').split())))
    # unparsed executable lines (gate-06 f91) are part of the identity: two
    # graphs differing in an unparsed call are never interchangeable
    sig.append(('unparsed', tuple(g.get('unparsed') or ())))
    return tuple(sig)


def select_variants(graphs, observed):
    """Executed variant(s): the ORDERED per-pass Triton sequence must equal the
    concatenation of graph Triton sequences (graphs may repeat). Graphs whose
    kernel sequences AND call signatures are identical form one equivalence
    class (interchangeable for pricing). The decomposition into classes must
    be UNIQUE (counted exhaustively with memoisation); zero or several
    decompositions -> None. Graphs without Triton kernels are invisible to
    the decomposition (stated in the rule text)."""
    import sys
    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))
    seqs = [tuple(c['name'] for c in g['calls'] if c['kind'] == 'triton') for g in graphs]
    sigs = [graph_signature(g) for g in graphs]
    cls_of, reps = {}, []
    for i, sg in enumerate(sigs):
        for ci, r in enumerate(reps):
            if sigs[r] == sg:
                cls_of[i] = ci
                break
        else:
            reps.append(i)
            cls_of[i] = len(reps) - 1
    target = tuple(observed['triton_seq'])
    n = len(target)
    memo = {}

    def ways(pos):
        if pos == n:
            return 1, ()
        if pos in memo:
            return memo[pos]
        total, witness, seen = 0, None, set()
        for i, sq in enumerate(seqs):
            if not sq or cls_of[i] in seen or target[pos:pos + len(sq)] != sq:
                continue
            seen.add(cls_of[i])
            w, wit = ways(pos + len(sq))
            if w:
                total += w
                if witness is None:
                    witness = (i,) + wit
        memo[pos] = (min(total, 2), witness)     # cap: only 0 / 1 / many matters
        return memo[pos]

    count, witness = ways(0)
    if count == 0:
        return None, 'no concatenation of compiled graphs reproduces the per-pass Triton sequence'
    if count > 1:
        return None, 'the per-pass Triton sequence has several distinct decompositions into compiled graphs'
    chosen = [graphs[i] for i in witness]
    invisible = sum(1 for sq in seqs if not sq)
    # extern reconciliation: SDPA fallbacks must equal the traced SDPA class
    # count; GEMM class events must be >= extern mm/bmm/addmm calls (cuBLAS may
    # launch more than one kernel per call, never fewer)
    n_sdpa = sum(1 for g in chosen for c in g['calls'] if c['kind'] == 'fallback' and 'scaled_dot_product' in c['name'])
    n_gemm = sum(1 for g in chosen for c in g['calls'] if c['kind'] == 'extern' and any(t in c['name'] for t in ('mm', 'bmm', 'addmm')))
    cc = observed['class_counts']
    if n_sdpa != cc.get('SDPA', 0):
        return None, f'SDPA reconciliation failed: wrapper {n_sdpa} vs trace {cc.get("SDPA", 0)}'
    if cc.get('GEMM', 0) < n_gemm:
        return None, f'GEMM reconciliation failed: trace {cc.get("GEMM", 0)} events < {n_gemm} extern calls'
    return chosen, (f'{len(chosen)} graph executions per pass from {len(set(witness))} distinct graphs '
                    f'({len(reps)} signature classes among {len(graphs)} compiled); unique decomposition; '
                    f'SDPA {n_sdpa} reconciled; GEMM events {cc.get("GEMM", 0)} >= {n_gemm} extern calls; '
                    f'{invisible} compiled graphs without Triton kernels are invisible to the decomposition')


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


def storage_extent(shape, stride):
    """Elements spanned by an allocation's address range: 1 + sum((size-1)*|stride|)."""
    if not shape:
        return 0
    ext = 1
    for d, st in zip(shape, stride or [0] * len(shape)):
        if int(d) == 0:
            return 0
        ext += (int(d) - 1) * abs(int(st))
    return ext


def ideal_us(flops, by, dtype, bw_l2, l2_bytes, footprint=None):
    """Per CALL. footprint = the call's unique resident bytes (max of read /
    write sets is used as a lower bound when aliases are unknown); the L2
    ideal applies only when footprint <= l2_bytes."""
    p = (P_FP32_TFLOPS if dtype in ('torch.float32', 'torch.float64') else P_FP16_TFLOPS) * 1e12
    t_flop = flops / p * 1e6
    t_hbm = by / (BW_HBM_TBPS * 1e12) * 1e6
    rec = {'ideal_us_hbm': max(t_flop, t_hbm)}
    fp = footprint if footprint is not None else by
    if bw_l2 and l2_bytes and fp <= l2_bytes:
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
    n_pass = max(1, args.passes)
    if n_pass != REGISTERED_PASSES:
        # unregistered shakedown (gate-06 f90): an explicit ineligible record, no rows
        with open(args.out, 'w') as f:
            json.dump({'arm': args.arm, 'chunk': args.chunk, 'passes': n_pass, 'rows': [], 'gemm_sites': [],
                       'measurement_status': 'UNREGISTERED_PROTOCOL',
                       'measurement_note': f'{n_pass} passes requested; registered {REGISTERED_PASSES}',
                       'variant_selection': {'status': 'UNREGISTERED_PROTOCOL'}}, f, indent=1)
        print(f'[ideal] {args.arm}@{args.chunk}: UNREGISTERED_PROTOCOL ({n_pass} passes) -> no rows')
        return
    out = {'arm': args.arm, 'chunk': args.chunk, 'passes': n_pass,
           'constants': {'P_fp16_tflops': P_FP16_TFLOPS, 'P_fp32_tflops': P_FP32_TFLOPS,
                         'BW_hbm_tbps': BW_HBM_TBPS, 'BW_l2_tbps': args.bw_l2_tbps, 'l2_bytes': args.l2_bytes},
           'variant_selection': None, 'rows': [], 'gemm_sites': [], 'unresolved': [],
           'measurement_status': None, 'measurement_note': None}
    # every measured value comes from the VALIDATED per-pass windows (gate-04 f51);
    # the aggregate kernel summary is reconciled against them, never used as a source
    observed, err = per_pass_sequence(args.base, n_pass)
    ks_path = args.base + '_cuda_gpu_kern_sum.csv'
    ks_agg = read_kern_sum(ks_path) if os.path.exists(ks_path) else {}
    if observed is None:
        out['measurement_status'] = 'UNRESOLVED'
        out['measurement_note'] = f'pass windows not validated: {err}'
        out['variant_selection'] = {'status': 'UNRESOLVED', 'reason': out['measurement_note']}
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=1)
        print(f"[ideal] {args.arm}@{args.chunk}: UNRESOLVED ({out['measurement_note']}) -> no rows published")
        return
    ks = {name: {'instances': v['instances'], 'total_ns': v['total_ns']} for name, v in observed['by_name'].items()}
    # complete reconciliation (gate-05 f71): name sets, counts AND durations
    mismatches = []
    for name in sorted(set(ks) | set(ks_agg)):
        v, agg = ks.get(name), ks_agg.get(name)
        if v is None or agg is None:
            mismatches.append({'kernel': name[:100], 'window_instances': v and v['instances'],
                               'aggregate_instances': agg and agg.get('instances'), 'kind': 'only in one inventory'})
        elif agg.get('instances') != v['instances'] or abs(agg.get('total_ns', 0) - v['total_ns']) > 0.01 * max(1, v['total_ns']):
            mismatches.append({'kernel': name[:100], 'window_instances': v['instances'], 'aggregate_instances': agg.get('instances'),
                               'window_ns': v['total_ns'], 'aggregate_ns': agg.get('total_ns'), 'kind': 'count/duration'})
    # the numerical discrepancy is always reported (gate-06 f92), tolerance or not
    rel = []
    for name, v in ks.items():
        agg = ks_agg.get(name)
        if agg is not None and v['total_ns']:
            rel.append(abs(agg.get('total_ns', 0) - v['total_ns']) / v['total_ns'])
    out['measurement_status'] = 'ok'
    out['measurement_note'] = (f'{observed["passes"]} validated windows; aggregate summary: '
                               f'{len(mismatches)} kernels outside the 1% count/duration tolerance, '
                               f'max relative duration discrepancy {max(rel) if rel else 0.0:.4%} (windowed values used)')
    out['aggregate_reconciliation'] = {'mismatches': mismatches[:50], 'n_mismatches': len(mismatches),
                                       'max_rel_duration_discrepancy': max(rel) if rel else None,
                                       'tolerance': 0.01, 'aggregate_available': bool(ks_agg)}

    fused_bytes, gemm_sites = {}, []
    if args.dump_dir and os.path.isdir(args.dump_dir):
        graphs = parse_output_code(args.dump_dir)
        if False:
            pass
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
                            footprints = {}
                            # iteration space from the call's trailing numeric args
                            # (poi: xnumel; red/per: xnumel, rnumel): a pointer is
                            # touched at most (sites x space) times per launch
                            # iteration space bound structurally (gate-04 f52): the
                            # positional literals after the pointer args, one per
                            # numeric kernel parameter, all integer; else unresolved
                            lits = c.get('numeric_literals') or []
                            nparams = k.get('numeric_params') or []
                            space = None
                            if len(lits) == len(nparams) and all(re.fullmatch(r'\d+', x) for x in lits):
                                space = 1
                                for x in lits:
                                    space *= int(x)
                            if space is None or space == 0:
                                unresolved = True
                            for a in c['args']:
                                r, w, unres = refined_access(k, a.get('role', ''))
                                m = arg_meta(g, a)
                                # element size from the kernel ABI signature first
                                tsz = TRITON_ITEMSIZE.get(k.get('ptr_dtypes', {}).get(a.get('role', '')))
                                if unres or not m['shape'] or (tsz is None and m['dtype'] is None):
                                    unresolved = True
                                    continue
                                numel = accessed_elems(m['shape'], m['stride'])
                                if numel == 0:
                                    unresolved = True
                                    continue
                                isz = tsz if tsz else nbytes((1,), m['dtype'])
                                rl = a.get('role', '')
                                if space is None:
                                    continue
                                if r:
                                    b_read += min(numel, max(1, k.get('load_sites', {}).get(rl, 1)) * space) * isz
                                if w:
                                    b_write += min(numel, max(1, k.get('store_sites', {}).get(rl, 1)) * space) * isz
                                # resident footprint (gate-04 f53): the UNION of the distinct
                                # base allocations touched, each at its full allocated size
                                base = m.get('base')
                                if base not in footprints:
                                    # FULL allocated storage (gate-05 f69): address extent of the
                                    # base allocation from its own shape/stride/dtype; unknown -> None
                                    alloc = g['allocs'].get(base, {})
                                    ash, ast_ = alloc.get('shape'), alloc.get('stride')
                                    aisz = nbytes((1,), alloc['dtype']) if alloc.get('dtype') else None
                                    footprints[base] = (storage_extent(ash, ast_) * aisz
                                                        if (ash and ast_ is not None and aisz) else None)
                            fp = (None if (unresolved or any(v is None for v in footprints.values()))
                                  else sum(footprints.values()))
                            fused_bytes.setdefault(c['name'], []).append(
                                None if unresolved else {'bytes': b_read + b_write, 'footprint': fp})
                        elif c['kind'] == 'extern' and any(t in c['name'] for t in ('mm', 'bmm', 'addmm')):
                            gs = gemm_site(c, g)
                            if gs is None:
                                out['unresolved'].append({'call': c['name'], 'line': c['line'][:160],
                                                          'reason': 'GEMM shapes/dtypes unresolved'})
                            else:
                                gs['name'] = c['name']
                                gs.update(ideal_us(gs['flops'], gs['bytes'], gs['dtype'], args.bw_l2_tbps,
                                                   args.l2_bytes, footprint=gs['bytes']))
                                gemm_sites.append(gs)
    else:
        out['variant_selection'] = {'status': 'n/a', 'reason': 'eager arm: no generated code'}

    rows = []
    for name, v in ks.items():
        cls = classify(name)
        template_gemm = name.startswith('triton_tem_')
        if template_gemm:
            cls = 'GEMM'          # Inductor GEMM templates (max-autotune) are GEMMs, not fused pointwise
        row = {'kernel': name[:120], 'kernel_full': name, 'class': cls, 'launches_per_pass': v['instances'] / n_pass,
               'time_us_per_pass': v['total_ns'] / n_pass / 1e3, 'flops_per_pass': None,
               'bytes_per_pass': None, 'ideal_us_hbm': None, 'ideal_us_l2': None, 'resolution': 'UNRESOLVED'}
        if template_gemm:
            row['resolution'] = ('UNRESOLVED: Inductor GEMM template kernel (max-autotune); not priced '
                                 'by the byte model — GEMM class only')
        elif cls == 'INDUCTOR_FUSED' and name in fused_bytes:
            vals = fused_bytes[name]
            if all(x is not None for x in vals):
                by = sum(v['bytes'] for v in vals)
                # per-call ideals, summed (each call's own L2 residency test)
                hbm = sum(ideal_us(0.0, v['bytes'], 'torch.float16', None, None)['ideal_us_hbm'] for v in vals)
                l2 = [ideal_us(0.0, v['bytes'], 'torch.float16', args.bw_l2_tbps,
                               args.l2_bytes if v['footprint'] is not None else None,
                               footprint=v['footprint']).get('ideal_us_l2') for v in vals]
                row.update(flops_per_pass=0.0, bytes_per_pass=by, ideal_us_hbm=hbm,
                           ideal_us_l2=(sum(l2) if all(x is not None for x in l2) else None),
                           resolution='wrapper access bytes, per-call ideals summed '
                                      '(zero-FLOP pricing: registered deviation for fused pointwise/reduction kernels)')
                if hbm > row['time_us_per_pass']:
                    row['bound_not_tight'] = ('HBM-model ideal exceeds the measured time: the byte '
                                              'footprint over-counts (slice reads) and/or L2 absorbed '
                                              'traffic; recoverable share under the HBM model = 0')
                row['recoverable_share_hbm_model'] = max(0.0, 1.0 - hbm / row['time_us_per_pass']) if row['time_us_per_pass'] else None
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
        keys = []
        for r in rows:
            keys += [k for k in r.keys() if k not in keys]
        w = csv.DictWriter(f, fieldnames=keys or ['kernel'], extrasaction='ignore')
        w.writeheader()
        for r in sorted(rows, key=lambda r: -r['time_us_per_pass']):
            w.writerow(r)
    sel = out['variant_selection']
    print(f"[ideal] {args.arm}@{args.chunk}: variants={sel.get('status')} classes="
          f"{json.dumps({k: round(v, 1) for k, v in out['class_totals_us_per_pass'].items()})} "
          f"gemm_sites={len(gemm_sites)} unresolved={len(out['unresolved'])}")


if __name__ == '__main__':
    main()
