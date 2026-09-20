"""Boundary-cost microbenchmark (plan 0.4; prereg_amendments.md §4).

Protocol (registered):
  chains of N DEPENDENT kernels, N ∈ {10, 100, 1000, 2000}:
    trivial : one Triton program loads ONE int32 element, adds 1, stores it
              back to the SAME buffer -> kernel k+1 depends on kernel k; the
              final counter value must equal the total number of launches
              (dependency proof, exact integer arithmetic)
    elem37  : in-place fp16 elementwise op on a 37 MB tensor (x.mul_(1.0)),
              value-preserving, N ∈ {10, 100}
    gemm    : alternating dependent GEMMs 660x384x1536 / 660x1536x384 fp16 into
              PREALLOCATED ping-pong outputs (torch.matmul(out=)), weights scaled
              by 1/sqrt(K) so magnitudes stay bounded, N ∈ {10, 100}
  modes:
    stream     : back-to-back launches on the current stream
    graph      : the chain captured into one CUDA graph, replayed
    pdl_stream : trivial kernels launched with programmatic dependent launch
                 (Triton 3.4 `launch_pdl=True`; gdc_wait() BEFORE the dependent
                 load, gdc_launch_dependents() AFTER the store); PTX checked
                 for griddepcontrol instructions
    pdl_graph  : the PDL chain captured into a CUDA graph
    (real-kernel PDL: not available through cuBLAS/ATen launches -> UNSUPPORTED)
  timing: unprofiled DEVICE time via CUDA events around the chain, median of
  REPEATS, min/max; T(N) per (chain, mode); least-squares slope and intercept
  over N (slope = effective per-node service time INCLUDING kernel execution;
  intercept = fixed chain overhead). This script does NOT claim a pure
  boundary cost: `--analyze-trace` reads the node-level trace of a separate
  run of this script and reports, per chain segment (NVTX range), the median
  kernel duration and the median serial inter-kernel gap; the gap is the
  boundary estimate for non-PDL modes; for PDL modes kernels overlap and
  no subtraction is attempted (UNRESOLVED as a pure boundary). Perturbation
  of the traced run vs the unprofiled T(N) is reported.
"""

import argparse
import csv
import json
import os
import re
import statistics
import traceback

import torch
import triton
import triton.language as tl

NS = (10, 100, 1000, 2000)
REPEATS = 20
WARMUP = 5

# Triton JIT resolves kernel globals at MODULE scope: the gdc intrinsics and
# both kernels must be module-level. The triton 3.4.0 wheel ships
# language/extra/cuda/gdc.py with the pre-3.4 `_builder=` signature while the
# 3.4 code generator passes `_semantic=` ("gdc_wait() got an unexpected keyword
# argument '_semantic'", ml-beast 2026-09-20), so the SAME two one-line PTX
# intrinsics are defined here with the 3.4 signature (griddepcontrol.wait /
# griddepcontrol.launch_dependents — identical asm strings to the wheel's).
from triton.language import core as _core

try:
    @_core.extern
    def gdc_wait(_semantic=None):
        _core.inline_asm_elementwise("griddepcontrol.wait; // dummy $0", "=r", [],
                                     dtype=_core.int32, is_pure=False, pack=1,
                                     _semantic=_semantic)

    @_core.extern
    def gdc_launch_dependents(_semantic=None):
        _core.inline_asm_elementwise("griddepcontrol.launch_dependents; // dummy $0", "=r", [],
                                     dtype=_core.int32, is_pure=False, pack=1,
                                     _semantic=_semantic)
    HAVE_GDC = True
except Exception:                                                            # pragma: no cover
    HAVE_GDC = False


@triton.jit
def add_one(x_ptr):
    v = tl.load(x_ptr)
    tl.store(x_ptr, v + 1)


if HAVE_GDC:
    @triton.jit
    def add_one_pdl(x_ptr):
        gdc_wait()                       # wait for the primary before the dependent load
        v = tl.load(x_ptr)
        tl.store(x_ptr, v + 1)
        gdc_launch_dependents()          # let the next kernel launch early
else:
    add_one_pdl = None


def _trivial_kernels():
    return add_one, add_one_pdl


def pdl_ptx_has_griddepcontrol(kernel):
    """Inspect the compiled PTX of the PDL kernel (after at least one launch)."""
    try:
        caches = getattr(kernel, 'device_caches', None) or {}
        for dev, entry in caches.items():
            cache = entry[0] if isinstance(entry, (tuple, list)) else entry
            for ck in cache.values():
                asm = getattr(ck, 'asm', None) or {}
                ptx = asm.get('ptx', '')
                if ptx:
                    has_wait = 'griddepcontrol.wait' in ptx
                    has_launch = 'griddepcontrol.launch_dependents' in ptx
                    return True if (has_wait and has_launch) else \
                        f'FAILED: wait={has_wait} launch_dependents={has_launch}'
        return 'UNRESOLVED: no compiled kernel in device cache'
    except Exception as e:
        return f'UNRESOLVED: {type(e).__name__}'


def time_chain(launch, n, capture, label):
    for _ in range(WARMUP):
        launch()
    torch.cuda.synchronize()
    graph = None
    if capture:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            launch()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        graph.replay()
        torch.cuda.synchronize()
    times = []
    torch.cuda.nvtx.range_push(label)
    for _ in range(REPEATS):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        if graph is not None:
            graph.replay()
        else:
            launch()
        e1.record()
        torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1) * 1e3)      # us
    torch.cuda.nvtx.range_pop()
    return {'chain_us_median': statistics.median(times),
            'chain_us_min': min(times), 'chain_us_max': max(times),
            'per_launch_us': statistics.median(times) / n, 'n': n, 'repeats': REPEATS}


def fit(points):
    """least squares T = a + b*N over (N, T_median) points"""
    if len(points) < 2:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else None
    return {'slope_us_per_node': b, 'intercept_us': (my - b * mx) if b is not None else None,
            'points': points}


def analyze_trace(base, out, traced_json=None):
    """Per NVTX chain segment: kernel durations and serial inter-kernel gaps.
    Joined with the traced run's own JSON (gate-03 f40): a segment is only
    accepted when its case status is 'ok' (dependency proof + PTX passed) and
    the segment holds exactly N x REPEATS kernels; else UNRESOLVED."""
    cases, qual_note, pdl_ptx = {}, None, None
    if traced_json and os.path.exists(traced_json):
        tj = json.load(open(traced_json))
        cases = tj.get('chains', {})
        pdl_ptx = tj.get('pdl_ptx')
    if not cases:
        qual_note = 'UNRESOLVED: traced run qualification JSON missing — no calibration value derived'
    def read(path):
        with open(path, newline='') as f:
            rows = list(csv.reader(f))
        return rows[0], rows[1:]
    head, rows = read(base + '_cuda_gpu_trace.csv')
    c_s = next(i for i, h in enumerate(head) if 'Start' in h)
    c_d = next(i for i, h in enumerate(head) if 'Duration' in h)
    c_n = next(i for i, h in enumerate(head) if h.strip().strip('"') == 'Name')
    kern = []
    for r in rows:
        try:
            kern.append((int(float(r[c_s].replace(',', ''))), int(float(r[c_d].replace(',', ''))), r[c_n]))
        except (ValueError, IndexError):
            continue
    kern.sort()
    head, rows = read(base + '_nvtx_gpu_proj_trace.csv')
    c_name = next(i for i, h in enumerate(head) if h.strip().strip('"') in ('Name', 'Range'))
    c_start = next(i for i, h in enumerate(head) if 'Start' in h)
    c_dur = next(i for i, h in enumerate(head) if 'Duration' in h)
    segs = {}
    for r in rows:
        nm = r[c_name].strip().strip('"').split(':')[-1]
        if '/' not in nm:
            continue
        try:
            s = int(float(r[c_start].replace(',', '')))
            d = int(float(r[c_dur].replace(',', '')))
        except ValueError:
            continue
        segs[nm] = (s, s + d)
    res = {}
    for nm, (s, e) in segs.items():
        ks = [k for k in kern if s <= k[0] < e]
        case = cases.get(nm)
        n = int(nm.rsplit('N', 1)[1]) if 'N' in nm else None
        expected = n * REPEATS if n else None
        if case is None:                       # gate-04 f54: qualification is mandatory
            res[nm] = {'kernels': len(ks), 'status': 'UNRESOLVED: no qualification case for this segment'}
            continue
        if case.get('status') != 'ok':
            res[nm] = {'kernels': len(ks), 'status': f"UNRESOLVED: case status {case.get('status')}"}
            continue
        # the proof DATA decides, not the label (gate-05 f72): trivial chains carry the
        # exact dependency counter; PDL modes additionally need the PTX qualification
        if nm.startswith('trivial/'):
            # the expected counter is DERIVED here from N, mode, warmup and repeats
            # (gate-06 f93): 25N for stream modes, 27N for graph modes (capture adds 2)
            mode = nm.split('/')[1]
            derived = n * (WARMUP + REPEATS + (2 if 'graph' in mode else 0)) if n else None
            if not (case.get('dependency_proof') is True and derived is not None
                    and case.get('counter_delta') == derived and case.get('counter_expected') == derived):
                res[nm] = {'kernels': len(ks), 'expected_counter': derived,
                           'status': 'UNRESOLVED: dependency proof data missing, failed, or not the derived count'}
                continue
        if '/pdl_' in nm and pdl_ptx is not True:
            res[nm] = {'kernels': len(ks), 'status': f'UNRESOLVED: PDL PTX qualification {pdl_ptx}'}
            continue
        if expected is not None and len(ks) != expected:
            res[nm] = {'kernels': len(ks), 'expected_kernels': expected,
                       'status': 'UNRESOLVED: kernel count != N x REPEATS'}
            continue
        if len(ks) < 2:
            res[nm] = {'kernels': len(ks), 'status': 'UNRESOLVED: too few kernels in range'}
            continue
        durs = [k[1] for k in ks]
        gaps = [ks[i + 1][0] - (ks[i][0] + ks[i][1]) for i in range(len(ks) - 1)]
        res[nm] = {'kernels': len(ks), 'expected_kernels': expected, 'status': 'ok',
                   'case_status': (case or {}).get('status'),
                   'kernel_us_median': statistics.median(durs) / 1e3,
                   'gap_us_median': statistics.median(gaps) / 1e3,
                   'gap_us_min': min(gaps) / 1e3, 'gap_us_max': max(gaps) / 1e3,
                   'negative_gap_fraction': sum(1 for g in gaps if g < 0) / len(gaps),
                   'note': ('overlapping kernels (PDL): gap is not a boundary estimate'
                            if 'pdl' in nm else 'serial gap = boundary estimate for this mode')}
    res['_qualification'] = {'traced_json': traced_json, 'cases': len(cases), 'note': qual_note,
                             'accepted_segments': sum(1 for k, v in res.items() if not k.startswith('_') and v.get('status') == 'ok')}
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)
    for nm, r in sorted(res.items()):
        if 'gap_us_median' in r:
            print(f"[boundary/trace] {nm}: kernel {r['kernel_us_median']:.3f} us, gap {r['gap_us_median']:.3f} us "
                  f"({r['kernels']} kernels)")
    print(f'-> {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--analyze-trace', default=None, help='export basename of a traced run')
    ap.add_argument('--traced-json', default=None, help='the traced run\'s own boundary JSON')
    args = ap.parse_args()
    if args.analyze_trace:
        analyze_trace(args.analyze_trace, args.out, args.traced_json)
        return
    assert torch.cuda.is_available()
    dev = 'cuda'
    result = {'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'ns': list(NS), 'chains': {}, 'fits': {}}
    add_one, add_one_pdl = _trivial_kernels()
    import triton
    result['triton'] = triton.__version__

    x = torch.zeros(1, device=dev, dtype=torch.int32)
    big = torch.ones(37 * 1024 * 1024 // 2, device=dev, dtype=torch.float16)   # 37 MB fp16
    a = torch.randn(660, 384, device=dev, dtype=torch.float16)
    w1 = (torch.randn(384, 1536, device=dev) / 384 ** 0.5).half()
    w2 = (torch.randn(1536, 384, device=dev) / 1536 ** 0.5).half()
    h1 = torch.empty(660, 1536, device=dev, dtype=torch.float16)
    h2 = torch.empty(660, 384, device=dev, dtype=torch.float16)

    def launch_trivial(n, pdl):
        k = add_one_pdl if pdl else add_one
        def go():
            for _ in range(n):
                if pdl:
                    k[(1,)](x, launch_pdl=True)
                else:
                    k[(1,)](x)
        return go

    def launch_elem(n):
        def go():
            for _ in range(n):
                big.mul_(1.0)
        return go

    def launch_gemm(n):
        def go():
            torch.matmul(a, w1, out=h1)
            for i in range(1, n):
                if i % 2 == 1:
                    torch.matmul(h1, w2, out=h2)
                else:
                    torch.matmul(h2, w1, out=h1)
        return go

    specs = []
    for n in NS:
        specs.append(('trivial', 'stream', n, launch_trivial(n, False), False))
        specs.append(('trivial', 'graph', n, launch_trivial(n, False), True))
        specs.append(('trivial', 'pdl_stream', n, launch_trivial(n, True), False))
        specs.append(('trivial', 'pdl_graph', n, launch_trivial(n, True), True))
    for n in (10, 100):
        specs.append(('elem37', 'stream', n, launch_elem(n), False))
        specs.append(('elem37', 'graph', n, launch_elem(n), True))
        specs.append(('gemm', 'stream', n, launch_gemm(n), False))
        specs.append(('gemm', 'graph', n, launch_gemm(n), True))

    launches_expected = 0
    for chain, mode, n, launch, capture in specs:
        key = f'{chain}/{mode}/N{n}'
        if mode.startswith('pdl') and add_one_pdl is None:
            result['chains'][key] = {'status': 'UNTESTED', 'reason': 'triton gdc intrinsics unavailable'}
            print(f'[boundary] {key}: UNTESTED (no gdc)')
            continue
        try:
            before = int(x.item()) if chain == 'trivial' else None
            rec = time_chain(launch, n, capture, key)
            rec['status'] = 'ok'
            if chain == 'trivial':
                torch.cuda.synchronize()
                after = int(x.item())
                # executed chains: WARMUP eager + [capture mode: 1 side-stream
                # warmup + 1 verification replay; the capture pass itself
                # records without executing] + REPEATS timed chains
                expected = n * (WARMUP + REPEATS + (2 if capture else 0))
                rec['counter_delta'] = after - before
                rec['counter_expected'] = expected
                rec['dependency_proof'] = (after - before == expected)
                if not rec['dependency_proof']:
                    rec['status'] = 'FAILED_PROOF'      # excluded from fits/derived values
            print(f"[boundary] {key}: {rec['per_launch_us']:.3f} us/launch (chain {rec['chain_us_median']:.1f} us, "
                  f"min {rec['chain_us_min']:.1f}, max {rec['chain_us_max']:.1f})"
                  + (f" counter {rec['counter_delta']}/{rec['counter_expected']}" if chain == 'trivial' else ''))
        except Exception as e:
            rec = {'status': 'UNTESTED', 'reason': f'{type(e).__name__}: {e}'[:400],
                   'traceback': traceback.format_exc()[-2000:]}
            print(f'[boundary] {key}: UNTESTED ({type(e).__name__})')
            torch.cuda.synchronize()
        result['chains'][key] = rec
    # PDL qualification: the compiled PTX must contain BOTH griddepcontrol
    # instructions; otherwise every PDL case is UNRESOLVED and excluded
    if add_one_pdl is not None:
        result['pdl_ptx'] = pdl_ptx_has_griddepcontrol(add_one_pdl)
    else:
        result['pdl_ptx'] = 'UNTESTED: gdc intrinsics unavailable'
    pdl_ok = result['pdl_ptx'] is True
    for key, rec in result['chains'].items():
        if '/pdl_' in key and rec.get('status') == 'ok' and not pdl_ok:
            rec['status'] = 'UNRESOLVED'
            rec['reason'] = f"PDL PTX qualification failed: {result['pdl_ptx']}"
    for chain in ('trivial', 'elem37', 'gemm'):
        for mode in ('stream', 'graph', 'pdl_stream', 'pdl_graph'):
            pts = [(n, result['chains'][f'{chain}/{mode}/N{n}']['chain_us_median'])
                   for n in NS if f'{chain}/{mode}/N{n}' in result['chains']
                   and result['chains'][f'{chain}/{mode}/N{n}'].get('status') == 'ok']
            if pts:
                result['fits'][f'{chain}/{mode}'] = fit(pts)
            excluded = [f'{chain}/{mode}/N{n}' for n in NS if f'{chain}/{mode}/N{n}' in result['chains']
                        and result['chains'][f'{chain}/{mode}/N{n}'].get('status') != 'ok']
            if excluded:
                result.setdefault('excluded_from_fits', []).extend(excluded)
    result['real_kernel_pdl'] = 'UNSUPPORTED: cuBLAS/ATen launches expose no PDL attribute'
    # plan 1.4 input: the 37 MB in-place elementwise chain fits the device L2
    # (recorded from device properties), so its graph-mode per-launch time is
    # an L2-resident bandwidth measurement (read + write of 37 MB per launch)
    props = torch.cuda.get_device_properties(0)
    l2 = getattr(props, 'L2_cache_size', None)
    e = result['chains'].get('elem37/graph/N100', {})
    if e.get('status') == 'ok' and l2 and big.numel() * 2 <= l2:
        result['l2_resident_bandwidth'] = {
            'l2_cache_bytes': l2, 'working_set_bytes': big.numel() * 2,
            'bytes_moved_per_launch': 2 * big.numel() * 2,
            'per_launch_us': e['per_launch_us'],
            'tbps': (2 * big.numel() * 2) / (e['per_launch_us'] * 1e-6) / 1e12,
            'note': 'measured on the 37 MB fp16 in-place chain in graph mode; use as BW_L2 for '
                    'kernels whose working set fits L2 (plan 1.4)'}
    else:
        result['l2_resident_bandwidth'] = {'status': 'UNRESOLVED', 'l2_cache_bytes': l2}
    result['prior_points'] = {
        'hazy_h100_stream_us': 2.1, 'hazy_h100_graph_us': 1.3,
        'nvidia_cuda126_graph_rtx3060_us_per_node': 0.55,
        'note': 'comparison points only (the NVIDIA figure is empty-kernel graph device service '
                'time per node, not CPU launch cost); no per-boundary number is used until measured'}
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=1)
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
