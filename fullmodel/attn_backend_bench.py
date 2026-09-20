"""Attention backend microbenchmark (plan 1.5; prereg_amendments.md §5), v2.

Two layout families per (operating point, axis):
  model      : the layouts the compiled model actually feeds to SDPA (read
               from the C3 generated code, axial-layout-scoping evidence):
               q and k are (b, n, h, d)-contiguous rotary outputs presented
               as (b, h, n, d) views — strides (n*h*d, d, h*d, 1); v is a
               view into the fused qkv GEMM output (b, n, 3*h*d) — strides
               (n*3*h*d, d, 3*h*d, 1), storage offset 2*h*d
  contiguous : (b, h, n, d) contiguous (separate experiment, not the model)
Shapes: time-axis batch = 60 bands, seq = T frames; freq-axis batch = T,
seq = 60; heads 8, dim_head 64, fp16; scale = default (dim_head**-0.5),
dropout 0 — the model's Attend settings. Backends forced one at a time
(flash / efficient / cudnn / math); unsupported combinations are recorded.
Reference for the error column: fp32 inputs with the MATH backend forced.
Timing: CUDA-graph replay of BLOCK launches, events, median of REPEATS.
The model's deployed backend is read from a trace kernel summary
(--model-kern-sum, e.g. the C3 node trace): flash if flash kernels dominate
the SDPA class, cudnn if cuDNN kernels do, else efficient; without a trace
the ratio is reported against 'flash' and flagged ASSUMED.
"""

import argparse
import csv
import json
import statistics

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

BLOCK = 20
REPEATS = 20
FRAMES = {352800: 801, 4410: 11}
BACKENDS = {'flash': SDPBackend.FLASH_ATTENTION, 'efficient': SDPBackend.EFFICIENT_ATTENTION,
            'cudnn': SDPBackend.CUDNN_ATTENTION, 'math': SDPBackend.MATH}
H, D = 8, 64


def model_layout(b, n, gen):
    """q, k: views of (b, n, h, d)-contiguous buffers; v: view into (b, n, 3hd)."""
    qk_store = [torch.randn(b, n, H, D, generator=gen).to('cuda', torch.float16) for _ in range(2)]
    q = qk_store[0].permute(0, 2, 1, 3)
    k = qk_store[1].permute(0, 2, 1, 3)
    qkv = torch.randn(b, n, 3 * H * D, generator=gen).to('cuda', torch.float16)
    v = qkv.view(b, n, 3, H, D)[:, :, 2].permute(0, 2, 1, 3)
    assert q.stride() == (n * H * D, D, H * D, 1) and v.stride() == (n * 3 * H * D, D, 3 * H * D, 1)
    return q, k, v, [qk_store, qkv]


def contiguous_layout(b, n, gen):
    t = [torch.randn(b, H, n, D, generator=gen).to('cuda', torch.float16) for _ in range(3)]
    return t[0], t[1], t[2], t


def bench(q, k, v, backend):
    def fn():
        with sdpa_kernel([backend]):
            return F.scaled_dot_product_attention(q, k, v)
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(BLOCK):
            fn()
    times = []
    for _ in range(REPEATS):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1) * 1e3 / BLOCK)
    return statistics.median(times), min(times), max(times)


def _family(name):
    """Attention EXECUTION kernel family; Triton/preprocessing names are never
    attention executions."""
    n = name.lower()
    if n.startswith('triton') or 'triton_' in n:
        return None
    if 'flash' in n and ('fwd' in n or 'attn' in n or 'kernel' in n):
        return 'flash'
    if 'cudnn' in n or 'fmha_v2' in n or 'sm100_fmha' in n or 'sm90_fmha' in n:
        return 'cudnn'
    if 'fmha' in n or 'mem_eff' in n or 'efficient' in n or 'attention_kernel' in n:
        return 'efficient'
    return None


def deployed_backends(base, expected_passes=50):
    """Per (axis) at the trace's operating point: the attention execution
    family of the 12 SDPA events per pass, in order (time, freq) x 6 layers.
    Returns {'time': family|None, 'freq': family|None, 'note': ...} or None
    (UNRESOLVED) when the per-pass SDPA structure is not the registered one."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'attrib'))
    from classify import classify
    from kernel_map import read_gpu_trace, nvtx_pass_windows, split_passes
    rows = read_gpu_trace(base + '_cuda_gpu_trace.csv')
    windows, err = nvtx_pass_windows(base + '_nvtx_gpu_proj_trace.csv')
    if err or len(windows) != expected_passes:
        return None
    passes, _ = split_passes(rows, windows, classify)
    fams = None
    for p in passes:
        sd = [r['name'] for r in p if classify(r['name']) == 'SDPA' and _family(r['name'])]
        if len(sd) != 12:
            return None
        f = [_family(n) for n in sd]
        if fams is None:
            fams = f
        elif f != fams:
            return None
    time_f = {fams[i] for i in range(0, 12, 2)}
    freq_f = {fams[i] for i in range(1, 12, 2)}
    return {'time': time_f.pop() if len(time_f) == 1 else None,
            'freq': freq_f.pop() if len(freq_f) == 1 else None,
            'note': '12 SDPA executions per pass in (time, freq) x 6 order (registered axis order)'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--model-trace-352800', default=None, help='export basename of the C3 8-s node trace')
    ap.add_argument('--model-trace-4410', default=None, help='export basename of the C3 4410 node trace')
    ap.add_argument('--passes', type=int, default=50)
    args = ap.parse_args()
    assert torch.cuda.is_available()
    gen = torch.Generator(device='cpu').manual_seed(3)
    # deployed backend per (chunk, axis) from the C3 traces; anything not
    # established from a trace is UNRESOLVED (no ratio), never assumed
    model_backend = {}
    for chunk, base in ((352800, args.model_trace_352800), (4410, args.model_trace_4410)):
        model_backend[chunk] = deployed_backends(base, args.passes) if base else None
    result = {'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'dtype': 'float16', 'heads': H, 'dim_head': D, 'bands': 60,
              'scale': 'default dim_head**-0.5', 'dropout': 0.0,
              'model_backend_per_chunk': {str(k): v for k, v in model_backend.items()}, 'cases': {}}
    for chunk, T in FRAMES.items():
        for axis, (b, n) in {'time': (60, T), 'freq': (T, 60)}.items():
            for layout, make in (('model', model_layout), ('contiguous', contiguous_layout)):
                q, k, v, keep = make(b, n, gen)
                case = {'chunk': chunk, 'axis': axis, 'layout': layout, 'batch': b, 'seq': n,
                        'q_stride': list(q.stride()), 'v_stride': list(v.stride()),
                        'v_storage_offset': v.storage_offset(), 'backends': {}}
                with torch.no_grad():
                    with sdpa_kernel([SDPBackend.MATH]):
                        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
                    for name, be in BACKENDS.items():
                        try:
                            with sdpa_kernel([be]):
                                out = F.scaled_dot_product_attention(q, k, v)
                            err = float((out.float() - ref).abs().max())
                            med, mn, mx = bench(q, k, v, be)
                            case['backends'][name] = {'us_median': med, 'us_min': mn, 'us_max': mx,
                                                      'max_abs_err_vs_fp32_math': err, 'status': 'ok'}
                            print(f'[attn] c{chunk} {axis} {layout} {name}: {med:.2f} us (err {err:.2e})')
                        except Exception as e:
                            case['backends'][name] = {'status': 'UNAVAILABLE',
                                                      'reason': f'{type(e).__name__}: {e}'[:300]}
                            print(f'[attn] c{chunk} {axis} {layout} {name}: UNAVAILABLE ({type(e).__name__})')
                            torch.cuda.synchronize()
                ok = {k2: v2['us_median'] for k2, v2 in case['backends'].items() if v2['status'] == 'ok'}
                if ok:
                    best = min(ok, key=ok.get)
                    case['best_backend'] = best
                    case['best_us'] = ok[best]
                    dep = (model_backend.get(chunk) or {}).get(axis) if model_backend.get(chunk) else None
                    case['model_backend'] = dep
                    if dep and dep in ok:
                        case['model_backend_us'] = ok[dep]
                        case['ratio_best_over_model'] = ok[best] / ok[dep]
                        case['ratio_note'] = f'deployed backend from the C3 trace at {chunk}/{axis}: {dep}'
                    else:
                        case['model_backend_us'] = None
                        case['ratio_best_over_model'] = None
                        case['ratio_note'] = 'UNRESOLVED: deployed backend not established from a trace'
                        case['status'] = 'UNRESOLVED'
                    if layout == 'contiguous':
                        case['note'] = 'separate experiment (not the model layout)'
                result['cases'][f'c{chunk}_{axis}_{layout}'] = case
                del keep
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=1)
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
