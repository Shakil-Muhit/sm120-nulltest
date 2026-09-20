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


def deployed_backend(kern_sum_csv):
    """From a node-level kernel summary: which SDPA kernel family carries the
    SDPA class time."""
    totals = {'flash': 0, 'cudnn': 0, 'efficient': 0}
    with open(kern_sum_csv, newline='') as f:
        rd = csv.reader(f)
        head = next(rd)
        c_t = next(i for i, h in enumerate(head) if 'Total Time' in h)
        c_n = next(i for i, h in enumerate(head) if h.strip().strip('"') == 'Name')
        for r in rd:
            try:
                t = int(float(r[c_t].replace(',', '')))
            except (ValueError, IndexError):
                continue
            n = r[c_n].lower()
            if 'flash' in n:
                totals['flash'] += t
            elif 'cudnn' in n and ('attn' in n or 'fmha' in n or 'sdpa' in n):
                totals['cudnn'] += t
            elif 'fmha' in n or 'mem_eff' in n or 'efficient' in n:
                totals['efficient'] += t
    if not any(totals.values()):
        return None, totals
    return max(totals, key=totals.get), totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--model-kern-sum', default=None,
                    help='cuda_gpu_kern_sum csv of the deployed compiled arm (C3) trace')
    args = ap.parse_args()
    assert torch.cuda.is_available()
    gen = torch.Generator(device='cpu').manual_seed(3)
    if args.model_kern_sum:
        mb, totals = deployed_backend(args.model_kern_sum)
        model_backend = {'backend': mb, 'source': args.model_kern_sum, 'sdpa_class_ns': totals,
                         'assumed': mb is None}
        if mb is None:
            model_backend['backend'] = 'flash'
    else:
        model_backend = {'backend': 'flash', 'source': None, 'assumed': True,
                         'note': 'Attend cuda_config=(True,False,False) + repaired context (flash, cudnn)'}
    result = {'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'dtype': 'float16', 'heads': H, 'dim_head': D, 'bands': 60,
              'scale': 'default dim_head**-0.5', 'dropout': 0.0,
              'model_backend': model_backend, 'cases': {}}
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
                    mbk = model_backend['backend']
                    case['best_backend'] = best
                    case['best_us'] = ok[best]
                    case['model_backend_us'] = ok.get(mbk)
                    case['ratio_best_over_model'] = (ok[best] / ok[mbk]) if mbk in ok else None
                    case['ratio_note'] = ('model backend ASSUMED (no trace given)' if model_backend.get('assumed')
                                          else f'model backend from trace: {mbk}')
                result['cases'][f'c{chunk}_{axis}_{layout}'] = case
                del keep
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=1)
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
