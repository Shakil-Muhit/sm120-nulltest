"""Probe B1 (Stage 2, engineering): do the model's small-M GEMMs at 100 ms run slower when the fp16 weight is
NOT freshly written by a cast kernel immediately before the GEMM? Four conditions per distinct nn.Linear shape,
each iteration started cold by an L2 flush:
  a: fp32 weight -> .half() cast kernel -> GEMM         (the autocast / C3 pattern)
  b: fp16 weight, GEMM alone                            (pre-cast / frozen pattern)
  c: fp16 weight, read-only touch kernel over the weight -> GEMM   (warms L2 without producing the operand)
  d: fp16 weight, cast of an UNRELATED fp32 tensor of the same size -> GEMM (same extra work, no warming)
GEMM-only time is measured with CUDA events around the GEMM call alone (the preceding kernel is outside the events).
Usage: python probe_b1_gemm_locality.py <chunk> <out.json>"""
import collections, json, os, sys, time
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(_ROOT, 'fullmodel'))
import torch
import loader, timing as tm
chunk, out = int(sys.argv[1]), sys.argv[2]
paths = loader.paths_from_env()
model, config, use_amp, ctx, ctx_nocache = loader.load_public_model(paths['msst'], paths['ckpt'], paths['config'])
model.eval()
x = tm.make_input(chunk, tm.TIMING_SEED)
shapes = collections.Counter()
def hook(mod, inp, outp):
    a = inp[0]
    shapes[(tuple(a.shape), tuple(mod.weight.shape), str(a.dtype), str(mod.weight.dtype), mod.bias is not None)] += 1
hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, torch.nn.Linear)]
with torch.no_grad(), ctx():
    model(x)
for h in hs: h.remove()
torch.cuda.synchronize()
print(f'[probe_b1] {sum(shapes.values())} Linear calls, {len(shapes)} distinct shapes', flush=True)
flush_buf = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device='cuda')
def l2_flush():
    flush_buf.fill_(1)
res = []
ITERS = int(os.environ.get('PROBE_B1_ITERS', '100'))
for (ishape, wshape, idt, wdt, has_bias), n in sorted(shapes.items(), key=lambda kv: -kv[1]):
    M = 1
    for d in ishape[:-1]: M *= d
    K = ishape[-1]; N = wshape[0]
    a = torch.randn(*ishape, device='cuda', dtype=torch.float16)
    w32 = torch.randn(*wshape, device='cuda', dtype=torch.float32)
    w16 = w32.half().contiguous()
    other32 = torch.randn(*wshape, device='cuda', dtype=torch.float32)
    bias = torch.randn(N, device='cuda', dtype=torch.float16) if has_bias else None
    def gemm(w):
        return torch.nn.functional.linear(a, w, bias)
    def run(cond):
        ts = []
        for i in range(ITERS + 10):
            l2_flush()
            if cond == 'a_cast_then_gemm':
                w = w32.half()
            elif cond == 'b_precast_alone':
                w = w16
            elif cond == 'c_touch_then_gemm':
                _ = torch.sum(w16, dim=(0, 1))     # read-only pass over the weight (one kernel)
                w = w16
            else:
                _ = other32.half()
                w = w16
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            e0.record(); y = gemm(w); e1.record(); torch.cuda.synchronize()
            if i >= 10: ts.append(e0.elapsed_time(e1) * 1000.0)
        ts.sort(); return {'median_us': ts[len(ts) // 2], 'p10_us': ts[len(ts) // 10], 'p90_us': ts[9 * len(ts) // 10]}
    row = {'input_shape': ishape, 'weight_shape': wshape, 'M': M, 'N': N, 'K': K, 'calls_per_pass': n, 'iters': ITERS}
    for cond in ('a_cast_then_gemm', 'b_precast_alone', 'c_touch_then_gemm', 'd_other_cast_then_gemm'):
        row[cond] = run(cond)
    # kernel names for a and b from the profiler (short runs)
    from torch.profiler import profile, ProfilerActivity
    names = {}
    for cond, w_fn in (('a', lambda: w32.half()), ('b', lambda: w16)):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                l2_flush(); y = gemm(w_fn()); torch.cuda.synchronize()
        ks = [e.key for e in prof.key_averages() if any(t in e.key.lower() for t in ('gemm', 'cutlass', 'sm120', 'xmma', 'nvjet'))]
        names[cond] = sorted(set(ks))[:4]
    row['gemm_kernels'] = names
    res.append(row)
    print(f"[probe_b1] M={M} N={N} K={K} x{n}: a={row['a_cast_then_gemm']['median_us']:.1f} b={row['b_precast_alone']['median_us']:.1f} c={row['c_touch_then_gemm']['median_us']:.1f} d={row['d_other_cast_then_gemm']['median_us']:.1f} us | same kernel: {names['a'] == names['b']}", flush=True)
json.dump({'chunk': chunk, 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'note': 'engineering probe on ml-beast; cold-L2 iterations (256 MB flush before each); GEMM-only CUDA-event time; never evidence', 'rows': res}, open(out, 'w'), indent=1)
print('[probe_b1] done ->', out)
