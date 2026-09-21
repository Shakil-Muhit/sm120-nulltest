"""2.5(viii): which condition produces the per-pass weight casts. Engineering data (ml-beast GPU 4)."""
import os, sys, time, collections
os.chdir(os.path.expanduser('~/axial/sm120-nulltest'))
sys.path.insert(0, 'fullmodel')
import torch, loader, timing
paths = loader.paths_from_env()
model, config, use_amp, ctx, ctx_nocache = loader.load_public_model(paths['msst'], paths['ckpt'], paths['config'])
ps = list(model.parameters())
print('params', len(ps), 'requires_grad True:', sum(p.requires_grad for p in ps), 'fp32 leaves:', sum(p.dtype == torch.float32 and p.is_leaf for p in ps),
      'views:', sum(p._is_view() for p in ps), 'use_amp', use_amp, 'grad_enabled', torch.is_grad_enabled())
x = timing.make_input(4410, 0)
from torch.profiler import profile, ProfilerActivity
def count_casts(rg, cache):
    for p in ps: p.requires_grad_(rg)
    c = torch.autocast('cuda', enabled=use_amp, cache_enabled=cache)
    with torch.no_grad(), c:
        for _ in range(3): model(x)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(2): model(x)
            torch.cuda.synchronize()
    cnt = collections.Counter()
    for e in prof.key_averages():
        n = e.key
        if 'copy' in n.lower() or 'nocast' in n.lower():
            cnt[n[:70]] += e.count
    tot = sum(cnt.values()) / 2
    return tot, dict(cnt.most_common(4))
for rg in (False, True):
    for cache in (True, False):
        tot, top = count_casts(rg, cache)
        print(f'requires_grad={rg} cache_enabled={cache}: copy/cast-like kernels per pass = {tot:.0f}; top {top}')
