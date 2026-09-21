"""E5 — structural variants built through the pinned construction path
(get_model_from_config on a derived yaml), random weights (seeded), no
checkpoint. Timing only; validation = self-consistency against the variant's
own eager-A reference (ref_variant_<name>/). Never headline-eligible."""

import hashlib
import os
import re
import sys

import torch

VARIANTS = {          # name: (depth, mlp_expansion_factor)
    'd1_mfull': (1, 4), 'd1_mquarter': (1, 1), 'd6_mfull': (6, 4), 'd6_mquarter': (6, 1)}
SEED = 20260921


def variant_config(public_config_path, name, out_dir):
    depth, mlp = VARIANTS[name]
    txt = open(public_config_path).read()
    assert re.search(r'^  depth: \d+\s*$', txt, flags=re.M), 'public config has no model.depth line'
    txt = re.sub(r'^  depth: \d+\s*$', f'  depth: {depth}', txt, count=1, flags=re.M)
    if re.search(r'^  mlp_expansion_factor: ', txt, flags=re.M):
        txt = re.sub(r'^  mlp_expansion_factor: .*$', f'  mlp_expansion_factor: {mlp}', txt, count=1, flags=re.M)
    else:
        txt = re.sub(r'^(  mask_estimator_depth: .*)$', rf'\1\n  mlp_expansion_factor: {mlp}', txt, count=1, flags=re.M)
    assert f'  mlp_expansion_factor: {mlp}' in txt
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'variant_{name}.yaml')
    with open(path, 'w') as f:
        f.write(txt)
    return path


def load_variant_model(msst_dir, public_config_path, name, out_dir, device='cuda'):
    import functools
    if msst_dir not in sys.path:
        sys.path.insert(0, msst_dir)
    from utils.settings import get_model_from_config
    vpath = variant_config(public_config_path, name, out_dir)
    torch.manual_seed(SEED)
    model, config = get_model_from_config('mel_band_roformer', vpath)
    model = model.to(device).eval()
    use_amp = config.training.use_amp
    assert isinstance(use_amp, bool)
    ctx = functools.partial(torch.autocast, 'cuda', enabled=use_amp)
    ctx_nocache = functools.partial(torch.autocast, 'cuda', enabled=use_amp, cache_enabled=False)
    n = sum(p.numel() for p in model.parameters())
    me = sum(p.numel() for p in model.mask_estimators.parameters())
    depth, mlp = VARIANTS[name]
    rec = {'variant': name, 'depth': depth, 'mlp_expansion_factor': mlp, 'seed': SEED,
           'param_count': n, 'param_bytes_fp16': n * 2,
           'mask_estimator_param_count': me, 'mask_estimator_bytes_fp16': me * 2,
           'variant_config': vpath, 'variant_config_sha256': hashlib.sha256(open(vpath, 'rb').read()).hexdigest(),
           'public_config_sha256': hashlib.sha256(open(public_config_path, 'rb').read()).hexdigest()}
    return model, config, use_amp, ctx, ctx_nocache, rec
