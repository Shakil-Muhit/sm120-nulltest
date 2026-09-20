"""Public construction path for the full model (prereg_amendments.md §1).

COPIED from axial-layout-scoping@a6f3e6f scoping/build_model.py::load_public_model
(itself mirroring sm120-nulltest scripts/p8_trace.py, AUDIT-MAP[5]): the model
comes ONLY from the pinned public clone via its own path —
utils.settings.get_model_from_config -> torch.load(weights_only=False,
map_location='cpu') -> utils.model_utils.load_start_checkpoint(type_='inference'),
the 3-line pattern of upstream inference.py at the pin. Precision: the repo's
own inference autocast, torch.autocast('cuda', enabled=config.training.use_amp).
"""

import argparse
import functools
import os
import sys

import torch


def load_public_model(msst_dir, ckpt, config_path,
                      model_type='mel_band_roformer', device='cuda'):
    if msst_dir not in sys.path:
        sys.path.insert(0, msst_dir)
    from utils.settings import get_model_from_config
    from utils.model_utils import load_start_checkpoint

    model, config = get_model_from_config(model_type, config_path)
    ckpt_ns = argparse.Namespace(
        start_check_point=ckpt, model_type=model_type,
        lora_checkpoint='', lora_checkpoint_loralib='')
    checkpoint = torch.load(ckpt, weights_only=False, map_location='cpu')
    load_start_checkpoint(ckpt_ns, model, checkpoint, type_='inference')
    model = model.to(device).eval()

    use_amp = config.training.use_amp
    if not isinstance(use_amp, bool):
        raise TypeError('config.training.use_amp must be an explicit boolean')
    autocast_ctx = functools.partial(torch.autocast, 'cuda', enabled=use_amp)
    # capture arms: the weight-cast cache must be off during graph capture
    # (torch CUDA Graphs + AMP guidance) — prereg_amendments.md §1, arm B
    autocast_nocache = functools.partial(torch.autocast, 'cuda', enabled=use_amp,
                                         cache_enabled=False)
    return model, config, use_amp, autocast_ctx, autocast_nocache


def paths_from_env():
    """Pod layout (setup.sh / fetch_checkpoint.sh): MSST_DIR, CKPT_DIR with
    manifest.env; env overrides win; ml-beast shakedown sets AXIAL-style paths."""
    home = os.path.expanduser('~')
    msst = os.environ.get('MSST_DIR', os.path.join(home, 'work', 'msst'))
    ckpt_dir = os.environ.get('CKPT_DIR', os.path.join(home, 'work', 'ckpt'))
    ckpt = os.environ.get('CKPT_FILE')
    config = os.environ.get('CONFIG_FILE')
    manifest = os.path.join(ckpt_dir, 'manifest.env')
    if (ckpt is None or config is None) and os.path.exists(manifest):
        for line in open(manifest):
            if '=' in line:
                k, v = line.strip().split('=', 1)
                v = v.strip('"')
                if k == 'CKPT_FILE' and ckpt is None:
                    ckpt = v
                if k == 'CONFIG_FILE' and config is None:
                    config = v
    if ckpt is None:
        ckpt = os.path.join(ckpt_dir, 'MelBandRoformer.ckpt')
    if config is None:
        config = os.path.join(msst, 'configs', 'KimberleyJensen',
                              'config_vocals_mel_band_roformer_kj.yaml')
    for p in (msst, ckpt, config):
        if not os.path.exists(p):
            raise FileNotFoundError(f'provenance path missing: {p} (run setup.sh)')
    return {'msst': msst, 'ckpt': ckpt, 'config': config}
