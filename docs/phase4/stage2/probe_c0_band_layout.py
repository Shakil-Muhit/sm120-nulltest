"""Probe C0 (Stage 2, CPU, analysis only): the band layout of the pinned MelBandRoformer config and the
pad-and-batch padding fraction. Usage: python probe_c0_band_layout.py <out.json>"""
import json, os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(_ROOT, 'fullmodel'))
import torch
import loader
paths = loader.paths_from_env()
if paths['msst'] not in sys.path: sys.path.insert(0, paths['msst'])
from utils.settings import get_model_from_config
model, config = get_model_from_config('mel_band_roformer', paths['config'])
attrs = {k: (tuple(v.shape) if torch.is_tensor(v) else (v if isinstance(v, (int, float, str, bool)) else type(v).__name__)) for k, v in vars(model).items() if not k.startswith('_')}
cand = {k: v for k, v in attrs.items() if any(s in k.lower() for s in ('band', 'freq', 'mel', 'dim'))}
print('[probe_c0] candidate attributes:', cand)
widths = None
v = getattr(model, 'num_freqs_per_band', None)
if v is not None:
    widths = [int(t) for t in v.reshape(-1).tolist()]
fi = getattr(model, 'freq_indices', None)
extra = {'freq_indices_len': int(fi.numel()) if fi is not None else None, 'dim_freqs_in': getattr(config.model, 'dim_freqs_in', None) if hasattr(config, 'model') else None}
rec = {'config': paths['config'], 'attrs': cand, **extra}
if widths:
    mx = max(widths); tot = sum(widths)
    rec.update({'num_bands': len(widths), 'widths': widths, 'max_width': mx, 'sum_widths': tot,
                'padding_fraction_of_padded_total': round((len(widths) * mx - tot) / (len(widths) * mx), 4),
                'padded_over_useful': round(len(widths) * mx / tot, 4),
                'note': 'per-frame constant: the fraction does not depend on the number of frames'})
    print('[probe_c0] bands', len(widths), 'max', mx, 'sum', tot, 'padded/useful', rec['padded_over_useful'])
json.dump(rec, open(sys.argv[1], 'w'), indent=1, default=str)
