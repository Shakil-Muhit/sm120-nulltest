"""E1(b) — pre-cast wrapper: the band-split and mask-estimator Linear weights
are cast to fp16 ONCE at install (external module swap; the pinned modules keep
their fp32 parameters registered and untouched). Under autocast an fp16 weight
needs no cast, so the per-pass `aten._to_copy` parameter casts that C3 replays
inside its graph (X6a) disappear for these modules. Everything else is the
public path. Registered contingency for E1 when freezing (C3F) does not
compose with cudagraph trees or leaves the casts in place."""

import hashlib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreCastLinear(nn.Module):
    def __init__(self, orig):
        super().__init__()
        assert isinstance(orig, nn.Linear)
        self.orig = orig
        self.register_buffer('w16', orig.weight.detach().to(torch.float16).contiguous(), persistent=False)
        if orig.bias is not None:
            self.register_buffer('b16', orig.bias.detach().to(torch.float16).contiguous(), persistent=False)
        else:
            self.b16 = None

    def forward(self, x):
        return F.linear(x, self.w16, self.b16)


def install(model):
    n = 0
    nbytes = 0
    for seq in model.band_split.to_features:            # Sequential(RMSNorm, Linear)
        assert isinstance(seq[1], nn.Linear)
        seq[1] = PreCastLinear(seq[1])
        n += 1
        nbytes += seq[1].w16.numel() * 2 + (seq[1].b16.numel() * 2 if seq[1].b16 is not None else 0)
    for me in model.mask_estimators:
        for mlp in me.to_freqs:                          # Sequential(MLP-Sequential, GLU)
            inner = mlp[0]
            for i, m in enumerate(inner):
                if isinstance(m, nn.Linear):
                    inner[i] = PreCastLinear(m)
                    n += 1
                    nbytes += inner[i].w16.numel() * 2 + (inner[i].b16.numel() * 2 if inner[i].b16 is not None else 0)
    assert n > 0
    src = os.path.abspath(__file__)
    return {'precast_linear_modules': n, 'fp16_weight_bytes': nbytes,
            'untouched': ['transformer Linear/RMSNorm parameters (still fp32, cast by autocast per pass)'],
            'source_sha256': hashlib.sha256(open(src, 'rb').read()).hexdigest()}
