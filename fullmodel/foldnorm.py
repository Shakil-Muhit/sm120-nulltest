"""E4a — norm-scale folding wrapper (Phase 3 registration E4a; prereg_amendments.md v0.2).

Mechanism (first principles, no code imported from any other codebase):
the public RMSNorm is  norm(x) = F.normalize(x, dim=-1) * scale * gamma
                              = x * s(x) * gamma,   s(x) = scale / max(||x||_2, 1e-12)  (per token).
A per-token scalar commutes with the following Linear, so
    Linear(norm(x)) = s(x) * (x @ (W * gamma[None, :]).T) + b .
The wrapper folds gamma into the following Linear's weight ONCE at install,
runs the GEMM on the un-normalized input, and applies s(x) (and the bias)
after the GEMM, in the epilogue position (fused into the GEMM by a Triton
template under max-autotune, else the smallest pointwise pass Inductor makes).
s(x) is one small reduction per token, which Inductor fuses into the kernel
that last wrote x when it can (the trace records which).
The residual add of the public Transformer ('x = attn(x) + x; x = ff(x) + x')
already follows the output projection; the wrapper leaves Transformer.forward
untouched, so the residual lands in to_out's epilogue whenever the compiler
fuses it. The Transformer's output RMSNorm (norm_output) is left untouched.
Folded weights are kept as fp32 buffers so that autocast casts them exactly as
it casts the public fp32 parameters (the E1/X6a cast behaviour is NOT changed
by this wrapper; the two levers stay separable).

Claim: ALGEBRAIC equivalence. Rounding points move (fp16 GEMM on x instead of
on the fp32-normalized x; scale applied after accumulation), so agreement is
verified at the loose tier plus the SDR gate on all eight clips (registered).
The scale is never applied inside the GEMM inner loop (design note in the
registration).
"""

import hashlib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _rms_scale(x, scale):
    n = torch.linalg.vector_norm(x.float(), dim=-1, keepdim=True)
    return scale / torch.clamp(n, min=1e-12)          # (..., 1) fp32, matches F.normalize's eps


def _is_rmsnorm(m):
    return hasattr(m, 'gamma') and hasattr(m, 'scale') and isinstance(m.gamma, torch.Tensor)


class FoldedAttention(nn.Module):
    def __init__(self, orig):
        super().__init__()
        assert getattr(orig, 'pope_embed', None) is None, 'PoPE path not covered by the fold'
        assert _is_rmsnorm(orig.norm) and isinstance(orig.to_qkv, nn.Linear) and orig.to_qkv.bias is None
        assert isinstance(orig.to_gates, nn.Linear) and orig.to_gates.bias is not None
        self.orig = orig                                  # parameters stay registered, untouched
        g = orig.norm.gamma.detach().float()
        self.scale = float(orig.norm.scale)
        self.heads = int(orig.heads)
        self.register_buffer('w_qkv', (orig.to_qkv.weight.detach().float() * g[None, :]).contiguous(), persistent=False)
        self.register_buffer('w_gates', (orig.to_gates.weight.detach().float() * g[None, :]).contiguous(), persistent=False)
        self.register_buffer('b_gates', orig.to_gates.bias.detach().float().clone(), persistent=False)

    def forward(self, x):
        o = self.orig
        s = _rms_scale(x, self.scale)
        qkv = F.linear(x, self.w_qkv)
        qkv = (qkv.float() * s).to(qkv.dtype)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)
        if o.rotary_embed is not None:
            q = o.rotary_embed.rotate_queries_or_keys(q)
            k = o.rotary_embed.rotate_queries_or_keys(k)
        out = o.attend(q, k, v)
        gates = F.linear(x, self.w_gates)
        gates = (gates.float() * s + self.b_gates).to(gates.dtype)
        out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()
        out = rearrange(out, 'b h n d -> b n (h d)')
        return o.to_out(out)


class FoldedFeedForward(nn.Module):
    def __init__(self, orig):
        super().__init__()
        net = orig.net
        assert isinstance(net, nn.Sequential) and len(net) == 6
        assert _is_rmsnorm(net[0]) and isinstance(net[1], nn.Linear) and net[1].bias is not None
        assert isinstance(net[2], nn.GELU) and isinstance(net[4], nn.Linear)
        self.orig = orig
        g = net[0].gamma.detach().float()
        self.scale = float(net[0].scale)
        self.register_buffer('w1', (net[1].weight.detach().float() * g[None, :]).contiguous(), persistent=False)
        self.register_buffer('b1', net[1].bias.detach().float().clone(), persistent=False)

    def forward(self, x):
        net = self.orig.net
        s = _rms_scale(x, self.scale)
        h = F.linear(x, self.w1)
        h = (h.float() * s + self.b1).to(h.dtype)
        h = net[2](h)
        h = net[3](h)
        return net[5](net[4](h))


def install(model):
    """Swap every (Attention, FeedForward) pair of every Transformer in the
    pinned model for its folded wrapper via public ModuleList item assignment.
    Returns the install record (counts, untouched parts, wrapper source hash)."""
    n_attn = n_ff = n_tf = 0
    for block in model.layers:                          # ModuleList([time_tf, freq_tf]) per depth
        for tf in block:
            if not hasattr(tf, 'layers'):
                continue
            n_tf += 1
            for i, pair in enumerate(tf.layers):
                attn, ff = pair[0], pair[1]
                assert type(attn).__name__ == 'Attention', f'unexpected attention module {type(attn).__name__}'
                assert type(ff).__name__ == 'FeedForward'
                tf.layers[i] = nn.ModuleList([FoldedAttention(attn), FoldedFeedForward(ff)])
                n_attn += 1
                n_ff += 1
    assert n_attn > 0
    src = os.path.abspath(__file__)
    return {'folded_attention': n_attn, 'folded_feedforward': n_ff, 'transformers': n_tf,
            'untouched': ['band_split', 'mask_estimators', 'Transformer.norm (norm_output)', 'Transformer.forward residual adds'],
            'folded_weight_dtype': 'float32 (autocast casts as for the public parameters)',
            'source_sha256': hashlib.sha256(open(src, 'rb').read()).hexdigest()}
