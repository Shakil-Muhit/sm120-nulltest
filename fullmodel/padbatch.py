"""Arm H — pad-and-batch band modules (prereg_amendments.md §1.1).

Generic reimplementation of the band-bmm idea (production code not used).
External wrappers holding the ORIGINAL modules (parameters stay registered;
nothing is modified) plus derived zero-padded weight buffers built once.

Exactness (registered): padded entries are zero on every path —
  band_split : F.normalize over a zero-padded row == per-band normalize (zeros
               add nothing to the L2 norm); per-band scale K_i**0.5 as a
               (60,1,1) vector; zero-padded gamma; bmm with zero-padded weight
               rows contributes exactly 0; output order == torch.stack(dim=-2).
  mask_est.  : three batched bmm chains; the last weight's columns are laid
               out [a-half | b-half], each half zero-padded to K_max, so
               GLU(dim=-1) over the padded width equals the per-band GLU
               (pad columns: 0 * sigmoid(0) = 0); valid columns are gathered
               back in band order == torch.cat(dim=-1).
The only numerical difference is cuBLAS summation order (loose tier by the
classification rule; H is compiled anyway).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _band_split_parts(orig):
    dims = tuple(int(d) for d in orig.dim_inputs)
    parts = []
    for seq in orig.to_features:
        assert isinstance(seq, nn.Sequential) and len(seq) == 2
        rms, lin = seq[0], seq[1]
        assert isinstance(lin, nn.Linear) and lin.bias is not None
        assert hasattr(rms, 'gamma') and hasattr(rms, 'scale')
        parts.append((rms, lin))
    assert len(parts) == len(dims)
    return dims, parts


class PadBatchBandSplit(nn.Module):
    """Drop-in for the public BandSplit: (1, T, sum K_i) -> (1, T, 60, dim)."""

    def __init__(self, orig):
        super().__init__()
        self.orig = orig                       # parameters stay registered
        dims, parts = _band_split_parts(orig)
        n, kmax = len(dims), max(dims)
        dim_out = parts[0][1].out_features
        total = sum(dims)
        dev = parts[0][1].weight.device
        idx = torch.full((n, kmax), total, dtype=torch.long)   # -> zero column
        gamma = torch.zeros(n, kmax, dtype=torch.float32)
        w = torch.zeros(n, kmax, dim_out, dtype=torch.float32)
        b = torch.zeros(n, 1, dim_out, dtype=torch.float32)
        scale = torch.zeros(n, 1, 1, dtype=torch.float32)
        off = 0
        for i, ((rms, lin), k) in enumerate(zip(parts, dims)):
            assert lin.in_features == k and lin.out_features == dim_out
            assert rms.gamma.shape == (k,)
            idx[i, :k] = torch.arange(off, off + k)
            gamma[i, :k] = rms.gamma.detach().float().cpu()
            w[i, :k, :] = lin.weight.detach().float().cpu().t()
            b[i, 0, :] = lin.bias.detach().float().cpu()
            scale[i] = float(rms.scale)
            off += k
        assert off == total
        self.n, self.kmax, self.total = n, kmax, total
        self.register_buffer('idx', idx.to(dev), persistent=False)
        self.register_buffer('gamma_pad', gamma.to(dev), persistent=False)
        self.register_buffer('w_pad', w.to(dev), persistent=False)
        self.register_buffer('b_pad', b.to(dev), persistent=False)
        self.register_buffer('scale', scale.to(dev), persistent=False)

    def forward(self, x):                      # (1, T, total)
        assert x.shape[0] == 1 and x.shape[-1] == self.total
        t = x.shape[1]
        x_ext = torch.cat([x[0], x.new_zeros(t, 1)], dim=-1)   # (T, total+1)
        xp = x_ext[:, self.idx]                                 # (T, n, kmax)
        xp = xp.transpose(0, 1)                                 # (n, T, kmax)
        xn = F.normalize(xp, dim=-1) * self.scale * self.gamma_pad[:, None, :]
        y = torch.baddbmm(self.b_pad, xn, self.w_pad)           # (n, T, dim)
        return y.transpose(0, 1).unsqueeze(0)                   # (1, T, n, dim)


class PadBatchMaskEstimator(nn.Module):
    """Drop-in for the public MaskEstimator: (1, T, 60, dim) -> (1, T, sum K_i)."""

    def __init__(self, orig):
        super().__init__()
        self.orig = orig
        dims = tuple(int(d) for d in orig.dim_inputs)
        n, kmax = len(dims), max(dims)
        chains = []
        for mlp in orig.to_freqs:
            assert isinstance(mlp, nn.Sequential) and len(mlp) == 2
            inner, glu = mlp[0], mlp[1]
            assert isinstance(glu, nn.GLU) and glu.dim == -1
            layers = list(inner)
            lins = [m for m in layers if isinstance(m, nn.Linear)]
            others = [m for m in layers if not isinstance(m, nn.Linear)]
            assert all(isinstance(m, nn.Tanh) for m in others), others
            # public MLP: Linear (Tanh Linear)* — activation after all but last
            assert len(others) == len(lins) - 1
            assert [type(m) for m in layers] == \
                [nn.Linear if i % 2 == 0 else nn.Tanh for i in range(len(layers))]
            chains.append(lins)
        depth = len(chains[0])
        assert all(len(c) == depth for c in chains)
        dev = chains[0][0].weight.device
        self.n, self.kmax, self.depth = n, kmax, depth
        self.total = sum(dims)
        for j in range(depth - 1):             # batched hidden layers
            din, dout = chains[0][j].in_features, chains[0][j].out_features
            w = torch.zeros(n, din, dout)
            b = torch.zeros(n, 1, dout)
            for i, lins in enumerate(chains):
                assert lins[j].in_features == din and lins[j].out_features == dout
                w[i] = lins[j].weight.detach().float().cpu().t()
                b[i, 0] = lins[j].bias.detach().float().cpu()
            self.register_buffer(f'w{j}', w.to(dev), persistent=False)
            self.register_buffer(f'b{j}', b.to(dev), persistent=False)
        # last layer: out = 2*K_i, laid out [a | b] halves each padded to kmax
        din = chains[0][-1].in_features
        w = torch.zeros(n, din, 2 * kmax)
        b = torch.zeros(n, 1, 2 * kmax)
        out_idx = torch.zeros(self.total, dtype=torch.long)
        off = 0
        for i, (lins, k) in enumerate(zip(chains, dims)):
            last = lins[-1]
            assert last.in_features == din and last.out_features == 2 * k
            wt = last.weight.detach().float().cpu().t()         # (din, 2k)
            bb = last.bias.detach().float().cpu()
            w[i, :, :k] = wt[:, :k]
            w[i, :, kmax:kmax + k] = wt[:, k:2 * k]
            b[i, 0, :k] = bb[:k]
            b[i, 0, kmax:kmax + k] = bb[k:2 * k]
            out_idx[off:off + k] = i * kmax + torch.arange(k)
            off += k
        assert off == self.total
        self.register_buffer('w_last', w.to(dev), persistent=False)
        self.register_buffer('b_last', b.to(dev), persistent=False)
        self.register_buffer('out_idx', out_idx.to(dev), persistent=False)

    def forward(self, x):                      # (1, T, n, dim)
        assert x.shape[0] == 1 and x.shape[2] == self.n
        t = x.shape[1]
        h = x[0].transpose(0, 1)                                # (n, T, dim)
        for j in range(self.depth - 1):
            h = torch.tanh(torch.baddbmm(getattr(self, f'b{j}'), h,
                                         getattr(self, f'w{j}')))
        o = torch.baddbmm(self.b_last, h, self.w_last)           # (n, T, 2kmax)
        o = F.glu(o, dim=-1)                                     # (n, T, kmax)
        flat = o.transpose(0, 1).reshape(t, self.n * self.kmax)
        return flat[:, self.out_idx].unsqueeze(0)                # (1, T, total)


def install(model):
    """Swap band_split and every mask estimator; returns a record."""
    from models.bs_roformer.mel_band_roformer import BandSplit, MaskEstimator
    assert type(model.band_split) is BandSplit
    model.band_split = PadBatchBandSplit(model.band_split)
    names = []
    for i, me in enumerate(model.mask_estimators):
        assert type(me) is MaskEstimator
        model.mask_estimators[i] = PadBatchMaskEstimator(me)
        names.append(f'mask_estimators.{i}')
    return {'status': 'PASS', 'band_split': 'PadBatchBandSplit',
            'mask_estimators': names, 'kmax_band_split': model.band_split.kmax,
            'kmax_mask': model.mask_estimators[0].kmax,
            'mask_depth': model.mask_estimators[0].depth,
            'production_code_used': False}
