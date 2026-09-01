"""Standalone MelBandRoformer front-end, for kernel-level profiling with nsys.

Reproduces — with public PyTorch ops only, no repo code on the profiled path —
the exact op chain of MelBandRoformer.forward() up to and including band_split:

    window       torch.hann_window   regenerated on-device EVERY forward (the model
                                     does this too — watch for it in the trace)
    stft         torch.stft          center-pad (reflect) + framing*window + batched
                                     FFT; the FFT lands in cuFFT library kernels
    view_as_real / merge stereo      complex -> (re, im), stereo folded into freq dim
    band gather  advanced indexing   mel-scaled band grouping via freq_indices;
                                     overlapping bands DUPLICATE frequency bins
    fold complex into freq           permute + copy
    band_split   per-band RMSNorm + Linear(dim_in -> dim), a Python loop over
                                     num_bands skinny matmuls, then stack

NOTE: this model keeps the complex STFT (re, im) — it never takes magnitude,
and the "mel" is only in how bins are grouped into bands. There is no mel matmul.

Config matches config/config_aug.yaml — the original/teacher architecture, same
front-end as the public MelBandRoformer vocal checkpoint (provenance-clean for
publication):
    sample_rate 44100, n_fft 2048, hop 441, win 2048, normalized False,
    stereo, num_bands 60, dim 384, chunk_size 352800 (8 s), fp32.

Profile (only the measured iterations are captured):

    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
        -o frontend_trace python frontend.py
    nsys stats -r cuda_gpu_kern_sum frontend_trace.nsys-rep

Sanity check vs the repo model's front-end (imports repo code; run from repo root,
works on CPU too):

    python frontend.py --sanity
"""

import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa import filters

# config/config_aug.yaml (original/teacher — public-model front-end)
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 441
WIN_LENGTH = 2048
NORMALIZED = False
NUM_BANDS = 60
DIM = 384
CHUNK_SIZE = 352800  # samples (8 s @ 44.1 kHz); override with --chunk N
                     # e.g. --chunk 4410 = 100 ms streaming-sized chunk
BATCH = 1
CHANNELS = 2  # stereo

WARMUP_ITERS = 10
MEASURED_ITERS = 20

if torch.cuda.is_available():
    _nvtx_push, _nvtx_pop = torch.cuda.nvtx.range_push, torch.cuda.nvtx.range_pop
else:
    _nvtx_push, _nvtx_pop = (lambda name: None), (lambda: None)


def build_band_indices():
    """Mel-scaled band grouping, identical to MelBandRoformer.__init__ (setup-time
    only — never on the profiled path)."""
    mel_fb = filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=NUM_BANDS)
    mel_fb[0, 0] = 1.0
    mel_fb[-1, -1] = 1.0
    freqs_per_band = mel_fb > 0  # (num_bands, n_fft//2+1)
    assert freqs_per_band.any(axis=0).all(), 'every freq bin must belong to some band'

    freq_indices = np.concatenate([np.flatnonzero(row) for row in freqs_per_band])
    # stereo was merged into the freq dim as (f s): bin f -> rows f*2 (L) and f*2+1 (R)
    freq_indices = np.stack([freq_indices * 2, freq_indices * 2 + 1], axis=-1).reshape(-1)

    num_freqs_per_band = freqs_per_band.sum(axis=1)
    dim_inputs = tuple(int(2 * CHANNELS * f) for f in num_freqs_per_band)  # 2 = (re, im)
    return torch.from_numpy(freq_indices).long(), dim_inputs


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class BandSplit(nn.Module):
    def __init__(self, dim, dim_inputs):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = nn.ModuleList(
            nn.Sequential(RMSNorm(dim_in), nn.Linear(dim_in, dim)) for dim_in in dim_inputs
        )

    def forward(self, x):
        parts = x.split(self.dim_inputs, dim=-1)
        outs = [f(part) for part, f in zip(parts, self.to_features)]
        return torch.stack(outs, dim=-2)


class FrontEnd(nn.Module):
    def __init__(self):
        super().__init__()
        freq_indices, dim_inputs = build_band_indices()
        self.register_buffer('freq_indices', freq_indices, persistent=False)
        self.band_split = BandSplit(DIM, dim_inputs)

    def forward(self, raw_audio):  # (b, s, t)
        b, s, t = raw_audio.shape
        device = raw_audio.device

        _nvtx_push('1_window')
        window = torch.hann_window(WIN_LENGTH, device=device)
        _nvtx_pop()

        _nvtx_push('2_stft')
        spec = torch.stft(
            raw_audio.reshape(b * s, t),
            n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
            normalized=NORMALIZED, window=window, return_complex=True,
        )  # (b*s, f, frames) complex
        _nvtx_pop()

        _nvtx_push('3_view_as_real_merge_stereo')
        spec = torch.view_as_real(spec)  # (b*s, f, frames, 2)
        f, frames = spec.shape[1], spec.shape[2]
        # 'b s f t c -> b (f s) t c'
        spec = (
            spec.reshape(b, s, f, frames, 2)
            .permute(0, 2, 1, 3, 4)
            .reshape(b, f * s, frames, 2)
        )
        _nvtx_pop()

        _nvtx_push('4_band_gather')
        batch_arange = torch.arange(b, device=device)[:, None]
        x = spec[batch_arange, self.freq_indices]  # (b, F_gather, frames, 2)
        _nvtx_pop()

        _nvtx_push('5_fold_complex_into_freq')
        # 'b f t c -> b t (f c)'
        x = x.permute(0, 2, 1, 3).reshape(b, frames, -1)
        _nvtx_pop()

        _nvtx_push('6_band_split_norm_linear')
        x = self.band_split(x)  # (b, frames, num_bands, dim)
        _nvtx_pop()
        return x


def cli_chunk():
    """Chunk size in samples from --chunk N, else the 8 s default. The op chain
    has a hard floor: center=True reflect-pads n_fft/2 = 1024 samples each side,
    and reflect padding requires pad < signal length."""
    chunk = CHUNK_SIZE
    if '--chunk' in sys.argv:
        chunk = int(sys.argv[sys.argv.index('--chunk') + 1])
    assert chunk > N_FFT // 2, \
        f'chunk must be > {N_FFT // 2} samples: center=True reflect padding ' \
        f'cannot exceed the signal length'
    return chunk


def sanity(device, chunk):
    """Bitwise-compare this file's front-end against the repo model's own
    buffers/modules driven by its forward() lines. Imports repo code — keep
    this OFF the profiled path."""
    from einops import rearrange
    from models.bs_roformer.mel_band_roformer import MelBandRoformer

    torch.manual_seed(0)
    # depth / mask_estimator_depth don't influence the front-end; keep them
    # minimal so instantiation is cheap
    model = MelBandRoformer(
        dim=DIM, depth=1, stereo=True, num_stems=1, num_bands=NUM_BANDS,
        sample_rate=SAMPLE_RATE, stft_n_fft=N_FFT, stft_hop_length=HOP_LENGTH,
        stft_win_length=WIN_LENGTH, stft_normalized=NORMALIZED,
        flash_attn=False, mask_estimator_depth=1,
    ).to(device).eval()

    fe = FrontEnd().to(device).eval()

    assert torch.equal(fe.freq_indices, model.freq_indices), 'freq_indices mismatch'
    assert tuple(fe.band_split.dim_inputs) == tuple(model.band_split.dim_inputs), \
        'band dim_inputs mismatch'
    fe.band_split.load_state_dict(model.band_split.state_dict())

    x = torch.randn(BATCH, CHANNELS, chunk, device=device)
    with torch.no_grad():
        mine = fe(x)

        # the model's forward() front-end lines, using its own modules/buffers
        window = model.stft_window_fn(device=device)
        ref = torch.stft(x.reshape(BATCH * CHANNELS, -1), **model.stft_kwargs,
                         window=window, return_complex=True)
        ref = torch.view_as_real(ref)
        ref = ref.reshape(BATCH, CHANNELS, *ref.shape[1:])
        ref = rearrange(ref, 'b s f t c -> b (f s) t c')
        ref = ref[torch.arange(BATCH, device=device)[..., None], model.freq_indices]
        ref = rearrange(ref, 'b f t c -> b t (f c)')
        ref = model.band_split(ref)

    diff = (mine - ref).abs().max().item()
    print(f'sanity [{device}]: shapes {tuple(mine.shape)} vs {tuple(ref.shape)}, '
          f'max abs diff = {diff:.3e}')
    assert mine.shape == ref.shape and diff == 0.0, 'front-end diverges from repo model'
    print('sanity: bitwise exact')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    chunk = cli_chunk()

    if '--sanity' in sys.argv:
        sanity(device, chunk)
        return

    torch.manual_seed(0)
    fe = FrontEnd().to(device).eval()
    x = torch.randn(BATCH, CHANNELS, chunk, device=device)

    dims = fe.band_split.dim_inputs
    print(f'device={device}  input (b s t)={tuple(x.shape)}  '
          f'({1e3 * chunk / SAMPLE_RATE:.1f} ms of audio)  fp32')
    print(f'stft: n_fft={N_FFT} hop={HOP_LENGTH} -> freqs={N_FFT // 2 + 1} '
          f'frames={1 + chunk // HOP_LENGTH}')
    print(f'gather: {len(fe.freq_indices)} rows from '
          f'{CHANNELS * (N_FFT // 2 + 1)} unique freq*channel rows '
          f'(mel band overlap duplicates bins)')
    print(f'band_split: {NUM_BANDS} bands, per-band in-dim min={min(dims)} '
          f'max={max(dims)}, out dim={DIM}')

    with torch.no_grad():
        for _ in range(WARMUP_ITERS):  # cuFFT plans, allocator, autotune
            fe(x)

        if device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.profiler.start()  # opens the nsys capture range
            t0 = time.perf_counter()
            for _ in range(MEASURED_ITERS):
                out = fe(x)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            torch.cuda.profiler.stop()
        else:
            t0 = time.perf_counter()
            for _ in range(MEASURED_ITERS):
                out = fe(x)
            elapsed = time.perf_counter() - t0

    print(f'output (b t bands dim)={tuple(out.shape)}')
    print(f'{MEASURED_ITERS} passes: {1e3 * elapsed / MEASURED_ITERS:.3f} ms/pass')


if __name__ == '__main__':
    main()
