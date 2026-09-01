"""Shared machinery for arms A-G (prereg arms table). Everything here traces
to prereg_null_test.md.pdf; PREREG-cited comments mark the constraint encoded.

Layout (in audit-map order — see README):
    build ......................... seeded model construction, identical weights
                                    for every arm and the eager reference
    band_split substitutions ...... FusedBandSplit (D/E/G), Recording/Replay
                                    wrappers for the capture fallback
    capture_graph ................. CUDA Graphs protocol (B/E/G): static I/O
                                    buffers, side-stream warmup, capture outside
                                    the timed region (per torch CUDA Graphs docs)
    build_arm_step ................ arm -> step(x) closure + capture-scope record
    settle_compile ................ C1/C2: run until recompilation stops (logged)
    validate ...................... AUDIT-MAP[2]: >=2-pass different-data check
    timed_run ..................... AUDIT-MAP[1]: prereg 5a timing protocol
    result schema / env ........... one JSON per run

frontend.py is imported, never modified: arm variants swap the fe.band_split
submodule from outside (public nn.Module attribute assignment), so stages 1-5
and every NVTX stage name are frontend.py's own code on every arm.
"""

import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

# repo root (frontend.py, frontend_rms_fused.py) and arms/ (streams.py)
_ARMS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_ARMS_DIR)
for _p in (_REPO_ROOT, _ARMS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn.functional as F

import frontend                              # provided harness — canonical model
from frontend_rms_fused import rms_fused     # session owner's fused RMSNorm kernel
import streams

ARMS = ('A', 'B', 'C1', 'C2', 'D', 'E', 'F', 'G')
CHUNKS = (352800, 4410)          # PREREG: the two regimes — 8 s and 100 ms
DEFAULT_ITERS = 50               # PREREG: >= 50 timed iters after warmup
DEFAULT_WARMUP = 10              # matches frontend.py WARMUP_ITERS
DEFAULT_STREAMS = 4              # arms F/G default (session-owner spec)

# Two-tier validation tolerance, keyed by arm class (prereg amendment /
# fix-pass 2026-08-18 — session-owner rationale recorded verbatim below):
#
#   REPLAY class (tight, rtol 1e-4 / atol 1e-5): the only kernel computing
#   different arithmetic from eager is the fused front-end, and its risky
#   reductions accumulate in fp32 on both sides (tl.dot's fp32 accumulator vs
#   cuBLAS's fp32 accumulation); everything downstream of a capture is
#   bitwise-identical binaries — captured, not regenerated — so the tiny
#   perturbation propagates through identical arithmetic with no fresh noise.
#   Replay arms keep the tight gate because B measured bitwise and G's Triton
#   front-end measured within 1e-4 in the 2026-08-17 smoke (fp32 accumulation
#   both sides). D, E and F join this class (D/F: eager variants of the same
#   kernels; E: fused + replay — the same numeric path as G, minus streams).
#
#   RECOMPILED class (loose, rtol 1e-2 / atol 1e-3): fp16 eps ~ 9.77e-4;
#   Inductor legitimately reorders reductions over K <= 512, honest drift
#   ~ sqrt(K)*eps ~ 1e-2 relative; atol governs near-silent stem regions.
#   Divergence is injected at every regenerated kernel and compounds with
#   depth: different reduction trees, different fusion boundaries, fused
#   fp32-register chains rounding in different places than eager's
#   materialized intermediates, and libdevice approximations that don't match
#   eager bit-for-bit.
#
#   CLASSIFICATION RULE (the tier assignment IS this rule, not an arm list):
#   tier = compiler-regenerated kernels present or not. Corrected 2026-09-01:
#   E was initially placed loose from the owner's arm-ledger memory, but
#   repo-E as built is fused-Triton + whole-forward replay of the fp32
#   front-end — NO Inductor-generated kernels on its path — so the rule puts
#   it in the tight tier; the rule outranks the ledger. If the amended prereg
#   later redefines E as an Inductor arm, build_arm_step's E branch and this
#   tier assignment move together under the same rule — flag it then.
VAL_TOL = {                      # arm -> (rtol, atol)
    'A':  (1e-4, 1e-5), 'B':  (1e-4, 1e-5), 'D': (1e-4, 1e-5),
    'E':  (1e-4, 1e-5), 'F':  (1e-4, 1e-5), 'G': (1e-4, 1e-5),
    'C1': (1e-2, 1e-3), 'C2': (1e-2, 1e-3),
}
VAL_SEEDS = (1001, 2002)         # two passes, DIFFERENT data (asserted)
TIMING_SEED = 0


# ---------------------------------------------------------------- build ----

def build_stock_frontend(device):
    """manual_seed(0) then FrontEnd(): identical weights for every arm model
    and for the eager reference (run_arm.py asserts state_dict equality)."""
    torch.manual_seed(0)
    return frontend.FrontEnd().to(device).eval()


def make_input(chunk, device, seed):
    """Deterministic input, independent of device RNG state: drawn on CPU from
    a dedicated generator, then moved. Same seed => same data in every process
    and on every arm."""
    gen = torch.Generator(device='cpu').manual_seed(seed)
    x = torch.randn(frontend.BATCH, frontend.CHANNELS, chunk, generator=gen)
    return x.to(device)


# ---------------------------------------------- band_split substitutions ----

def fused_band_fns(band_split):
    """One closure per band: session owner's rms_fused kernel + F.linear, on
    the ORIGINAL parameter tensors (no copies)."""
    fns = []
    for seq in band_split.to_features:       # nn.Sequential(RMSNorm, Linear)
        rms, lin = seq[0], seq[1]
        def fn(part, _g=rms.gamma, _s=rms.scale, _w=lin.weight, _b=lin.bias):
            return F.linear(rms_fused(part, _g, _s), _w, _b)
        fns.append(fn)
    return fns


class FusedBandSplit(torch.nn.Module):
    """Arm D/E/G band_split: per-band fused-RMSNorm + Linear, serial loop
    (arm G wraps the same closures in MultiStreamBandSplit instead)."""

    def __init__(self, orig):
        super().__init__()
        self.orig = orig                     # keeps original parameters registered
        self.band_fns = fused_band_fns(orig)

    def forward(self, x):
        # rms_fused squeezes the batch dim (see frontend_rms_fused.rms_fused)
        assert x.shape[0] == 1, 'rms_fused wrapper assumes batch size 1'
        parts = x.split(self.orig.dim_inputs, dim=-1)
        outs = [fn(p) for fn, p in zip(self.band_fns, parts)]
        return torch.stack(outs, dim=-2)     # same as frontend.BandSplit.forward


class RecordingBandSplit(torch.nn.Module):
    """Setup-time-only wrapper: remembers band_split's input tensor (the folded
    spectrogram) so the capture fallback knows the static buffer's shape."""

    def __init__(self, orig):
        super().__init__()
        self.orig = orig
        self.last_input = None

    def forward(self, x):
        self.last_input = x
        return self.orig(x)


class ReplayBandSplit(torch.nn.Module):
    """Capture-fallback band_split (prereg P4 pre-authorized scope): stages 1-5
    stay eager in frontend.py's own forward; this module copies the folded
    tensor into the static input buffer and replays the captured graph."""

    def __init__(self, orig, graph, static_in, static_out):
        super().__init__()
        self.orig = orig                     # parameters stay registered
        self.graph = graph
        self.static_in = static_in
        self.static_out = static_out

    def forward(self, x):
        self.static_in.copy_(x)              # fresh data into the static buffer
        self.graph.replay()
        return self.static_out               # reused buffer — validate() clones


# -------------------------------------------------------------- capture ----

def capture_graph(fn, example, warmup_iters=3):
    """PREREG graphs mechanics: static I/O buffers, warmup on a side stream,
    capture OUTSIDE the timed region (this runs at build time; timed_run is
    called later). Pattern follows the torch CUDA Graphs docs.

    Returns (graph, static_in, static_out). Raises on capture failure —
    build_arm_step() catches and falls back per prereg P4.
    """
    static_in = example.detach().clone()     # private contiguous input buffer
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.no_grad():
        for _ in range(warmup_iters):
            fn(static_in)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        static_out = fn(static_in)
    return graph, static_in, static_out


def _capture_with_fallback(fe, x_example, info):
    """B/E/G shared protocol. Try whole-forward capture; on failure (prereg P4
    predicts the complex STFT breaks it) fall back to capturing the band_split
    stage only. The scope that actually captured is RECORDED in the result
    JSON (info['capture_scope']) and stated in the log."""
    try:
        graph, static_in, static_out = capture_graph(fe, x_example)
        info['capture_scope'] = 'whole_forward'

        def step(x):
            static_in.copy_(x)               # AUDIT-MAP[2]: fresh input each replay
            graph.replay()
            return static_out
        return step
    except Exception:
        info['notes'].append(
            'whole-forward capture FAILED (prereg P4 predicted this at the '
            'complex STFT); falling back to band_split-only capture. '
            'Traceback:\n' + traceback.format_exc())
        # a failed capture can leave the ambient stream pointing at the graph
        # ctx's internal capture stream (torch 2.8: graph.__exit__ ends the
        # capture BEFORE restoring the stream, with no try/finally) — restore
        # the default stream before building the fallback
        torch.cuda.set_stream(torch.cuda.default_stream())
        torch.cuda.synchronize()

    # fallback: find the folded tensor's shape with one recorded eager pass,
    # then capture ONLY the current band_split module (stock, fused, or
    # multi-stream — whatever this arm installed)
    band_split = fe.band_split
    recorder = RecordingBandSplit(band_split)
    fe.band_split = recorder
    with torch.no_grad():
        fe(x_example)
    folded = recorder.last_input.detach().clone()
    fe.band_split = band_split

    graph, static_in, static_out = capture_graph(band_split, folded)
    fe.band_split = ReplayBandSplit(band_split, graph, static_in, static_out)
    info['capture_scope'] = 'band_split_only'
    return lambda x: fe(x)


# -------------------------------------------------------------- the arms ----

def build_arm_step(arm, fe, chunk, device, num_streams, info):
    """Turn a freshly built stock FrontEnd into the requested arm.

    Returns step: step(x) -> output tensor runs ONE pass. `info` is the
    CALLER'S dict (run_arm.py) and is mutated in place — capture_scope
    (B/E/G), stream_count (F/G), compile mode (C1/C2), notes — so that
    everything recorded before a mid-build exception (e.g. the P4
    whole-forward-capture traceback) still reaches the result JSON.
    All build/compile/capture work happens HERE, before validation and before
    the timed region (AUDIT-MAP[1]).
    """
    x_example = make_input(chunk, device, TIMING_SEED)

    if arm == 'A':                # eager baseline, stock F.normalize chain
        return lambda x: fe(x)

    if arm == 'B':                # A + manual whole-forward capture (P4 fallback)
        return _capture_with_fallback(fe, x_example, info)

    if arm in ('C1', 'C2'):       # PREREG: torch.compile(..., dynamic=False)
        mode = 'reduce-overhead' if arm == 'C1' else 'max-autotune'
        info['compile_mode'] = mode
        compiled = torch.compile(fe, mode=mode, dynamic=False)
        return lambda x: compiled(x)

    if arm == 'D':                # fused RMSNorm kernel, eager elsewhere
        fe.band_split = FusedBandSplit(fe.band_split)
        return lambda x: fe(x)

    if arm == 'E':                # D + capture (same fallback protocol as B)
        fe.band_split = FusedBandSplit(fe.band_split)
        return _capture_with_fallback(fe, x_example, info)

    if arm == 'F':                # A + multi-stream band loop
        info['stream_count'] = num_streams
        loop = streams.MultiStreamBandLoop(num_streams, device)
        stock_fns = list(fe.band_split.to_features)      # original modules
        fe.band_split = streams.MultiStreamBandSplit(fe.band_split, stock_fns, loop)
        return lambda x: fe(x)

    if arm == 'G':
        # AUDIT-MAP[3]: honest counterfactual — fused kernel + multi-stream
        # fork/join band loop, the whole stream-ordered structure captured
        # into ONE graph (whole-forward first, band_split-only on P4 fallback;
        # either way a single graph replay covers the multi-stream loop).
        info['stream_count'] = num_streams
        loop = streams.MultiStreamBandLoop(num_streams, device)
        fns = fused_band_fns(fe.band_split)
        fe.band_split = streams.MultiStreamBandSplit(fe.band_split, fns, loop)
        return _capture_with_fallback(fe, x_example, info)

    raise ValueError(f'unknown arm {arm!r}')


# ------------------------------------------------------------- settling ----

def settle_compile(step, x, max_iters=60, consecutive=5):
    """PREREG: 'let autotune fully settle before timing' — run the compiled
    model until recompilation stops, and log it. Detector: torch._dynamo's
    counters unchanged for `consecutive` successive passes (compilations,
    graph breaks and recompiles all move these counters). Runs BEFORE
    validation and the timed region. TORCH_LOGS=recompiles,perf_hints goes to
    stderr and is captured to evidence/ by run_all.sh."""
    import torch._dynamo.utils as dynamo_utils

    def snap():
        # str() keys: the snapshot lands in the result JSON, and counter keys
        # are not guaranteed to be strings on every torch version
        return {str(k): {str(kk): int(vv) for kk, vv in v.items()}
                for k, v in dynamo_utils.counters.items()}

    t0 = time.perf_counter()
    prev, stable, it = snap(), 0, 0
    with torch.no_grad():
        while stable < consecutive and it < max_iters:
            step(x)
            it += 1
            cur = snap()
            stable = stable + 1 if cur == prev else 0
            prev = cur
    torch.cuda.synchronize()
    return {'settle_iters': it,
            'settle_stable_tail': stable,
            'settle_capped': stable < consecutive,    # hit max_iters still unsettled
            'settle_wall_s': round(time.perf_counter() - t0, 3),
            'dynamo_counters': prev}


# ------------------------------------------------------------ validation ----

def validate(step, reference, arm, chunk, device):
    """AUDIT-MAP[2]. PREREG: >= 2-pass different-data validation on EVERY arm,
    including D/E/G — stale-buffer aliasing under capture passes single-pass
    checks while being wrong. Runs OUTSIDE the timed region (before it).

    Tolerance is per arm class (VAL_TOL — two-tier, prereg amendment
    2026-08-18): replay class tight, recompiled class loose. NOTE: C1/C2/E
    have never been validated; first validation is a pod-session opening item
    (D and F likewise have never run on hardware).

    Pass 2 genuinely uses different data: distinct seeds, and torch.equal
    inequality is asserted. For capture arms the step closure performs
    static_in.copy_(x) before graph replay (see _capture_with_fallback /
    ReplayBandSplit), so a broken copy path shows up here as pass 2 returning
    pass 1's answer. A FAIL marks the run INVALID in the JSON; it never
    crashes the session (run_arm.py catches and records)."""
    rtol, atol = VAL_TOL[arm]
    passes = []
    prev_x = None 
    for seed in VAL_SEEDS:
        x = make_input(chunk, device, seed)
        if prev_x is not None:
            assert not torch.equal(x, prev_x), \
                'validation passes must use different input data'
        prev_x = x
        with torch.no_grad():
            ref = reference(x)
            out = step(x)
            torch.cuda.synchronize()
            out = out.clone()    # replay arms reuse the output buffer — snapshot
        ok = bool(torch.allclose(out, ref, rtol=rtol, atol=atol))
        passes.append({'seed': seed, 'ok': ok,
                       'max_abs_diff': float((out - ref).abs().max().item())})
    return {'status': 'PASS' if all(p['ok'] for p in passes) else 'FAIL',
            'rtol': rtol, 'atol': atol, 'tolerance_class':
                ('recompiled' if arm in ('C1', 'C2') else 'replay'),
            'passes': passes}


# ---------------------------------------------------------------- timing ----

def timed_run(step, x, iters):
    """AUDIT-MAP[1]. PREREG constraint 5a — TIMING PROTOCOL (cited verbatim
    per session-owner instruction; see prereg_null_test.md.pdf, methodology).

    PRIMARY (defines "wall" for the decision rule): per-iteration LATENCY —
    torch.cuda.synchronize() after every pass, per-iteration times returned
    (run_arm.py prints and saves the full list; never only aggregates).
    Rationale (5a): streaming semantics — chunks arrive on real-time cadence,
    back-to-back pipelining is unavailable in deployment, so latency (not
    throughput) is the decision quantity in the 4410 regime.

    SECONDARY: one free-running block (sync, N iters, sync; total/N) recorded
    as throughput_ms_per_pass — the honest offline metric for the 8 s regime,
    the continuity bridge to the historical free-running baseline (5.5-7.5 ms),
    and the empirical measure of the serialization delta. 5a known bias:
    per-iteration sync can only inflate G's wall -> inflates R -> biases
    AGAINST the kill verdict; expected < 2%. 5a escalation clause: if the
    free-running delta at the 4410 regime exceeds 5%, the decision rule is
    evaluated under BOTH protocols and any disagreement is escalated
    (analysis/collect.py surfaces serialization_delta_pct per arm x regime).

    Timed-region boundary (PREREG): model build, compile/autotune settling,
    graph capture and warmup all happen BEFORE this function — see
    run_arm.py main(). The nsys capture range (cudaProfilerApi, frontend.py's
    pattern) brackets the LATENCY block only, so a traced run's kernel sum
    pairs with that same run's latency wall — "busy from nsys kernel sum,
    same run" (prereg decision rule).
    """
    with torch.no_grad():
        torch.cuda.synchronize()
        torch.cuda.profiler.start()              # nsys capture range: open
        latency_ms = []
        for _ in range(iters):
            t0 = time.perf_counter()
            step(x)
            torch.cuda.synchronize()             # 5a PRIMARY: sync every pass
            latency_ms.append((time.perf_counter() - t0) * 1e3)
        torch.cuda.profiler.stop()               # nsys capture range: close

        torch.cuda.synchronize()                 # 5a SECONDARY: free-running
        t0 = time.perf_counter()
        for _ in range(iters):
            step(x)
        torch.cuda.synchronize()
        throughput_ms = (time.perf_counter() - t0) * 1e3 / iters
    return latency_ms, throughput_ms


# --------------------------------------------------------- result schema ----

def env_fingerprint():
    try:
        driver = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
    except Exception:
        driver = 'unavailable'
    import triton
    return {'torch': torch.__version__,
            'triton': triton.__version__,
            'cuda_build': torch.version.cuda,
            'driver': driver,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none',
            'capability': list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
            'python': platform.python_version(),
            'hostname': platform.node()}


def result_json(arm, chunk, tag, iters, latency_ms, throughput_ms, validation,
                info, settle, status, error=None):
    """One JSON per run — the prereg result schema (session-owner list):
    arm, chunk, versions, GPU, timestamp, iters, per-iter list, median/min/max,
    validation status, capture scope, stream count, notes; plus the 5a fields
    latency_*/throughput_ms_per_pass/serialization_delta_pct."""
    med = mn = mx = delta = None
    if latency_ms:
        s = sorted(latency_ms)
        n = len(s)
        med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
        mn, mx = s[0], s[-1]
        if throughput_ms:
            delta = (med - throughput_ms) / throughput_ms * 100.0
    return {
        'arm': arm,
        'chunk': chunk,
        'tag': tag,
        'status': status,                        # VALID | INVALID
        'error': error,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'argv': sys.argv,
        'env': env_fingerprint(),
        'torch_logs_env': os.environ.get('TORCH_LOGS'),
        'iters': iters,
        'latency_ms_per_iter': latency_ms,       # PREREG: per-iteration list, always
        'latency_median_ms': med,
        'latency_min_ms': mn,
        'latency_max_ms': mx,
        'throughput_ms_per_pass': throughput_ms, # PREREG 5a secondary
        'serialization_delta_pct': delta,        # PREREG 5a escalation input
        'validation': validation,
        'capture_scope': info.get('capture_scope'),
        'stream_count': info.get('stream_count'),
        'compile_mode': info.get('compile_mode'),
        'settle': settle,
        'notes': info.get('notes', []),
    }
