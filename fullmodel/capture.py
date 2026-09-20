"""Whole-forward CUDA-graph capture ladder for arms B/E/G (prereg_amendments §1).

Ladder (scope RECORDED in the run JSON, every failure's traceback kept):
  1. stock forward as-is;
  2. the single external constant-hoist adapter: the forward's DC-filter index
     `tensor(0, device=device)` is replaced by a pre-allocated device tensor.
     A pageable host->device copy is illegal under stream capture — verified
     empirically on ml-beast (torch 2.8.0+cu128): capturing
     `x.index_fill(1, torch.tensor(0, device='cuda'), 0.)` fails with
     "operation failed due to a previous error during capture". The adapter is
     an AST-verified single-expression substitution of the public forward
     (same pattern as axial-layout-scoping extensions/c3/native_tail.py); no
     arithmetic changes;
  2b. the iSTFT boundary adapter: with the DC index hoisted, capture still
     fails at capture_end (ml-beast shakedown 2026-09-20). torch.istft performs
     a host-side window-envelope check (ATen istft: the envelope minimum is
     read with .item()) — a device->host sync, illegal under stream capture
     (the C3 study recorded the same: native iSTFT is capture-unsafe). Rung 2b
     keeps rung 2 and additionally replaces the forward's single
     `torch.istft(...)` call by `_capture_istft(...)` with the SAME argument
     list (AST-verified; reverse substitution restores the pinned source).
     During capture, _capture_istft stashes its input (the graph's static
     pre-iSTFT complex spectrogram) and returns a placeholder; each replay
     is followed by the SAME public torch.istft call run eagerly on that
     static buffer, then the forward's own post-iSTFT rearranges. Scope label:
     'whole_forward_to_istft' — everything up to the iSTFT is graphed, the
     iSTFT + two reshapes are eager. No arithmetic changes.
  3. band_split-only capture (prereg P4 fallback), via the sm120 Recording/
     ReplayBandSplit wrappers.

Capture mechanics = sm120 arms/common.py::capture_graph (static I/O buffers,
side-stream warmup, capture outside the timed region) with the repo's
autocast context, cache_enabled=False during capture (torch AMP+graphs
guidance).
"""

import ast
import hashlib
import inspect
import os
import textwrap
import traceback
import types

import torch

import common   # sm120 arms/common.py (Recording/ReplayBandSplit) — on sys.path via fullarms


def capture_graph_amp(fn, example, ctx, warmup_iters=3):
    """sm120 capture_graph with the autocast context on every pass."""
    static_in = example.detach().clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.no_grad(), ctx():
        for _ in range(warmup_iters):
            fn(static_in)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), ctx(), torch.cuda.graph(graph):
        static_out = fn(static_in)
    return graph, static_in, static_out


def _reset_after_failed_capture():
    # torch 2.8: graph.__exit__ can leave the ambient stream on the capture
    # stream after a failure (sm120 _capture_with_fallback note)
    torch.cuda.set_stream(torch.cuda.default_stream())
    torch.cuda.synchronize()


def install_dc_index_hoist(model, out_dir):
    """External forward copy differing ONLY in the DC-filter index expression."""
    from models.bs_roformer.mel_band_roformer import MelBandRoformer
    assert type(model) is MelBandRoformer
    assert 'forward' not in model.__dict__, 'forward already replaced'
    original = textwrap.dedent(inspect.getsource(MelBandRoformer.forward))
    tree = ast.parse(original)
    replacements = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and
                ast.unparse(node.func) == 'stft_repr.index_fill'):
            assert len(node.args) == 3 and not node.keywords
            assert ast.unparse(node.args[0]) == '1'
            assert ast.unparse(node.args[1]) == 'tensor(0, device=device)'
            assert isinstance(node.args[2], ast.Constant) and node.args[2].value == 0.
            node.args[1] = ast.Name(id='_capture_dc_index', ctx=ast.Load())
            replacements += 1
    assert replacements == 1, 'expected the single public DC-filter call'
    source = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    restored = ast.parse(source)
    for node in ast.walk(restored):
        if (isinstance(node, ast.Call) and
                ast.unparse(node.func) == 'stft_repr.index_fill'):
            node.args[1] = ast.parse('tensor(0, device=device)', mode='eval').body
    assert ast.dump(restored) == ast.dump(ast.parse(original)), \
        'external forward differs beyond the index hoist'
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, 'generated_capture_forward.py')
    with open(dest, 'w') as f:
        f.write(source)
    namespace = dict(MelBandRoformer.forward.__globals__)
    namespace['_capture_dc_index'] = torch.tensor(0, device='cuda')
    exec(compile(source, dest, 'exec'), namespace)
    model.forward = types.MethodType(namespace['forward'], model)
    return {'status': 'PASS', 'replacements': replacements,
            'original_forward_sha256': hashlib.sha256(original.encode()).hexdigest(),
            'external_forward_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'generated_source': dest,
            'scope': 'public forward differs only in the DC-filter index expression'}


class _IstftBoundary:
    """Capture-time stand-in for torch.istft (rung 2b). Same signature."""

    def __init__(self):
        self.stash = None

    def __call__(self, input, *args, **kwargs):
        if torch.cuda.is_current_stream_capturing():
            self.stash = {'input': input, 'args': args, 'kwargs': dict(kwargs)}
            length = kwargs.get('length')
            if length is None:
                # torch.istft output length with center=True: hop * (frames - 1)
                assert kwargs.get('center', True), 'istft boundary assumes center=True'
                length = int(kwargs['hop_length']) * (int(input.shape[-1]) - 1)
            self.placeholder_len = length
            return torch.zeros(input.shape[0], length, device=input.device,
                               dtype=torch.float32)
        return torch.istft(input, *args, **kwargs)


def install_istft_boundary(model, out_dir, boundary):
    """External forward copy differing ONLY in the iSTFT callee (on top of the
    DC-index hoist, which must already be installed)."""
    from models.bs_roformer.mel_band_roformer import MelBandRoformer
    assert type(model) is MelBandRoformer
    assert 'forward' in model.__dict__, 'install the DC-index hoist first'
    src_path = os.path.join(out_dir, 'generated_capture_forward.py')
    original = open(src_path).read()
    tree = ast.parse(original)
    replacements = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == 'torch.istft':
            assert ast.unparse(node.args[0]) == 'stft_repr'
            assert any(k.arg == 'length' for k in node.keywords)
            node.func = ast.Name(id='_capture_istft', ctx=ast.Load())
            replacements += 1
    assert replacements == 1, 'expected the single public torch.istft call'
    source = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    restored = ast.parse(source)
    for node in ast.walk(restored):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == '_capture_istft':
            node.func = ast.parse('torch.istft', mode='eval').body
    assert ast.dump(restored) == ast.dump(ast.parse(original)), \
        'external forward differs beyond the iSTFT callee'
    dest = os.path.join(out_dir, 'generated_capture_forward_istft.py')
    with open(dest, 'w') as f:
        f.write(source)
    namespace = dict(MelBandRoformer.forward.__globals__)
    namespace['_capture_dc_index'] = torch.tensor(0, device='cuda')
    namespace['_capture_istft'] = boundary
    exec(compile(source, dest, 'exec'), namespace)
    model.forward = types.MethodType(namespace['forward'], model)
    return {'status': 'PASS', 'replacements': replacements,
            'external_forward_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'generated_source': dest,
            'scope': 'public forward differs only in the DC-filter index expression '
                     'and the iSTFT callee (same argument list)'}


def capture_ladder(model, x_example, ctx_nocache, info, out_dir):
    """Returns step(x). Mutates info: capture_scope, capture_adapter,
    capture_attempts (tracebacks of the failed rungs)."""
    info['capture_attempts'] = []

    def whole_step(graph, static_in, static_out):
        def step(x):
            static_in.copy_(x)           # AUDIT-MAP[2]: fresh input each replay
            graph.replay()
            return static_out            # borrowed — validate() clones
        return step

    try:
        graph, sin, sout = capture_graph_amp(model, x_example, ctx_nocache)
        info['capture_scope'] = 'whole_forward'
        info['capture_adapter_active'] = 'stock'
        return whole_step(graph, sin, sout)
    except Exception:
        info['capture_attempts'].append(
            {'rung': 'whole_forward_stock', 'traceback': traceback.format_exc()})
        _reset_after_failed_capture()

    try:
        info['capture_adapter'] = install_dc_index_hoist(model, out_dir)
        graph, sin, sout = capture_graph_amp(model, x_example, ctx_nocache)
        info['capture_scope'] = 'whole_forward+dc_index_hoist'
        info['capture_adapter_active'] = 'dc_hoist'
        return whole_step(graph, sin, sout)
    except Exception:
        info['capture_attempts'].append(
            {'rung': 'whole_forward_dc_index_hoist', 'traceback': traceback.format_exc()})
        _reset_after_failed_capture()

    try:
        boundary = _IstftBoundary()
        info['capture_adapter_istft'] = install_istft_boundary(model, out_dir, boundary)
        graph, sin, sout = capture_graph_amp(model, x_example, ctx_nocache)
        assert boundary.stash is not None, 'iSTFT boundary never reached during capture'
        st = boundary.stash
        from einops import rearrange
        n_stems = len(model.mask_estimators)
        channels = model.audio_channels

        def step_istft(x):
            sin.copy_(x)                     # AUDIT-MAP[2]: fresh input each replay
            graph.replay()
            audio = torch.istft(st['input'], *st['args'], **st['kwargs'])
            assert audio.shape[-1] == boundary.placeholder_len, \
                (audio.shape, boundary.placeholder_len)
            audio = rearrange(audio, '(b n s) t -> b n s t', b=1, s=channels, n=n_stems)
            if n_stems == 1:
                audio = rearrange(audio, 'b 1 s t -> b s t')
            return audio
        info['capture_scope'] = 'whole_forward_to_istft'
        info['capture_adapter_active'] = 'dc_hoist+istft_boundary'
        return step_istft
    except Exception:
        info['capture_attempts'].append(
            {'rung': 'whole_forward_to_istft', 'traceback': traceback.format_exc()})
        _reset_after_failed_capture()

    # rung 3 runs the STOCK forward eagerly (adapters removed) with only the
    # band_split module replaced by its replay wrapper
    if 'forward' in model.__dict__:
        del model.__dict__['forward']
    info['capture_adapter_active'] = None
    band_split = model.band_split
    recorder = common.RecordingBandSplit(band_split)
    model.band_split = recorder
    with torch.no_grad(), ctx_nocache():
        model(x_example)
    folded = recorder.last_input.detach().clone()
    model.band_split = band_split
    graph, sin, sout = capture_graph_amp(band_split, folded, ctx_nocache)
    model.band_split = common.ReplayBandSplit(band_split, graph, sin, sout)
    info['capture_scope'] = 'band_split_only'
    return lambda x: model(x)
