"""DOCUMENTED ADAPTATION of axial-layout-scoping@a6f3e6f extensions/c3/native_tail.py
(the verbatim original is kept beside this file as native_tail.py and is NOT
imported by the harness). Registered in prereg_amendments.md §1 (arm C3).

The original registers the DC-filter custom op for the single 8-s shape
(2, 1025, 801) with hard-coded strides. This session runs C3/H at frame
counts T ∈ {11, 51, 201, 801} (chunks 4410, 22050, 88200, 352800) and, for the
Phase-3 E9 held-out lengths, T ∈ {401, 601} (176400, 264600; qualified on the pod only). The ONLY
changes: (1) the op name (both modules would otherwise register the same
name); (2) the shape/stride checks are parametrised by T = input.shape[-1]
and accept exactly the registered set; (3) verify_operator() checks every
registered T. The public-forward AST substitution, the native
Tensor.index_fill call, and the no-new-kernel / no-arithmetic properties are
unchanged. Native output metadata, fake metadata, bitwise eager equivalence,
CUDA-graph capture/replay and isolated compilation are verified per shape.
"""
import ast
import hashlib
import inspect
from pathlib import Path
import textwrap
import types

import torch

FRAMES = (11, 51, 201, 801, 401, 601)      # registered operating points + sweep + Phase-3 E9 held-out (401, 601)


def _check_meta(t):
    assert t.dim() == 3 and t.shape[0] == 2 and t.shape[1] == 1025, tuple(t.shape)
    T = int(t.shape[-1])
    assert T in FRAMES, T
    assert t.dtype == torch.complex64, t.dtype
    assert t.stride() in ((T, 2 * T, 1), (1025 * T, T, 1)), (t.stride(), T)
    return T


@torch.library.custom_op('fullmodel_c3::native_dc_fill_t', mutates_args=())
def native_dc_fill(input: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    _check_meta(input)
    result = input.index_fill(1, index, 0.)
    assert result.shape == input.shape
    assert result.stride() == input.stride() and result.storage_offset() == 0
    assert result.dtype == input.dtype == torch.complex64
    return result


@native_dc_fill.register_fake
def native_dc_fill_fake(input, index):
    _check_meta(input)
    assert index.shape == () and index.dtype == torch.int64
    return torch.empty_strided(input.shape, input.stride(),
                               device=input.device, dtype=input.dtype)


def call_native_dc_fill(input, dim, index, value):
    assert dim == 1 and value == 0.
    return native_dc_fill(input, index)


def install(model, out_dir):
    from models.bs_roformer.mel_band_roformer import MelBandRoformer
    assert type(model) is MelBandRoformer
    assert model.forward.__func__ is MelBandRoformer.forward
    assert 'forward' not in model.__dict__
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
            node.func = ast.Name(id='c3_native_dc_fill', ctx=ast.Load())
            node.args.insert(0, ast.Name(id='stft_repr', ctx=ast.Load()))
            replacements += 1
    assert replacements == 1, 'expected the single public DC-filter call'
    source = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    restored = ast.parse(source)
    for node in ast.walk(restored):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == 'c3_native_dc_fill':
            base = node.args.pop(0)
            node.func = ast.Attribute(value=base, attr='index_fill', ctx=ast.Load())
    assert ast.dump(restored) == ast.dump(ast.parse(original))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    dest = Path(out_dir) / 'generated_public_forward.py'
    dest.write_text(source)
    namespace = dict(MelBandRoformer.forward.__globals__)
    namespace['c3_native_dc_fill'] = call_native_dc_fill
    exec(compile(source, str(dest), 'exec'), namespace)
    model.forward = types.MethodType(namespace['forward'], model)
    return {'status': 'PASS', 'replacements': replacements,
            'original_forward_sha256': hashlib.sha256(original.encode()).hexdigest(),
            'external_forward_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'generated_source': str(dest), 'native_call': 'Tensor.index_fill',
            'custom_op': 'fullmodel_c3::native_dc_fill_t', 'new_gpu_kernel': False,
            'cudagraph_unsafe_tag': False, 'frames_supported': list(FRAMES),
            'adaptation_of': 'axial-layout-scoping@a6f3e6f extensions/c3/native_tail.py',
            'scope': 'public forward differs only in DC-filter callee; op checks parametrised by frames'}


def verify_operator(frames=FRAMES):
    """Per registered T: schema/fake checks, bitwise native equality,
    CUDA-graph capture/replay, isolated full-graph compile."""
    out = {}
    for T in frames:
        gen = torch.Generator(device='cpu').manual_seed(4242)
        x = torch.randn((1025, 2, T), generator=gen, dtype=torch.complex64).to('cuda')
        x = x.permute(1, 0, 2)
        index = torch.tensor(0, device='cuda')
        with torch.no_grad(), torch.autocast('cuda', enabled=True):
            checks = torch.library.opcheck(native_dc_fill, (x, index),
                                          test_utils=('test_schema', 'test_faketensor'))
            reference = x.index_fill(1, index, 0.)
            actual = native_dc_fill(x, index)
            assert torch.equal(actual, reference)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = native_dc_fill(x, index)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(captured, reference)
            torch._dynamo.reset()
            compiled = torch.compile(native_dc_fill, mode='reduce-overhead',
                                     dynamic=False, fullgraph=True)
            for _ in range(6):
                result = compiled(x, index).clone()
            torch.cuda.synchronize()
            assert torch.equal(result, reference)
        out[str(T)] = {'status': 'PASS', 'opcheck': checks, 'native_bitwise_equal': True,
                       'cuda_graph_capture_and_replay': 'PASS', 'compiled_fullgraph': 'PASS',
                       'output_shape': list(actual.shape), 'output_stride': list(actual.stride())}
    torch._dynamo.reset()
    return out
