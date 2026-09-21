"""Full-model arm builder (prereg_amendments.md §1).

Every arm is an EXTERNAL transformation of the pinned public model: module
swaps (public nn.Module attribute assignment), external methods installed
with AST-verified source checks, graph capture, or torch.compile. The clone
is never edited. build_arm_step(...) returns step(x) -> output for ONE pass;
all build/compile/capture work happens here, before validation and before
the timed region (sm120 AUDIT-MAP[1]).

Mechanics reused verbatim from the sm120 harness (arms/common.py,
arms/streams.py): FusedBandSplit / fused_band_fns (D/E/G), MultiStreamBandLoop /
MultiStreamBandSplit (F/G). C3 repairs: fullmodel/c3_fix (vendored verbatim
from axial-layout-scoping@a6f3e6f). Arm H: fullmodel/padbatch.py.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, 'arms'), os.path.join(_ROOT, 'scripts'),
           os.path.join(_HERE, 'c3_fix')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                  # noqa: E402
import common                                 # noqa: E402  sm120 arms/common.py
import streams                                # noqa: E402  sm120 arms/streams.py
import capture as capture_mod                 # noqa: E402
import padbatch                               # noqa: E402

ARMS = ('A', 'ANVTX', 'B', 'C1', 'C2', 'C3', 'C1A', 'C1D', 'D', 'E', 'F', 'G', 'H',
        'C3F', 'C3P', 'C3M', 'C1P', 'C3N')          # Phase 3: E1a, E1b, E2, E3, E4a
COMPILE_ARMS = ('C1', 'C2', 'C3', 'C1A', 'C1D', 'H', 'C3F', 'C3P', 'C3M', 'C1P', 'C3N')
CAPTURE_ARMS = ('B', 'E', 'G')
REQUIRE_ZERO_BREAKS = ('C3', 'H', 'C3F', 'C3P', 'C3M', 'C3N')   # gate: zero graph breaks or INVALID
# classification rule (sm120 VAL_TOL rationale): compiler-regenerated kernels
# present -> loose; else tight.
TIER = {a: ('loose' if a in COMPILE_ARMS else 'tight') for a in ARMS}
CHUNKS = (352800, 4410, 22050, 88200, 176400, 264600)   # registered operating points + sweep + E9 held-out
HELD_OUT_CHUNKS = (176400, 264600)            # E9: run ONLY inside the registered pod session (SESSION_POD=1)


def held_out_guard(chunk):
    """E9 rule (Phase 3 handoff R10/§4): the two held-out chunk lengths are never
    run on the shakedown host or in any ad-hoc process; only the pod session
    exports SESSION_POD=1."""
    if int(chunk) in HELD_OUT_CHUNKS and os.environ.get('SESSION_POD') != '1':
        raise RuntimeError(f'chunk {chunk} is an E9 held-out length: it runs only inside the registered pod session')


def base_arm(label):
    """'C3#2' (second instance of an A/A pair) -> 'C3'."""
    return str(label).split('#')[0]
DEFAULT_STREAMS = 4
PLANTED_FAULT_CONST = 1e-3                    # §1.2 registered planted fault


def _c3_fixes(model, out_dir, attention=True, dc_filter=True):
    rec = {}
    if attention:
        from graphbreak_fix import install as install_attention
        rec['attention_repair'] = install_attention(model)
    if dc_filter:
        import native_tail_adapted as native_tail     # frames-parametrised adaptation (§1)
        rec['dc_filter_boundary'] = native_tail.install(model, out_dir)
    return rec


def verify_c3_operator():
    import native_tail_adapted
    return native_tail_adapted.verify_operator()


def _compile(model, mode, info):
    info['compile_mode'] = mode
    compiled = torch.compile(model, mode=mode, dynamic=False)
    return lambda x: compiled(x)


def build_arm_step(arm, model, x_example, ctx_nocache, num_streams, info, out_dir):
    arm = base_arm(arm)
    """info: caller's dict, mutated in place (capture_scope, capture_adapter,
    capture_attempts, stream_count, compile_mode, c3_fixes, padbatch,
    nvtx_boundary_mode, notes)."""
    if arm == 'A':
        return lambda x: model(x)

    if arm == 'ANVTX':
        from p8_trace import install_nvtx_hooks       # sm120 scripts/p8_trace.py
        handles, mode = install_nvtx_hooks(model)
        info['nvtx_boundary_mode'] = mode
        info['nvtx_hook_count'] = len(handles)
        return lambda x: model(x)

    if arm == 'B':
        return capture_mod.capture_ladder(model, x_example, ctx_nocache, info, out_dir)

    if arm in ('C1', 'C2'):
        return _compile(model, 'reduce-overhead' if arm == 'C1' else 'max-autotune', info)

    if arm == 'C3':
        info['c3_fixes'] = _c3_fixes(model, out_dir, attention=True, dc_filter=True)
        return _compile(model, 'reduce-overhead', info)

    if arm == 'C1A':
        info['c3_fixes'] = _c3_fixes(model, out_dir, attention=True, dc_filter=False)
        return _compile(model, 'reduce-overhead', info)

    if arm == 'C1D':
        info['c3_fixes'] = _c3_fixes(model, out_dir, attention=False, dc_filter=True)
        return _compile(model, 'reduce-overhead', info)

    if arm == 'D':
        model.band_split = common.FusedBandSplit(model.band_split)
        return lambda x: model(x)

    if arm == 'E':
        model.band_split = common.FusedBandSplit(model.band_split)
        return capture_mod.capture_ladder(model, x_example, ctx_nocache, info, out_dir)

    if arm == 'F':
        info['stream_count'] = num_streams
        loop = streams.MultiStreamBandLoop(num_streams, 'cuda')
        stock_fns = list(model.band_split.to_features)
        model.band_split = streams.MultiStreamBandSplit(model.band_split, stock_fns, loop)
        return lambda x: model(x)

    if arm == 'G':
        info['stream_count'] = num_streams
        loop = streams.MultiStreamBandLoop(num_streams, 'cuda')
        fns = common.fused_band_fns(model.band_split)
        model.band_split = streams.MultiStreamBandSplit(model.band_split, fns, loop)
        return capture_mod.capture_ladder(model, x_example, ctx_nocache, info, out_dir)

    if arm == 'H':
        info['c3_fixes'] = _c3_fixes(model, out_dir, attention=True, dc_filter=True)
        info['padbatch'] = padbatch.install(model)
        return _compile(model, 'reduce-overhead', info)

    # ---- Phase 3 arms (prereg_amendments.md v0.2) ----
    if arm == 'C3F':                                  # E1(a): C3 + Inductor freezing
        import torch._inductor.config as icfg
        icfg.freezing = True
        info['inductor_config'] = {'freezing': True}
        info['c3_fixes'] = _c3_fixes(model, out_dir, attention=True, dc_filter=True)
        return _compile(model, 'reduce-overhead', info)

    if arm == 'C3P':                                  # E1(b): C3 + pre-cast band/mask Linear weights
        import precast
        info['c3_fixes'] = _c3_fixes(model, out_dir, attention=True, dc_filter=True)
        info['precast'] = precast.install(model)
        return _compile(model, 'reduce-overhead', info)

    if arm == 'C3M':                                  # E2: repair x max-autotune (with cudagraphs)
        info['c3_fixes'] = _c3_fixes(model, out_dir, attention=True, dc_filter=True)
        return _compile(model, 'max-autotune', info)

    if arm == 'C1P':                                  # E3: C1 + Inductor graph partition
        import torch._inductor.config as icfg
        icfg.graph_partition = True
        info['inductor_config'] = {'graph_partition': True}
        return _compile(model, 'reduce-overhead', info)

    if arm == 'C3N':                                  # E4a: C3 + norm-scale folding wrapper
        import foldnorm
        mode = os.environ.get('E4A_MODE', 'reduce-overhead')
        assert mode in ('reduce-overhead', 'max-autotune'), mode
        info['inductor_config'] = {'e4a_mode': mode}
        info['c3_fixes'] = _c3_fixes(model, out_dir, attention=True, dc_filter=True)
        info['foldnorm'] = foldnorm.install(model)
        return _compile(model, mode, info)

    raise ValueError(f'unknown arm {arm!r}')


def plant_fault(step, const=PLANTED_FAULT_CONST):
    """§1.2: a deliberate, deterministic output corruption the validator
    MUST catch (both seeds FAIL, seed-invariant max_abs_diff = const);
    1e-3 exercises the tight comparator, 5e-2 the loose one."""
    return lambda x: step(x) + const
