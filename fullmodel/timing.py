"""5a timing, settle, validation, environment — full-model session.

Sources (prereg rule: reuse the audited harness, cite it):
  timed_run / settle_compile   sm120-nulltest arms/common.py (AUDIT-MAP[1]),
                               adapted as in axial-layout-scoping@a6f3e6f
                               scoping/common_timing.py: the forward runs inside
                               the caller's autocast context (the repo's own
                               inference precision), five repeats per process
                               with min-of-medians + spread (plan: protocol 5a).
  validate                     sm120-nulltest arms/common.py (AUDIT-MAP[2]):
                               >= 2 passes with DIFFERENT data (asserted),
                               compared against SAVED eager-A outputs written
                               by the session's reference step; hardened
                               compare (exact shape + finiteness, gate-02 f9 of
                               the scoping study).
"""

import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone

import torch

import provenance

TIMING_SEED = 0                  # sm120 arms/common.py::TIMING_SEED
VAL_SEEDS = (1001, 2002)         # sm120 arms/common.py::VAL_SEEDS
TIERS = {'loose': (1e-2, 1e-3), 'tight': (1e-4, 1e-5)}   # sm120 VAL_TOL values
CHANNELS = 2                     # model.stereo=True (KJ config)


def median(xs):
    return statistics.median(xs)


def make_input(chunk, seed, device='cuda'):
    """Deterministic input independent of device RNG (sm120 make_input)."""
    gen = torch.Generator(device='cpu').manual_seed(seed)
    return torch.randn(1, CHANNELS, chunk, generator=gen).to(device)


def settle_compile(step, x, ctx, max_iters=60, consecutive=5):
    """Run until torch._dynamo counters are unchanged for `consecutive`
    passes (sm120 settle_compile; str-keyed snapshot)."""
    import torch._dynamo.utils as dynamo_utils

    def snap():
        return {str(k): {str(kk): int(vv) for kk, vv in v.items()}
                for k, v in dynamo_utils.counters.items()}

    t0 = time.perf_counter()
    prev, stable, it = snap(), 0, 0
    with torch.no_grad(), ctx():
        while stable < consecutive and it < max_iters:
            step(x)
            it += 1
            cur = snap()
            stable = stable + 1 if cur == prev else 0
            prev = cur
    torch.cuda.synchronize()
    return {'settle_iters': it, 'settle_stable_tail': stable,
            'settle_capped': stable < consecutive,
            'settle_wall_s': round(time.perf_counter() - t0, 3),
            'dynamo_counters': prev}


def dynamo_counters():
    from torch._dynamo.utils import counters
    return {str(k): {str(kk): int(vv) for kk, vv in v.items()}
            for k, v in counters.items()}


STABILITY_KEYS = ('graph_break', 'unimplemented')


def compile_stable(before, after):
    """Recording/compilation stability (gate-02 f21): no NEW graph breaks or
    unimplemented ops, the same number of unique Dynamo graphs, and EVERY
    Inductor cudagraph counter unchanged (recordings, skips, non-static
    input recordings). Per-call counters (calls, frames) are ignored."""
    for k in STABILITY_KEYS:
        if before.get(k, {}) != after.get(k, {}):
            return False
    if before.get('stats', {}).get('unique_graphs') != after.get('stats', {}).get('unique_graphs'):
        return False
    b_ind = {k: v for k, v in before.get('inductor', {}).items() if 'cudagraph' in k}
    a_ind = {k: v for k, v in after.get('inductor', {}).items() if 'cudagraph' in k}
    return b_ind == a_ind


def cudagraph_tree_state():
    """Recording observer (gate-03 f36): the cudagraph-trees manager's recorded
    graph NODES (roots and all descendants) — the identity set that must not
    change between measured phases. torch 2.8: TreeManager.roots is
    {FunctionID: [CUDAGraphNode]}, each node has .children of the same shape.
    'available': False -> the observer is missing and a compiled run cannot
    be accepted (unresolved recording state)."""
    try:
        from torch._inductor import cudagraph_trees
        container = cudagraph_trees.get_container(0)
        mgr = getattr(container, 'tree_manager', None)
        if mgr is None:
            return {'available': False, 'reason': 'no tree manager on device 0'}
        roots = getattr(mgr, 'roots', None)
        if not isinstance(roots, dict):
            return {'available': False, 'reason': 'roots not a dict'}
        n_nodes, n_roots, stack = 0, 0, []
        for lst in roots.values():
            for node in lst:
                n_roots += 1
                stack.append(node)
        seen = set()
        while stack:
            node = stack.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            n_nodes += 1
            ch = getattr(node, 'children', None)
            if isinstance(ch, dict):
                for lst in ch.values():
                    stack.extend(lst)
        ids = getattr(mgr, 'ids_to_funcs', None) or getattr(mgr, 'ids_to_stack', None)
        return {'available': True, 'roots': n_roots, 'nodes': n_nodes,
                'functions': len(ids) if ids is not None else None,
                'current_gen': getattr(mgr, 'current_gen', None)}
    except Exception as e:
        return {'available': False, 'reason': f'{type(e).__name__}: {e}'[:120]}


def recording_state():
    """Counters + tree node identity, compared together by phase."""
    return {'counters': dynamo_counters(), 'tree': cudagraph_tree_state()}


def recording_stable(a, b):
    if not (a['tree'].get('available') and b['tree'].get('available')):
        return None                      # unresolved: observer missing
    return (compile_stable(a['counters'], b['counters']) and
            a['tree']['nodes'] == b['tree']['nodes'] and a['tree']['roots'] == b['tree']['roots'])


def graph_gate(require_zero_breaks):
    """Compile-arm gate (prereg_amendments.md §1, C3/H): graph breaks and
    unimplemented ops from the Dynamo counters; cudagraph skips from the
    Inductor counters (plan 0.3 input)."""
    snap = dynamo_counters()
    breaks = sum(snap.get('graph_break', {}).values())
    unsupported = sum(snap.get('unimplemented', {}).values())
    inductor = snap.get('inductor', {})
    skips = {k: v for k, v in inductor.items() if 'cudagraph' in k}
    rec = {'graph_break_count': breaks, 'unimplemented_count': unsupported,
           'unique_graphs': snap.get('stats', {}).get('unique_graphs', 0),
           'cudagraph_counters': skips,
           'suppress_errors': bool(torch._dynamo.config.suppress_errors),
           'require_zero_breaks': require_zero_breaks,
           'graph_break_reasons': list(snap.get('graph_break', {}).keys())[:40]}
    rec['status'] = ('PASS' if (not require_zero_breaks or
                                (breaks == 0 and unsupported == 0))
                     and not rec['suppress_errors'] else 'FAIL')
    return rec


def timed_run(step, x, ctx, iters=50, nvtx_pass=False):
    """PREREG 5a — PRIMARY per-iteration latency (sync every pass) inside the
    cudaProfilerApi capture range; SECONDARY one free-running block outside it.
    nvtx_pass=True pushes an OUTER eager NVTX range per pass (traced runs:
    pass windows for the exports; never inside compiled code)."""
    with torch.no_grad(), ctx():
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        latency_ms = []
        for i in range(iters):
            if nvtx_pass:
                torch.cuda.nvtx.range_push(f'pass_{i}')
            t0 = time.perf_counter()
            step(x)
            torch.cuda.synchronize()
            latency_ms.append((time.perf_counter() - t0) * 1e3)
            if nvtx_pass:
                torch.cuda.nvtx.range_pop()
        torch.cuda.profiler.stop()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            step(x)
        torch.cuda.synchronize()
        throughput_ms = (time.perf_counter() - t0) * 1e3 / iters
    return latency_ms, throughput_ms


def five_repeats(step, x, ctx, iters=50, repeats=5, nvtx_pass=False):
    """Five repeats of the 5a block; min-of-medians + median spread; every
    per-iteration list preserved; block wall timestamps for the clock log."""
    blocks = []
    for r in range(repeats):
        t_start = time.time()
        lat, thr = timed_run(step, x, ctx, iters, nvtx_pass=nvtx_pass)
        t_end = time.time()
        s = sorted(lat)
        blocks.append({'repeat': r, 'latency_ms_per_iter': lat,
                       'latency_median_ms': median(lat),
                       'latency_min_ms': s[0], 'latency_max_ms': s[-1],
                       'throughput_ms_per_pass': thr,
                       'unix_start': t_start, 'unix_end': t_end})
    meds = [b['latency_median_ms'] for b in blocks]
    return {'blocks': blocks, 'latency_medians_ms': meds,
            'min_of_medians_ms': min(meds),
            'median_spread_ms': max(meds) - min(meds),
            'throughput_ms_per_pass_best': min(b['throughput_ms_per_pass'] for b in blocks),
            'serialization_delta_pct': [
                (b['latency_median_ms'] - b['throughput_ms_per_pass'])
                / b['throughput_ms_per_pass'] * 100.0 for b in blocks],
            'iters_per_block': iters, 'repeats': repeats}


def compare_outputs(out, ref, tier):
    rtol, atol = TIERS[tier]
    out = out.float()
    ref = ref.float()
    rec = {'ok': False, 'rtol': rtol, 'atol': atol,
           'shape_out': list(out.shape), 'shape_ref': list(ref.shape)}
    if out.shape != ref.shape:
        rec['reason'] = 'shape mismatch'
        rec['max_abs_diff'] = None
        return rec
    fin_out, fin_ref = bool(torch.isfinite(out).all()), bool(torch.isfinite(ref).all())
    rec['finite_out'], rec['finite_ref'] = fin_out, fin_ref
    rec['max_abs_diff'] = float((out - ref).abs().max().item())
    rec['bitwise_equal'] = bool(torch.equal(out, ref))
    if not (fin_out and fin_ref):
        rec['reason'] = 'non-finite values present'
        return rec
    rec['ok'] = bool(torch.allclose(out, ref, rtol=rtol, atol=atol))
    if not rec['ok']:
        rec['reason'] = 'tolerance exceeded'
    return rec


def ref_path(ref_dir, chunk, seed):
    return os.path.join(ref_dir, f'ref_A_c{chunk}_s{seed}.pt')


def bind_refs(ref_dir, manifest_id, write=False):
    """Directory marker (kept for the session preflight); every reference
    file additionally carries its own lineage sidecar (provenance.bind_artifact)."""
    path = os.path.join(ref_dir, 'manifest_id.txt')
    if write:
        os.makedirs(ref_dir, exist_ok=True)
        with open(path, 'w') as f:
            f.write(manifest_id + '\n')
        return
    if manifest_id is None:
        return
    if not os.path.exists(path) or open(path).read().strip() != manifest_id:
        raise RuntimeError('reference outputs are not bound to this session manifest')


def validate(step, ctx, tier, chunk, ref_dir, save_reference=False, manifest_id=None):
    """AUDIT-MAP[2] mechanics: two passes, different data (asserted), each
    compared to the SAVED eager-A output for that (chunk, seed). With
    save_reference=True (arm A reference step) the outputs are written
    instead of compared."""
    passes, prev_x = [], None
    for seed in VAL_SEEDS:
        x = make_input(chunk, seed)
        if prev_x is not None:
            assert not torch.equal(x, prev_x), \
                'validation passes must use different input data'
        prev_x = x
        with torch.no_grad(), ctx():
            out = step(x)
            torch.cuda.synchronize()
            out = out.clone()            # borrow discipline (build_spec rule 1)
        path = ref_path(ref_dir, chunk, seed)
        if save_reference:
            os.makedirs(ref_dir, exist_ok=True)
            torch.save(out.float().cpu(), path)
            if manifest_id is not None:
                provenance.bind_artifact(path, manifest_id)
            passes.append({'seed': seed, 'saved': path,
                           'finite': bool(torch.isfinite(out).all())})
            continue
        if not os.path.exists(path):
            raise FileNotFoundError(f'reference output missing: {path} '
                                    '(run the arm-A reference step first)')
        if manifest_id is not None:
            provenance.check_artifact(path, manifest_id)
        ref = torch.load(path, map_location='cuda', weights_only=True)
        rec = compare_outputs(out, ref, tier)
        rec['seed'] = seed
        passes.append(rec)
        del ref
    if save_reference:
        return {'status': 'REFERENCE_SAVED', 'tier': tier, 'passes': passes}
    ok = all(p['ok'] for p in passes)
    return {'status': 'PASS' if ok else 'FAIL', 'tier': tier,
            'rtol': TIERS[tier][0], 'atol': TIERS[tier][1], 'passes': passes}


def env_fingerprint():
    try:
        driver = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
    except Exception:
        driver = 'unavailable'
    import triton
    try:
        uuid = subprocess.run(
            ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader', '-i',
             os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
    except Exception:
        uuid = 'unavailable'
    props = torch.cuda.get_device_properties(0)
    return {'torch': torch.__version__, 'triton': triton.__version__,
            'cuda_build': torch.version.cuda, 'driver': driver,
            'gpu': torch.cuda.get_device_name(0), 'gpu_uuid': uuid,
            'l2_cache_bytes': getattr(props, 'L2_cache_size', None),
            'sm_count': props.multi_processor_count,
            'total_memory_bytes': props.total_memory,
            'capability': list(torch.cuda.get_device_capability(0)),
            'python': platform.python_version(), 'hostname': platform.node(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
            'timestamp_utc': datetime.now(timezone.utc).isoformat()}


def write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, allow_nan=False)
    print(f'-> {path}')
