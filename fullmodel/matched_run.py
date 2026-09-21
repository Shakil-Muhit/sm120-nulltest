"""Matched block-alternation runner (plan 1.2; prereg_amendments.md §5), v2.

Two arms in ONE fresh process. Lifecycle (gate-02 f19):
  provenance + manifest → build A (settle, gate) → build B (settle, gate) →
  JOINT settle of the actual alternation (a, b, a, b, … until the compile/
  recording counters are stable across two full alternation rounds) →
  validate A and B (after the joint settle) → alternating 5a blocks
  a, b, a, b, … (five each) with wall timestamps and a counter snapshot per
  block → pair verdict.
Pair eligibility requires: both builds OK, both graph gates PASS, no capped
settle, joint settle stable, both validations PASS, counters stable across
every block. The SDR qualification at this operating point is joined at
collection time from the arms' matrix cells (collect_full.py). Output:
<evidence>/matched_<A>_<B>_c<chunk>_<tag>.json
"""

import argparse
import json
import os
import statistics
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                      # noqa: E402
import fullarms as arms_mod       # noqa: E402
import loader                     # noqa: E402
import provenance
import qualify                 # noqa: E402
import timing as tm               # noqa: E402


def build_one(arm, chunk, streams, gen_dir, paths, settle_max, variant=None):
    from torch._dynamo.utils import counters
    counters.clear()               # this arm's gate reflects only its own compilation
    if variant:
        import variants
        model, config, use_amp, ctx, ctx_nocache, _vrec = variants.load_variant_model(
            paths['msst'], paths['config'], variant, gen_dir)
    else:
        model, config, use_amp, ctx, ctx_nocache = loader.load_public_model(
            paths['msst'], paths['ckpt'], paths['config'])
    x = tm.make_input(chunk, tm.TIMING_SEED)
    info = {'capture_scope': None, 'capture_adapter': None, 'capture_attempts': [],
            'stream_count': None, 'compile_mode': None, 'notes': []}
    step = arms_mod.build_arm_step(arm, model, x, ctx_nocache, streams, info, gen_dir)
    settle = gate = gate_snapshot = None
    if arms_mod.base_arm(arm) in arms_mod.COMPILE_ARMS:
        settle = tm.settle_compile(step, x, ctx, settle_max, 5)
        gate = tm.graph_gate(arms_mod.base_arm(arm) in arms_mod.REQUIRE_ZERO_BREAKS)
        gate_snapshot = tm.recording_state()      # reconciliation reference for this arm's gate (gate-06 f83)
    else:
        with torch.no_grad(), ctx():
            for _ in range(10):
                step(x)
        torch.cuda.synchronize()
    return {'model': model, 'step': step, 'ctx': ctx, 'x': x, 'info': info,
            'settle': settle, 'gate': gate, 'gate_snapshot': gate_snapshot, 'use_amp': use_amp}


def joint_settle(built, arms, rounds=2, passes=3):
    """Alternate the two steps. The recording state is snapshotted BEFORE the
    first round and after every round; stable iff unchanged across ALL rounds
    (a re-recording of the arm built first shows up in round 1), and no new
    graph breaks / unimplemented ops appear at any point."""
    snaps = [tm.recording_state()]
    for r in range(rounds + 1):
        for arm in arms:
            b = built[arm]
            with torch.no_grad(), b['ctx']():
                for _ in range(passes):
                    b['step'](b['x'])
        torch.cuda.synchronize()
        snaps.append(tm.recording_state())
    verdicts = [tm.recording_stable(a, c) for a, c in zip(snaps, snaps[1:])]
    breaks = [sum(x['counters'].get('graph_break', {}).values()) +
              sum(x['counters'].get('unimplemented', {}).values()) for x in snaps]
    return {'stable': (None if any(v is None for v in verdicts) else all(verdicts)),
            'new_breaks_during_joint_settle': breaks[-1] - breaks[0],
            'rounds': rounds, 'snapshots': snaps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', nargs=2, required=True, metavar=('ARM_A', 'ARM_B'))
    ap.add_argument('--chunk', type=int, required=True, choices=list(arms_mod.CHUNKS))
    ap.add_argument('--tag', default='m1')
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--streams', type=int, default=arms_mod.DEFAULT_STREAMS)
    ap.add_argument('--settle-max', type=int, default=60)
    ap.add_argument('--evidence-dir', '--out-dir', dest='evidence_dir',
                    default=os.path.join(_ROOT, 'evidence'))
    ap.add_argument('--ref-dir', default=None)
    ap.add_argument('--no-manifest', action='store_true')
    ap.add_argument('--variant', default=None, help='E5 structural variant (both arms); never a headline')
    args = ap.parse_args()
    arms_mod.held_out_guard(args.chunk)                 # E9: held-out lengths run only in the pod session
    if (args.iters, args.repeats) != (50, 5):
        print('[matched] WARNING: not the registered 5 x 50 protocol — pair ineligible')
    assert torch.cuda.is_available()
    os.makedirs(args.evidence_dir, exist_ok=True)
    ref_dir = args.ref_dir or os.path.join(args.evidence_dir, 'ref')
    a, b = args.pair
    out_path = os.path.join(args.evidence_dir, f'matched_{a}_{b}_c{args.chunk}_{args.tag}.json')
    if b == a:                                          # E6 A/A: the same arm built twice in one process
        b = f'{a}#2'
    result = {'pair': [a, b], 'chunk': args.chunk, 'tag': args.tag, 'status': 'VALID',
              'aa_pair': (arms_mod.base_arm(a) == arms_mod.base_arm(b)), 'variant': args.variant,
              'error': None, 'env': tm.env_fingerprint(), 'arms': {}, 'blocks': [],
              'iters': args.iters, 'repeats': args.repeats,
              'torch_logs_env': os.environ.get('TORCH_LOGS'), 'manifest_id': None,
              'pair_eligible': None, 'ineligibility_reasons': []}
    reasons = result['ineligibility_reasons']
    try:
        paths = loader.paths_from_env()
        result['provenance'] = provenance.verify(paths)
        if not args.no_manifest:
            result['manifest_id'] = provenance.load_manifest(
                args.evidence_dir, paths, tm.env_fingerprint())['manifest_id']
            tm.bind_refs(ref_dir, result['manifest_id'])
        else:
            reasons.append('not bound to a session manifest')
        if (args.iters, args.repeats) != (50, 5):
            reasons.append(f'protocol {args.repeats}x{args.iters} is not the registered 5x50')
        if args.variant:
            reasons.append(f'structural variant {args.variant} (E5): never a headline')
            ref_dir = args.ref_dir or os.path.join(args.evidence_dir, f'ref_variant_{args.variant}')
        built = {}
        for arm in (a, b):
            gen_dir = os.path.join(args.evidence_dir, 'generated', f'matched_{arm}_c{args.chunk}_{args.tag}')
            try:
                built[arm] = build_one(arm, args.chunk, args.streams, gen_dir, paths, args.settle_max, args.variant)
                bi = built[arm]
                result['arms'][arm] = {'build_status': 'OK', 'settle': bi['settle'], 'graph_gate': bi['gate'],
                                       'gate_snapshot': bi.get('gate_snapshot'),
                                       'capture_scope': bi['info'].get('capture_scope'),
                                       'capture_adapter_active': bi['info'].get('capture_adapter_active'),
                                       'compile_mode': bi['info'].get('compile_mode'),
                                       'stream_count': bi['info'].get('stream_count')}
                if bi['settle'] and bi['settle']['settle_capped']:
                    reasons.append(f'{arm}: settle capped')
                if bi['gate'] and bi['gate']['status'] != 'PASS':
                    reasons.append(f'{arm}: graph gate FAIL')
                print(f'[matched] {arm}: built (scope {bi["info"].get("capture_scope")}, gate '
                      f'{(bi["gate"] or {}).get("status")})')
            except Exception:
                result['arms'][arm] = {'build_status': 'FAILED', 'error': traceback.format_exc()}
                reasons.append(f'{arm}: build failed')
                print(f'[matched] {arm}: BUILD FAILED (recorded)', file=sys.stderr)
        if not all(result['arms'][x]['build_status'] == 'OK' for x in (a, b)):
            result['status'] = 'INVALID'
        else:
            js = joint_settle(built, (a, b))
            result['joint_settle'] = js
            if js['stable'] is None:
                reasons.append('joint settle: recording observer unavailable (UNRESOLVED)')
            elif not js['stable']:
                reasons.append('joint alternation did not settle (recording state changed)')
            if js['new_breaks_during_joint_settle']:
                reasons.append(f"{js['new_breaks_during_joint_settle']} new graph breaks/unimplemented "
                               'during the joint settle')
            # per-arm zero-break requalification after the joint settle (global delta)
            before_val = tm.recording_state()
            for arm in (a, b):
                v = tm.validate(built[arm]['step'], built[arm]['ctx'], arms_mod.TIER[arms_mod.base_arm(arm)],
                                args.chunk, ref_dir, manifest_id=result['manifest_id'])
                result['arms'][arm]['validation'] = v
                if v['status'] != 'PASS':
                    reasons.append(f'{arm}: validation {v["status"]}')
                print(f'[matched] {arm}: validation {v["status"]} '
                      f'{[p.get("max_abs_diff") for p in v["passes"]]}')
            after_val = tm.recording_state()
            result['snapshots'] = {'joint_settle': js['snapshots'], 'before_validation': before_val,
                                   'after_validation': after_val, 'blocks': []}
            sv = tm.recording_stable(before_val, after_val)
            if sv is None:
                reasons.append('validation: recording observer unavailable')
            elif not sv:
                reasons.append('recording state changed during validation')
            per = {a: [], b: []}
            prev = after_val
            block_stable = True
            for r in range(args.repeats):
                for arm in (a, b):
                    t0 = time.time()
                    lat, thr = tm.timed_run(built[arm]['step'], built[arm]['x'],
                                            built[arm]['ctx'], args.iters)
                    t1 = time.time()
                    cur = tm.recording_state()
                    stable = tm.recording_stable(prev, cur)
                    block_stable = block_stable and bool(stable)
                    result['snapshots']['blocks'].append(cur)
                    prev = cur
                    blk = {'repeat': r, 'arm': arm, 'latency_median_ms': tm.median(lat),
                           'throughput_ms_per_pass': thr, 'latency_ms_per_iter': lat,
                           'unix_start': t0, 'unix_end': t1, 'counters_stable': stable}
                    result['blocks'].append(blk)
                    per[arm].append(blk['latency_median_ms'])
                    print(f'  block {r} {arm}: median {blk["latency_median_ms"]:.4f} ms')
            if not block_stable:
                reasons.append('compile/recording counters changed during the blocks')
            for arm in (a, b):
                result['arms'][arm].update({
                    'latency_medians_ms': per[arm], 'min_of_medians_ms': min(per[arm]),
                    'median_spread_ms': max(per[arm]) - min(per[arm])})
            diffs = [pa - pb for pa, pb in zip(per[a], per[b])]
            result['matched_delta_ms'] = {'a_minus_b_per_block': diffs,
                                          'median': statistics.median(diffs),
                                          'min': min(diffs), 'max': max(diffs),
                                          'median_pct_of_a': statistics.median(diffs) /
                                          statistics.median(per[a]) * 100.0}
            print(f'[matched] {a} - {b}: median {statistics.median(diffs):+.4f} ms')
    except Exception:
        result['status'], result['error'] = 'INVALID', traceback.format_exc()
        reasons.append('exception')
        print(result['error'], file=sys.stderr)
    result['effective_variants'] = {k: qualify.variant(v) for k, v in result['arms'].items()}
    result['qualify_version'] = qualify.QUALIFY_VERSION
    # independent re-derivation from the record itself (gate-04 f47)
    for prob in qualify.pair_problems(result, args.iters, args.repeats):
        if prob not in reasons:
            reasons.append(prob)
    if (args.iters, args.repeats) != (qualify.REGISTERED_ITERS, qualify.REGISTERED_REPEATS):
        reasons.append(f'protocol {args.repeats}x{args.iters} is not the registered 5x50') if \
            f'protocol {args.repeats}x{args.iters} is not the registered 5x50' not in reasons else None
    result['pair_eligible'] = result['status'] == 'VALID' and not reasons
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=1)
    print(f'[matched] status={result["status"]} pair_eligible={result["pair_eligible"]} {reasons} -> {out_path}')


if __name__ == '__main__':
    main()
