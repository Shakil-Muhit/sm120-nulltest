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
import provenance                 # noqa: E402
import timing as tm               # noqa: E402


def build_one(arm, chunk, streams, gen_dir, paths, settle_max):
    from torch._dynamo.utils import counters
    counters.clear()               # this arm's gate reflects only its own compilation
    model, config, use_amp, ctx, ctx_nocache = loader.load_public_model(
        paths['msst'], paths['ckpt'], paths['config'])
    x = tm.make_input(chunk, tm.TIMING_SEED)
    info = {'capture_scope': None, 'capture_adapter': None, 'capture_attempts': [],
            'stream_count': None, 'compile_mode': None, 'notes': []}
    step = arms_mod.build_arm_step(arm, model, x, ctx_nocache, streams, info, gen_dir)
    settle = gate = None
    if arm in arms_mod.COMPILE_ARMS:
        settle = tm.settle_compile(step, x, ctx, settle_max, 5)
        gate = tm.graph_gate(arm in arms_mod.REQUIRE_ZERO_BREAKS)
    else:
        with torch.no_grad(), ctx():
            for _ in range(10):
                step(x)
        torch.cuda.synchronize()
    return {'model': model, 'step': step, 'ctx': ctx, 'x': x, 'info': info,
            'settle': settle, 'gate': gate, 'use_amp': use_amp}


def joint_settle(built, arms, rounds=2, passes=3):
    """Alternate the two steps; stable iff counters unchanged across the
    last two alternation rounds."""
    snaps = []
    for r in range(rounds + 1):
        for arm in arms:
            b = built[arm]
            with torch.no_grad(), b['ctx']():
                for _ in range(passes):
                    b['step'](b['x'])
        torch.cuda.synchronize()
        snaps.append(tm.dynamo_counters())
    stable = all(tm.compile_stable(a, c) for a, c in zip(snaps, snaps[1:]))
    return {'stable': stable, 'rounds': rounds, 'snapshots': snaps}


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
    args = ap.parse_args()
    assert torch.cuda.is_available()
    os.makedirs(args.evidence_dir, exist_ok=True)
    ref_dir = args.ref_dir or os.path.join(args.evidence_dir, 'ref')
    a, b = args.pair
    out_path = os.path.join(args.evidence_dir, f'matched_{a}_{b}_c{args.chunk}_{args.tag}.json')
    result = {'pair': [a, b], 'chunk': args.chunk, 'tag': args.tag, 'status': 'VALID',
              'error': None, 'env': tm.env_fingerprint(), 'arms': {}, 'blocks': [],
              'torch_logs_env': os.environ.get('TORCH_LOGS'), 'manifest_id': None,
              'pair_eligible': False, 'ineligibility_reasons': []}
    reasons = result['ineligibility_reasons']
    try:
        paths = loader.paths_from_env()
        result['provenance'] = provenance.verify(paths)
        if not args.no_manifest:
            result['manifest_id'] = provenance.load_manifest(args.evidence_dir)['manifest_id']
            tm.bind_refs(ref_dir, result['manifest_id'])
        else:
            reasons.append('not bound to a session manifest')
        built = {}
        for arm in (a, b):
            gen_dir = os.path.join(args.evidence_dir, 'generated', f'matched_{arm}_c{args.chunk}_{args.tag}')
            try:
                built[arm] = build_one(arm, args.chunk, args.streams, gen_dir, paths, args.settle_max)
                bi = built[arm]
                result['arms'][arm] = {'build_status': 'OK', 'settle': bi['settle'], 'graph_gate': bi['gate'],
                                       'capture_scope': bi['info'].get('capture_scope'),
                                       'capture_adapter_active': bi['info'].get('capture_adapter_active'),
                                       'compile_mode': bi['info'].get('compile_mode')}
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
            result['joint_settle'] = {'stable': js['stable'], 'rounds': js['rounds']}
            if not js['stable']:
                reasons.append('joint alternation did not settle')
            for arm in (a, b):
                v = tm.validate(built[arm]['step'], built[arm]['ctx'], arms_mod.TIER[arm],
                                args.chunk, ref_dir)
                result['arms'][arm]['validation'] = v
                if v['status'] != 'PASS':
                    reasons.append(f'{arm}: validation {v["status"]}')
                print(f'[matched] {arm}: validation {v["status"]} '
                      f'{[p.get("max_abs_diff") for p in v["passes"]]}')
            per = {a: [], b: []}
            prev = tm.dynamo_counters()
            block_stable = True
            for r in range(args.repeats):
                for arm in (a, b):
                    t0 = time.time()
                    lat, thr = tm.timed_run(built[arm]['step'], built[arm]['x'],
                                            built[arm]['ctx'], args.iters)
                    t1 = time.time()
                    cur = tm.dynamo_counters()
                    stable = tm.compile_stable(prev, cur)
                    block_stable &= stable
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
    result['pair_eligible'] = result['status'] == 'VALID' and not reasons
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=1)
    print(f'[matched] status={result["status"]} pair_eligible={result["pair_eligible"]} {reasons} -> {out_path}')


if __name__ == '__main__':
    main()
