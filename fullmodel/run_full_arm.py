"""Run ONE (arm x chunk) full-model measurement in ONE fresh process.

PREREG: separate OS process per run — scripts/session*.sh launch this
freshly for every run; this file never loops over arms or chunks.

    python fullmodel/run_full_arm.py --arm C3 --chunk 352800 --tag r1 --sdr
    python fullmodel/run_full_arm.py --arm A --chunk 352800 --tag ref --save-reference --sdr --skip-timing

Phase order in main() (timed region excludes everything before step 6):
    0 provenance gate (pins verified) + session-manifest binding
    1 load the public model (loader.py)
    2 build the arm step (fullarms.py): compile / capture / swaps happen here
    3 settle (compile arms: until Dynamo counters stop moving, logged) or warmup
    4 graph gate (compile arms; C3/H require zero graph breaks or INVALID)
    5 validate vs the SAVED eager-A outputs (two seeds, different data)
    6 timed region: five repeats of the 5a block (timing.five_repeats)
    7 SDR gate (--sdr) AFTER the timed region (registered placement)
    8 one JSON per run: <evidence>/full_<arm>_c<chunk>_<tag>.json with
      headline_eligible + machine-readable reasons

Counter snapshots are taken after settle, after validation, before and
after timing, after SDR; all are preserved. A validation FAIL, a graph-gate
FAIL, an unstable recording state, or an exception marks the run INVALID /
ineligible in its JSON and exits 0 (the session continues, loudly).
"""

import argparse
import json
import os
import sys
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
import sdr_gate                   # noqa: E402
import timing as tm               # noqa: E402

REGISTERED_ITERS, REGISTERED_REPEATS = 50, 5


def parse_args():
    p = argparse.ArgumentParser(description='one (arm x chunk) full-model run')
    p.add_argument('--arm', required=True, choices=list(arms_mod.ARMS))
    p.add_argument('--chunk', required=True, type=int, choices=list(arms_mod.CHUNKS))
    p.add_argument('--tag', default='adhoc')
    p.add_argument('--iters', type=int, default=REGISTERED_ITERS)
    p.add_argument('--repeats', type=int, default=REGISTERED_REPEATS)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--streams', type=int, default=arms_mod.DEFAULT_STREAMS)
    p.add_argument('--settle-max', type=int, default=60)
    p.add_argument('--settle-consecutive', type=int, default=5)
    p.add_argument('--evidence-dir', '--out-dir', dest='evidence_dir',
                   default=os.path.join(_ROOT, 'evidence'))
    p.add_argument('--ref-dir', default=None, help='default <evidence-dir>/ref')
    p.add_argument('--no-manifest', action='store_true',
                   help='shakedown only: do not bind to a session manifest (run is ineligible)')
    p.add_argument('--save-reference', action='store_true',
                   help='arm A only: write the validation/SDR reference outputs')
    p.add_argument('--sdr', action='store_true', help='run the SDR gate at this operating point')
    p.add_argument('--clips-dir', default=os.environ.get('CLIPS_DIR',
                   os.path.join(os.path.expanduser('~'), 'work', 'sdr_clips')))
    p.add_argument('--clips-manifest', default=os.path.join(_ROOT, 'configs', 'sdr_clips.json'))
    p.add_argument('--planted-fault', action='store_true')
    p.add_argument('--planted-const', type=float, default=None)
    p.add_argument('--force-tier', choices=['tight', 'loose'], default=None,
                   help='validator-check only: evaluate arm A under the loose comparator')
    p.add_argument('--skip-timing', action='store_true')
    p.add_argument('--nvtx-pass', action='store_true',
                   help='outer NVTX range per pass (traced runs; use --repeats 1)')
    args = p.parse_args()
    if args.iters < REGISTERED_ITERS and not args.skip_timing:
        p.error('--iters must be >= 50 (prereg 5a)')
    if args.save_reference and args.arm != 'A':
        p.error('--save-reference is for arm A only')
    if args.force_tier and not args.planted_fault:
        p.error('--force-tier is only for the planted-fault validator check')
    if args.ref_dir is None:
        args.ref_dir = os.path.join(args.evidence_dir, 'ref')
    return args


def main():
    args = parse_args()
    assert torch.cuda.is_available(), 'this harness measures a CUDA GPU'
    os.makedirs(args.evidence_dir, exist_ok=True)
    out_path = os.path.join(args.evidence_dir, f'full_{args.arm}_c{args.chunk}_{args.tag}.json')
    gen_dir = os.path.join(args.evidence_dir, 'generated', f'{args.arm}_c{args.chunk}_{args.tag}')
    print(f'[full] arm={args.arm} chunk={args.chunk} tag={args.tag} iters={args.iters} '
          f'repeats={args.repeats} streams={args.streams} planted={args.planted_fault}')

    info = {'capture_scope': None, 'capture_adapter': None, 'capture_attempts': [],
            'stream_count': None, 'compile_mode': None, 'notes': []}
    status, error = 'VALID', None
    settle = gate = validation = sdr = timing_rec = None
    snapshots = {}
    use_amp = None
    prov = manifest_id = None
    tier = args.force_tier or arms_mod.TIER[args.arm]
    try:
        paths = loader.paths_from_env()
        prov = provenance.verify(paths)
        if args.no_manifest:
            info['notes'].append('--no-manifest: shakedown run, never headline-eligible')
        else:
            manifest = provenance.load_manifest(args.evidence_dir)
            manifest_id = manifest['manifest_id']
        if args.save_reference and manifest_id:
            tm.bind_refs(args.ref_dir, manifest_id, write=True)
        elif not args.save_reference:
            tm.bind_refs(args.ref_dir, manifest_id)

        model, config, use_amp, ctx, ctx_nocache = loader.load_public_model(
            paths['msst'], paths['ckpt'], paths['config'])
        x = tm.make_input(args.chunk, tm.TIMING_SEED)
        step = arms_mod.build_arm_step(args.arm, model, x, ctx_nocache,
                                       args.streams, info, gen_dir)
        if args.planted_fault:
            const = args.planted_const if args.planted_const is not None else arms_mod.PLANTED_FAULT_CONST
            step = arms_mod.plant_fault(step, const)
            info['planted_fault_const'] = const
        if info['capture_scope'] is not None:
            print(f'[full] capture scope: {info["capture_scope"]} (adapter {info.get("capture_adapter_active")})')

        compiled = args.arm in arms_mod.COMPILE_ARMS
        if compiled:
            settle = tm.settle_compile(step, x, ctx, args.settle_max, args.settle_consecutive)
            print(f'[full] compile settled: {settle["settle_iters"]} passes, '
                  f'capped={settle["settle_capped"]} ({settle["settle_wall_s"]} s)')
            if settle['settle_capped']:
                info['notes'].append('compile settle hit --settle-max without stabilizing')
            gate = tm.graph_gate(args.arm in arms_mod.REQUIRE_ZERO_BREAKS)
            print(f'[full] graph gate: {gate["status"]} breaks={gate["graph_break_count"]} '
                  f'unimplemented={gate["unimplemented_count"]} cudagraph={gate["cudagraph_counters"]}')
            if gate['status'] == 'FAIL':
                status = 'INVALID'
                info['notes'].append('graph gate FAILED (registered zero-break arm)')
            snapshots['after_settle'] = settle['dynamo_counters']
            snapshots['tree_after_settle'] = tm.cudagraph_tree_state()
        else:
            with torch.no_grad(), ctx():
                for _ in range(args.warmup):
                    step(x)
            torch.cuda.synchronize()

        validation = tm.validate(step, ctx, tier, args.chunk, args.ref_dir,
                                 save_reference=args.save_reference)
        print(f'[full] validation: {validation["status"]} '
              f'{[p.get("max_abs_diff") for p in validation["passes"]]} tier={validation["tier"]}')
        if validation['status'] == 'FAIL':
            status = 'INVALID'
        if compiled:
            snapshots['after_validation'] = tm.dynamo_counters()

        if args.skip_timing:
            info['notes'].append('--skip-timing: no timed region executed')
        else:
            if compiled:
                snapshots['before_timing'] = tm.dynamo_counters()
            timing_rec = tm.five_repeats(step, x, ctx, args.iters, args.repeats,
                                         nvtx_pass=args.nvtx_pass)
            if compiled:
                snapshots['after_timing'] = tm.dynamo_counters()
                snapshots['tree_after_timing'] = tm.cudagraph_tree_state()
            for b in timing_rec['blocks']:
                print(f'  block {b["repeat"]}: median {b["latency_median_ms"]:.4f} ms '
                      f'(min {b["latency_min_ms"]:.4f}, max {b["latency_max_ms"]:.4f}) '
                      f'throughput {b["throughput_ms_per_pass"]:.4f} ms/pass')

        if args.sdr:                                   # registered: after the timed region
            manifest_clips, clips = sdr_gate.load_clips(args.clips_dir, args.clips_manifest)
            sdr = sdr_gate.run_gate(step, ctx, clips, args.ref_dir, args.chunk,
                                    save_reference=args.save_reference)
            sdr['manifest_revision'] = manifest_clips.get('revision')
            print(f'[full] SDR gate: {sdr["status"]} mean_sdr_ref={sdr.get("mean_sdr_ref_db")} '
                  f'mean_delta={sdr.get("mean_delta_sdr_ref_db")}')
            if compiled:
                snapshots['after_sdr'] = tm.dynamo_counters()

        if compiled and settle is not None:
            seq = [snapshots.get(k) for k in ('after_settle', 'after_validation', 'before_timing',
                                              'after_timing', 'after_sdr') if snapshots.get(k)]
            info['counters_stable'] = all(tm.compile_stable(a, b) for a, b in zip(seq, seq[1:]))
            if not info['counters_stable']:
                info['notes'].append('compile/recording counters changed between measured phases')
    except Exception:
        status, error = 'INVALID', traceback.format_exc()
        print(f'[full] ERROR (recorded, not fatal to the session):\n{error}', file=sys.stderr)

    reasons = []
    if status != 'VALID':
        reasons.append(f'status {status}')
    if manifest_id is None:
        reasons.append('not bound to a session manifest')
    if prov is None:
        reasons.append('provenance not verified')
    if not timing_rec:
        reasons.append('no timed region')
    else:
        if timing_rec['repeats'] < REGISTERED_REPEATS or timing_rec['iters_per_block'] < REGISTERED_ITERS:
            reasons.append(f"timing protocol incomplete ({timing_rec['repeats']} repeats x "
                           f"{timing_rec['iters_per_block']} iters; registered 5 x 50)")
        if any(len(b['latency_ms_per_iter']) < REGISTERED_ITERS for b in timing_rec['blocks']):
            reasons.append('a latency block completed fewer than 50 passes')
    if args.nvtx_pass:
        reasons.append('traced/attribution run (never a headline)')
    if args.planted_fault:
        reasons.append('planted fault run')
    if validation and validation.get('status') != 'PASS':
        reasons.append(f"validation {validation.get('status')}")
    if sdr is not None and sdr.get('status') != 'PASS':
        reasons.append(f"SDR gate {sdr.get('status')} (excluded from speed comparisons)")
    if info.get('counters_stable') is False:
        reasons.append('compile/recording counters unstable during measurement')
    if settle and settle.get('settle_capped'):
        reasons.append('settle capped')
    if gate and gate.get('status') == 'FAIL':
        reasons.append('graph gate FAIL')

    result = {
        'arm': args.arm, 'chunk': args.chunk, 'tag': args.tag, 'status': status,
        'error': error, 'argv': sys.argv, 'env': tm.env_fingerprint(),
        'manifest_id': manifest_id, 'provenance': prov,
        'torch_logs_env': os.environ.get('TORCH_LOGS'),
        'use_amp': use_amp, 'tier': tier, 'registered_tier': arms_mod.TIER[args.arm],
        'forced_tier': args.force_tier,
        'iters_requested': args.iters, 'repeats_requested': args.repeats,
        'timing': timing_rec,
        'min_of_medians_ms': timing_rec['min_of_medians_ms'] if timing_rec else None,
        'median_spread_ms': timing_rec['median_spread_ms'] if timing_rec else None,
        'throughput_ms_per_pass_best': timing_rec['throughput_ms_per_pass_best'] if timing_rec else None,
        'validation': validation, 'sdr_gate': sdr, 'settle': settle, 'graph_gate': gate,
        'counter_snapshots': snapshots, 'counters_stable': info.get('counters_stable'),
        'capture_scope': info.get('capture_scope'),
        'capture_adapter': info.get('capture_adapter'),
        'capture_adapter_active': info.get('capture_adapter_active'),
        'capture_adapter_istft': info.get('capture_adapter_istft'),
        'capture_attempts': info.get('capture_attempts'),
        'stream_count': info.get('stream_count'), 'compile_mode': info.get('compile_mode'),
        'c3_fixes': info.get('c3_fixes'), 'padbatch': info.get('padbatch'),
        'nvtx_boundary_mode': info.get('nvtx_boundary_mode'),
        'planted_fault_const': info.get('planted_fault_const'),
        'nvtx_pass': args.nvtx_pass,
        'generated_dir': gen_dir if os.path.isdir(gen_dir) else None,
        'notes': info.get('notes', []),
        'headline_eligible': not reasons, 'ineligibility_reasons': reasons,
    }
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=1)
    if timing_rec:
        print(f'[full] min-of-medians = {timing_rec["min_of_medians_ms"]:.4f} ms | '
              f'spread = {timing_rec["median_spread_ms"]:.4f} ms | '
              f'best throughput = {timing_rec["throughput_ms_per_pass_best"]:.4f} ms/pass')
    print(f'[full] status={status} eligible={not reasons} {reasons} -> {out_path}')


if __name__ == '__main__':
    main()
