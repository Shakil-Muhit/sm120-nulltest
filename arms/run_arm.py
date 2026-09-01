"""Run ONE (arm x regime) measurement in ONE fresh process.

PREREG: separate OS process per (arm x chunk regime) — scripts/run_all.sh
launches `python arms/run_arm.py` freshly for every run; this file never loops
over arms or regimes.

    python arms/run_arm.py --arm G --chunk 4410 --tag r1
    python arms/run_arm.py --arm C2 --chunk 352800 --tag outputcode --skip-timing

Phase order in main() (AUDIT-MAP[1] — the timed region excludes everything
before step 6):
    1 build reference (eager stock FrontEnd, seed 0)
    2 build arm model (same seed — state_dict equality ASSERTED), then the
      arm step: compile (C1/C2) or graph capture (B/E/G) happens here
    3 settle: compile arms run until recompilation stops (logged); other
      arms run a plain warmup loop
    4 validate: >= 2 passes, different data, vs the eager reference
    5 (--skip-timing stops here — used for the C2 output_code reads)
    6 timed region: common.timed_run — prereg 5a latency block (inside the
      cudaProfilerApi capture range) + free-running throughput block
    7 per-iteration times PRINTED and saved; one result JSON written

A validation FAIL or a build/run error marks the result INVALID in the JSON
and exits 0 — it never kills the session (prereg: does not crash the session).
"""

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402  (also puts the repo root on sys.path)

import torch  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description='one (arm x regime) run, one process')
    p.add_argument('--arm', required=True, choices=list(common.ARMS))
    p.add_argument('--chunk', required=True, type=int, choices=list(common.CHUNKS),
                   help='PREREG regimes only: 352800 (8 s) or 4410 (100 ms)')
    p.add_argument('--iters', type=int, default=common.DEFAULT_ITERS,
                   help='timed iterations per block (PREREG: >= 50)')
    p.add_argument('--warmup', type=int, default=common.DEFAULT_WARMUP,
                   help='warmup passes for non-compile arms')
    p.add_argument('--streams', type=int, default=common.DEFAULT_STREAMS,
                   help='stream count for arms F/G')
    p.add_argument('--tag', default='adhoc',
                   help='run label: r1..rN for the restart set (collect.py '
                        'headlines any r<digits> tag), trace for the nsys '
                        'run, outputcode for the kernel-inventory run; the '
                        '"adhoc" default keeps manual runs out of headlines')
    p.add_argument('--settle-max', type=int, default=60,
                   help='C1/C2: max settle passes before declaring capped')
    p.add_argument('--settle-consecutive', type=int, default=5,
                   help='C1/C2: passes with unchanged dynamo counters => settled')
    p.add_argument('--out-dir', default=os.path.join(common._REPO_ROOT, 'evidence'))
    p.add_argument('--skip-timing', action='store_true',
                   help='build + settle + validate only (C2 output_code reads)')
    args = p.parse_args()
    if args.iters < common.DEFAULT_ITERS and not args.skip_timing:
        # PREREG: >= 50 timed iters — refuse quietly-degraded runs
        p.error(f'--iters must be >= {common.DEFAULT_ITERS} (prereg)')
    return args


def main():
    args = parse_args()
    assert torch.cuda.is_available(), 'this harness measures a CUDA GPU'
    device = 'cuda'
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir, f'arm_{args.arm}_c{args.chunk}_{args.tag}.json')

    print(f'[run_arm] arm={args.arm} chunk={args.chunk} tag={args.tag} '
          f'iters={args.iters} streams={args.streams}')

    latency_ms, throughput_ms, validation, settle = [], None, None, None
    info = {'capture_scope': None, 'stream_count': None,
            'compile_mode': None, 'notes': []}
    status, error = 'VALID', None
    try:
        # 1-2: build (everything heavy happens here, outside the timed region)
        reference = common.build_stock_frontend(device)
        fe = common.build_stock_frontend(device)
        ref_sd, fe_sd = reference.state_dict(), fe.state_dict()
        assert ref_sd.keys() == fe_sd.keys() and all(
            torch.equal(ref_sd[k], fe_sd[k]) for k in ref_sd), \
            'arm model and eager reference must start from identical weights'
        step = common.build_arm_step(
            args.arm, fe, args.chunk, device, args.streams, info)
        if info['capture_scope'] is not None:
            print(f'[run_arm] capture scope: {info["capture_scope"]}')

        x = common.make_input(args.chunk, device, common.TIMING_SEED)

        # 3: settle / warmup (excluded from the timed region)
        if args.arm in ('C1', 'C2'):
            settle = common.settle_compile(
                step, x, args.settle_max, args.settle_consecutive)
            print(f'[run_arm] compile settled: {settle["settle_iters"]} passes, '
                  f'capped={settle["settle_capped"]} '
                  f'({settle["settle_wall_s"]} s)')
            if settle['settle_capped']:
                info['notes'].append(
                    'compile settle hit --settle-max without stabilizing; '
                    'see TORCH_LOGS capture for recompile reasons')
        else:
            with torch.no_grad():
                for _ in range(args.warmup):
                    step(x)
            torch.cuda.synchronize()

        # 4: validation (outside the timed region, before it)
        validation = common.validate(step, reference, args.arm, args.chunk, device)
        print(f'[run_arm] validation: {validation["status"]} '
              f'{[p["max_abs_diff"] for p in validation["passes"]]} '
              f'(rtol={validation["rtol"]}, atol={validation["atol"]})')
        if validation['status'] == 'FAIL':
            status = 'INVALID'   # PREREG: marked invalid, session continues

        # 5-6: timed region (prereg 5a; skipped for output_code runs)
        if args.skip_timing:
            info['notes'].append('--skip-timing run (output_code capture): '
                                 'no timed region executed')
        else:
            latency_ms, throughput_ms = common.timed_run(step, x, args.iters)
            # 7: PREREG — per-iteration times printed, never only aggregates
            for i, ms in enumerate(latency_ms):
                print(f'  iter {i:3d}: {ms:9.4f} ms')

    except Exception:
        status, error = 'INVALID', traceback.format_exc()
        print(f'[run_arm] ERROR (recorded, not fatal to the session):\n{error}',
              file=sys.stderr)

    # 'iters' records passes ACTUALLY completed in the latency block (0 on
    # crash / --skip-timing) — busy-per-pass extraction must never divide by
    # a requested count the run did not reach
    result = common.result_json(
        args.arm, args.chunk, args.tag, len(latency_ms),
        latency_ms, throughput_ms, validation, info, settle, status, error)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=1)

    if latency_ms:
        print(f'[run_arm] latency median/min/max = '
              f'{result["latency_median_ms"]:.4f} / {result["latency_min_ms"]:.4f} / '
              f'{result["latency_max_ms"]:.4f} ms | '
              f'throughput = {throughput_ms:.4f} ms/pass | '
              f'serialization delta = {result["serialization_delta_pct"]:+.2f}%')
    print(f'[run_arm] status={status} -> {out_path}')


if __name__ == '__main__':
    main()
