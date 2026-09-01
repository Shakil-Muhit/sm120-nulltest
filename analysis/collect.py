"""Merge per-run JSONs from evidence/ into the arms x regimes table.

PREREG anchors:
  - ">= 3 process restarts for the headline numbers": the headline latency per
    (arm x regime) is the MEDIAN OF THE PER-RESTART MEDIANS over tags r1/r2/r3
    (conservative reading — the prereg does not name the cross-restart
    statistic; flagged in README "Deviations & ambiguities"). Per-iteration
    lists from every run are preserved verbatim in summary.json.
  - decision rule: R = median wall-time / GPU-busy-time on arm G at the 4410
    regime, busy from the nsys kernel sum of the SAME run — so R pairs the
    trace-tagged run's own latency median with its trace's busy time
    (analysis/traces.sh writes busy_*.json; headline walls stay untraced).
  - 5a escalation: serialization_delta_pct surfaced per cell; > 5% at the
    4410 regime is flagged for evaluation under both protocols.

No plots, no further analysis — the collection table only (session-owner
instruction). Usage:
    python analysis/collect.py                  # table + evidence/summary.json
    python analysis/collect.py --best-generic   # print best generic arm @4410
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVIDENCE = os.path.join(REPO_ROOT, 'evidence')

ARMS = ('A', 'B', 'C1', 'C2', 'D', 'E', 'F', 'G')
CHUNKS = (352800, 4410)
GENERIC_ARMS = ('A', 'B', 'C1', 'C2', 'F')   # no custom kernel: the generic-
                                             # tooling side of the null test
# headline set = every r<digits> tag (r1..rN — run_all.sh RESTARTS knob may
# exceed 3); 'trace' pairs with busy; 'outputcode'/'adhoc' never headline
RESTART_TAG_RE = re.compile(r'^r\d+$')

# GRAPH-TRACE-FAILURE GUARD (fix-pass 2026-08-18): arms whose timed region
# replays a CUDA graph (manual capture: B/E/G; inductor cudagraph trees under
# reduce-overhead / max-autotune in torch 2.8: C1/C2). If such an arm's trace
# shows < KERNELS_PER_PASS_FLOOR kernels/pass, --cuda-graph-trace=node did not
# take effect and the busy numbers are wrong in the thesis-favoring direction:
# the cell's busy is marked INVALID and never feeds R. Mirrors the primary
# guard in analysis/traces.sh (defense in depth against hand-made busy files).
GRAPH_ARMS = ('B', 'E', 'G', 'C1', 'C2')
KERNELS_PER_PASS_FLOOR = 100


def load_runs(evidence_dir):
    runs = {}
    for path in sorted(glob.glob(os.path.join(evidence_dir, 'arm_*.json'))):
        with open(path) as f:
            r = json.load(f)
        runs.setdefault((r['arm'], r['chunk']), {})[r['tag']] = r
    return runs


def load_busy(evidence_dir):
    busy = {}
    for path in sorted(glob.glob(os.path.join(evidence_dir, 'busy_*.json'))):
        with open(path) as f:
            b = json.load(f)
        busy[(b['arm'], b['chunk'])] = b
    return busy


def cell_summary(arm, tag_runs, busy_entry):
    """Aggregate one (arm x chunk) cell from its runs."""
    busy_rejected = None
    if busy_entry and arm in GRAPH_ARMS:
        kpp = busy_entry.get('kernels_per_pass')
        # None/absent counts as failing the floor: a graph arm with NO kernel
        # count must not contribute busy either (defense in depth is only
        # defense if a hand-made file can't dodge it by omitting the field)
        if kpp is None or kpp < KERNELS_PER_PASS_FLOOR:
            shown = 'absent' if kpp is None else f'{kpp:.1f}'
            busy_rejected = (
                f'INVALID busy: {shown} kernels/pass (floor '
                f'{KERNELS_PER_PASS_FLOOR}) on a graph-replay arm — '
                f'--cuda-graph-trace=node likely ineffective; busy rejected')
            busy_entry = None            # never contributes busy numbers or R
    restarts = [run for tag, run in sorted(tag_runs.items())
                if RESTART_TAG_RE.match(tag)]
    valid = [r for r in restarts if r['status'] == 'VALID' and r['latency_ms_per_iter']]
    cell = {
        'n_restarts_found': len(restarts),
        'n_restarts_valid': len(valid),
        'validation': sorted({(r.get('validation') or {}).get('status', 'ERROR')
                              for r in restarts}) if restarts else [],
        'capture_scope': sorted({str(r.get('capture_scope')) for r in restarts})
                         if restarts else [],
        'stream_count': sorted({str(r.get('stream_count')) for r in restarts})
                        if restarts else [],
        'headline_latency_median_ms': None,
        'per_restart_latency_median_ms': [],
        'latency_min_ms': None,
        'latency_max_ms': None,
        'throughput_ms_per_pass': None,
        'serialization_delta_pct': None,
        'busy_ms_per_pass': None,
        'kernels_per_pass': None,
        'busy_rejected': busy_rejected,   # graph-trace-failure guard verdict
        'trace_status': None,
        'trace_validation': None,
        'trace_latency_median_ms': None,
        'wall_over_busy_same_run': None,
    }
    if valid:
        meds = [r['latency_median_ms'] for r in valid]
        cell['per_restart_latency_median_ms'] = meds
        cell['headline_latency_median_ms'] = statistics.median(meds)
        cell['latency_min_ms'] = min(r['latency_min_ms'] for r in valid)
        cell['latency_max_ms'] = max(r['latency_max_ms'] for r in valid)
        cell['throughput_ms_per_pass'] = statistics.median(
            r['throughput_ms_per_pass'] for r in valid)
        cell['serialization_delta_pct'] = statistics.median(
            r['serialization_delta_pct'] for r in valid)
    if busy_entry:
        cell['busy_ms_per_pass'] = busy_entry.get('busy_ms_per_pass')
        cell['kernels_per_pass'] = busy_entry.get('kernels_per_pass')
    trace_run = tag_runs.get('trace')
    if trace_run:
        cell['trace_status'] = trace_run.get('status')
        cell['trace_validation'] = (trace_run.get('validation') or {}).get('status')
        # R only from a VALID, validation-PASS trace run — an INVALID run must
        # never silently produce the pre-registered decision number
        if (trace_run.get('status') == 'VALID'
                and cell['trace_validation'] == 'PASS'
                and trace_run.get('latency_median_ms')
                and cell['busy_ms_per_pass']):
            cell['trace_latency_median_ms'] = trace_run['latency_median_ms']
            cell['wall_over_busy_same_run'] = (
                trace_run['latency_median_ms'] / cell['busy_ms_per_pass'])
    return cell


def best_generic(cells):
    """Best generic arm at the 4410 regime: lowest headline latency median
    among A/B/C1/C2/F cells whose validation passed."""
    candidates = []
    for arm in GENERIC_ARMS:
        c = cells.get((arm, 4410))
        if c and c['headline_latency_median_ms'] is not None \
                and c['validation'] == ['PASS']:
            candidates.append((c['headline_latency_median_ms'], arm))
    if not candidates:
        return None
    return min(candidates)[1]


def fmt(v, spec='.4f'):
    return format(v, spec) if isinstance(v, (int, float)) else '-'


def print_table(cells):
    hdr = (f'{"arm":<4}{"chunk":>7}{"runs":>5}{"val":>6}'
           f'{"lat med ms":>12}{"lat min":>9}{"lat max":>9}'
           f'{"thr ms":>9}{"delta%":>8}{"busy ms":>9}{"kern/pass":>10}'
           f'  scope/streams')
    print(hdr)
    print('-' * len(hdr))
    for chunk in CHUNKS:
        for arm in ARMS:
            c = cells.get((arm, chunk))
            if not c:
                print(f'{arm:<4}{chunk:>7}{"-":>5}{"-":>6}  (no runs found)')
                continue
            val = '/'.join(c['validation']) or '-'
            scope = '/'.join(s for s in c['capture_scope'] if s != 'None')
            strm = '/'.join(s for s in c['stream_count'] if s != 'None')
            extra = ' '.join(x for x in (scope, f'{strm}str' if strm else '') if x)
            print(f'{arm:<4}{chunk:>7}{c["n_restarts_valid"]:>5}{val:>6}'
                  f'{fmt(c["headline_latency_median_ms"]):>12}'
                  f'{fmt(c["latency_min_ms"]):>9}{fmt(c["latency_max_ms"]):>9}'
                  f'{fmt(c["throughput_ms_per_pass"]):>9}'
                  f'{fmt(c["serialization_delta_pct"], "+.2f"):>8}'
                  f'{fmt(c["busy_ms_per_pass"]):>9}'
                  f'{fmt(c["kernels_per_pass"], ".1f"):>10}  {extra}')
        print()


def print_decision_rule(cells):
    """PREREG decision rule, computed at the 100 ms regime on arm G —
    R = median wall / GPU-busy, both from the trace-tagged run."""
    g = cells.get(('G', 4410))
    if not g or g['wall_over_busy_same_run'] is None:
        why = 'need arm G 4410 trace run + busy extraction (analysis/traces.sh)'
        if g and g.get('busy_rejected'):
            why = (f'{g["busy_rejected"]} — re-trace with a graph-node-capable '
                   f'nsys (docs/profiling_crib.md section 4.1)')
        elif g and g.get('trace_status') and g['trace_status'] != 'VALID':
            why = (f'arm G 4410 trace run is {g["trace_status"]} '
                   f'(validation: {g.get("trace_validation")}) — R must not '
                   f'be computed from an invalid run; re-run the trace')
        print(f'decision rule: R NOT COMPUTABLE — {why}')
        return
    r = g['wall_over_busy_same_run']
    print(f'DECISION RULE (prereg, arm G @ 4410, same-run pairing):')
    print(f'  R = wall {g["trace_latency_median_ms"]:.4f} ms / '
          f'busy {g["busy_ms_per_pass"]:.4f} ms = {r:.3f}')
    # thresholds are approximate ('~') in the prereg — bucket labels only
    if r <= 1.25:
        print('  bucket: R <= ~1.25 -> gaps closed by overlap+replay; '
              'launch-collapse headroom is small')
    elif r >= 1.5:
        print('  bucket: R >= ~1.5 -> measured headroom stands; grouped kernel '
              'has a quantitative target')
    else:
        print('  bucket: between -> gray zone; week-4 gate weighs it against '
              'candidate-(b) evidence')
    print('  (structural criterion is separate: inspect the best-generic and G '
          'timelines for the ~60 separate GEMM kernels per pass)')


def print_escalation(cells):
    flagged = [(arm, c['serialization_delta_pct'])
               for (arm, chunk), c in cells.items()
               if chunk == 4410 and c['serialization_delta_pct'] is not None
               and abs(c['serialization_delta_pct']) > 5.0]
    if flagged:
        print('PREREG 5a ESCALATION: free-running delta > 5% at the 4410 '
              f'regime on: {flagged}')
        print('  -> evaluate the decision rule under BOTH protocols; escalate '
              'any verdict disagreement (human step).')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence-dir', default=EVIDENCE)
    ap.add_argument('--best-generic', action='store_true',
                    help='print only the best generic arm at 4410 (for traces.sh)')
    args = ap.parse_args()

    runs = load_runs(args.evidence_dir)
    busy = load_busy(args.evidence_dir)
    cells = {key: cell_summary(key[0], tag_runs, busy.get(key))
             for key, tag_runs in runs.items()}

    if args.best_generic:
        b = best_generic(cells)
        if b is None:
            print('NONE', file=sys.stderr)
            sys.exit(1)
        print(b)
        return

    print_table(cells)
    for (arm, chunk), c in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if c.get('busy_rejected'):
            print(f'GRAPH-TRACE GUARD: {arm}@{chunk}: {c["busy_rejected"]}')
    print_decision_rule(cells)
    print_escalation(cells)

    # summary.json preserves everything, per-iteration lists included
    summary = {
        'cells': {f'{arm}_{chunk}': cells[(arm, chunk)]
                  for (arm, chunk) in cells},
        'best_generic_arm_4410': best_generic(cells),
        'runs': {f'{arm}_{chunk}_{tag}': run
                 for (arm, chunk), tag_runs in runs.items()
                 for tag, run in tag_runs.items()},
    }
    out = os.path.join(args.evidence_dir, 'summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=1)
    print(f'\nwritten: {out}')


if __name__ == '__main__':
    main()
