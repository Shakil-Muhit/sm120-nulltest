"""Merge full_*.json / matched_*.json into the arms x chunks table.

Headline rule (registered, gate-02 f20): a cell publishes a HEADLINE only if
restarts r1, r2 AND r3 (exactly the registered set) all exist, are
headline_eligible (which already requires the 5x50 protocol, validation
PASS, stable counters, manifest binding), share the same effective variant
(capture scope / adapter / compile mode), and the cell has an explicit SDR
PASS at this operating point (from any restart that ran the gate). A failed
required restart is never replaced by a later tag. Otherwise the cell is
INCOMPLETE with reasons; diagnostic medians are shown separately.
Matched pairs are eligible only if the pair verdict is True AND both arms'
cells hold an SDR PASS at that operating point.
"""

import argparse
import glob
import json
import os
import statistics

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ('A', 'B', 'C1', 'C2', 'C3', 'C1A', 'C1D', 'D', 'E', 'F', 'G', 'H', 'ANVTX')
CHUNKS = (352800, 4410, 22050, 88200)
REQUIRED_RESTARTS = ('r1', 'r2', 'r3')


def load(evidence_dir):
    runs = {}
    for path in sorted(glob.glob(os.path.join(evidence_dir, 'full_*.json'))):
        with open(path) as f:
            r = json.load(f)
        runs.setdefault((r['arm'], r['chunk']), {})[r['tag']] = r
    return runs


def sdr_status(tag_runs):
    recs = [r.get('sdr_gate') for r in tag_runs.values() if r.get('sdr_gate')]
    statuses = {r.get('status') for r in recs}
    if not recs:
        return 'NONE', None
    if 'FAIL' in statuses:
        return 'FAIL', next(r.get('mean_delta_sdr_ref_db') for r in recs if r.get('status') == 'FAIL')
    if 'PASS' in statuses:
        return 'PASS', next(r.get('mean_delta_sdr_ref_db') for r in recs if r.get('status') == 'PASS')
    return '/'.join(sorted(s for s in statuses if s)), None


def variant(r):
    return (str(r.get('capture_scope')), str(r.get('capture_adapter_active')), str(r.get('compile_mode')))


def cell(tag_runs):
    req = {t: tag_runs.get(t) for t in REQUIRED_RESTARTS}
    sdr, sdr_delta = sdr_status(tag_runs)
    c = {'restarts_found': sorted(t for t in tag_runs if t.startswith('r')),
         'statuses': {t: (r or {}).get('status', 'MISSING') for t, r in req.items()},
         'tier': sorted({r.get('registered_tier') or r.get('tier') for r in tag_runs.values() if r.get('tier')}),
         'validation': {t: ((r or {}).get('validation') or {}).get('status', 'NONE') for t, r in req.items()},
         'sdr_gate': sdr, 'mean_delta_sdr_ref_db': sdr_delta,
         'variants': sorted({'/'.join(variant(r)) for r in req.values() if r}),
         'graph_breaks': sorted({str(((r or {}).get('graph_gate') or {}).get('graph_break_count')) for r in req.values() if r}),
         'per_restart_min_of_medians_ms': {t: (r or {}).get('min_of_medians_ms') for t, r in req.items()},
         'per_restart_spread_ms': {t: (r or {}).get('median_spread_ms') for t, r in req.items()},
         'headline_status': 'INCOMPLETE', 'headline_ms': None, 'restart_spread_ms': None,
         'throughput_best_ms': None, 'reasons': []}
    for t, r in req.items():
        if r is None:
            c['reasons'].append(f'{t} missing')
        elif not r.get('headline_eligible'):
            c['reasons'].append(f"{t}: {'; '.join(r.get('ineligibility_reasons') or ['ineligible'])}")
    if len(c['variants']) > 1:
        c['reasons'].append(f"effective variants differ: {c['variants']}")
    if sdr != 'PASS':
        c['reasons'].append(f'SDR gate at this operating point: {sdr}')
    if not c['reasons']:
        vals = [req[t]['min_of_medians_ms'] for t in REQUIRED_RESTARTS]
        c['headline_status'] = 'HEADLINE'
        c['headline_ms'] = statistics.median(vals)
        c['restart_spread_ms'] = max(vals) - min(vals)
        c['throughput_best_ms'] = min(req[t]['throughput_ms_per_pass_best'] for t in REQUIRED_RESTARTS)
    vals = [v for v in c['per_restart_min_of_medians_ms'].values() if v]
    c['diagnostic_median_ms'] = statistics.median(vals) if vals else None
    return c


def fmt(v, spec='.4f'):
    return format(v, spec) if isinstance(v, (int, float)) else '-'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence-dir', default=os.path.join(REPO_ROOT, 'evidence'))
    args = ap.parse_args()
    runs = load(args.evidence_dir)
    cells = {k: cell(v) for k, v in runs.items()}
    print('| arm | chunk | r1/r2/r3 status | tier | SDR (mean ΔSDR dB) | headline | ms | restart spread | best thr ms | variant / breaks | reasons |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for chunk in CHUNKS:
        for arm in ARMS:
            c = cells.get((arm, chunk))
            if not c:
                continue
            st = '/'.join(c['statuses'][t][:4] for t in REQUIRED_RESTARTS)
            print(f"| {arm} | {chunk} | {st} | {'/'.join(c['tier'])} | {c['sdr_gate']} "
                  f"({fmt(c['mean_delta_sdr_ref_db'], '+.3f')}) | {c['headline_status']} | "
                  f"{fmt(c['headline_ms'])} | {fmt(c['restart_spread_ms'])} | {fmt(c['throughput_best_ms'])} | "
                  f"{' '.join(c['variants'])} b={'/'.join(c['graph_breaks'])} | {'; '.join(c['reasons'])[:120]} |")
    pairs = {}
    for path in sorted(glob.glob(os.path.join(args.evidence_dir, 'matched_*.json'))):
        with open(path) as f:
            m = json.load(f)
        a, b = m['pair']
        key = f"{a}_vs_{b}_c{m['chunk']}"
        sdr_ok = all(cells.get((x, m['chunk']), {}).get('sdr_gate') == 'PASS' for x in (a, b))
        eligible = bool(m.get('pair_eligible')) and sdr_ok
        pairs.setdefault(key, []).append({
            'tag': m['tag'], 'status': m['status'], 'pair_eligible': m.get('pair_eligible'),
            'sdr_both_pass': sdr_ok, 'eligible': eligible, 'delta': m.get('matched_delta_ms'),
            'reasons': m.get('ineligibility_reasons'),
            'arms': {k: {kk: v.get(kk) for kk in ('min_of_medians_ms', 'build_status', 'capture_scope', 'compile_mode')}
                     for k, v in (m.get('arms') or {}).items()}})
    if pairs:
        print('\nmatched pairs (median over alternating blocks of a - b, ms):')
        for key, lst in sorted(pairs.items()):
            for e in lst:
                d = e['delta']
                print(f"  {key} {e['tag']}: eligible={e['eligible']} "
                      + (f"delta {d['median']:+.4f} ms ({d['median_pct_of_a']:+.2f}% of a) range [{d['min']:+.4f}, {d['max']:+.4f}]" if d else 'no delta')
                      + (f" reasons={e['reasons']}" if e['reasons'] else ''))
        for key, lst in sorted(pairs.items()):
            el = [e['delta']['median'] for e in lst if e['eligible'] and e['delta']]
            if el:
                print(f"  {key}: eligible restarts {len(el)}/{len(lst)}; median of medians {statistics.median(el):+.4f} ms")
    out = os.path.join(args.evidence_dir, 'summary_full.json')
    with open(out, 'w') as f:
        json.dump({'cells': {f'{a}_{c}': v for (a, c), v in cells.items()},
                   'matched_pairs': pairs,
                   'runs': {f'{a}_{c}_{t}': r for (a, c), tr in runs.items() for t, r in tr.items()}},
                  f, indent=1)
    print(f'\nwritten: {out}')


if __name__ == '__main__':
    main()
