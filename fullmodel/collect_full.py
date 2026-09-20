"""Merge full_*.json / matched_*.json into the arms x chunks table.

Headline rule (registered): a cell publishes a HEADLINE only if restarts
r1, r2 AND r3 (exactly the registered set) all exist, are headline_eligible,
share ONE session manifest and ONE effective variant (capture scope /
adapter / compile mode / stream count), and the cell's SDR qualification —
taken ONLY from the r1 record (the registered lifecycle: gate in r1), same
manifest and variant — is PASS. A failed required restart is never replaced
by a later tag. Otherwise INCOMPLETE with reasons.
Matched pairs: a pair headline requires m1, m2 AND m3 all pair_eligible with
one manifest and one variant pair, and SDR PASS (r1 records, same manifest
and variant) for both arms' cells at that operating point.
The completeness audit (session status) lives in completeness.py.
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
REQUIRED_MATCHED = ('m1', 'm2', 'm3')


def load(evidence_dir):
    runs = {}
    for path in sorted(glob.glob(os.path.join(evidence_dir, 'full_*.json'))):
        with open(path) as f:
            r = json.load(f)
        runs.setdefault((r['arm'], r['chunk']), {})[r['tag']] = r
    return runs


def variant(r):
    return r.get('effective_variant') or '/'.join(str(r.get(k)) for k in
                                                   ('capture_scope', 'capture_adapter_active',
                                                    'compile_mode', 'stream_count'))


def sdr_from_r1(tag_runs, manifest_id, var):
    """The registered qualification record: r1, same manifest, same variant."""
    r1 = tag_runs.get('r1')
    if not r1 or not r1.get('sdr_gate'):
        return 'NONE', None, 'no SDR record in r1'
    if manifest_id and r1.get('manifest_id') != manifest_id:
        return 'NONE', None, 'r1 SDR record belongs to another manifest'
    if var and variant(r1) != var:
        return 'NONE', None, 'r1 SDR record has another effective variant'
    g = r1['sdr_gate']
    if g.get('operating_point') not in (None, r1['chunk']):
        return 'NONE', None, 'r1 SDR record operating point mismatch'
    return g.get('status'), g.get('mean_delta_sdr_ref_db'), None


def cell(tag_runs):
    req = {t: tag_runs.get(t) for t in REQUIRED_RESTARTS}
    present = [r for r in req.values() if r]
    manifests = sorted({str(r.get('manifest_id')) for r in present})
    variants = sorted({variant(r) for r in present})
    manifest_id = manifests[0] if len(manifests) == 1 else None
    var = variants[0] if len(variants) == 1 else None
    sdr, sdr_delta, sdr_note = sdr_from_r1(tag_runs, manifest_id, var)
    c = {'restarts_found': sorted(t for t in tag_runs if t.startswith('r')),
         'statuses': {t: (r or {}).get('status', 'MISSING') for t, r in req.items()},
         'tier': sorted({r.get('registered_tier') or r.get('tier') for r in present if r.get('tier')}),
         'validation': {t: ((r or {}).get('validation') or {}).get('status', 'NONE') for t, r in req.items()},
         'sdr_gate': sdr, 'mean_delta_sdr_ref_db': sdr_delta, 'sdr_note': sdr_note,
         'manifests': manifests, 'variants': variants,
         'graph_breaks': sorted({str(((r or {}).get('graph_gate') or {}).get('graph_break_count')) for r in present}),
         'per_restart_min_of_medians_ms': {t: (r or {}).get('min_of_medians_ms') for t, r in req.items()},
         'per_restart_spread_ms': {t: (r or {}).get('median_spread_ms') for t, r in req.items()},
         'headline_status': 'INCOMPLETE', 'headline_ms': None, 'restart_spread_ms': None,
         'throughput_best_ms': None, 'reasons': []}
    for t, r in req.items():
        if r is None:
            c['reasons'].append(f'{t} missing')
        elif not r.get('headline_eligible'):
            c['reasons'].append(f"{t}: {'; '.join(r.get('ineligibility_reasons') or ['ineligible'])}")
    if len(manifests) != 1 or manifests == ['None']:
        c['reasons'].append(f'restarts do not share one session manifest: {manifests}')
    if len(variants) != 1:
        c['reasons'].append(f'effective variants differ: {variants}')
    if sdr != 'PASS':
        c['reasons'].append(f'SDR qualification (r1) at this operating point: {sdr}' + (f' ({sdr_note})' if sdr_note else ''))
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
    print('| arm | chunk | r1/r2/r3 | tier | SDR r1 (mean ΔSDR dB) | headline | ms | restart spread | best thr ms | variant / breaks | reasons |')
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
                  f"{' '.join(c['variants'])} b={'/'.join(c['graph_breaks'])} | {'; '.join(c['reasons'])[:140]} |")
    pairs = {}
    for path in sorted(glob.glob(os.path.join(args.evidence_dir, 'matched_*.json'))):
        with open(path) as f:
            m = json.load(f)
        a, b = m['pair']
        key = f"{a}_vs_{b}_c{m['chunk']}"
        pairs.setdefault(key, {})[m['tag']] = m
    pair_summary = {}
    if pairs:
        print('\nmatched pairs (median over alternating blocks of a - b, ms):')
        for key, by_tag in sorted(pairs.items()):
            a, b = key.split('_vs_')[0], key.split('_vs_')[1].rsplit('_c', 1)[0]
            chunk = int(key.rsplit('_c', 1)[1])
            reasons = []
            req = {t: by_tag.get(t) for t in REQUIRED_MATCHED}
            for t, m in req.items():
                if m is None:
                    reasons.append(f'{t} missing')
                elif not m.get('pair_eligible'):
                    reasons.append(f"{t}: {'; '.join(m.get('ineligibility_reasons') or ['ineligible'])}")
            present = [m for m in req.values() if m]
            mans = sorted({str(m.get('manifest_id')) for m in present})
            vars_ = sorted({json.dumps(m.get('effective_variants'), sort_keys=True) for m in present})
            if len(mans) != 1 or mans == ['None']:
                reasons.append(f'matched restarts do not share one manifest: {mans}')
            if len(vars_) != 1:
                reasons.append('matched restarts differ in effective variants')
            for arm in (a, b):
                ca = cells.get((arm, chunk))
                if not ca or ca['sdr_gate'] != 'PASS':
                    reasons.append(f'{arm}: SDR qualification at {chunk} is {(ca or {}).get("sdr_gate", "NONE")}')
                elif mans and ca['manifests'] != mans:
                    reasons.append(f'{arm}: SDR qualification from another manifest')
            deltas = {t: (m.get('matched_delta_ms') or {}).get('median') for t, m in by_tag.items()}
            entry = {'required': {t: (m or {}).get('pair_eligible') for t, m in req.items()},
                     'deltas_ms': deltas, 'reasons': reasons,
                     'pair_headline': 'HEADLINE' if not reasons else 'INCOMPLETE'}
            if not reasons:
                entry['median_delta_ms'] = statistics.median(req[t]['matched_delta_ms']['median'] for t in REQUIRED_MATCHED)
                entry['median_delta_pct_of_a'] = statistics.median(req[t]['matched_delta_ms']['median_pct_of_a'] for t in REQUIRED_MATCHED)
            pair_summary[key] = entry
            print(f"  {key}: {entry['pair_headline']} deltas {deltas} "
                  + (f"-> median {entry['median_delta_ms']:+.4f} ms ({entry['median_delta_pct_of_a']:+.2f}% of a)" if not reasons else f"reasons {reasons}"))
    out = os.path.join(args.evidence_dir, 'summary_full.json')
    with open(out, 'w') as f:
        json.dump({'cells': {f'{a}_{c}': v for (a, c), v in cells.items()},
                   'matched_pairs': pair_summary,
                   'runs': {f'{a}_{c}_{t}': r for (a, c), tr in runs.items() for t, r in tr.items()}},
                  f, indent=1)
    print(f'\nwritten: {out}')


if __name__ == '__main__':
    main()
