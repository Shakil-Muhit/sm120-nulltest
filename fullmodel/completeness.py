"""Session completeness audit (gate-03 f43): the registered mandatory cells
per session, each marked COMPLETE / INCOMPLETE with its reason, and a
mechanical session verdict written to <evidence>/session_status.json.
An INCOMPLETE session is reported as such; it is never salvaged by relaxing
validation or timing requirements.
"""

import argparse
import json
import os

MATRIX_S1 = [(a, c) for c in (352800, 4410) for a in ('A', 'C3', 'C1', 'H', 'G', 'B', 'E', 'D', 'F')] + \
            [('C1D', 352800), ('C2', 352800), ('C2', 4410)]
PAIRS_S1 = [('C1', 'C3'), ('C1', 'C1D'), ('C3', 'H')]
SWEEP_S2 = [(a, c) for c in (22050, 88200) for a in ('A', 'C3', 'G', 'H')]
TRACES_S2 = [(a, c) for c in (352800, 4410) for a in ('A', 'ANVTX', 'C3', 'G', 'H', 'C1', 'B', 'E', 'F')] + [('C2', 4410)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence-dir', required=True)
    ap.add_argument('--session', choices=['session1', 'session2'], required=True)
    args = ap.parse_args()
    ev = args.evidence_dir
    summary = json.load(open(os.path.join(ev, 'summary_full.json')))
    cells, pairs = summary['cells'], summary.get('matched_pairs', {})
    items = []
    if args.session == 'session1':
        for a, c in MATRIX_S1:
            cell = cells.get(f'{a}_{c}')
            ok = bool(cell and cell['headline_status'] == 'HEADLINE')
            note = None
            if not ok and a == 'C2':
                # registered C2 rule: a timed-out cold attempt is not retried and the
                # remaining restarts are UNTESTED (unavailable performance, not a
                # session defect) — complete iff every required restart is a terminal
                # UNTESTED record citing the allowance / no-retry / timeout rule
                recs = [summary['runs'].get(f'C2_{c}_r{i}') for i in (1, 2, 3)]
                if all(r and r.get('status') == 'UNTESTED' and any(
                        t in (r.get('error') or '') for t in ('timeout', 'allowance', 'no-retry')) for r in recs):
                    ok, note = True, 'UNTESTED under the registered C2 allowance rule'
            items.append({'item': f'matrix {a}@{c}', 'complete': ok, 'note': note,
                          'reason': None if ok else (cell or {}).get('reasons', ['no runs'])})
        for a, b in PAIRS_S1:
            p = pairs.get(f'{a}_vs_{b}_c352800')
            ok = bool(p and p['pair_headline'] == 'HEADLINE')
            items.append({'item': f'matched {a} vs {b} @352800', 'complete': ok,
                          'reason': None if ok else (p or {}).get('reasons', ['no pair runs'])})
        for name in ('full_A_c4410_planted_tight.json', 'full_A_c4410_planted_loose.json',
                     'c3_operator_qualification.json'):
            ok = os.path.exists(os.path.join(ev, name))
            items.append({'item': name, 'complete': ok, 'reason': None if ok else ['missing']})
    else:
        for a, c in SWEEP_S2:
            cell = cells.get(f'{a}_{c}')
            ok = bool(cell and cell['headline_status'] == 'HEADLINE')
            items.append({'item': f'sweep {a}@{c}', 'complete': ok,
                          'reason': None if ok else (cell or {}).get('reasons', ['no runs'])})
        for a, c in TRACES_S2:
            exp = os.path.join(ev, 'traces', f'trace_{a}_{c}_trace_exports.json')
            ok = os.path.exists(exp) and json.load(open(exp)).get('complete') is True
            items.append({'item': f'trace {a}@{c}', 'complete': bool(ok), 'reason': None if ok else ['exports incomplete or missing']})
        for name in ('attn_backend_bench.json', 'boundary_bench.json', 'boundary_bench_trace_analysis.json',
                     'perturbation_and_source_map.json'):
            ok = os.path.exists(os.path.join(ev, name))
            items.append({'item': name, 'complete': ok, 'reason': None if ok else ['missing']})
    n_ok = sum(1 for i in items if i['complete'])
    status = {'session': args.session, 'items': items, 'complete_items': n_ok, 'total_items': len(items),
              'verdict': 'COMPLETE' if n_ok == len(items) else 'INCOMPLETE'}
    with open(os.path.join(ev, 'session_status.json'), 'w') as f:
        json.dump(status, f, indent=1)
    print(f"[completeness] {args.session}: {status['verdict']} ({n_ok}/{len(items)} mandatory items)")
    for i in items:
        if not i['complete']:
            print(f"   INCOMPLETE {i['item']}: {i['reason']}")


if __name__ == '__main__':
    main()
