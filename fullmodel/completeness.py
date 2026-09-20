"""Session completeness audit (gate-03 f43, gate-04 f56): the registered
mandatory attempts and products per session (prereg_amendments.md §5),
each validated by CONTENT (never by file existence alone), marked
COMPLETE / INCOMPLETE with its reason, and a mechanical session verdict in
<evidence>/session_status.json. An INCOMPLETE session is reported as such;
it is never salvaged by relaxing validation or timing requirements.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qualify  # noqa: E402

MATRIX_ARMS = ('A', 'C3', 'C1', 'H', 'G', 'B', 'E', 'D', 'F')
MATRIX_S1 = [(a, c) for c in (352800, 4410) for a in MATRIX_ARMS] + [('C1D', 352800), ('C2', 352800), ('C2', 4410)]
PAIRS_S1 = [('C1', 'C3'), ('C1', 'C1D'), ('C3', 'H')]
SWEEP_S2 = [(a, c) for c in (22050, 88200) for a in ('A', 'C3', 'G', 'H')]
TRACE_ARMS = ('A', 'ANVTX', 'C3', 'G', 'H', 'C1', 'B', 'E', 'F')
TRACES_S2 = [(a, c) for c in (352800, 4410) for a in TRACE_ARMS] + [('C2', 4410)]
COMPILED_TRACED = ('C3', 'H', 'C1', 'C2')
PERT_S2 = [(a, c, m) for c in (352800, 4410) for a in ('A', 'C3', 'G') for m in ('unprofiled', 'graph', 'node')]
C3_FRAMES = {'11', '51', '201', '801'}
ATTN_CASES = [f'c{c}_{ax}_{lay}' for c in (352800, 4410) for ax in ('time', 'freq') for lay in ('model', 'contiguous')]
BOUNDARY_CHAINS = [f'{ch}/{m}/N{n}' for ch in ('trivial',) for m in ('stream', 'graph', 'pdl_stream', 'pdl_graph')
                   for n in (10, 100, 1000, 2000)] + \
                  [f'{ch}/{m}/N{n}' for ch in ('elem37', 'gemm') for m in ('stream', 'graph') for n in (10, 100)]


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def item(items, name, ok, reason=None, note=None):
    items.append({'item': name, 'complete': bool(ok), 'reason': None if ok else (reason or ['missing']),
                  'note': note})


def audit_lineage(ev, items, manifest_id):
    """Every reference file carries a lineage sidecar bound to THIS manifest."""
    ref = os.path.join(ev, 'ref')
    marker = os.path.join(ref, 'manifest_id.txt')
    ok = os.path.exists(marker) and open(marker).read().strip() == manifest_id
    item(items, 'reference directory bound to the session manifest', ok, ['ref/manifest_id.txt missing or foreign'])
    files = [f for f in glob.glob(os.path.join(ref, '*.pt')) if not f.endswith('.lineage.json')]
    bad = []
    for f in files:
        side = load(f + '.lineage.json')
        if not side or side.get('manifest_id') != manifest_id:
            bad.append(os.path.basename(f))
    item(items, f'reference lineage sidecars ({len(files)} files)', files and not bad, [f'unbound: {bad[:5]}'] if bad else ['no reference files'])


def audit_aux(ev, items, session):
    for path in sorted(glob.glob(os.path.join(ev, 'aux_*.json'))):
        rec = load(path) or {}
        item(items, f"auxiliary step {rec.get('item', os.path.basename(path))}", rec.get('status') == 'OK',
             [f"{rec.get('status')}: {rec.get('reason')}"])
    clocks = os.path.join(ev, f'clocks_{session}.csv')
    item(items, 'clock log', os.path.exists(clocks) and os.path.getsize(clocks) > 0, ['clock log missing or empty'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence-dir', required=True)
    ap.add_argument('--session', choices=['session1', 'session2'], required=True)
    args = ap.parse_args()
    ev = args.evidence_dir
    items = []
    manifest = load(os.path.join(ev, 'session_manifest.json')) or {}
    manifest_id = manifest.get('manifest_id')
    item(items, 'session manifest', bool(manifest_id), ['session_manifest.json missing'])
    summary = load(os.path.join(ev, 'summary_full.json'))
    item(items, 'collection summary', summary is not None, ['summary_full.json missing or invalid'])
    cells = (summary or {}).get('cells', {})
    pairs = (summary or {}).get('matched_pairs', {})
    runs = (summary or {}).get('runs', {})
    audit_lineage(ev, items, manifest_id)

    if args.session == 'session1':
        for a, c in MATRIX_S1:
            cell = cells.get(f'{a}_{c}')
            ok = bool(cell and cell['headline_status'] == 'HEADLINE')
            note = None
            if not ok and a == 'C2':
                # registered (§5): a timed-out cold C2 attempt is not retried; the cell's
                # restarts are terminal UNTESTED records citing the allowance/no-retry/
                # timeout rule — permitted terminal unavailability, not missing work
                recs = [runs.get(f'C2_{c}_r{i}') for i in (1, 2, 3)]
                if all(r and r.get('status') == 'UNTESTED' and any(
                        t in (r.get('error') or '') for t in ('timeout', 'allowance', 'no-retry')) for r in recs):
                    ok, note = True, 'UNTESTED under the registered C2 allowance rule (unavailable performance)'
            item(items, f'matrix {a}@{c}', ok, (cell or {}).get('reasons', ['no runs']), note)
        for a, b in PAIRS_S1:
            p = pairs.get(f'{a}_vs_{b}_c352800')
            ok = bool(p and p['pair_headline'] == 'HEADLINE')
            item(items, f'matched {a} vs {b} @352800 (m1..m3)', ok, (p or {}).get('reasons', ['no pair runs']))
        # once-only (C1, C1A) attempt: a terminal record must exist (expected compile failure)
        m = load(os.path.join(ev, 'matched_C1_C1A_c352800_m1.json'))
        ok = bool(m and m.get('status') in ('VALID', 'INVALID') and (m.get('arms') or {}).get('C1A', {}).get('build_status'))
        item(items, 'matched C1 vs C1A @352800 attempted once (terminal record)', ok,
             ['no terminal record of the once-only C1A attempt'],
             note=(f"C1A build {m['arms']['C1A'].get('build_status')}" if ok else None))
        for tier, const in (('tight', 0.001), ('loose', 0.05)):
            r = load(os.path.join(ev, f'full_A_c4410_planted_{tier}.json'))
            ok = bool(r and r.get('status') == 'INVALID' and r.get('planted_fault_const') is not None
                      and abs(r['planted_fault_const'] - const) < 1e-9 and r.get('tier') == tier
                      and (r.get('validation') or {}).get('status') == 'FAIL'
                      and (r['validation'].get('passes') or [])
                      and all(abs((p.get('max_abs_diff') or 0) - const) < 1e-6 for p in r['validation']['passes']))
            item(items, f'planted fault {tier} caught ({const})', ok, ['planted-fault record missing or not a FAIL at the planted magnitude'])
        q = load(os.path.join(ev, 'c3_operator_qualification.json'))
        ok = bool(q and set(q) == C3_FRAMES and all(v.get('status') == 'PASS' for v in q.values()))
        item(items, 'C3 operator qualification (frames 11/51/201/801)', ok, ['qualification JSON missing, wrong frame set, or not all PASS'])
    else:
        for a, c in SWEEP_S2:
            cell = cells.get(f'{a}_{c}')
            ok = bool(cell and cell['headline_status'] == 'HEADLINE')
            item(items, f'sweep {a}@{c}', ok, (cell or {}).get('reasons', ['no runs']))
        for a, c in TRACES_S2:
            base = os.path.join(ev, 'traces', f'trace_{a}_{c}_trace')
            exp = load(base + '_exports.json') or {}
            run = load(os.path.join(ev, f'full_{a}_c{c}_trace.json'))
            probs = []
            if exp.get('complete') is not True or exp.get('graph_trace_mode') != 'node':
                probs.append('node-level exports incomplete')
            if not run or run.get('status') != 'VALID':
                probs.append('traced run missing or not VALID')
            else:
                t = run.get('timing') or {}
                if t.get('iters_per_block') != 50 or len(t.get('blocks') or []) != 1 or \
                        len((t['blocks'][0].get('latency_ms_per_iter') or [])) != 50:
                    probs.append('traced run is not one block of 50 passes')
                if (run.get('validation') or {}).get('status') != 'PASS' and a != 'ANVTX':
                    probs.append('traced run validation not PASS')
                if run.get('manifest_id') != manifest_id:
                    probs.append('traced run not bound to this manifest')
            item(items, f'trace {a}@{c}', not probs, probs)
            if a != 'ANVTX':
                gv = load(os.path.join(ev, 'traces', f'graph_verify_{a}_{c}.json')) or {}
                item(items, f'graph verification {a}@{c}', gv.get('verdict') in ('FULLY_GRAPHED', 'NOT_FULLY_GRAPHED'),
                     [f"verdict {gv.get('verdict', 'MISSING')}"])
                tv = load(os.path.join(ev, 'traces', f'timing_views_{a}_{c}.json')) or {}
                item(items, f'timing views {a}@{c}', tv.get('passes') == 50, ['timing views missing or not 50 passes'])
            if a in COMPILED_TRACED and not (a == 'C2' and c == 352800):
                ic = load(os.path.join(ev, 'traces', f'ideal_costs_{a}_{c}.json')) or {}
                sel = (ic.get('variant_selection') or {}).get('status')
                if a == 'C3':
                    ok = sel == 'ok' and ic.get('measurement_status') == 'ok'
                    item(items, f'ideal costs {a}@{c} (unique variant, windowed)', ok, [f'selection {sel}, measurement {ic.get("measurement_status")}'])
                else:
                    item(items, f'ideal costs {a}@{c} (record)', bool(ic.get('measurement_status')), ['ideal-cost record missing'])
        for a, c, m in PERT_S2:
            r = load(os.path.join(ev, f'full_{a}_c{c}_pert_{m}.json'))
            ok = bool(r and r.get('status') == 'VALID' and (r.get('validation') or {}).get('status') == 'PASS'
                      and r.get('manifest_id') == manifest_id)
            item(items, f'perturbation run {a}@{c} {m}', ok, ['run missing, not VALID, validation not PASS, or foreign manifest'])
        pm = load(os.path.join(ev, 'perturbation_and_source_map.json')) or {}
        for a, c in [(a, c) for c in (352800, 4410) for a in ('A', 'C3', 'G')]:
            rec = (pm.get('perturbation') or {}).get(f'{a}_{c}') or {}
            item(items, f'perturbation ratio {a}@{c}', rec.get('status') == 'ok', [f"status {rec.get('status', 'MISSING')}: {rec.get('problems')}"])
        for c in (352800, 4410):
            sm = (pm.get('source_map') or {}).get(f'C3_{c}') or {}
            cov = (sm.get('coverage') or {}).get('share')
            item(items, f'kernel source map coverage C3@{c}', isinstance(cov, (int, float)), ['no coverage share'])
        ab = load(os.path.join(ev, 'attn_backend_bench.json')) or {}
        cases = ab.get('cases') or {}
        missing = [k for k in ATTN_CASES if k not in cases]
        unres = [k for k in ATTN_CASES if k in cases and k.endswith('_model') and cases[k].get('ratio_best_over_model') is None]
        item(items, 'attention backend bench (8 cases, model-layout ratios established)', not missing and not unres,
             [f'missing {missing}', f'unresolved ratio {unres}'])
        bb = load(os.path.join(ev, 'boundary_bench.json')) or {}
        chains = bb.get('chains') or {}
        bad = [k for k in BOUNDARY_CHAINS if chains.get(k, {}).get('status') not in ('ok', 'UNTESTED')]
        item(items, 'boundary bench chains (terminal status per registered chain)', not bad, [f'non-terminal {bad[:6]}'])
        ta = load(os.path.join(ev, 'boundary_bench_trace_analysis.json')) or {}
        qual = ta.get('_qualification') or {}
        item(items, 'boundary trace analysis qualified', bool(qual.get('cases')) and qual.get('accepted_segments', 0) > 0,
             [f"qualification {qual}"])
    audit_aux(ev, items, args.session)
    n_ok = sum(1 for i in items if i['complete'])
    status = {'session': args.session, 'manifest_id': manifest_id, 'items': items,
              'complete_items': n_ok, 'total_items': len(items),
              'verdict': 'COMPLETE' if n_ok == len(items) else 'INCOMPLETE',
              'qualify_version': qualify.QUALIFY_VERSION}
    with open(os.path.join(ev, 'session_status.json'), 'w') as f:
        json.dump(status, f, indent=1)
    print(f"[completeness] {args.session}: {status['verdict']} ({n_ok}/{len(items)} mandatory items)")
    for i in items:
        if not i['complete']:
            print(f"   INCOMPLETE {i['item']}: {i['reason']}")
    return 0 if status['verdict'] == 'COMPLETE' else 3


if __name__ == '__main__':
    sys.exit(main())
