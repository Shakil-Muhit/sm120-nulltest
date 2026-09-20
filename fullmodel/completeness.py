"""Session completeness audit (gate-03 f43, gate-04 f56, gate-05 f65, gate-06 f85–f87).

The registered mandatory attempts and products per session (prereg_amendments.md
§5), every one validated by its PAYLOAD through the shared validators
(fullmodel/qualify.py) and by re-deriving cells and pairs from the run records —
never from the collector's labels. Reference artifacts are enumerated per
operating point / seed / clip, re-hashed, and joined to the hashes the runs
recorded. The verdict (COMPLETE / INCOMPLETE with per-item reasons) is written
to <evidence>/session_status.json; an INCOMPLETE session is reported as such.
A non-zero exit means the audit itself crashed.
"""

import argparse
import bisect
import datetime as _dt
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_full  # noqa: E402
import provenance  # noqa: E402
import qualify  # noqa: E402

MATRIX_ARMS = ('A', 'C3', 'C1', 'H', 'G', 'B', 'E', 'D', 'F')
MATRIX_S1 = [(a, c) for c in (352800, 4410) for a in MATRIX_ARMS] + [('C1D', 352800), ('C2', 352800), ('C2', 4410)]
PAIRS_S1 = [('C1', 'C3'), ('C1', 'C1D'), ('C3', 'H')]
SWEEP_S2 = [(a, c) for c in (22050, 88200) for a in ('A', 'C3', 'G', 'H')]
TRACE_ARMS = ('A', 'ANVTX', 'C3', 'G', 'H', 'C1', 'B', 'E', 'F')
TRACES_S2 = [(a, c) for c in (352800, 4410) for a in TRACE_ARMS] + [('C2', 4410)]
COMPILED_TRACED = ('C3', 'H', 'C1', 'C2')
EAGER_NO_REPLAY = ('A', 'D', 'F')
PERT_S2 = [(a, c, m) for c in (352800, 4410) for a in ('A', 'C3', 'G') for m in ('unprofiled', 'graph', 'node')]
C3_FRAMES = {'11', '51', '201', '801'}
ATTN_CASES = [f'c{c}_{ax}_{lay}' for c in (352800, 4410) for ax in ('time', 'freq') for lay in ('model', 'contiguous')]
BOUNDARY_CHAINS = [f'trivial/{m}/N{n}' for m in ('stream', 'graph', 'pdl_stream', 'pdl_graph') for n in (10, 100, 1000, 2000)] + \
                  [f'{ch}/{m}/N{n}' for ch in ('elem37', 'gemm') for m in ('stream', 'graph') for n in (10, 100)]
OPS = {'session1': (352800, 4410), 'session2': (352800, 4410, 22050, 88200)}
MANIFEST_FIELDS = ('manifest_id', 'session', 'created_utc', 'repo_commit', 'registration_sha256',
                   'clips_manifest_sha256', 'code', 'provenance', 'runtime', 'env')
STATIC_AUX = {
    'session1': ['c3_operator_qualification', 'collect'],
    'session2': ['attn_backend_bench', 'boundary_bench', 'boundary_bench_trace', 'boundary_bench_export',
                 'boundary_bench_analysis', 'perturbation', 'collect'],
}


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def item(items, name, ok, reason=None, note=None):
    items.append({'item': name, 'complete': bool(ok), 'reason': None if ok else (reason or ['missing']),
                  'note': note})


def fin(x):
    return qualify._finite(x)


def load_runs(ev):
    runs = {}
    for path in sorted(glob.glob(os.path.join(ev, 'full_*.json'))):
        r = load(path)
        if r and 'arm' in r and 'chunk' in r and 'tag' in r:
            runs.setdefault((r['arm'], r['chunk']), {})[r['tag']] = r
    return runs


def load_matched(ev):
    out = {}
    for path in sorted(glob.glob(os.path.join(ev, 'matched_*.json'))):
        m = load(path)
        if m and m.get('pair') and 'chunk' in m and 'tag' in m:
            out[(tuple(m['pair']), m['chunk'], m['tag'])] = m
    return out


def audit_manifest(ev, items):
    manifest = load(os.path.join(ev, 'session_manifest.json')) or {}
    missing = [k for k in MANIFEST_FIELDS if not manifest.get(k)]
    prov = manifest.get('provenance') or {}
    ok = not missing and isinstance(manifest.get('code'), dict) and len(manifest['code']) > 10 \
        and prov.get('msst_commit') and prov.get('ckpt_sha256') and (manifest.get('runtime') or {}).get('torch')
    item(items, 'session manifest (complete identity)', ok, [f'missing or empty fields: {missing}'])
    return manifest.get('manifest_id') if ok else None


def audit_references(ev, items, manifest_id, session, runs, matched):
    """Enumerate the required references, re-hash them, and join every recorded
    hash to the corresponding artifact (gate-06 f86)."""
    ref = os.path.join(ev, 'ref')
    marker = os.path.join(ref, 'manifest_id.txt')
    item(items, 'reference directory bound to the session manifest',
         os.path.exists(marker) and open(marker).read().strip() == manifest_id, ['ref/manifest_id.txt missing or foreign'])
    hashes = {}
    bad = []
    for chunk in OPS[session]:
        for seed in qualify.VALIDATION_SEEDS:
            f = os.path.join(ref, f'ref_A_c{chunk}_s{seed}.pt')
            key = ('val', chunk, seed)
            side = load(f + '.lineage.json')
            if not os.path.exists(f) or not side or side.get('manifest_id') != manifest_id:
                bad.append(os.path.basename(f))
                continue
            h = provenance.sha256_file(f)
            if side.get('sha256') != h:
                bad.append(os.path.basename(f) + ' (hash)')
                continue
            hashes[key] = h
        for i in range(qualify.SDR_CLIPS):
            f = os.path.join(ref, f'sdr_A_c{chunk}_clip{i:02d}.pt')
            key = ('clip', chunk, i)
            side = load(f + '.lineage.json')
            if not os.path.exists(f) or not side or side.get('manifest_id') != manifest_id:
                bad.append(os.path.basename(f))
                continue
            h = provenance.sha256_file(f)
            if side.get('sha256') != h:
                bad.append(os.path.basename(f) + ' (hash)')
                continue
            hashes[key] = h
    n_req = len(OPS[session]) * (len(qualify.VALIDATION_SEEDS) + qualify.SDR_CLIPS)
    item(items, f'required reference inventory ({n_req} artifacts, bound and re-hashed)', not bad,
         [f'missing/unbound/changed: {bad[:8]}'])
    # join: every recorded hash must equal the re-hashed artifact of its operating point
    joined = 0
    mismatch = []
    for (arm, chunk), tr in runs.items():
        for tag, r in tr.items():
            for p in (r.get('validation') or {}).get('passes') or []:
                if p.get('ref_sha256'):
                    joined += 1
                    if hashes.get(('val', chunk, p.get('seed'))) != p['ref_sha256']:
                        mismatch.append(f'{arm}@{chunk}/{tag} seed {p.get("seed")}')
            for row in (r.get('sdr_gate') or {}).get('clips') or []:
                if row.get('a_output_sha256'):
                    joined += 1
                    if hashes.get(('clip', chunk, row.get('index'))) != row['a_output_sha256']:
                        mismatch.append(f'{arm}@{chunk}/{tag} clip {row.get("index")}')
    for (pair, chunk, tag), m in matched.items():
        for arm, a in (m.get('arms') or {}).items():
            for p in (a.get('validation') or {}).get('passes') or []:
                if p.get('ref_sha256'):
                    joined += 1
                    if hashes.get(('val', chunk, p.get('seed'))) != p['ref_sha256']:
                        mismatch.append(f'matched {arm}@{chunk}/{tag} seed {p.get("seed")}')
    item(items, f'recorded reference hashes join the audited artifacts ({joined} joins)', joined > 0 and not mismatch,
         [f'{len(mismatch)} mismatches: {mismatch[:6]}'] if mismatch else ['no recorded reference hashes'])


def audit_clock(ev, items, session, runs, matched):
    """This session's clock log must cover every timing block that belongs to
    this session (blocks started after the log began), including matched
    blocks; every block needs finite timestamps (gate-06 f87)."""
    clocks = os.path.join(ev, f'clocks_{session}.csv')
    times = []
    bad_lines = 0
    if os.path.exists(clocks):
        for line in open(clocks):
            parts = [x.strip() for x in line.split(',')]
            if len(parts) != 7:
                bad_lines += 1
                continue
            for fmt in ('%Y/%m/%d %H:%M:%S.%f', '%Y/%m/%d %H:%M:%S'):
                try:
                    times.append(_dt.datetime.strptime(parts[0], fmt).timestamp())
                    break
                except ValueError:
                    continue
    times.sort()
    blocks = []
    for rs in runs.values():
        for r in rs.values():
            blocks += [(b.get('unix_start'), b.get('unix_end')) for b in ((r.get('timing') or {}).get('blocks') or [])]
    for m in matched.values():
        blocks += [(b.get('unix_start'), b.get('unix_end')) for b in (m.get('blocks') or [])]
    if not times:
        item(items, 'clock log covers every timing block of this session', False, ['clock log missing or unparsable'])
        return
    t0 = times[0] - 60
    own = [(a, b) for a, b in blocks if not (fin(a) and fin(b)) or a >= t0]
    nonfinite = sum(1 for a, b in own if not (fin(a) and fin(b)))
    uncovered = 0
    for a, b in own:
        if not (fin(a) and fin(b)):
            continue
        i = bisect.bisect_left(times, a - 30)
        if i >= len(times) or times[i] > b + 30:
            uncovered += 1
    ok = bool(own) and nonfinite == 0 and uncovered == 0 and bad_lines == 0
    item(items, f'clock log covers every timing block of this session ({len(own)} blocks)', ok,
         [f'{uncovered} uncovered, {nonfinite} without finite timestamps, {bad_lines} malformed clock lines, blocks={len(own)}'])


def audit_aux(ev, items, session, dynamic):
    required = STATIC_AUX[session] + dynamic
    for name in required:
        rec = load(os.path.join(ev, f'aux_{name}.json')) or {}
        item(items, f'auxiliary step {name}', rec.get('status') == 'OK', [f"{rec.get('status', 'MISSING')}: {rec.get('reason')}"])
    for path in sorted(glob.glob(os.path.join(ev, 'aux_*.json'))):
        rec = load(path) or {}
        name = rec.get('item', os.path.basename(path)[4:-5])
        if name in required or name == 'completeness':
            continue
        item(items, f'auxiliary step {name}', rec.get('status') == 'OK', [f"{rec.get('status')}: {rec.get('reason')}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence-dir', required=True)
    ap.add_argument('--session', choices=['session1', 'session2'], required=True)
    args = ap.parse_args()
    ev = args.evidence_dir
    items = []
    manifest_id = audit_manifest(ev, items)
    item(items, 'collection summary present', load(os.path.join(ev, 'summary_full.json')) is not None,
         ['summary_full.json missing or invalid'])
    run_map = load_runs(ev)
    matched = load_matched(ev)
    # cells and pairs re-derived from the records through the validators
    cells = {}
    for (a, c), v in run_map.items():
        cell = collect_full.cell(v)
        for t in ('r1', 'r2', 'r3'):
            r = v.get(t)
            if r and r.get('manifest_id') != manifest_id:
                cell['headline_status'] = 'INCOMPLETE'
                cell['reasons'].append(f'{t}: foreign manifest')
        cells[f'{a}_{c}'] = cell
    pairs = collect_full.pair_summaries(ev, cells, run_map)
    for key, p in pairs.items():
        for t in ('m1', 'm2', 'm3'):
            a, rest = key.split('_vs_')
            b, chunk = rest.rsplit('_c', 1)
            m = matched.get(((a, b), int(chunk), t))
            if m and m.get('manifest_id') != manifest_id:
                p['pair_headline'] = 'INCOMPLETE'
                p['reasons'].append(f'{t}: foreign manifest')
    audit_references(ev, items, manifest_id, args.session, run_map, matched)
    dynamic_aux = []

    if args.session == 'session1':
        for a, c in MATRIX_S1:
            cell = cells.get(f'{a}_{c}')
            ok = bool(cell and cell['headline_status'] == 'HEADLINE')
            note = None
            if not ok and a == 'C2':
                recs = [(run_map.get((a, c)) or {}).get(f'r{i}') for i in (1, 2, 3)]
                if all(r and r.get('status') == 'UNTESTED' and any(
                        t in (r.get('error') or '') for t in ('timeout', 'allowance', 'no-retry')) for r in recs):
                    ok, note = True, 'UNTESTED under the registered C2 allowance rule (unavailable performance)'
            item(items, f'matrix {a}@{c}', ok, (cell or {}).get('reasons', ['no runs']), note)
        for a, b in PAIRS_S1:
            p = pairs.get(f'{a}_vs_{b}_c352800')
            ok = bool(p and p['pair_headline'] == 'HEADLINE')
            item(items, f'matched {a} vs {b} @352800 (m1..m3)', ok, (p or {}).get('reasons', ['no pair runs']))
        m = matched.get((('C1', 'C1A'), 352800, 'm1'))
        ok = bool(m and m.get('status') in ('VALID', 'INVALID') and m.get('manifest_id') == manifest_id
                  and (m.get('arms') or {}).get('C1A', {}).get('build_status'))
        item(items, 'matched C1 vs C1A @352800 attempted once (terminal record)', ok,
             ['no terminal record of the once-only C1A attempt bound to this manifest'],
             note=(f"C1A build {m['arms']['C1A'].get('build_status')}" if ok else None))
        for tier, const in (('tight', 0.001), ('loose', 0.05)):
            r = (run_map.get(('A', 4410)) or {}).get(f'planted_{tier}')
            v = (r or {}).get('validation') or {}
            passes = v.get('passes') or []
            ok = bool(r and r.get('status') == 'INVALID' and r.get('planted_fault_const') is not None
                      and abs(r['planted_fault_const'] - const) < 1e-9 and r.get('tier') == tier
                      and v.get('status') == 'FAIL' and r.get('manifest_id') == manifest_id
                      and sorted(p.get('seed') for p in passes) == sorted(qualify.VALIDATION_SEEDS)
                      and all(p.get('ok') is False and fin(p.get('max_abs_diff')) and abs(p['max_abs_diff'] - const) < 1e-6
                              and (p.get('violations') or 0) > 0 for p in passes))
            item(items, f'planted fault {tier} caught ({const})', ok,
                 ['planted-fault record missing, foreign, or not a per-pass FAIL at the planted magnitude'])
        q = load(os.path.join(ev, 'c3_operator_qualification.json')) or {}
        ok = bool(set(q) == C3_FRAMES and all(v.get('status') == 'PASS' and v.get('native_bitwise_equal') is True
                                               and v.get('opcheck') for v in q.values()))
        item(items, 'C3 operator qualification (frames 11/51/201/801, opcheck + bitwise)', ok,
             ['qualification JSON missing, wrong frame set, or payload not PASS/bitwise/opcheck'])
    else:
        for a, c in SWEEP_S2:
            cell = cells.get(f'{a}_{c}')
            ok = bool(cell and cell['headline_status'] == 'HEADLINE')
            item(items, f'sweep {a}@{c}', ok, (cell or {}).get('reasons', ['no runs']))
        for a, c in TRACES_S2:
            base = os.path.join(ev, 'traces', f'trace_{a}_{c}_trace')
            exp = load(base + '_exports.json') or {}
            run = (run_map.get((a, c)) or {}).get('trace')
            probs = []
            if exp.get('complete') is not True or exp.get('graph_trace_mode') != 'node':
                probs.append('node-level exports incomplete')
            for rep in ('cuda_gpu_trace', 'cuda_gpu_kern_sum', 'cuda_api_trace', 'nvtx_gpu_proj_trace'):
                f = f'{base}_{rep}.csv'
                if not (os.path.exists(f) and os.path.getsize(f) > 0):
                    probs.append(f'export {rep} missing')
            probs += qualify.trace_problems(run)
            if run and run.get('manifest_id') != manifest_id:
                probs.append('traced run bound to a foreign manifest')
            if run and not run.get('nvtx_pass'):
                probs.append('traced run has no NVTX pass markers')
            item(items, f'trace {a}@{c}', not probs, probs)
            if a != 'ANVTX':
                gv = load(os.path.join(ev, 'traces', f'graph_verify_{a}_{c}.json')) or {}
                accepted = ('FULLY_GRAPHED', 'NOT_FULLY_GRAPHED') + (('NO_REPLAY_EVIDENCE',) if a in EAGER_NO_REPLAY else ())
                pp = gv.get('per_pass') or []
                ok = (gv.get('verdict') in accepted and gv.get('qualified') is True and gv.get('registered_protocol') is True
                      and gv.get('pass_count_complete') is True and len(pp) == 50
                      and all(isinstance(p.get('captured_model_kernels'), int) for p in pp))
                item(items, f'graph verification {a}@{c}', ok,
                     [f"verdict {gv.get('verdict', 'MISSING')} qualified={gv.get('qualified')} passes={len(pp)}"])
                tv = load(os.path.join(ev, 'traces', f'timing_views_{a}_{c}.json')) or {}
                med = tv.get('median_us') or {}
                ok = (tv.get('passes') == 50 and all(fin(med.get(k)) and med[k] > 0 for k in
                                                     ('sum_ns', 'union_ns', 'critical_path_observed_order_ns', 'span_ns')))
                item(items, f'timing views {a}@{c}', ok, ['timing views missing, not 50 passes, or non-finite medians'])
                dynamic_aux += [f'graph_verify_{a}_{c}', f'timing_views_{a}_{c}']
            if a in COMPILED_TRACED and not (a == 'C2' and c == 352800):
                ic = load(os.path.join(ev, 'traces', f'ideal_costs_{a}_{c}.json')) or {}
                sel = (ic.get('variant_selection') or {}).get('status')
                rows = ic.get('rows') or []
                priced = [r for r in rows if fin(r.get('ideal_us_hbm'))]
                if a == 'C3':
                    ok = (sel == 'ok' and ic.get('measurement_status') == 'ok' and ic.get('passes') == 50
                          and len(priced) > 0 and all(fin(r.get('time_us_per_pass')) for r in rows))
                    item(items, f'ideal costs {a}@{c} (unique variant, 50 windows, priced rows)', ok,
                         [f'selection {sel}, measurement {ic.get("measurement_status")}, priced {len(priced)}'])
                else:
                    ok = ic.get('measurement_status') in ('ok', 'UNRESOLVED') and ic.get('passes') == 50
                    item(items, f'ideal costs {a}@{c} (record, 50 windows)', ok, ['ideal-cost record missing or not the registered protocol'])
                dynamic_aux.append(f'ideal_costs_{a}_{c}')
        for a, c, m in PERT_S2:
            r = (run_map.get((a, c)) or {}).get(f'pert_{m}')
            probs = qualify.trace_problems(r)
            if r and r.get('manifest_id') != manifest_id:
                probs.append('foreign manifest')
            if r and m == 'unprofiled' and r.get('nvtx_pass'):
                probs.append('unprofiled run carries NVTX markers')
            item(items, f'perturbation run {a}@{c} {m}', not probs, probs)
        pm = load(os.path.join(ev, 'perturbation_and_source_map.json')) or {}
        for a, c in [(a, c) for c in (352800, 4410) for a in ('A', 'C3', 'G')]:
            rec = (pm.get('perturbation') or {}).get(f'{a}_{c}') or {}
            ok = (rec.get('status') == 'ok' and fin(rec.get('graph_over_unprofiled')) and fin(rec.get('node_over_unprofiled'))
                  and rec.get('caveat_required') in (True, False))
            item(items, f'perturbation ratio {a}@{c}', ok, [f"status {rec.get('status', 'MISSING')}: {rec.get('problems')}"])
        for c in (352800, 4410):
            sm = (pm.get('source_map') or {}).get(f'C3_{c}') or {}
            cov = (sm.get('coverage') or {}).get('share')
            item(items, f'kernel source map coverage C3@{c}', fin(cov) and 0 <= cov <= 1, ['no finite coverage share'])
        ab = load(os.path.join(ev, 'attn_backend_bench.json')) or {}
        cases = ab.get('cases') or {}
        missing = [k for k in ATTN_CASES if k not in cases]
        unres = [k for k in ATTN_CASES if k in cases and k.endswith('_model') and not fin(cases[k].get('ratio_best_over_model'))]
        item(items, 'attention backend bench (8 cases, model-layout ratios established)', not missing and not unres,
             [f'missing {missing}', f'unresolved ratio {unres}'])
        bb = load(os.path.join(ev, 'boundary_bench.json')) or {}
        chains = bb.get('chains') or {}
        bad = [k for k in BOUNDARY_CHAINS if chains.get(k, {}).get('status') not in ('ok', 'UNTESTED')
               or (chains.get(k, {}).get('status') == 'ok' and not fin(chains[k].get('per_launch_us')))]
        item(items, 'boundary bench chains (terminal status per registered chain)', not bad, [f'non-terminal {bad[:6]}'])
        ta = load(os.path.join(ev, 'boundary_bench_trace_analysis.json')) or {}
        qual = ta.get('_qualification') or {}
        item(items, 'boundary trace analysis qualified', bool(qual.get('cases')) and qual.get('accepted_segments', 0) > 0,
             [f"qualification {qual}"])
    audit_clock(ev, items, args.session, run_map, matched)
    audit_aux(ev, items, args.session, dynamic_aux)
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
    return 0


if __name__ == '__main__':
    sys.exit(main())
