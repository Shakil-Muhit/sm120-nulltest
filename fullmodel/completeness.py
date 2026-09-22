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
BOUNDARY_CHAINS_S3 = [f'prologue/{m}/N{n}' for m in ('stream', 'graph', 'pdl_stream', 'pdl_graph') for n in (100, 1000)]
BOUNDARY_CHAINS = [f'trivial/{m}/N{n}' for m in ('stream', 'graph', 'pdl_stream', 'pdl_graph') for n in (10, 100, 1000, 2000)] + \
                  [f'{ch}/{m}/N{n}' for ch in ('elem37', 'gemm') for m in ('stream', 'graph') for n in (10, 100)]
OPS = {'session1': (352800, 4410), 'session2': (352800, 4410, 22050, 88200), 'session3': (352800, 4410), 'session4': (352800, 4410, 44100, 176400, 264600)}
SDR_OPS = {'session4': (352800, 4410)}          # sessions without an entry: SDR clip references at every OPS point
HELD_OUT_S3 = (176400, 264600)
ANALYSIS_TOOLS = ('completeness.py', 'collect_full.py', 'qualify.py', 'ledger.py', 'session_result.py', 'e_tools.py', 'session_result_v2.py', 'session3_result.py', 'figures.py', 'session4_result.py', 'stats_v2.py',
                  'ideal_costs.py', 'graph_verify.py', 'timing_views.py', 'perturbation.py', 'kernel_map.py', 'classify.py')
MANIFEST_FIELDS = ('manifest_id', 'session', 'created_utc', 'repo_commit', 'registration_sha256',
                   'clips_manifest_sha256', 'code', 'provenance', 'runtime', 'env')
STATIC_AUX = {
    'session1': ['c3_operator_qualification', 'collect'],
    'session2': ['attn_backend_bench', 'boundary_bench', 'boundary_bench_trace', 'boundary_bench_export',
                 'boundary_bench_analysis', 'perturbation', 'collect'],
    'session3': ['c3_operator_qualification', 'aa_stats', 'collect'],
    'session4': ['c3_operator_qualification', 'aa_stats', 'stats_v2', 'collect'],
}
OPTIONAL_AUX_S3 = ('boundary_bench', 'boundary_bench_trace', 'boundary_bench_export', 'boundary_bench_analysis', 'e5_summary')


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def item(items, name, ok, reason=None, note=None, optional=False):
    items.append({'item': name, 'complete': bool(ok), 'optional': bool(optional), 'reason': None if ok else (reason or ['missing']),
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
    """The manifest must be the identity of THIS tree, registration and pins
    (gate-07 f100): every field re-derived here and compared, not just present."""
    manifest = load(os.path.join(ev, 'session_manifest.json')) or {}
    missing = [k for k in MANIFEST_FIELDS if not manifest.get(k)]
    probs = [f'missing or empty fields: {missing}'] if missing else []
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        if manifest.get('repo_commit') != provenance._git(root, 'rev-parse', 'HEAD'):
            probs.append('repo_commit is not this tree')
        if manifest.get('registration_sha256') != provenance.sha256_file(os.path.join(root, 'prereg_amendments.md')):
            probs.append('registration hash differs from prereg_amendments.md')
        if manifest.get('clips_manifest_sha256') != provenance.sha256_file(os.path.join(root, 'configs', 'sdr_clips.json')):
            probs.append('clip manifest hash differs from configs/sdr_clips.json')
        # the MEASURING code is immutable for the session; the post-hoc analysis
        # tools (collector, validators, audit, report generators) may be corrected
        # after the session — their identity is recorded separately below
        live = provenance.code_hashes()
        mcode = manifest.get('code') or {}
        diff = sorted(k for k in set(live) | set(mcode) if live.get(k) != mcode.get(k) and os.path.basename(k) not in ANALYSIS_TOOLS)
        if diff:
            probs.append(f'measuring-code hashes differ from this tree: {diff[:6]}')
    except Exception as e:
        probs.append(f'tree identity could not be derived: {type(e).__name__}')
    rt = manifest.get('runtime') or {}
    if (rt.get('torch'), rt.get('triton')) != (provenance.REGISTERED_TORCH, provenance.REGISTERED_TRITON):
        probs.append('runtime pins are not the registered torch/triton')
    prov = manifest.get('provenance') or {}
    try:
        import loader
        live = provenance.verify(loader.paths_from_env())
        for k in ('msst_commit', 'ckpt_sha256', 'config_sha256', 'torch', 'triton'):
            if prov.get(k) != live.get(k):
                probs.append(f'provenance field {k} differs from the verified pins')
    except Exception as e:
        probs.append(f'pins could not be verified in this environment: {type(e).__name__}')
    item(items, 'session manifest (identity verified against tree, registration, pins)', not probs, probs)
    return manifest.get('manifest_id') if not missing else None, manifest


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
        for i in (range(qualify.SDR_CLIPS) if chunk in SDR_OPS.get(session, OPS[session]) else ()):
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
    n_req = len(OPS[session]) * len(qualify.VALIDATION_SEEDS) + len(SDR_OPS.get(session, OPS[session])) * qualify.SDR_CLIPS
    item(items, f'required reference inventory ({n_req} artifacts, bound and re-hashed)', not bad,
         [f'missing/unbound/changed: {bad[:8]}'])
    # Phase-3 namespaces (gate-09 f133): held-out validation references (pod) and the
    # variant reference directories (E5) are inventoried and joined separately
    vhashes = {}
    if session == 'session3':
        cfg3 = load(os.path.join(ev, 'session3_config.json')) or {}
        if cfg3.get('session_pod') and not cfg3.get('shakedown'):
            for chunk in HELD_OUT_S3:
                for seed in qualify.VALIDATION_SEEDS:
                    f = os.path.join(ref, f'ref_A_c{chunk}_s{seed}.pt'); side = load(f + '.lineage.json')
                    if os.path.exists(f) and side and side.get('manifest_id') == manifest_id and side.get('sha256') == provenance.sha256_file(f):
                        hashes[('val', chunk, seed)] = side['sha256']
        for d in sorted(glob.glob(os.path.join(ev, 'ref_variant_*'))):
            v = os.path.basename(d)[len('ref_variant_'):]
            for chunk in OPS[session]:
                for seed in qualify.VALIDATION_SEEDS:
                    f = os.path.join(d, f'ref_A_c{chunk}_s{seed}.pt'); side = load(f + '.lineage.json')
                    if os.path.exists(f) and side and side.get('manifest_id') == manifest_id and side.get('sha256') == provenance.sha256_file(f):
                        vhashes[(v, chunk, seed)] = side['sha256']
    # join: every recorded hash must equal the re-hashed artifact of its operating point
    joined = 0
    mismatch = []
    for (arm, chunk), tr in runs.items():
        if chunk not in OPS[session] and not (session == 'session3' and chunk in HELD_OUT_S3):
            continue                      # records of the other session's operating points
        for tag, r in tr.items():
            vname = ((r.get('variant') or {}).get('variant')) if isinstance(r.get('variant'), dict) else None
            for p in (r.get('validation') or {}).get('passes') or []:
                if p.get('ref_sha256'):
                    joined += 1
                    want_h = vhashes.get((vname, chunk, p.get('seed'))) if vname else hashes.get(('val', chunk, p.get('seed')))
                    if want_h != p['ref_sha256']:
                        mismatch.append(f'{arm}@{chunk}/{tag} seed {p.get("seed")}' + (f' (variant {vname})' if vname else ''))
            for row in (r.get('sdr_gate') or {}).get('clips') or []:
                if row.get('a_output_sha256'):
                    joined += 1
                    if hashes.get(('clip', chunk, row.get('index'))) != row['a_output_sha256']:
                        mismatch.append(f'{arm}@{chunk}/{tag} clip {row.get("index")}')
    for (pair, chunk, tag), m in matched.items():
        vname = m.get('variant')
        for arm, a in (m.get('arms') or {}).items():
            for p in (a.get('validation') or {}).get('passes') or []:
                if p.get('ref_sha256'):
                    joined += 1
                    want_h = vhashes.get((vname, chunk, p.get('seed'))) if vname else hashes.get(('val', chunk, p.get('seed')))
                    if want_h != p['ref_sha256']:
                        mismatch.append(f'matched {arm}@{chunk}/{tag} seed {p.get("seed")}')
    item(items, f'recorded reference hashes join the audited artifacts ({joined} joins)', joined > 0 and not mismatch,
         [f'{len(mismatch)} mismatches: {mismatch[:6]}'] if mismatch else ['no recorded reference hashes'])


def session_blocks(session, runs, matched):
    """Timing blocks that belong to a session by REGISTERED tag/operating point
    (gate-07 f101): session 1 = r1–r3 at 352800/4410 and the matched pairs;
    session 2 = r1–r3 at the sweep points, every trace and perturbation run."""
    out = []
    for (a, c), rs in runs.items():
        for t, r in rs.items():
            blocks = ((r.get('timing') or {}).get('blocks') or [])
            if session == 'session1' and c in (352800, 4410) and t in ('r1', 'r2', 'r3'):
                out += [(b.get('unix_start'), b.get('unix_end'), f'{a}@{c}/{t}') for b in blocks]
            if session == 'session2' and ((c in (22050, 88200) and t in ('r1', 'r2', 'r3')) or t == 'trace' or t.startswith('pert_')):
                out += [(b.get('unix_start'), b.get('unix_end'), f'{a}@{c}/{t}') for b in blocks]
            if session == 'session3' and c in OPS['session3'] + HELD_OUT_S3:
                out += [(b.get('unix_start'), b.get('unix_end'), f'{a}@{c}/{t}') for b in blocks]
            if session == 'session4' and c in OPS['session4']:
                out += [(b.get('unix_start'), b.get('unix_end'), f'{a}@{c}/{t}') for b in blocks]
    if session in ('session1', 'session3', 'session4'):
        for (pair, c, t), m in matched.items():
            out += [(b.get('unix_start'), b.get('unix_end'), f'matched {pair}/{t}') for b in (m.get('blocks') or [])]
    return out


def clock_series(ev, session, gpu_uuid):
    """(timestamp, SM MHz) samples of this GPU from the session clock log (UTC unless CLOCK_TZ=local)."""
    clocks = os.path.join(ev, f'clocks_{session}.csv')
    out = []
    if not os.path.exists(clocks):
        return out
    for line in open(clocks):
        parts = [x.strip() for x in line.split(',')]
        if len(parts) != 7 or (gpu_uuid and parts[1] != gpu_uuid):
            continue
        try:
            sm = float(parts[2].replace('MHz', '').strip())
        except ValueError:
            continue
        ts = None
        for fmt in ('%Y/%m/%d %H:%M:%S.%f', '%Y/%m/%d %H:%M:%S'):
            try:
                d = _dt.datetime.strptime(parts[0], fmt)
                ts = d.timestamp() if os.environ.get('CLOCK_TZ', 'utc') == 'local' else d.replace(tzinfo=_dt.timezone.utc).timestamp()
                break
            except ValueError:
                continue
        if ts is not None:
            out.append((ts, sm))
    out.sort()
    return out


def block_sm_median(series, a, b, window=3.0):
    # with the 5 Hz session-3 sampler most 100 ms blocks hold their own samples; the window is the fallback
    """Median SM clock over the samples inside [a, b]; a block shorter than the 1 Hz
    sampling interval (50 passes at 100 ms take 0.15 s) uses the nearest samples within
    `window` seconds on either side instead."""
    if not series or not (fin(a) and fin(b)):
        return None
    ts = [t for t, _ in series]
    i = bisect.bisect_left(ts, a); j = bisect.bisect_right(ts, b)
    vals = [v for _, v in series[i:j]]
    if not vals:
        i = bisect.bisect_left(ts, a - window); j = bisect.bisect_right(ts, b + window)
        vals = [v for _, v in series[i:j]]
    return qualify._median(vals) if vals else None


def drift_flags(ev, session, runs, matched, gpu_uuid, limit=0.05):
    """Registered rule (prereg v0.2): clocks may not drift more than 5 % between COMPARED
    blocks. Matched pairs: the a-block vs the b-block of each repeat; the flag is
    REPEAT-scoped ({(pair, chunk, tag): {repeat: message}}). Headline cells: the per-block
    SM medians across r1..rN ({(arm, chunk): message}). Blocks shorter than the sampling
    interval use the nearest samples (block_sm_median). Shared by the audit and by every
    publisher (gate-10 f139/f140), so published aggregates exclude exactly what the audit flags."""
    series = clock_series(ev, session, gpu_uuid)
    flagged = {'pairs': {}, 'cells': {}, 'unresolved': [], 'checked': 0}
    for (pair, c, t), m in matched.items():
        if m.get('status') not in ('VALID',):
            continue
        by_rep = {}
        for b in (m.get('blocks') or []):
            by_rep.setdefault(b.get('repeat'), []).append(b)
        for rep, bs in sorted(by_rep.items(), key=lambda kv: (kv[0] is None, kv[0])):
            if len(bs) != 2:
                continue
            sa = block_sm_median(series, bs[0].get('unix_start'), bs[0].get('unix_end'))
            sb = block_sm_median(series, bs[1].get('unix_start'), bs[1].get('unix_end'))
            if sa is None or sb is None:
                flagged['unresolved'].append(f'matched {pair}@{c}/{t} rep {rep}')
                continue
            flagged['checked'] += 1
            drift = abs(sa - sb) / ((sa + sb) / 2)
            if drift > limit:
                flagged['pairs'].setdefault((pair, c, t), {})[rep] = f'{sa:.0f} vs {sb:.0f} MHz ({drift * 100:.1f} %)'
    for (a, c), rs in runs.items():
        meds = []
        for t, r in rs.items():
            if not t.startswith('r') or not t[1:].isdigit() or r.get('status') != 'VALID':
                continue
            for b in ((r.get('timing') or {}).get('blocks') or []):
                sm = block_sm_median(series, b.get('unix_start'), b.get('unix_end'))
                if sm is None:
                    flagged['unresolved'].append(f'{a}@{c}/{t} block {b.get("repeat")}')
                else:
                    meds.append(sm)
        if len(meds) >= 2:
            flagged['checked'] += 1
            drift = (max(meds) - min(meds)) / (sum(meds) / len(meds))
            if drift > limit:
                flagged['cells'][(a, c)] = f'restarts: {min(meds):.0f}–{max(meds):.0f} MHz ({drift * 100:.1f} %)'
    return flagged


def audit_clock_drift(ev, items, session, runs, matched, gpu_uuid, limit=0.05):
    """Adds the informational drift item; the consequences are scoped per comparison by the callers."""
    flagged = drift_flags(ev, session, runs, matched, gpu_uuid, limit)
    n_bad = len(flagged['pairs']) + len(flagged['cells'])
    unresolved, checked = flagged['unresolved'], flagged['checked']
    item(items, f'clock drift <= {limit * 100:.0f} % between compared blocks: {checked} comparisons, {n_bad} flagged (informational; consequences are scoped per comparison)',
         checked > 0 and not unresolved, [f'unresolved (no samples near block) {unresolved[:4]}'] if unresolved else ['no comparisons'],
         note=(('flagged: ' + '; '.join(f'{k}: {v}' for k, v in list(flagged['pairs'].items())[:8] + list(flagged['cells'].items())[:4])) if n_bad else 'no violations'),
         optional=True)
    return flagged


def audit_clock(ev, items, session, runs, matched, gpu_uuid):
    """This session's clock log (samples of THIS GPU, numeric fields, 1 Hz) must
    cover every block of this session with bounded gaps (gate-07 f101)."""
    clocks = os.path.join(ev, f'clocks_{session}.csv')
    times, bad_lines, other_gpu = [], 0, 0
    if os.path.exists(clocks):
        for line in open(clocks):
            parts = [x.strip() for x in line.split(',')]
            if len(parts) != 7:
                bad_lines += 1
                continue
            if gpu_uuid and parts[1] != gpu_uuid:
                other_gpu += 1
                continue
            try:
                for tok, unit in ((parts[2], 'MHz'), (parts[3], 'MHz'), (parts[4], 'W'), (parts[6], '%')):
                    float(tok.replace(unit, '').strip())
                int(parts[5])
            except ValueError:
                bad_lines += 1
                continue
            ts = None
            for fmt in ('%Y/%m/%d %H:%M:%S.%f', '%Y/%m/%d %H:%M:%S'):
                try:
                    # the clock log carries the measuring host's local time; the pod runs UTC. CLOCK_TZ=local keeps
                    # the old behaviour for logs written by a host in this machine's zone (ml-beast smokes)
                    d = _dt.datetime.strptime(parts[0], fmt)
                    ts = d.timestamp() if os.environ.get('CLOCK_TZ', 'utc') == 'local' else d.replace(tzinfo=_dt.timezone.utc).timestamp()
                    break
                except ValueError:
                    continue
            if ts is None:
                bad_lines += 1
            else:
                times.append(ts)
    times.sort()
    blocks = session_blocks(session, runs, matched)
    if not times:
        item(items, f'clock log covers every timing block of this session ({len(blocks)} blocks)', False,
             ['clock log missing, unparsable, or without samples of this GPU'])
        return
    nonfinite = [n for a, b, n in blocks if not (fin(a) and fin(b) and b >= a)]
    uncovered = []
    for a, b, n in blocks:
        if not (fin(a) and fin(b) and b >= a):
            continue
        i = bisect.bisect_left(times, a - 5)
        j = bisect.bisect_right(times, b + 5)
        seg = times[i:j]
        if not seg or seg[0] > a + 5 or seg[-1] < b - 5 or any(seg[k + 1] - seg[k] > 5 for k in range(len(seg) - 1)):
            uncovered.append(n)
    ok = bool(blocks) and not nonfinite and not uncovered and bad_lines == 0
    item(items, f'clock log covers every timing block of this session ({len(blocks)} blocks, gaps <= 5 s)', ok,
         [f'uncovered {uncovered[:5]}, non-finite {nonfinite[:3]}, malformed lines {bad_lines}, blocks {len(blocks)}'])


def audit_prerequisites(ev, items, manifest_id, run_map):
    """Planted-fault validator checks and the C3 operator qualification, bound to
    THIS manifest, are prerequisites of BOTH sessions (gate-07 f98/f99)."""
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
    probs = []
    cfg3 = load(os.path.join(ev, 'session3_config.json')) or {}
    want = set(C3_FRAMES)
    if cfg3 and cfg3.get('session_pod') and not cfg3.get('shakedown'):
        want = want | {'401', '601'}                     # E9 held-out frame counts, pod only
    if os.path.exists(os.path.join(ev, 'session4_config.json')):
        want = {'11', '51', '201', '801', '401', '601', '101'}
    if set(q) != want:
        probs.append(f'frame set {sorted(q)} != registered {sorted(want)}')
    for T, v in q.items():
        oc = v.get('opcheck')
        oc_ok = isinstance(oc, dict) and bool(oc) and all(x == 'SUCCESS' for x in oc.values())
        if not (v.get('status') == 'PASS' and v.get('native_bitwise_equal') is True and oc_ok
                and v.get('cuda_graph_capture_and_replay') == 'PASS' and v.get('compiled_fullgraph') == 'PASS'
                and v.get('output_shape') == [2, 1025, int(T)]):
            probs.append(f'frame {T}: component results not all successful')
    item(items, 'C3 operator qualification (every registered component per frame)', not probs, probs)


def graph_verdict_rederived(gv):
    """The graph-verification verdict re-derived from the 50 per-pass records (session-2 rule, gate-07 f97)."""
    pp = gv.get('per_pass') or []
    keys = ('graph_launch_calls', 'captured_model_kernels', 'uncaptured_model_kernels', 'unresolved_kernels')
    well = len(pp) == 50 and all(all(isinstance(p.get(k), int) and p[k] >= 0 for k in keys) for p in pp)
    if not well:
        return None
    fully = (min(p['graph_launch_calls'] for p in pp) >= 1 and min(p['captured_model_kernels'] for p in pp) > 0
             and max(p['uncaptured_model_kernels'] for p in pp) == 0 and max(p['unresolved_kernels'] for p in pp) == 0)
    return ('FULLY_GRAPHED' if fully else 'NOT_FULLY_GRAPHED' if min(p['graph_launch_calls'] for p in pp) >= 1 else 'NO_REPLAY_EVIDENCE')


def _pair_ok(m, manifest_id):
    return bool(m and m.get('status') == 'VALID' and m.get('manifest_id') == manifest_id
                and m.get('pair_eligible') and not qualify.pair_problems(m))


def _untested_reason(m, needle):
    return bool(m and m.get('status') == 'UNTESTED' and needle in str(m.get('error') or ''))


def audit_session3(ev, items, manifest_id, run_map, matched, cells, pairs, dynamic_aux, drift=None):
    """Phase-3 registrations (prereg_amendments.md v0.2): mandatory items E6, E3, E2, E1,
    E9 (pod; UNTESTED-by-rule on a shakedown), E4a; optional E5 and E7 (registered as
    skippable under the deadline)."""
    cfg = load(os.path.join(ev, 'session3_config.json')) or {}
    shakedown = bool(cfg.get('shakedown'))
    drift = drift or {'pairs': {}, 'cells': {}}
    n_restart = int(cfg.get('headline_restarts') or 5)
    n_pair = int(cfg.get('pair_restarts') or 3)
    audit_prerequisites(ev, items, manifest_id, run_map)
    # ---- E6
    n_pair_reps = int(qualify.REGISTERED_REPEATS)
    def clean_reps(pair, c, t):
        fl = drift['pairs'].get((pair, c, t)) or {}
        return [r for r in range(n_pair_reps) if r not in fl]
    for c in OPS['session3']:
        m = matched.get((('C3', 'C3#2'), c, 'm1'))
        fl = drift['pairs'].get((('C3', 'C3#2'), c, 'm1'))
        ok = _pair_ok(m, manifest_id) and bool(clean_reps(('C3', 'C3#2'), c, 'm1'))
        item(items, f'E6 A/A matched C3 vs C3 @{c} (m1, block-alternating; clock-clean repeats required)', ok,
             (qualify.pair_problems(m) if m else ['no A/A record']) + ([f'clock-drift flagged repeats {fl}'] if fl else []),
             note=(f'clock-drift flagged repeats (registered 5 % rule): {fl}' if fl else None))
    for a in ('C3', 'H', 'A', 'C1'):
        for c in OPS['session3']:
            cell = cells.get(f'{a}_{c}')
            fl = drift['cells'].get((a, c))
            ok = bool(cell and cell['headline_status'] == 'HEADLINE') and not fl
            item(items, f'E6 headline {a}@{c} (r1..r{n_restart}, SDR in r1; clock drift across restarts <= 5 %)', ok,
                 (cell or {}).get('reasons', ['no runs']) + ([f'clock-drift flagged across restarts (registered 5 % rule): {fl}'] if fl else []),
                 note=(f'clock-drift flagged: {fl}; the headline is published as clock-flagged' if fl else None))
    aa = load(os.path.join(ev, 'aa_stats.json')) or {}
    aa_chunks = {v.get('chunk') for v in (aa.get('pairs') or {}).values() if 'median_ms' in v}
    need_aa = set(OPS['session3']) | (set(HELD_OUT_S3) if (cfg.get('session_pod') and not shakedown) else set())
    item(items, 'E6/E9 A/A statistics (block-level bootstrap intervals) for every A/A operating point', need_aa <= aa_chunks,
         [f'A/A statistics present for {sorted(aa_chunks)}, required {sorted(need_aa)}'])
    # ---- pairs helper
    def pair_items(label, a, b, c, tags):
        clean = 0
        for t in tags:
            m = matched.get(((a, b), c, t))
            fl = drift['pairs'].get(((a, b), c, t))
            ok = _pair_ok(m, manifest_id)
            item(items, f'{label} matched {a} vs {b} @{c} ({t})', ok, qualify.pair_problems(m) if m else ['record missing'],
                 note=(f'clock-drift flagged repeats (registered 5 % rule; excluded from published aggregates): {fl}' if fl else None))
            if ok and clean_reps((a, b), c, t):
                clean += 1
        item(items, f'{label} {a} vs {b} @{c}: at least one eligible restart with clock-clean repeats', clean > 0,
             [f'every eligible restart has every repeat clock-drift flagged: {[drift["pairs"].get(((a, b), c, t)) for t in tags]}'])
    def trace_items(label, a, c, sfx='', need_graphed=True, optional=False):
        base = os.path.join(ev, 'traces', f'trace_{a}_{c}_' + (sfx or 'trace'))
        exp = load(base + '_exports.json') or {}
        run = (run_map.get((a, c)) or {}).get(sfx or 'trace')
        probs = [] if exp.get('complete') is True and exp.get('graph_trace_mode') == 'node' else ['node-level exports incomplete']
        probs += qualify.trace_problems(run)
        if run and run.get('manifest_id') != manifest_id:
            probs.append('foreign manifest')
        for rep, cols in (('cuda_gpu_trace', ('Start', 'Duration', 'CorrId', 'Name')), ('cuda_gpu_kern_sum', ('Total Time', 'Name')),
                          ('cuda_api_trace', ('Start', 'CorrId', 'Name')), ('nvtx_gpu_proj_trace', ('Start', 'Name'))):
            if not csv_rows(f'{base}_{rep}.csv', cols):
                probs.append(f'export {rep} missing, unparsable, or empty')
        item(items, f'{label} trace {a}@{c}' + (f' [{sfx}]' if sfx else ''), not probs, probs, optional=optional)
        gsfx = f'_{sfx}' if sfx else ''
        gv = load(os.path.join(ev, 'traces', f'graph_verify_{a}_{c}{gsfx}.json')) or {}
        derived = graph_verdict_rederived(gv)
        accepted = ('FULLY_GRAPHED',) if need_graphed else ('FULLY_GRAPHED', 'NOT_FULLY_GRAPHED', 'NO_REPLAY_EVIDENCE')
        ok = (derived is not None and gv.get('verdict') == derived and derived in accepted and gv.get('qualified') is True
              and gv.get('registered_protocol') is True and gv.get('pass_count_complete') is True)
        item(items, f'{label} graph verification {a}@{c}' + (f' [{sfx}]' if sfx else '') + (' FULLY_GRAPHED' if need_graphed else ' (verdict recorded)'),
             ok, [f"verdict {gv.get('verdict', 'MISSING')} (re-derived {derived}) qualified={gv.get('qualified')} passes={len(gv.get('per_pass') or [])}"], optional=optional)
        names = [f'graph_verify_{a}_{c}{gsfx}', f'timing_views_{a}_{c}{gsfx}', f'cast_count_{a}_{c}{gsfx}', f'ideal_costs_{a}_{c}{gsfx}', f'template_count_{a}_{c}{gsfx}']
        dynamic_aux.extend(names)
        if optional:
            audit_session3.optional_aux = getattr(audit_session3, 'optional_aux', set()) | set(names)
    def sdr_ok(a, c):
        r = (run_map.get((a, c)) or {}).get('r1')
        return r is not None and not qualify.sdr_problems(r, c, manifest_id, qualify.variant(r))
    def sdr_item(label, a, c):
        r = (run_map.get((a, c)) or {}).get('r1')
        g = (r or {}).get('sdr_gate') or {}
        probs = qualify.sdr_problems(r, c, manifest_id, qualify.variant(r)) if r else ['no r1 record']
        recomputed = qualify.sdr_recomputed(g)[0] if g else 'NONE'
        # a FAIL is a registered outcome (kill branches), so a validated FAIL payload is 'recorded'; a PASS must be a validated PASS
        ok = bool(r and r.get('manifest_id') == manifest_id and ((not probs and recomputed == 'PASS') or (g.get('status') == 'FAIL' and recomputed == 'FAIL')))
        item(items, f'{label} SDR gate {a}@{c} recorded and validated (r1; eight clips, lineage-hashed)', ok, probs or [f'recomputed {recomputed}'],
             note=(f"SDR {recomputed} mean delta {g.get('mean_delta_sdr_ref_db')}" if ok else None))
    tags = [f'm{i}' for i in range(1, n_pair + 1)]
    # ---- E3
    pair_items('E3', 'C1', 'C1P', 352800, tags)
    trace_items('E3', 'C1P', 352800, need_graphed=False)
    # ---- E2
    for c in OPS['session3']:
        pair_items('E2', 'C3', 'C3M', c, tags)
        sdr_item('E2', 'C3M', c)
        trace_items('E2', 'C3M', c, need_graphed=True)
        for a in ('C3', 'C3M'):
            tc = load(os.path.join(ev, 'traces', f'template_count_{a}_{c}.json')) or {}
            item(items, f'E2 template count {a}@{c}', fin(tc.get('triton_template_kernels_per_pass')), ['template count missing'])
            dynamic_aux.append(f'template_count_{a}_{c}')
    # ---- E1
    branch = load(os.path.join(ev, 'e1_branch.json')) or {}
    item(items, 'E1 branch decision recorded (e1_branch.json)', 'run_C3P' in branch, ['missing'],
         note=(f"lane stopped by SDR: {branch.get('e1_lane_stopped_sdr')}" if branch.get('e1_lane_stopped_sdr') else None))
    for c in OPS['session3']:
        pair_items('E1', 'C3', 'C3F', c, tags)
        sdr_item('E1', 'C3F', c)
        trace_items('E1', 'C3', c, need_graphed=True)
        trace_items('E1', 'C3F', c, need_graphed=True)
        pair_items("E1(a')", 'C3', 'C3G', c, tags)
        sdr_item("E1(a')", 'C3G', c)
        trace_items("E1(a')", 'C3G', c, need_graphed=True)
        for a in ('C3', 'C3F', 'C3G'):
            cc = load(os.path.join(ev, 'traces', f'cast_count_{a}_{c}.json')) or {}
            item(items, f'E1 cast count {a}@{c}', fin(cc.get('cast_kernels_per_pass')), ['cast count missing'])
        if branch.get('run_C3P'):
            pair_items('E1(b)', 'C3', 'C3P', c, tags)
            sdr_item('E1(b)', 'C3P', c)
            trace_items('E1(b)', 'C3P', c, need_graphed=True)
    # ---- E9
    for c in HELD_OUT_S3:
        if shakedown or not cfg.get('session_pod'):
            for t in ['m1'] + tags:
                pair = ('C3', 'C3') if t == 'm1' else ('C3', 'H')
                m = matched.get((pair, c, t))
                item(items, f'E9 held-out @{c} {pair[0]} vs {pair[1]} ({t}): UNTESTED by rule (shakedown, never run off-pod)',
                     _untested_reason(m, 'held-out'), ['expected an UNTESTED terminal record citing the held-out rule'])
        else:
            m = matched.get((('C3', 'C3#2'), c, 'm1'))
            item(items, f'E9 A/A matched C3 vs C3 @{c} (m1)', _pair_ok(m, manifest_id), qualify.pair_problems(m) if m else ['record missing'])
            pair_items('E9', 'C3', 'H', c, tags)
            ref = os.path.join(ev, 'ref')
            bad = []
            for seed in qualify.VALIDATION_SEEDS:
                f = os.path.join(ref, f'ref_A_c{c}_s{seed}.pt'); side = load(f + '.lineage.json')
                if not os.path.exists(f) or not side or side.get('manifest_id') != manifest_id or side.get('sha256') != provenance.sha256_file(f):
                    bad.append(os.path.basename(f))
            item(items, f'E9 validation references @{c} bound and re-hashed', not bad, [f'{bad}'])
    if not (shakedown or not cfg.get('session_pod')):
        e9 = load(os.path.join(ev, 'e9_check.json')) or {}
        item(items, 'E9 registered predictions checked (e9_check.json, verdict recorded)', e9.get('verdict') in ('PASS', 'FAIL'),
             [f"verdict {e9.get('verdict', 'MISSING')}"], note=f"verdict {e9.get('verdict')}")
        dynamic_aux.append('e9_check')
        reg_sha = cfg.get('e9_registration_sha256')
        item(items, 'E9 registration file hash recorded in session config', bool(reg_sha), ['missing'])
    # ---- E4a
    dec = load(os.path.join(ev, 'e4a_decision.json')) or {}
    item(items, 'E4a control/mode decision recorded (e4a_decision.json)', dec.get('control') in ('C3', 'C3M'), ['missing'])
    ctrl = dec.get('control') or 'C3'
    # re-derive the registered decision from the ELIGIBLE E2 pairs (clock-clean repeats) and C3M's validated quality at both O
    import statistics as _st
    e2 = []
    for t in tags:
        m = matched.get((('C3', 'C3M'), 352800, t))
        if _pair_ok(m, manifest_id):
            rd = qualify.recomputed_matched_delta(m)
            fl = drift['pairs'].get((('C3', 'C3M'), 352800, t)) or {}
            if rd:
                diffs = [d for i, d in enumerate(rd['a_minus_b_per_block']) if i not in fl]
                if diffs:
                    e2.append(qualify._median(diffs))
    gain = _st.median(e2) if e2 else None
    q_ok = all(sdr_ok('C3M', c) and graph_verdict_rederived(load(os.path.join(ev, 'traces', f'graph_verify_C3M_{c}.json')) or {}) == 'FULLY_GRAPHED'
               and (((run_map.get(('C3M', c)) or {}).get('r1') or {}).get('status') == 'VALID') for c in OPS['session3'])
    expect = 'C3M' if (gain is not None and gain >= 1.0 and q_ok) else 'C3'
    item(items, f'E4a decision re-derived from eligible E2 pairs and validated C3M quality (expected control {expect}, gain {gain})',
         dec.get('control') == expect and dec.get('mode') == ('max-autotune' if expect == 'C3M' else 'reduce-overhead'),
         [f"recorded control {dec.get('control')} / mode {dec.get('mode')} vs re-derived {expect} (gain {gain}, quality_ok {q_ok})"])
    for c in OPS['session3']:
        pair_items('E4a', ctrl, 'C3N', c, tags)
        sdr_item('E4a', 'C3N', c)
        trace_items('E4a', 'C3N', c, need_graphed=True)
        modes = {}
        for t in tags:
            m = matched.get(((ctrl, 'C3N'), c, t)) or {}
            modes[f'matched {t}'] = ((m.get('arms') or {}).get('C3N') or {}).get('compile_mode')
        for t in ('r1', 'trace'):
            modes[t] = (((run_map.get(('C3N', c)) or {}).get(t)) or {}).get('compile_mode')
        item(items, f'E4a C3N@{c} compile mode matches the decision ({dec.get("mode")}) in every matched, headline and traced record',
             all(v == dec.get('mode') for v in modes.values()), [f'modes {modes} vs decision {dec.get("mode")}'])
    # ---- E5 (optional)
    variants_ = cfg.get('e5_variants') or []
    n_e5 = int(cfg.get('e5_restarts') or 2)
    for v in variants_:
        for a in ('C3', 'H'):
            for c in OPS['session3']:
                for i in range(1, n_e5 + 1):
                    r = (run_map.get((a, c)) or {}).get(f'v_{v}_r{i}')
                    ok = bool(r and ((r.get('status') == 'VALID' and (r.get('validation') or {}).get('status') == 'PASS'
                                      and r.get('manifest_id') == manifest_id and (r.get('variant') or {}).get('variant') == v)
                                     or _untested_reason(r, 'allowance') or _untested_reason(r, 'deadline')))
                    item(items, f'E5 variant {v} {a}@{c} r{i} (VALID+PASS, or UNTESTED under the allowance)', ok,
                         [f"status {(r or {}).get('status', 'MISSING')}"], optional=True)
            trace_items('E5', a, 4410, sfx=f'v_{v}_trace', need_graphed=True, optional=True)
    e5 = load(os.path.join(ev, 'e5_summary.json')) or {}
    item(items, 'E5 summary present', bool(e5.get('cells')), ['missing'], optional=True)
    # ---- E7 (optional)
    bb = load(os.path.join(ev, 'boundary_bench.json')) or {}
    chains = bb.get('chains') or {}
    bad = [k for k in BOUNDARY_CHAINS + BOUNDARY_CHAINS_S3 if chains.get(k, {}).get('status') not in ('ok', 'UNTESTED')
           or (chains.get(k, {}).get('status') == 'ok' and not fin(chains[k].get('per_launch_us')))]
    item(items, 'E7 boundary chains incl. prologue chains (terminal status per registered chain)', bool(chains) and not bad,
         [f'non-terminal {bad[:6]}'], optional=True)
    item(items, 'E7 PDL evidence recorded (launch attributes, PTX, SASS check)',
         bool(bb.get('pdl_launch_attributes')) and bb.get('pdl_ptx') is True and isinstance(bb.get('pdl_sass'), dict),
         [f"attrs {bool(bb.get('pdl_launch_attributes'))} ptx {bb.get('pdl_ptx')} sass {type(bb.get('pdl_sass')).__name__}"], optional=True,
         note=(f"SASS mnemonics {(bb.get('pdl_sass') or {}).get('griddep_mnemonics_found')}; e7 {bb.get('e7_pdl_saving')}" if bb else None))
    ta = load(os.path.join(ev, 'boundary_bench_trace_analysis.json')) or {}
    qual = ta.get('_qualification') or {}
    pdl_segs = {k: v.get('overlap_observed') for k, v in ta.items() if '/pdl_' in k and v.get('status') == 'ok'}
    item(items, 'E7 boundary trace analysis qualified (overlap recorded per PDL segment)', bool(qual.get('cases')) and qual.get('accepted_segments', 0) > 0,
         [f'qualification {qual}'], optional=True, note=f'overlap_observed {pdl_segs}' if pdl_segs else None)
    # ---- references at the held-out lengths are audited above; the standard inventory covers OPS['session3']


def audit_session4(ev, items, manifest_id, run_map, matched, cells, pairs, dynamic_aux, drift=None):
    """Phase-4 plan (docs/phase4/PLAN.md): P0 A/A, PA C3–H at four lengths, PB C1P diagnostics, PC1–PC4 pairs
    (mandatory); PQ gates/traces mandatory for the SDR gates and the 4410 traces, optional for the 8 s traces
    of the max-autotune arm; statistics files present."""
    cfg = load(os.path.join(ev, 'session4_config.json')) or {}
    drift = drift or {'pairs': {}, 'cells': {}}
    n_pair = int(cfg.get('pair_restarts') or 3)
    tags = [f'm{i}' for i in range(1, n_pair + 1)]
    audit_prerequisites(ev, items, manifest_id, run_map)
    def clean_reps(pair, c, t):
        fl = drift['pairs'].get((pair, c, t)) or {}
        return [r for r in range(int(qualify.REGISTERED_REPEATS)) if r not in fl]
    SDR_GATED = ('C3', 'C3I', 'C3G', 'C3M', 'C3MG', 'H', 'HG')
    def sdr_valid_pass(a_, c_):
        r = (run_map.get((a_, c_)) or {}).get('r1')
        if not r:
            return False
        g = r.get('sdr_gate') or {}
        return (not qualify.sdr_problems(r, c_, manifest_id, qualify.variant(r))) and qualify.sdr_recomputed(g)[0] == 'PASS'
    def pair_items(label, a, b, c, tags_, min_clean=1):
        clean = 0
        gate_ok = True
        if c in SDR_OPS.get('session4', ()):
            gate_ok = all(sdr_valid_pass(x, c) for x in (a, b) if x in SDR_GATED)
        for t in tags_:
            m = matched.get(((a, b), c, t)); fl = drift['pairs'].get(((a, b), c, t))
            ok = _pair_ok(m, manifest_id) and gate_ok
            item(items, f'{label} matched {a} vs {b} @{c} ({t})', ok, (qualify.pair_problems(m) if m else ['record missing']) + ([] if gate_ok else ['an arm of this pair has no validated SDR PASS at this operating point (registered stop rule)']),
                 note=(f'clock-drift flagged repeats (excluded from published aggregates): {fl}' if fl else None))
            if ok and clean_reps((a, b), c, t):
                clean += 1
        item(items, f'{label} {a} vs {b} @{c}: at least {min_clean} eligible restart(s) with clock-clean repeats', clean >= min_clean,
             [f'clean eligible restarts {clean}'])
    def trace_items(label, a, c, need_graphed=True, optional=False):
        base = os.path.join(ev, 'traces', f'trace_{a}_{c}_trace')
        exp = load(base + '_exports.json') or {}
        run = (run_map.get((a, c)) or {}).get('trace')
        probs = [] if exp.get('complete') is True and exp.get('graph_trace_mode') == 'node' else ['node-level exports incomplete']
        probs += qualify.trace_problems(run)
        if run and run.get('manifest_id') != manifest_id:
            probs.append('foreign manifest')
        for rep, cols in (('cuda_gpu_trace', ('Start', 'Duration', 'CorrId', 'Name')), ('cuda_gpu_kern_sum', ('Total Time', 'Name')),
                          ('cuda_api_trace', ('Start', 'CorrId', 'Name')), ('nvtx_gpu_proj_trace', ('Start', 'Name'))):
            if not csv_rows(f'{base}_{rep}.csv', cols):
                probs.append(f'export {rep} missing, unparsable, or empty')
        item(items, f'{label} trace {a}@{c}', not probs, probs, optional=optional)
        gv = load(os.path.join(ev, 'traces', f'graph_verify_{a}_{c}.json')) or {}
        derived = graph_verdict_rederived(gv)
        accepted = ('FULLY_GRAPHED',) if need_graphed else ('FULLY_GRAPHED', 'NOT_FULLY_GRAPHED', 'NO_REPLAY_EVIDENCE')
        ok = (derived is not None and gv.get('verdict') == derived and derived in accepted and gv.get('qualified') is True
              and gv.get('registered_protocol') is True and gv.get('pass_count_complete') is True)
        item(items, f'{label} graph verification {a}@{c}' + (' FULLY_GRAPHED' if need_graphed else ' (verdict recorded)'), ok,
             [f"verdict {gv.get('verdict', 'MISSING')} (re-derived {derived}) qualified={gv.get('qualified')} passes={len(gv.get('per_pass') or [])}"], optional=optional)
        names = [f'graph_verify_{a}_{c}', f'timing_views_{a}_{c}', f'cast_count_{a}_{c}', f'ideal_costs_{a}_{c}', f'template_count_{a}_{c}']
        dynamic_aux.extend(names)
        if optional:
            audit_session4.optional_aux = getattr(audit_session4, 'optional_aux', set()) | set(names)
    def sdr_item(label, a, c):
        r = (run_map.get((a, c)) or {}).get('r1')
        g = (r or {}).get('sdr_gate') or {}
        probs = qualify.sdr_problems(r, c, manifest_id, qualify.variant(r)) if r else ['no r1 record']
        recomputed = qualify.sdr_recomputed(g)[0] if g else 'NONE'
        ok = bool(r and r.get('manifest_id') == manifest_id and ((not probs and recomputed == 'PASS') or (g.get('status') == 'FAIL' and recomputed == 'FAIL')))
        item(items, f'{label} SDR gate {a}@{c} recorded and validated (r1)', ok, probs or [f'recomputed {recomputed}'],
             note=(f"SDR {recomputed} mean delta {g.get('mean_delta_sdr_ref_db')}" if ok else None))
    # P0
    for c in [int(x) for x in (cfg.get('aa_lengths') or [44100, 176400, 264600, 352800, 4410])]:
        m = matched.get((('C3', 'C3#2'), c, 'm1')); fl = drift['pairs'].get((('C3', 'C3#2'), c, 'm1'))
        item(items, f'P0 A/A matched C3 vs C3 @{c} (m1; clock-clean repeats required)', _pair_ok(m, manifest_id) and bool(clean_reps(('C3', 'C3#2'), c, 'm1')),
             (qualify.pair_problems(m) if m else ['no A/A record']) + ([f'flagged {fl}'] if fl else []))
    # PA
    for c in [int(x) for x in (cfg.get('pa_lengths') or [44100, 176400, 264600, 352800])]:
        pair_items('PA', 'C3', 'H', c, tags, min_clean=1)
    # PB
    cell = cells.get('C1P_352800')
    fl = drift['cells'].get(('C1P', 352800))
    item(items, 'PB standalone unprofiled C1P @352800 (r1..r3 HEADLINE-grade cell; clock drift across restarts <= 5 %)', bool(cell and cell['headline_status'] == 'HEADLINE') and not fl,
         (cell or {}).get('reasons', ['no runs']) + ([f'clock-drift flagged: {fl}'] if fl else []))
    pair_items('PB', 'C1', 'C1P', 352800, tags, min_clean=1)
    pair_items('PB (order swapped)', 'C1P', 'C1', 352800, ['m1', 'm2'], min_clean=1)
    m = matched.get((('C1', 'C1P'), 352800, 'p1'))
    mt = load(os.path.join(ev, 'traces', 'matched_trace_C1_C1P_352800_p1.json')) or {}
    exp = load(os.path.join(ev, 'traces', 'trace_matched_C1_C1P_352800_p1_exports.json')) or {}
    pbase = os.path.join(ev, 'traces', 'trace_matched_C1_C1P_352800_p1')
    exports_ok = all(csv_rows(f'{pbase}_{rep}.csv', cols) for rep, cols in (('cuda_gpu_trace', ('Start', 'Duration', 'Name')), ('cuda_api_trace', ('Start', 'Name')), ('nvtx_pushpop_trace', ('Start', 'Name'))))
    pprobs = qualify.pair_problems(m) if m else ['record missing']
    ok = bool(m and not pprobs and m.get('manifest_id') == manifest_id and m.get('profiled_repeat') == 0 and exp.get('complete') is True and exports_ok
              and mt.get('status') == 'ok' and all(a_ in (mt.get('arms') or {}) and sorted(p_.get('pass') for p_ in mt['arms'][a_].get('per_pass') or []) == list(range(50)) for a_ in ('C1', 'C1P')))
    item(items, 'PB profiled matched C1/C1P process (record passes the shared validator; repeat 0 captured; populated exports; aligned analysis with unique pass ids 0..49 per arm)', ok,
         pprobs + [f"profiled_repeat {(m or {}).get('profiled_repeat')} exports {exp.get('complete')} populated {exports_ok} analysis {mt.get('status')}"])
    dynamic_aux.append('matched_trace_C1_C1P_352800_p1')
    trace_items('PB', 'C1P', 352800, need_graphed=False)
    trace_items('PB', 'C1', 352800, need_graphed=False)
    # PC1–PC4
    for c in [int(x) for x in (cfg.get('pc_ops') or [4410, 352800])]:
        pair_items('PC1', 'C3', 'C3I', c, tags, min_clean=1)
        pair_items('PC2', 'C3I', 'C3G', c, tags, min_clean=1)
        pair_items('PC3', 'C3M', 'C3MG', c, tags, min_clean=1)
        pair_items('PC4', 'H', 'HG', c, tags, min_clean=1)
        for a in SDR_GATED:
            sdr_item('PQ', a, c)
    # PQ traces: mandatory at 4410 for C3I/HG/C3G/C3 and at 352800 for C3I/HG/C3G/C3; C3MG optional at both
    for a in ('C3I', 'HG', 'C3G', 'C3'):
        for c in (4410, 352800):
            trace_items('PQ', a, c, need_graphed=True)
    trace_items('PQ', 'C3MG', 4410, need_graphed=True, optional=False)
    trace_items('PQ', 'C3MG', 352800, need_graphed=True, optional=True)
    for name in ('aa_stats.json', 'stats_v2.json'):
        item(items, f'statistics file {name} present', bool(load(os.path.join(ev, name))), ['missing'])


def csv_rows(path, required_cols):
    """Parse an export: header must contain the required columns; returns row count."""
    import csv
    try:
        with open(path, newline='') as f:
            rd = csv.reader(f)
            head = [h.strip().strip('"').lower() for h in next(rd)]
            # nsys spells the correlation column 'CorrId' (GPU trace) and 'CorrID' (API trace)
            if not all(any(c.lower() in h for h in head) for c in required_cols):
                return None
            return sum(1 for _ in rd)
    except Exception:
        return None


def audit_aux(ev, items, session, dynamic):
    required = list(dict.fromkeys(STATIC_AUX[session] + dynamic))      # deduplicated, order kept
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
    ap.add_argument('--session', choices=['session1', 'session2', 'session3', 'session4'], required=True)
    args = ap.parse_args()
    ev = args.evidence_dir
    items = []
    manifest_id, manifest = audit_manifest(ev, items)
    item(items, 'collection summary present', load(os.path.join(ev, 'summary_full.json')) is not None,
         ['summary_full.json missing or invalid'])
    run_map = load_runs(ev)
    matched = load_matched(ev)
    if args.session == 'session3':
        # the headline restart count is bound to the session record, never to the environment of the re-audit
        cfg3 = load(os.path.join(ev, 'session3_config.json')) or {}
        n3 = int(cfg3.get('headline_restarts') or 5)
        collect_full.REQUIRED_RESTARTS = tuple(f'r{i}' for i in range(1, n3 + 1))
    if args.session == 'session4':
        collect_full.REQUIRED_RESTARTS = ('r1', 'r2', 'r3')
    mprov = (manifest.get('provenance') or {})
    foreign = []
    for (a, c), tr in run_map.items():
        for t, r in tr.items():
            rp = r.get('provenance') or {}
            if r.get('status') in ('VALID', 'INVALID') and any(rp.get(k) != mprov.get(k) for k in ('msst_commit', 'ckpt_sha256', 'config_sha256')):
                foreign.append(f'{a}@{c}/{t}')
    for (pair, c, t), m in matched.items():
        rp = m.get('provenance') or {}
        if m.get('status') in ('VALID', 'INVALID') and any(rp.get(k) != mprov.get(k) for k in ('msst_commit', 'ckpt_sha256', 'config_sha256')):
            foreign.append(f'matched {pair}@{c}/{t}')
    item(items, 'every run record carries the manifest provenance pins', not foreign, [f'{len(foreign)} records differ: {foreign[:6]}'])
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

    if args.session == 'session3':
        drift = audit_clock_drift(ev, items, args.session, run_map, matched, (manifest.get('env') or {}).get('gpu_uuid'))
        audit_session3(ev, items, manifest_id, run_map, matched, cells, pairs, dynamic_aux, drift)
    elif args.session == 'session4':
        drift = audit_clock_drift(ev, items, args.session, run_map, matched, (manifest.get('env') or {}).get('gpu_uuid'))
        audit_session4(ev, items, manifest_id, run_map, matched, cells, pairs, dynamic_aux, drift)
    elif args.session == 'session1':
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
        audit_prerequisites(ev, items, manifest_id, run_map)
    else:
        audit_prerequisites(ev, items, manifest_id, run_map)
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
            for rep, cols in (('cuda_gpu_trace', ('Start', 'Duration', 'CorrId', 'Name')),
                              ('cuda_gpu_kern_sum', ('Total Time', 'Name')),
                              ('cuda_api_trace', ('Start', 'CorrId', 'Name')),
                              ('nvtx_gpu_proj_trace', ('Start', 'Name'))):
                n = csv_rows(f'{base}_{rep}.csv', cols)
                if not n:
                    probs.append(f'export {rep} missing, unparsable, or empty')
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
                keys = ('graph_launch_calls', 'captured_model_kernels', 'uncaptured_model_kernels', 'unresolved_kernels')
                well = len(pp) == 50 and all(all(isinstance(p.get(k), int) and p[k] >= 0 for k in keys) for p in pp)
                if well:
                    # the verdict re-derived from the per-pass evidence (gate-07 f97)
                    fully = (min(p['graph_launch_calls'] for p in pp) >= 1 and min(p['captured_model_kernels'] for p in pp) > 0
                             and max(p['uncaptured_model_kernels'] for p in pp) == 0 and max(p['unresolved_kernels'] for p in pp) == 0)
                    derived = ('FULLY_GRAPHED' if fully else 'NOT_FULLY_GRAPHED' if min(p['graph_launch_calls'] for p in pp) >= 1
                               else 'NO_REPLAY_EVIDENCE')
                else:
                    derived = None
                ok = (well and gv.get('verdict') == derived and derived in accepted and gv.get('qualified') is True
                      and gv.get('registered_protocol') is True and gv.get('pass_count_complete') is True)
                item(items, f'graph verification {a}@{c}', ok,
                     [f"verdict {gv.get('verdict', 'MISSING')} (re-derived {derived}) qualified={gv.get('qualified')} passes={len(pp)}"])
                tv = load(os.path.join(ev, 'traces', f'timing_views_{a}_{c}.json')) or {}
                med = tv.get('median_us') or {}
                per = tv.get('per_pass') or []
                vk = ('sum_ns', 'union_ns', 'critical_path_observed_order_ns', 'span_ns', 'window_ns')
                per_ok = (len(per) == 50 and all(all(fin(p.get(k)) and p[k] >= 0 for k in vk) for p in per)
                          and all(p['critical_path_observed_order_ns'] <= p['union_ns'] + 1 and p['union_ns'] <= p['span_ns'] + 1
                                  and p['span_ns'] <= p['window_ns'] + 1 and p['sum_ns'] >= p['union_ns'] - 1 for p in per))
                med_ok = per_ok and all(fin(med.get(k)) and abs(med[k] - qualify._median([p[k] for p in per]) / 1e3) <= 1e-6 * max(1.0, abs(med[k]))
                                        for k in vk)
                item(items, f'timing views {a}@{c}', tv.get('passes') == 50 and per_ok and med_ok,
                     ['timing views missing, not 50 well-formed passes, inequality violated, or medians not those of the per-pass values'])
                dynamic_aux += [f'graph_verify_{a}_{c}', f'timing_views_{a}_{c}']
            if a in COMPILED_TRACED and not (a == 'C2' and c == 352800):
                ic = load(os.path.join(ev, 'traces', f'ideal_costs_{a}_{c}.json')) or {}
                sel = (ic.get('variant_selection') or {}).get('status')
                rows = ic.get('rows') or []
                priced = [r for r in rows if fin(r.get('ideal_us_hbm'))]
                rows_ok = all(fin(r.get('time_us_per_pass')) and r['time_us_per_pass'] >= 0 and fin(r.get('launches_per_pass'))
                              and r['launches_per_pass'] > 0 for r in rows)
                priced_ok = all(r['ideal_us_hbm'] >= 0 and (r.get('ideal_us_l2') is None or (fin(r['ideal_us_l2']) and 0 <= r['ideal_us_l2'] <= r['ideal_us_hbm'] + 1e-9))
                                and (r.get('recoverable_share_hbm_model') is None or (fin(r['recoverable_share_hbm_model']) and 0 <= r['recoverable_share_hbm_model'] <= 1))
                                for r in priced)
                totals = ic.get('class_totals_us_per_pass') or {}
                sums = {}
                for r in rows:
                    sums[r.get('class')] = sums.get(r.get('class'), 0.0) + (r.get('time_us_per_pass') or 0.0)
                totals_ok = all(abs(totals.get(k, 0.0) - sums.get(k, 0.0)) <= 1e-6 * max(1.0, sums.get(k, 0.0)) for k in set(totals) | set(sums))
                if a == 'C3':
                    ok = (sel == 'ok' and ic.get('measurement_status') == 'ok' and ic.get('passes_observed') == 50
                          and len(priced) > 0 and rows_ok and priced_ok and totals_ok)
                    item(items, f'ideal costs {a}@{c} (unique variant, 50 windows, consistent priced rows)', ok,
                         [f'selection {sel}, measurement {ic.get("measurement_status")}, priced {len(priced)}, rows_ok={rows_ok}, totals_ok={totals_ok}'])
                else:
                    ok = ic.get('measurement_status') in ('ok', 'UNRESOLVED') and ic.get('passes_observed') == 50 and rows_ok and totals_ok
                    item(items, f'ideal costs {a}@{c} (record, 50 windows, consistent rows)', ok, ['ideal-cost record missing, not the registered protocol, or inconsistent'])
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
            # ratios re-derived from the three run records' samples (gate-07 f97)
            meds = {}
            for m in ('unprofiled', 'graph', 'node'):
                rt = qualify.recomputed_timing(((run_map.get((a, c)) or {}).get(f'pert_{m}') or {}).get('timing'), 50, 1)
                meds[m] = rt['block_medians_ms'][0] if rt else None
            derived_ok = all(fin(v) and v > 0 for v in meds.values())
            g = meds['graph'] / meds['unprofiled'] if derived_ok else None
            n = meds['node'] / meds['unprofiled'] if derived_ok else None
            ok = (rec.get('status') == 'ok' and derived_ok and fin(rec.get('graph_over_unprofiled')) and fin(rec.get('node_over_unprofiled'))
                  and abs(rec['graph_over_unprofiled'] - g) <= 1e-9 and abs(rec['node_over_unprofiled'] - n) <= 1e-9
                  and rec.get('caveat_required') == (n > 1.05))
            item(items, f'perturbation ratio {a}@{c}', ok, [f"status {rec.get('status', 'MISSING')}: {rec.get('problems')}; re-derived graph {g} node {n}"])
        for c in (352800, 4410):
            sm = (pm.get('source_map') or {}).get(f'C3_{c}') or {}
            cov = (sm.get('coverage') or {})
            ic = load(os.path.join(ev, 'traces', f'ideal_costs_C3_{c}.json')) or {}
            fused = [r for r in (ic.get('rows') or []) if r.get('class') == 'INDUCTOR_FUSED']
            kern = sm.get('kernels') or {}
            tot = sum(r.get('time_us_per_pass') or 0.0 for r in fused)
            mapped = sum(r.get('time_us_per_pass') or 0.0 for r in fused
                         if (lambda k: bool(k) and k.get('ops_source') != 'AMBIGUOUS' and bool(k.get('fused_ops')))(kern.get(r.get('kernel_full') or r.get('kernel'))))
            share = (mapped / tot) if tot else None
            ok = (fin(cov.get('share')) and share is not None and abs(cov['share'] - share) <= 1e-9
                  and abs((cov.get('fused_time_us_per_pass') or 0) - tot) <= 1e-6 * max(1.0, tot))
            item(items, f'kernel source map coverage C3@{c} (re-derived)', ok, [f"stored {cov.get('share')} re-derived {share}"])
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
    audit_clock(ev, items, args.session, run_map, matched, (manifest.get('env') or {}).get('gpu_uuid'))
    audit_aux(ev, items, args.session, dynamic_aux)
    if args.session in ('session3', 'session4'):        # registered optional experiments never block the verdict
        opt_aux = set(OPTIONAL_AUX_S3) | set(getattr(audit_session3, 'optional_aux', set())) | set(getattr(audit_session4, 'optional_aux', set()))
        for i in items:
            if i['item'].startswith('auxiliary step '):
                nm = i['item'].split(' ', 2)[2]
                if nm in opt_aux or '_v_' in nm:
                    i['optional'] = True
    mandatory = [i for i in items if not i.get('optional')]
    n_ok = sum(1 for i in mandatory if i['complete'])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        analysis_commit = provenance._git(root, 'rev-parse', 'HEAD')
    except Exception:
        analysis_commit = None
    cfg3 = load(os.path.join(ev, 'session3_config.json')) or load(os.path.join(ev, 'session4_config.json')) or {}
    status = {'session': args.session, 'manifest_id': manifest_id, 'items': items,
              'complete_items': n_ok, 'total_items': len(mandatory),
              'optional_items': [{'item': i['item'], 'complete': i['complete'], 'reason': i['reason']} for i in items if i.get('optional')],
              'verdict': 'COMPLETE' if n_ok == len(mandatory) else 'INCOMPLETE',
              'shakedown': (bool(cfg3.get('shakedown')) or not bool(cfg3.get('session_pod'))) if args.session in ('session3', 'session4') else None,
              'evidence_grade': (('pod' if (cfg3.get('session_pod') and not cfg3.get('shakedown')) else 'engineering (off-pod or shakedown: never session evidence)')
                                 if args.session in ('session3', 'session4') else 'pod'),
              'qualify_version': qualify.QUALIFY_VERSION,
              'analysis_tools_commit': analysis_commit,
              'analysis_tools_sha256': {k: v for k, v in provenance.code_hashes().items() if os.path.basename(k) in ANALYSIS_TOOLS}}
    with open(os.path.join(ev, 'session_status.json'), 'w') as f:
        json.dump(status, f, indent=1)
    print(f"[completeness] {args.session}: {status['verdict']} ({n_ok}/{len(mandatory)} mandatory items"
          + (f"; shakedown={status['shakedown']}; evidence grade: {status['evidence_grade']}" if args.session in ('session3', 'session4') else '') + ')')
    for i in items:
        if not i['complete']:
            print(f"   {'optional-incomplete' if i.get('optional') else 'INCOMPLETE'} {i['item']}: {i['reason']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
