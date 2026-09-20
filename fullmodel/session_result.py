"""SESSION_RESULT.md — filled mechanically from evidence/ (docs/SESSION_RESULT_template.md).

Usage: python fullmodel/session_result.py --evidence-dir <dir> --out SESSION_RESULT.md
Every cell cites its file; missing inputs print UNRESOLVED.
"""

import argparse
import csv
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qualify  # noqa: E402

ARMS = ('A', 'C3', 'C1', 'H', 'G', 'B', 'E', 'D', 'F', 'C1D', 'C1A', 'C2', 'ANVTX')
OPS = (352800, 4410, 22050, 88200)


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def fmt(v, spec='.4f'):
    return format(v, spec) if isinstance(v, (int, float)) and not isinstance(v, bool) else 'UNRESOLVED'


def clocks(ev):
    out = {}
    for s in ('session1', 'session2'):
        p = os.path.join(ev, f'clocks_{s}.csv')
        sm, mem = [], []
        if os.path.exists(p):
            gpu = ((load(os.path.join(ev, 'session_manifest.json')) or {}).get('env') or {}).get('gpu_uuid')
            for line in open(p):
                parts = [x.strip() for x in line.split(',')]
                if len(parts) != 7 or (gpu and parts[1] != gpu):
                    continue
                try:
                    sm.append(float(parts[2].replace('MHz', '')))
                    mem.append(float(parts[3].replace('MHz', '')))
                except ValueError:
                    continue
        out[s] = {'sm_median': statistics.median(sm) if sm else None, 'sm_min': min(sm) if sm else None,
                  'sm_max': max(sm) if sm else None, 'mem_median': statistics.median(mem) if mem else None, 'n': len(sm)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence-dir', required=True)
    ap.add_argument('--out', default='SESSION_RESULT.md')
    ap.add_argument('--registration-commit', default='')
    ap.add_argument('--pushed-utc', default='')
    args = ap.parse_args()
    ev = args.evidence_dir
    man = load(os.path.join(ev, 'session_manifest.json')) or {}
    env, prov = man.get('env') or {}, man.get('provenance') or {}
    summary = load(os.path.join(ev, 'summary_full.json')) or {}
    cells, pairs = summary.get('cells') or {}, summary.get('matched_pairs') or {}
    status = {s: load(os.path.join(ev, f'session_status_{s}.json')) or {} for s in ('session1', 'session2')}
    if not status['session1'] and not status['session2']:
        status['session1'] = load(os.path.join(ev, 'session_status.json')) or {}
    ck = clocks(ev)
    L = []
    L.append('# Ceiling-ledger Phase 1 — SESSION_RESULT (filled mechanically from evidence/)')
    L.append('')
    L.append(f"repo: {man.get('repo_commit')}   registration: prereg_amendments.md @ {args.registration_commit or man.get('repo_commit')} "
             f"(sha256 {str(man.get('registration_sha256'))[:12]}; pushed {args.pushed_utc or 'see git log'})   msst @ {str(prov.get('msst_commit'))[:12]}   ckpt sha256 {str(prov.get('ckpt_sha256'))[:12]}")
    L.append(f"torch {env.get('torch')} · triton {env.get('triton')} · driver {env.get('driver')} · GPU {env.get('gpu')} {env.get('gpu_uuid')} · pod melband_profiling · manifest {man.get('manifest_id')} · created {man.get('created_utc')}")
    for s in ('session1', 'session2'):
        c = ck[s]
        L.append(f"clocks {s}: SM median {fmt(c['sm_median'], '.0f')} MHz (range {fmt(c['sm_min'], '.0f')}–{fmt(c['sm_max'], '.0f')}) · memory {fmt(c['mem_median'], '.0f')} MHz · {c['n']} samples — evidence/clocks_{s}.csv")
    L.append('')
    L.append('## Decision inputs (mechanical)')
    L.append('| input | value | evidence |')
    L.append('|---|---|---|')
    for tier, const in (('tight', 0.001), ('loose', 0.05)):
        r = load(os.path.join(ev, f'full_A_c4410_planted_{tier}.json')) or {}
        v = r.get('validation') or {}
        caught = v.get('status') == 'FAIL' and all(p.get('ok') is False for p in v.get('passes') or [])
        L.append(f"| planted fault {tier} ({const}) | {'CAUGHT' if caught else 'NOT CAUGHT'} max_abs_diff {[p.get('max_abs_diff') for p in v.get('passes') or []]} | evidence/full_A_c4410_planted_{tier}.json |")
    q = load(os.path.join(ev, 'c3_operator_qualification.json')) or {}
    L.append(f"| C3 operator qualification | {', '.join(f'T={k}: {v.get('status')}' for k, v in sorted(q.items(), key=lambda kv: int(kv[0])))} | evidence/c3_operator_qualification.json |")
    L.append('')
    L.append('### Arms × operating points (headline = median over r1–r3 of per-process min-of-medians; unprofiled)')
    L.append('| arm | O | headline ms | restart spread ms | best throughput ms/pass | tier | SDR (mean ΔSDR_ref dB) | variant | breaks | status / reasons |')
    L.append('|---|---|---|---|---|---|---|---|---|---|')
    for O in OPS:
        for arm in ARMS:
            c = cells.get(f'{arm}_{O}')
            if not c:
                continue
            L.append(f"| {arm} | {O} | {fmt(c.get('headline_ms'))} | {fmt(c.get('restart_spread_ms'))} | {fmt(c.get('throughput_best_ms'))} | {'/'.join(c.get('tier') or [])} | "
                     f"{c.get('sdr_gate')} ({fmt(c.get('mean_delta_sdr_ref_db'), '+.3f')}) | {' '.join(c.get('variants') or [])} | {'/'.join(c.get('graph_breaks') or [])} | "
                     f"{c.get('headline_status')} {('; '.join(c.get('reasons') or []))[:160]} |")
    L.append('')
    L.append('### Matched pairs (plan 1.2; median over alternating blocks, a − b)')
    L.append('| pair | m1 | m2 | m3 | median ms | % of a | status / reasons |')
    L.append('|---|---|---|---|---|---|---|')
    for key, p in sorted(pairs.items()):
        d = p.get('deltas_ms') or {}
        L.append(f"| {key} | {fmt(d.get('m1'))} | {fmt(d.get('m2'))} | {fmt(d.get('m3'))} | {fmt(p.get('median_delta_ms'))} | {fmt(p.get('median_delta_pct_of_a'), '+.2f')} | {p.get('pair_headline')} {('; '.join(p.get('reasons') or []))[:160]} |")
    L.append('')
    L.append('### Graph deployment (node-level traces)')
    L.append('| arm | O | verdict | graph launches/pass | captured | staging | uncaptured | unresolved | qualified |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for path in sorted(glob.glob(os.path.join(ev, 'traces', 'graph_verify_*.json'))):
        g = load(path) or {}
        m = g.get('medians') or g
        name = os.path.basename(path)[len('graph_verify_'):-5]
        arm, O = name.rsplit('_', 1)
        L.append(f"| {arm} | {O} | {g.get('verdict')} | {fmt(m.get('graph_launch_calls'), '.1f')} | {fmt(m.get('captured_model_kernels'), '.0f')} | {fmt(m.get('input_staging_kernels'), '.0f')} | "
                 f"{fmt(m.get('uncaptured_model_kernels'), '.0f')} | {fmt(m.get('unresolved_kernels'), '.0f')} | {g.get('qualified')} |")
    L.append('')
    pm = load(os.path.join(ev, 'perturbation_and_source_map.json')) or {}
    L.append('### Profiling perturbation (one block each; the only profiled/unprofiled ratios)')
    L.append('| arm@O | unprofiled ms | graph/unprofiled | node/unprofiled | caveat | status |')
    L.append('|---|---|---|---|---|---|')
    for k, r in sorted((pm.get('perturbation') or {}).items()):
        L.append(f"| {k} | {fmt(r.get('unprofiled_block_ms'))} | {fmt(r.get('graph_over_unprofiled'), '.3f')} | {fmt(r.get('node_over_unprofiled'), '.3f')} | {r.get('caveat_required')} | {r.get('status')} |")
    L.append('')
    ab = load(os.path.join(ev, 'attn_backend_bench.json')) or {}
    L.append('### Attention backend (per axis, per O; model layout = deployed layout)')
    L.append('| case | best backend | best µs | deployed backend | deployed µs | ratio best/deployed | status |')
    L.append('|---|---|---|---|---|---|---|')
    for k, c in sorted((ab.get('cases') or {}).items()):
        L.append(f"| {k} | {c.get('best_backend')} | {fmt(c.get('best_us'), '.2f')} | {c.get('model_backend')} | {fmt(c.get('model_backend_us'), '.2f')} | {fmt(c.get('ratio_best_over_model'), '.3f')} | {c.get('status', 'ok')} |")
    L.append('')
    bb = load(os.path.join(ev, 'boundary_bench.json')) or {}
    ta = load(os.path.join(ev, 'boundary_bench_trace_analysis.json')) or {}
    L.append('### Boundary cost (T(N) fits; serial gap from the node trace)')
    L.append('| mode | slope µs/node | intercept µs | trace gap µs (N1000) | PDL PTX |')
    L.append('|---|---|---|---|---|')
    for mode, f in sorted((bb.get('fits') or {}).items()):
        seg = ta.get(f'{mode}/N1000') or {}
        L.append(f"| {mode} | {fmt(f.get('slope_us_per_node'), '.3f')} | {fmt(f.get('intercept_us'), '.3f')} | {fmt(seg.get('gap_us_median'), '.3f')} ({seg.get('status', 'n/a')}) | {bb.get('pdl_ptx')} |")
    L.append(f"L2-resident bandwidth: {fmt((bb.get('l2_resident_bandwidth') or {}).get('tbps'), '.3f')} TB/s — evidence/boundary_bench.json")
    L.append('')
    L.append('### Three timing views (kernels only; memops separate)')
    L.append('| arm | O | sum µs | union µs | observed-order critical path µs | span µs | streams |')
    L.append('|---|---|---|---|---|---|---|')
    for path in sorted(glob.glob(os.path.join(ev, 'traces', 'timing_views_*.json'))):
        v = load(path) or {}
        m = v.get('median_us') or {}
        name = os.path.basename(path)[len('timing_views_'):-5]
        arm, O = name.rsplit('_', 1)
        L.append(f"| {arm} | {O} | {fmt(m.get('sum_ns'), '.1f')} | {fmt(m.get('union_ns'), '.1f')} | {fmt(m.get('critical_path_observed_order_ns'), '.1f')} | {fmt(m.get('span_ns'), '.1f')} | {v.get('streams_union') or ''} |")
    L.append('')
    L.append('### Per-kernel ideal costs (class totals µs/pass; GEMM/SDPA ideal vs measured)')
    L.append('| arm | O | measurement | variant | GEMM | SDPA | INDUCTOR_FUSED | COPY | FFT | OTHER | SDPA ideal µs | GEMM ideal µs (sum of sites) |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for path in sorted(glob.glob(os.path.join(ev, 'traces', 'ideal_costs_*.json'))):
        ic = load(path) or {}
        t = ic.get('class_totals_us_per_pass') or {}
        name = os.path.basename(path)[len('ideal_costs_'):-5]
        arm, O = name.rsplit('_', 1)
        L.append(f"| {arm} | {O} | {ic.get('measurement_status')} | {(ic.get('variant_selection') or {}).get('status')} | "
                 + ' | '.join(fmt(t.get(k), '.1f') for k in ('GEMM', 'SDPA', 'INDUCTOR_FUSED', 'COPY', 'FFT', 'OTHER'))
                 + f" | {fmt((ic.get('sdpa_class') or {}).get('ideal_us_hbm'), '.1f')} | {fmt((ic.get('gemm_class') or {}).get('ideal_us_hbm_sum'), '.1f')} |")
    L.append('')
    L.append('## Predictions vs measurements')
    L.append('| registered claim (prereg_amendments.md) | outcome | evidence |')
    L.append('|---|---|---|')
    pt = load(os.path.join(ev, 'full_A_c4410_planted_tight.json')) or {}
    pl = load(os.path.join(ev, 'full_A_c4410_planted_loose.json')) or {}
    def diffs(r):
        return [p.get('max_abs_diff') for p in ((r.get('validation') or {}).get('passes') or [])]
    L.append(f"| §1.2 planted faults FAIL both seeds, seed-invariant diff | tight {diffs(pt)} loose {diffs(pl)} | evidence/full_A_c4410_planted_*.json |")
    m = load(os.path.join(ev, 'matched_C1_C1A_c352800_m1.json')) or {}
    L.append(f"| §1 C1A compile failure expected | C1A build {((m.get('arms') or {}).get('C1A') or {}).get('build_status', 'no record')}, record status {m.get('status')}, pair_eligible {m.get('pair_eligible')} | evidence/matched_C1_C1A_c352800_m1.json |")
    b3 = {O: (cells.get(f'C3_{O}') or {}).get('graph_breaks') for O in OPS}
    L.append(f"| §1 C3 zero graph breaks | {b3} | evidence/full_C3_*_r*.json graph_gate |")
    best4410 = sorted(((c['headline_ms'], k) for k, c in cells.items() if k.endswith('_4410') and c.get('headline_status') == 'HEADLINE' and not k.startswith('ANVTX')))
    comp = [(v, k) for v, k in best4410 if k.split('_')[0] in ('C1', 'C2', 'C3', 'H', 'C1D')]
    gh = [(v, k) for v, k in best4410 if k.split('_')[0] in ('G', 'H')]
    L.append(f"| §6 exploratory: G/H faster than best verified-replay compiled arm at 4410 | best compiled {comp[0] if comp else 'UNRESOLVED'}; G/H {gh if gh else 'UNRESOLVED'} | evidence/summary_full.json |")
    ser = []
    for k, r in (summary.get('runs') or {}).items():
        if k.endswith('_4410_r1') and (r.get('timing') or {}).get('serialization_delta_pct'):
            ser.append((k.split('_')[0], max(r['timing']['serialization_delta_pct'])))
    L.append(f"| §5 5a serialization delta > 5 % at 4410 (escalation) | max per arm: {ser} | evidence/full_*_c4410_r1.json timing.serialization_delta_pct |")
    L.append('')
    L.append('## What failed / UNTESTED / INCOMPLETE')
    for s, st in status.items():
        if not st:
            continue
        L.append(f"- {s}: **{st.get('verdict')}** ({st.get('complete_items')}/{st.get('total_items')} mandatory items) — evidence/session_status.json")
        for it in st.get('items') or []:
            if not it.get('complete'):
                L.append(f"  - {it['item']}: {it.get('reason')}")
    for k, c in sorted(cells.items()):
        if c.get('headline_status') != 'HEADLINE':
            L.append(f"- cell {k}: {'; '.join(c.get('reasons') or [])}")
    L.append('')
    L.append('## Conventions')
    L.append('Headline walls unprofiled; PERTURBATION ratios are the only profiled/unprofiled ratios; anything trace-derived divided by an unprofiled wall is labelled MODELLED (feeds LEDGER.md, never a measured speedup). '
             'This file is generated by fullmodel/session_result.py; every number cites its evidence file.')
    with open(args.out, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
