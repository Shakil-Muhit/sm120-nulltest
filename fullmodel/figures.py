"""Figure CSVs (Phase 3 §6). Every CSV is computed from the cited evidence; nothing
is transcribed by hand. Outputs go to docs/phase3/figures/.

  fig_headlines.csv      arm × O headline (min-of-medians median over restarts) with the
                         block-level bootstrap interval (docs/phase3/stats/bootstrap_headlines.csv)
  fig_x1_decomposition.csv, fig_padbatch.csv   copies of the §2.3/§2.4 tables (single source)
  fig_e9_crossover.csv   registered model curve D(L) = wall(C3) − wall(H) over L, the
                         four sweep observations and (when present) the held-out observations
  fig_ledger.csv         one row per ledger row: ceiling Y, Z, conf, realized, realized/ceiling
  fig_boundary.csv       chain slopes/intercepts from the session-2 and (when present) session-3 benches
  fig_e5_variants.csv, fig_aa.csv   from e5_summary.json / aa_stats.json when present
"""
import argparse
import csv
import json
import math
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOP = 441


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def write(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f'{os.path.basename(path)}: {len(rows)} rows')


def e9_curve(reg, p3):
    v = reg['variants'][reg['registered_variant']]
    p = v['params']
    inp = reg['inputs']
    def q(T):
        return (lambda f: f / (T * math.ceil(f / T))) if T else (lambda f: 1.0)
    qH, qC = q(p['TH']), q(p['TC'])
    B, NF, MID = inp['B'], inp['NF'], inp['MID']
    rows = []
    for L in range(4410, 352801, 4410):
        f = L // HOP + 1
        rH = p['RH'] * f / (f + p['hH']) * qH(f)
        rC = p['RC'] * f / (f + p['hC']) * qC(f)
        pad = inp['pad_gflop_per_frame'] * f / rH
        eff = inp['true_gflop_per_frame'] * f * (1 / rH - 1 / rC)
        D = B - (NF[0] + NF[1] * f) - (MID[0] + MID[1] * f) - pad - eff
        rows.append([L, f, round(D, 4), 'model', round(B, 4), round(-(NF[0] + NF[1] * f), 4), round(-(MID[0] + MID[1] * f), 4), round(-pad, 4), round(-eff, 4)])
    for L, D in inp['sweep_obs_C3_minus_H'].items():
        rows.append([int(L), int(L) // HOP + 1, round(D, 4), 'observed_sweep (headline C3 − H)', '', '', '', '', ''])
    e9 = load(os.path.join(p3, 'e9_check.json')) if p3 else None
    for L, r in ((e9 or {}).get('held_out') or {}).items():
        obs = r.get('observed_C3_minus_H_ms (median over restarts of block medians)')
        if obs is not None:
            ci = r.get('observed_ci95_block_level_ms')
            rows.append([int(L), int(L) // HOP + 1, round(obs, 4), f'observed_held_out (matched C3 − H; CI95 {ci})', '', '', '', '', ''])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase3-dir', default=None)
    ap.add_argument('--ledger-json', default=os.path.join(ROOT, 'LEDGER.json'))
    ap.add_argument('--evidence-dir', default=None, help='session-2 evidence (boundary_bench.json)')
    ap.add_argument('--out-dir', default=os.path.join(ROOT, 'docs', 'phase3', 'figures'))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    # headlines
    src = os.path.join(ROOT, 'docs', 'phase3', 'stats', 'bootstrap_headlines.csv')
    shutil.copy(src, os.path.join(a.out_dir, 'fig_headlines.csv')); print('fig_headlines.csv: copied from docs/phase3/stats/bootstrap_headlines.csv')
    shutil.copy(os.path.join(ROOT, 'docs', 'phase3', 'x1', 'x1_decomposition_352800.csv'), os.path.join(a.out_dir, 'fig_x1_decomposition.csv'))
    shutil.copy(os.path.join(ROOT, 'docs', 'phase3', 'padbatch', 'decomposition_table.csv'), os.path.join(a.out_dir, 'fig_padbatch.csv'))
    print('fig_x1_decomposition.csv, fig_padbatch.csv: copied (single source in docs/phase3)')
    reg = load(os.path.join(ROOT, 'docs', 'phase3', 'e9', 'e9_cost_model.json'))
    write(os.path.join(a.out_dir, 'fig_e9_crossover.csv'),
          ['L_samples', 'frames', 'D_C3_minus_H_ms', 'kind', 'term_B', 'term_nonGEMM_fronttail', 'term_middle', 'term_padding', 'term_efficiency'],
          e9_curve(reg, a.phase3_dir))
    rows = load(a.ledger_json) or []
    write(os.path.join(a.out_dir, 'fig_ledger.csv'),
          ['X', 'O', 'status', 'T_attr_ms', 'f', 'Y_ms', 'Z_pct', 'conf', 'group', 'label', 'realized_ms', 'realized_ci95_ms', 'realized_over_ceiling', 'realized_source'],
          [[r.get('X'), r.get('O'), r.get('status'), r.get('T_attributable_ms'), r.get('f_recoverable'), r.get('Y_ms'), r.get('Z_pct'), r.get('conf'), r.get('group'),
            r.get('label'), r.get('realized_ms'), r.get('realized_ci95_ms'),
            (r['realized_ms'] / r['Y_ms']) if isinstance(r.get('realized_ms'), (int, float)) and r.get('Y_ms') else '', r.get('realized_source')] for r in rows])
    brows = []
    for name, d in (('session2', a.evidence_dir), ('session3', a.phase3_dir)):
        bb = load(os.path.join(d, 'boundary_bench.json')) if d else None
        for k, v in ((bb or {}).get('fits') or {}).items():
            brows.append([name, k, v.get('slope_us_per_node'), v.get('intercept_us'), json.dumps(v.get('points'))])
        for k, v in ((bb or {}).get('e7_pdl_saving') or {}).items():
            brows.append([name, 'E7 ' + k, v.get('pdl_saving_us_per_node'), '', ''])
    write(os.path.join(a.out_dir, 'fig_boundary.csv'), ['session', 'chain/mode', 'slope_us_per_node', 'intercept_us', 'points'], brows)
    if a.phase3_dir:
        e5 = load(os.path.join(a.phase3_dir, 'e5_summary.json')) or {}
        write(os.path.join(a.out_dir, 'fig_e5_variants.csv'),
              ['file', 'arm', 'chunk', 'variant', 'depth', 'mlp_expansion_factor', 'param_bytes_fp16', 'mask_estimator_bytes_fp16', 'min_of_medians_ms', 'kernel_sum_ms', 'span_ms', 'cast_time_ms_per_pass', 'boundary_count'],
              [[k] + [v.get(c) for c in ('arm', 'chunk', 'variant', 'depth', 'mlp_expansion_factor', 'param_bytes_fp16', 'mask_estimator_bytes_fp16', 'min_of_medians_ms', 'kernel_sum_ms', 'span_ms', 'cast_time_ms_per_pass', 'boundary_count')]
               for k, v in (e5.get('cells') or {}).items()])
        aa = load(os.path.join(a.phase3_dir, 'aa_stats.json')) or {}
        write(os.path.join(a.out_dir, 'fig_aa.csv'), ['file', 'chunk', 'median_ms', 'ci95_median_ms', 'abs_max_ms', 'per_block_ms'],
              [[k, v.get('chunk'), v.get('median_ms'), v.get('ci95_median_ms'), v.get('abs_max_ms'), v.get('a_minus_b_per_block_ms')] for k, v in (aa.get('pairs') or {}).items()])


if __name__ == '__main__':
    main()
