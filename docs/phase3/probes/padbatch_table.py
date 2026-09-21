"""2.4 — the decomposition table: batching benefit (gaps/launches), padding cost (extra FLOPs at the measured rate of
the padded GEMM kernels), kernel-efficiency change (rate of batched vs per-band GEMMs on the TRUE FLOPs), residual.
Inputs: docs/phase3/padbatch/regions_and_arithmetic.json, timing_views (span/union), matched pairs & headlines."""
import json, csv
EV = '/home/muhit/ms/evidence_pod/9tblar2zwo97ln_20260920_1855/evidence'
R = json.load(open('docs/phase3/padbatch/regions_and_arithmetic.json'))
S = json.load(open(EV + '/summary_full.json'))
def views(arm, O):
    v = json.load(open(f'{EV}/traces/timing_views_{arm}_{O}.json'))['median_us']; return v
rows = []
for O in (352800, 4410):
    a = R['arithmetic'][str(O)]
    c3, h = R['regions'][f'C3_{O}'], R['regions'][f'H_{O}']
    vc, vh = views('C3', O), views('H', O)
    gaps_c3 = (vc['span_ns'] - vc['union_ns']) / 1e3; gaps_h = (vh['span_ns'] - vh['union_ns']) / 1e3   # ms
    ksum_c3 = vc['sum_ns'] / 1e3; ksum_h = vh['sum_ns'] / 1e3
    # measured wall difference (unprofiled): matched pair at 8 s, headline difference at 4410 (C3 - H)
    if O == 352800:
        wall_delta = S['matched_pairs']['C3_vs_H_c352800']['median_delta_ms']
    else:
        wall_delta = S['cells']['C3_4410']['headline_ms'] - S['cells']['H_4410']['headline_ms']
    # padded GEMM kernels live in H's front (band split) and tail (mask estimator); achieved rate of those kernels:
    # H tail+front GEMM time vs H padded FLOPs (band split + mask last layer are the padded modules; the other tail GEMMs
    # (mask layers 1-2, 384->1536->1536) are unpadded but batched: included in the same kernels' time -> rate is region-level)
    true_flops = a['band_split']['true_flops'] + a['mask_last_layer']['true_flops']
    pad_flops = a['band_split']['padded_flops'] + a['mask_last_layer']['padded_flops']
    extra_flops = pad_flops - true_flops
    # region GEMM times (ms)
    g_c3 = c3['front']['GEMM'] + c3['tail']['GEMM']; g_h = h['front']['GEMM'] + h['tail']['GEMM']
    # the unpadded tail GEMMs (mask layers 1,2: 60 bands x 384->1536 and 1536->1536) FLOPs, same in both arms
    T = a['frames']
    unpadded_tail = 60 * 2 * T * (384 * 1536 + 1536 * 1536)
    rate_h = (pad_flops + unpadded_tail) / (g_h / 1e3)          # FLOP/s achieved by H's front+tail GEMMs (padded work counted)
    rate_c3 = (true_flops + unpadded_tail) / (g_c3 / 1e3)       # FLOP/s achieved by C3's per-band GEMMs (true work)
    padding_cost = extra_flops / rate_h * 1e3                    # ms: the padded FLOPs at the rate the padded kernels actually run
    efficiency_change = (true_flops + unpadded_tail) * (1 / rate_h - 1 / rate_c3) * 1e3   # ms: same true work at H's rate vs C3's rate (+ = H slower)
    batching_benefit = (gaps_c3 - gaps_h)                        # ms of GPU idle removed (launch/boundary floor) — positive = H benefits
    nongemm_delta = ((h['front']['total_ms'] - h['front']['GEMM']) + (h['tail']['total_ms'] - h['tail']['GEMM'])) - \
                    ((c3['front']['total_ms'] - c3['front']['GEMM']) + (c3['tail']['total_ms'] - c3['tail']['GEMM']))
    middle_delta = h['middle']['total_ms'] - c3['middle']['total_ms']
    # accounted wall delta C3 - H = batching_benefit - padding_cost - efficiency_change - nongemm_delta - middle_delta
    accounted = batching_benefit - padding_cost - efficiency_change - nongemm_delta - middle_delta
    span_delta = (vc['span_ns'] - vh['span_ns']) / 1e3
    rows.append({'O': O, 'wall_delta_C3_minus_H_ms (unprofiled)': wall_delta, 'trace_span_delta_ms': span_delta,
                 'batching_benefit_ms (gap/launch floor removed)': batching_benefit,
                 'padding_cost_ms (extra FLOPs at H rate)': padding_cost,
                 'kernel_efficiency_change_ms (true work at H rate vs C3 rate; + = H slower)': efficiency_change,
                 'non_GEMM_front_tail_delta_ms (H - C3)': nongemm_delta, 'transformer_middle_delta_ms (H - C3)': middle_delta,
                 'accounted_ms': accounted, 'residual_vs_trace_span_ms': span_delta - accounted, 'residual_vs_wall_ms': wall_delta - accounted,
                 'extra_padded_GFLOP': extra_flops / 1e9, 'H_fronttail_GEMM_rate_TFLOPS': rate_h / 1e12, 'C3_fronttail_GEMM_rate_TFLOPS': rate_c3 / 1e12,
                 'C3_kernels': int(c3['front']['n'] + c3['middle']['n'] + c3['tail']['n']), 'H_kernels': int(h['front']['n'] + h['middle']['n'] + h['tail']['n']),
                 'gaps_C3_ms': gaps_c3, 'gaps_H_ms': gaps_h})
with open('docs/phase3/padbatch/decomposition_table.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
json.dump(rows, open('docs/phase3/padbatch/decomposition_table.json', 'w'), indent=1)
for r in rows:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
