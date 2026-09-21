"""2.3 — X1 decomposition from the C1 node-level trace at 8 s (session 2 evidence).
Parts: input-staging copies, eager kernels between graph segments, launch gaps, lost fusion (C1 vs C3).
Every number cites the trace CSVs; the residual against the matched 6.741 ms is stated."""
import csv, json, os, statistics, sys
sys.path.insert(0, 'fullmodel'); sys.path.insert(0, 'fullmodel/attrib')
from classify import classify
from kernel_map import read_gpu_trace, nvtx_pass_windows, split_passes
EV = '/home/muhit/ms/evidence_pod/9tblar2zwo97ln_20260920_1855/evidence'
O = 352800
def med(xs): s = sorted(xs); n = len(s); return s[n//2] if n % 2 else 0.5*(s[n//2-1]+s[n//2])
def api_rows(base):
    with open(base + '_cuda_api_trace.csv', newline='') as f:
        rd = csv.DictReader(f)
        out = []
        for r in rd:
            try:
                out.append({'start': int(float(r['Start (ns)'].replace(',', ''))), 'dur': int(float(r['Duration (ns)'].replace(',', ''))),
                            'name': r['Name'], 'corr': r.get('CorrID') or r.get('CorrId')})
            except (ValueError, KeyError):
                continue
        return out
def gpu_rows_full(base):
    """GPU rows with CorrId and bytes (memcpy) from the raw CSV."""
    with open(base + '_cuda_gpu_trace.csv', newline='') as f:
        rd = csv.DictReader(f)
        out = []
        for r in rd:
            try:
                out.append({'start': int(float(r['Start (ns)'].replace(',', ''))), 'dur': int(float(r['Duration (ns)'].replace(',', ''))),
                            'name': r['Name'], 'corr': r['CorrId'], 'bytes_mb': float(r['Bytes (MB)'].replace(',', '')) if r.get('Bytes (MB)') else None})
            except (ValueError, KeyError):
                continue
        return out
res = {}
per = {}
for arm in ('C1', 'C3'):
    base = f'{EV}/traces/trace_{arm}_{O}_trace'
    windows, err = nvtx_pass_windows(base + '_nvtx_gpu_proj_trace.csv'); assert not err and len(windows) == 50, err
    gpu = gpu_rows_full(base); api = api_rows(base)
    launches = [a for a in api if 'cudaGraphLaunch' in a['name']]
    launch_corr = {a['corr'] for a in launches}
    passes = []
    for (s, e) in windows:
        prow = [r for r in gpu if s <= r['start'] < e]
        # graph launches whose CPU call falls in the pass window's CPU span are not exported per pass; use GPU-side CorrId membership
        graphed = [r for r in prow if r['corr'] in launch_corr]
        eager = [r for r in prow if r['corr'] not in launch_corr]
        staging = [r for r in eager if ('multi_tensor_apply' in r['name'] and 'Copy<' in r['name']) or 'direct_copy_kernel' in r['name'] or r['name'].startswith('[CUDA memcpy')]
        eager_k = [r for r in eager if r not in staging and not r['name'].startswith('[CUDA mem')]
        memops = [r for r in prow if r['name'].startswith('[CUDA mem')]
        # launch instances in this pass = distinct graph CorrIds seen
        n_launch = len({r['corr'] for r in graphed})
        # idle gaps on the GPU timeline inside the pass (union of busy intervals vs span)
        iv = sorted((r['start'], r['start'] + r['dur']) for r in prow)
        busy, cur_s, cur_e = 0, None, None
        for a, b in iv:
            if cur_e is None or a > cur_e:
                if cur_e is not None: busy += cur_e - cur_s
                cur_s, cur_e = a, b
            else:
                cur_e = max(cur_e, b)
        if cur_e is not None: busy += cur_e - cur_s
        span = max(b for a, b in iv) - min(a for a, b in iv)
        fused_attn = [r for r in prow if classify(r['name']) == 'INDUCTOR_FUSED' and 'scaled_dot_product' in r['name']]
        passes.append({'n_graph_launches': n_launch, 'staging_n': len(staging), 'staging_ms': sum(r['dur'] for r in staging)/1e6,
                       'staging_bytes_mb': sum(r['bytes_mb'] or 0 for r in staging), 'eager_n': len(eager_k), 'eager_ms': sum(r['dur'] for r in eager_k)/1e6,
                       'graphed_n': len(graphed), 'graphed_ms': sum(r['dur'] for r in graphed)/1e6, 'gap_ms': (span - busy)/1e6, 'span_ms': span/1e6,
                       'fused_n': sum(1 for r in prow if classify(r['name']) == 'INDUCTOR_FUSED'), 'fused_ms': sum(r['dur'] for r in prow if classify(r['name']) == 'INDUCTOR_FUSED')/1e6,
                       'gemm_ms': sum(r['dur'] for r in prow if classify(r['name']) == 'GEMM')/1e6, 'sdpa_ms': sum(r['dur'] for r in prow if classify(r['name']) == 'SDPA')/1e6,
                       'fused_attn_n': len(fused_attn), 'fused_attn_ms': sum(r['dur'] for r in fused_attn)/1e6,
                       'copy_class_ms': sum(r['dur'] for r in prow if classify(r['name']) == 'COPY' and r not in staging)/1e6,
                       'other_ms': sum(r['dur'] for r in prow if classify(r['name']) == 'OTHER')/1e6, 'kernel_sum_ms': sum(r['dur'] for r in prow if not r['name'].startswith('[CUDA mem'))/1e6})
    per[arm] = {k: med(p[k] for p in passes) for k in passes[0]}
    # host-side cudaGraphLaunch durations
    per[arm]['host_launch_us_median'] = med([a['dur'] for a in launches]) / 1e3 if launches else None
    per[arm]['host_launch_total_ms_per_pass'] = (sum(a['dur'] for a in launches) / 1e6 / 50) if launches else 0.0
c1, c3 = per['C1'], per['C3']
# Partition of the per-pass SPAN difference (trace-derived): kernel-sum difference by mechanism + GPU idle gaps.
# Eager kernels between segments are NOT a separate part: their compute exists in C3 inside the graph; their cost
# shows up as lost fusion (unfused elementwise = OTHER/COPY classes) and is reported as context.
parts = {
  'input_staging_copies_ms': c1['staging_ms'] - c3['staging_ms'],
  'lost_fusion_ms': (c1['fused_ms'] + c1['other_ms'] + c1['copy_class_ms']) - (c3['fused_ms'] + c3['other_ms'] + c3['copy_class_ms']),
  'gpu_idle_gaps_ms': c1['gap_ms'] - c3['gap_ms'],
  'gemm_sdpa_selection_delta_ms': (c1['gemm_ms'] + c1['sdpa_ms']) - (c3['gemm_ms'] + c3['sdpa_ms']),
}
matched = 6.7412
trace_span_delta = c1['span_ms'] - c3['span_ms']
out = {'operating_point': O, 'matched_delta_ms': matched, 'trace_span_delta_ms': trace_span_delta, 'per_pass_medians': per, 'parts_ms': parts,
       'parts_sum_ms': sum(parts.values()), 'residual_vs_span_delta_ms': trace_span_delta - sum(parts.values()),
       'residual_vs_matched_ms': matched - sum(parts.values()),
       'note': 'trace-derived (node-level, C1 perturbation not separately measured; C3 x1.013 at 8 s); the matched delta is unprofiled'}
os.makedirs('docs/phase3/x1', exist_ok=True)
json.dump(out, open('docs/phase3/x1/x1_decomposition_352800.json', 'w'), indent=1)
with open('docs/phase3/x1/x1_decomposition_352800.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['part', 'ms_per_pass', 'C1_value', 'C3_value'])
    w.writerow(['input staging copies', parts['input_staging_copies_ms'], f"{c1['staging_n']:.0f} kernels, {c1['staging_bytes_mb']:.1f} MB memcpy, {c1['staging_ms']:.3f} ms", f"{c3['staging_n']:.0f} kernels, {c3['staging_ms']:.3f} ms"])
    w.writerow(['lost fusion (fused + unfused elementwise classes)', parts['lost_fusion_ms'], f"fused {c1['fused_n']:.0f} kernels {c1['fused_ms']:.3f} ms + other/copy {c1['other_ms']+c1['copy_class_ms']:.3f} ms; eager kernels between segments: {c1['eager_n']:.0f}, {c1['eager_ms']:.3f} ms (context)", f"fused {c3['fused_n']:.0f} kernels {c3['fused_ms']:.3f} ms + other/copy {c3['other_ms']+c3['copy_class_ms']:.3f} ms; attention-neighbourhood fused {c3['fused_attn_n']:.0f} kernels {c3['fused_attn_ms']:.3f} ms"])
    w.writerow(['GPU idle gaps inside the pass (launch gaps)', parts['gpu_idle_gaps_ms'], f"{c1['n_graph_launches']:.0f} graph launch CorrIds with kernels (96 cudaGraphLaunch calls/pass), host launch median {c1['host_launch_us_median']:.1f} us, gaps {c1['gap_ms']:.3f} ms", f"{c3['n_graph_launches']:.0f} launch (host {c3['host_launch_us_median']:.0f} us), gaps {c3['gap_ms']:.3f} ms"])
    w.writerow(['GEMM + SDPA kernel-selection delta', parts['gemm_sdpa_selection_delta_ms'], f"GEMM {c1['gemm_ms']:.3f} SDPA {c1['sdpa_ms']:.3f}", f"GEMM {c3['gemm_ms']:.3f} SDPA {c3['sdpa_ms']:.3f}"])
    w.writerow(['SUM of parts', sum(parts.values()), '', ''])
    w.writerow(['trace span delta (C1 - C3, per pass)', trace_span_delta, f"{c1['span_ms']:.3f}", f"{c3['span_ms']:.3f}"])
    w.writerow(['matched-pair delta (unprofiled)', matched, '', ''])
    w.writerow(['residual (matched - sum)', out['residual_vs_matched_ms'], '', ''])
print(json.dumps({k: round(v, 4) for k, v in parts.items()}, indent=0))
print('sum', round(sum(parts.values()), 4), '| span delta', round(trace_span_delta, 4), '| matched', matched, '| residual vs matched', round(out['residual_vs_matched_ms'], 4))
print('C1:', {k: round(v, 3) if isinstance(v, float) else v for k, v in c1.items()})
print('C3:', {k: round(v, 3) if isinstance(v, float) else v for k, v in c3.items()})
