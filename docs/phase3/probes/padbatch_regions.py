"""2.4 — pad-and-batch decomposition: per-region kernel times (C3 vs H, both OPs) from the node traces, and the
padded-vs-true FLOP arithmetic from the checkpoint's band widths (parameter shapes in the wrappers).
Regions per pass: FRONT = kernels before the first attention execution; TAIL = kernels after the last one;
MIDDLE = transformer. Class split inside each region. Trace-derived (perturbation caveats: C3 x1.013 / x1.052)."""
import csv, json, os, sys
sys.path.insert(0, 'fullmodel'); sys.path.insert(0, 'fullmodel/attrib')
from classify import classify
from kernel_map import read_gpu_trace, nvtx_pass_windows, split_passes
EV = '/home/muhit/ms/evidence_pod/9tblar2zwo97ln_20260920_1855/evidence'
def med(xs): s = sorted(xs); n = len(s); return s[n//2] if n % 2 else 0.5*(s[n//2-1]+s[n//2])
def is_sdpa_exec(n): return classify(n) == 'SDPA' and not n.startswith('triton')
out = {'regions': {}, 'arithmetic': {}}
for arm in ('C3', 'H'):
    for O in (352800, 4410):
        base = f'{EV}/traces/trace_{arm}_{O}_trace'
        rows = [r for r in read_gpu_trace(base + '_cuda_gpu_trace.csv') if not r['name'].startswith('[CUDA mem')]
        windows, err = nvtx_pass_windows(base + '_nvtx_gpu_proj_trace.csv'); assert not err and len(windows) == 50
        passes, _ = split_passes(rows, windows, classify)
        acc = {reg: {cls: [] for cls in ('GEMM', 'SDPA', 'INDUCTOR_FUSED', 'COPY', 'FFT', 'OTHER', 'n')} for reg in ('front', 'middle', 'tail')}
        for p in passes:
            p = sorted(p, key=lambda r: r['start'])
            idx = [i for i, r in enumerate(p) if is_sdpa_exec(r['name'])]
            f, l = idx[0], idx[-1]
            for reg, seg in (('front', p[:f]), ('middle', p[f:l+1]), ('tail', p[l+1:])):
                for cls in ('GEMM', 'SDPA', 'INDUCTOR_FUSED', 'COPY', 'FFT', 'OTHER'):
                    acc[reg][cls].append(sum(r['dur'] for r in seg if classify(r['name']) == cls) / 1e6)
                acc[reg]['n'].append(len(seg))
        out['regions'][f'{arm}_{O}'] = {reg: {cls: med(v) for cls, v in d.items()} for reg, d in acc.items()}
        for reg in ('front', 'middle', 'tail'):
            d = out['regions'][f'{arm}_{O}'][reg]; d['total_ms'] = sum(d[c] for c in ('GEMM', 'SDPA', 'INDUCTOR_FUSED', 'COPY', 'FFT', 'OTHER'))
        print(arm, O, {reg: (round(d['total_ms'], 3), int(d['n'])) for reg, d in out['regions'][f'{arm}_{O}'].items()})
# arithmetic: band widths from the C3 wrapper (band_split Linear in = 4*freqs_i, out 384; mask last Linear out = 8*freqs_i? in 1536)
bw = json.load(open('docs/phase3/probes/band_shapes_probe.json'))['band_widths_from_C3_4410']
bs_in = sorted(bw['band_split_in_features']); mk_out = sorted(bw['mask_out_features'])
assert len(bs_in) == 60 and len(mk_out) == 60
FR = {352800: 801, 4410: 11}
for O in (352800, 4410):
    T = FR[O]
    true_bs = sum(2 * T * k * 384 for k in bs_in); pad_bs = 60 * 2 * T * max(bs_in) * 384
    true_mk = sum(2 * T * 1536 * n for n in mk_out); pad_mk = 60 * 2 * T * 1536 * max(mk_out)
    out['arithmetic'][str(O)] = {'frames': T, 'band_split': {'true_flops': true_bs, 'padded_flops': pad_bs, 'waste_frac': 1 - true_bs / pad_bs, 'K_max': max(bs_in), 'K_sum': sum(bs_in)},
                                 'mask_last_layer': {'true_flops': true_mk, 'padded_flops': pad_mk, 'waste_frac': 1 - true_mk / pad_mk, 'N_max': max(mk_out), 'N_sum': sum(mk_out)},
                                 'note': 'H pads band_split to K_max=520 (one (60,520,384) stack) and the mask last layer to N_max=1040 (one (60,1536,1040) stack): a single bucket, not per-bucket (wrapper parameter shapes)'}
    print(O, 'band_split waste', round(1 - true_bs / pad_bs, 3), 'mask-last waste', round(1 - true_mk / pad_mk, 3), 'padded extra GFLOP', round((pad_bs - true_bs + pad_mk - true_mk) / 1e9, 3))
os.makedirs('docs/phase3/padbatch', exist_ok=True)
json.dump(out, open('docs/phase3/padbatch/regions_and_arithmetic.json', 'w'), indent=1)
with open('docs/phase3/padbatch/region_times.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['arm', 'O', 'region', 'kernels', 'GEMM_ms', 'SDPA_ms', 'fused_ms', 'COPY_ms', 'FFT_ms', 'OTHER_ms', 'total_ms'])
    for k, regs in out['regions'].items():
        arm, O = k.split('_')
        for reg, d in regs.items():
            w.writerow([arm, O, reg, int(d['n']), *(round(d[c], 4) for c in ('GEMM', 'SDPA', 'INDUCTOR_FUSED', 'COPY', 'FFT', 'OTHER')), round(d['total_ms'], 4)])
