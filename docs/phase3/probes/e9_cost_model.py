"""E9 cost model: wall(C3, L) - wall(H, L) as a function of chunk length L.

Inputs (ONLY): the two node-level trace decompositions (docs/phase3/padbatch/
decomposition_table.json, from the 8 s and 100 ms traces) and the four existing
sweep points (SESSION_RESULT_v2.md headline walls at 4410, 22050, 88200, 352800).
Frames per chunk F(L) = L/441 + 1 (hop 441; 352800 -> 801, 4410 -> 11).

Term structure (all in ms, sign: + means H faster than C3):
  D(L) = B                      batching benefit (gap/launch floor removed): fixed, mean of the two traces
       - NF(F)                  non-GEMM front/tail delta (H - C3): linear in F through the two traces
       - MID(F)                 transformer-middle delta (H - C3): linear in F through the two traces
       - PAD(F)                 padded FLOPs (0.16094 GFLOP/frame) at rate_H(F)
       - EFF(F)                 true work (0.4083 GFLOP/frame) x (1/rate_H(F) - 1/rate_C3(F))
  rate_X(F) = R_X * F / (F + h_X) * q_X(F), q = tile-quantization factor F / (T ceil(F/T)) or 1.
Model variants differ in q and in whether (R, h) come from the two traces alone
(variant "traces_only") or are refit on the four sweep walls with the trace
rates as soft constraints (variant "sweep_refit"). Every variant's in-sample
residuals at the four sweep points are printed and stored; the registered
variant is chosen by the smallest max |residual| and the choice is recorded
BEFORE any held-out measurement. Held-out lengths: 176400 (F=401) and 264600
(F=601). Never run those lengths on the shakedown host.
"""
import json, math, sys, itertools
import numpy as np
from scipy.optimize import least_squares

ROOT = __file__.rsplit('/docs/', 1)[0]
dec = json.load(open(f'{ROOT}/docs/phase3/padbatch/decomposition_table.json'))
d8 = next(d for d in dec if d['O'] == 352800); d1 = next(d for d in dec if d['O'] == 4410)
HOP = 441
def F(L): return L // HOP + 1
F8, F1 = F(352800), F(4410)

# sweep walls (SESSION_RESULT_v2.md arms table: headline = median over restarts of min-of-medians)
walls = {'C3': {352800: 28.8714, 4410: 3.2752, 22050: 4.2173, 88200: 8.6252},
         'H':  {352800: 29.3624, 4410: 2.3386, 22050: 3.3184, 88200: 7.4558}}
obs = {L: walls['C3'][L] - walls['H'][L] for L in walls['C3']}   # + = H faster
matched = {352800: -0.4953803727403283, 4410: 0.936683383770287}  # matched-pair medians (same sign convention)

B = 0.5 * (d8['batching_benefit_ms (gap/launch floor removed)'] + d1['batching_benefit_ms (gap/launch floor removed)'])
def lin(y1, y8):
    a = (y8 - y1) / (F8 - F1); return (y1 - a * F1, a)
NF = lin(d1['non_GEMM_front_tail_delta_ms (H - C3)'], d8['non_GEMM_front_tail_delta_ms (H - C3)'])
MID = lin(d1['transformer_middle_delta_ms (H - C3)'], d8['transformer_middle_delta_ms (H - C3)'])
pad_gflop_per_frame = d8['extra_padded_GFLOP'] / F8
eff1 = d1['kernel_efficiency_change_ms (true work at H rate vs C3 rate; + = H slower)']
eff8 = d8['kernel_efficiency_change_ms (true work at H rate vs C3 rate; + = H slower)']
rH1, rH8 = d1['H_fronttail_GEMM_rate_TFLOPS'], d8['H_fronttail_GEMM_rate_TFLOPS']
rC1, rC8 = d1['C3_fronttail_GEMM_rate_TFLOPS'], d8['C3_fronttail_GEMM_rate_TFLOPS']
true_gflop_per_frame = (eff8 / (1 / rH8 - 1 / rC8)) / F8       # ms / (ms/GFLOP)
chk = (eff1 / (1 / rH1 - 1 / rC1)) / F1
assert abs(true_gflop_per_frame - chk) / true_gflop_per_frame < 0.02, (true_gflop_per_frame, chk)

def q_tile(T):
    return (lambda f: f / (T * math.ceil(f / T))) if T else (lambda f: 1.0)

def fit_rate_two_points(r1, r8, q):
    # r = R F/(F+h) q(F) through (F1, r1), (F8, r8)
    a1, a8 = r1 / q(F1), r8 / q(F8)
    # a1 (F1+h)/F1 = a8 (F8+h)/F8  ->  h = (a8 F1 - a1 F8) / (a1 - a8) * ... solve linear
    # a1*(F1+h)*F8 = a8*(F8+h)*F1 -> a1 F1 F8 + a1 F8 h = a8 F8 F1 + a8 F1 h -> h (a1 F8 - a8 F1) = F1 F8 (a8 - a1)
    h = F1 * F8 * (a8 - a1) / (a1 * F8 - a8 * F1)
    R = a1 * (F1 + h) / F1
    return R, h

def model_D(f, RH, hH, RC, hC, qH, qC):
    rH = RH * f / (f + hH) * qH(f); rC = RC * f / (f + hC) * qC(f)
    pad = pad_gflop_per_frame * f / rH
    eff = true_gflop_per_frame * f * (1 / rH - 1 / rC)
    return B - (NF[0] + NF[1] * f) - (MID[0] + MID[1] * f) - pad - eff, dict(rate_H=rH, rate_C3=rC, pad=pad, eff=eff,
            nf=NF[0] + NF[1] * f, mid=MID[0] + MID[1] * f, B=B)

variants = {}
for TH, TC in itertools.product([None, 64, 128], [None, 64, 128]):
    qH, qC = q_tile(TH), q_tile(TC)
    RH, hH = fit_rate_two_points(rH1, rH8, qH); RC, hC = fit_rate_two_points(rC1, rC8, qC)
    if min(RH, hH, RC, hC) <= 0: continue
    name = f'traces_only_TH{TH}_TC{TC}'
    pred = {L: model_D(F(L), RH, hH, RC, hC, qH, qC)[0] for L in obs}
    res = {L: pred[L] - obs[L] for L in obs}
    variants[name] = dict(params=dict(RH=RH, hH=hH, RC=RC, hC=hC, TH=TH, TC=TC), pred=pred, res=res,
                          max_abs_res=max(abs(v) for v in res.values()))
    # sweep refit: (RC, hC) free, H curve kept from traces; soft constraint on the two trace rates for C3
    def resid(p):
        RCx, hCx = p
        r = [model_D(F(L), RH, hH, RCx, hCx, qH, qC)[0] - obs[L] for L in obs]
        r += [0.2 * (RCx * F1 / (F1 + hCx) * qC(F1) - rC1) / rC1, 0.2 * (RCx * F8 / (F8 + hCx) * qC(F8) - rC8) / rC8]
        return r
    sol = least_squares(resid, x0=[RC, hC], bounds=([1, 0.1], [419, 5000]))
    RCx, hCx = sol.x
    pred = {L: model_D(F(L), RH, hH, RCx, hCx, qH, qC)[0] for L in obs}
    res = {L: pred[L] - obs[L] for L in obs}
    variants[name.replace('traces_only', 'sweep_refit')] = dict(params=dict(RH=RH, hH=hH, RC=RCx, hC=hCx, TH=TH, TC=TC),
        pred=pred, res=res, max_abs_res=max(abs(v) for v in res.values()),
        trace_rate_check=dict(rC1_model=RCx * F1 / (F1 + hCx) * qC(F1), rC1=rC1, rC8_model=RCx * F8 / (F8 + hCx) * qC(F8), rC8=rC8))

for name, v in sorted(variants.items(), key=lambda kv: kv[1]['max_abs_res']):
    print(f"{name:32s} max|res| {v['max_abs_res']:.3f}  res " + ' '.join(f"{L}:{v['res'][L]:+.3f}" for L in sorted(v['res'])))

HELD = [176400, 264600]
def crossover(v):
    p = v['params']; qH, qC = q_tile(p['TH']), q_tile(p['TC'])
    Ls = np.arange(4410, 352801, 441)
    Ds = np.array([model_D(F(int(L)), p['RH'], p['hH'], p['RC'], p['hC'], qH, qC)[0] for L in Ls])
    sign_changes = np.where(np.diff(np.sign(Ds)) != 0)[0]
    return [int(Ls[i]) for i in sign_changes], Ls, Ds
best_name = min(variants, key=lambda k: variants[k]['max_abs_res'])
out = {'tolerance_ms': 0.4, 'pass_rule': 'sign correct at both held-out lengths and |observed - predicted| <= tolerance_ms at both (prereg v0.2 E9)',
       'inputs': dict(B=B, NF=NF, MID=MID, pad_gflop_per_frame=pad_gflop_per_frame, true_gflop_per_frame=true_gflop_per_frame,
                      trace_rates=dict(rH1=rH1, rH8=rH8, rC1=rC1, rC8=rC8), sweep_obs_C3_minus_H=obs, matched=matched, hop=HOP),
       'variants': variants, 'registered_variant': best_name, 'held_out': {}}
for name in [best_name, 'traces_only_THNone_TCNone']:
    v = variants[name]; p = v['params']; qH, qC = q_tile(p['TH']), q_tile(p['TC'])
    xs, Ls, Ds = crossover(v)
    v['crossover_L_star'] = xs
    v['held_out_pred'] = {L: model_D(F(L), p['RH'], p['hH'], p['RC'], p['hC'], qH, qC) for L in HELD}
    v['held_out_pred'] = {L: dict(D_C3_minus_H_ms=val[0], terms=val[1]) for L, val in v['held_out_pred'].items()}
    print(f"\n{name}: L* = {xs}  held-out: " + ' '.join(f"{L}:{v['held_out_pred'][L]['D_C3_minus_H_ms']:+.3f}" for L in HELD))
json.dump(out, open(f'{ROOT}/docs/phase3/e9/e9_cost_model.json', 'w'), indent=1, default=float)
print('written docs/phase3/e9/e9_cost_model.json')
