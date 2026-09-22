"""Phase 4 §2A: independent verification that the registered E9 variant is affine in the frame
count F = L//441 + 1, reproduction of its stored predictions and in-sample residuals, the
calibration-only affine counterexample from the packaging review, and a plain piecewise-linear
interpolation baseline through the four sweep observations (retrospective comparators; nothing
is fitted to the 4 s / 6 s observations). Reads only the stored registration JSON."""
import json, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
reg = json.load(open(os.path.join(ROOT, 'docs/phase3/e9/e9_cost_model.json')))
inp, var = reg['inputs'], reg['variants'][reg['registered_variant']]
p = var['params']; assert p['TH'] is None and p['TC'] is None, 'tile factors must be 1 for the affine reduction'
B, (nf0, nf1), (mid0, mid1) = inp['B'], inp['NF'], inp['MID']
pad, w = inp['pad_gflop_per_frame'], inp['true_gflop_per_frame']
RH, hH, RC, hC = p['RH'], p['hH'], p['RC'], p['hC']
# D(F) = B - (nf0 + nf1 F) - (mid0 + mid1 F) - pad (F+hH)/RH - w [ (F+hH)/RH - (F+hC)/RC ]
a = B - nf0 - mid0 - pad * hH / RH - w * (hH / RH - hC / RC)
b = -(nf1 + mid1 + pad / RH + w * (1 / RH - 1 / RC))
def D_affine(F): return a + b * F
def F_of(L): return L // inp['hop'] + 1
out = {'affine_form': {'a_ms': a, 'b_ms_per_frame': b, 'note': 'D(F) = a + b*F exactly, because F/(R F/(F+h)) = (F+h)/R when the tile factor is 1'}}
# reproduce stored held-out predictions and in-sample residuals
rep = {}
for L, h in var['held_out_pred'].items():
    rep[L] = {'stored': h['D_C3_minus_H_ms'], 'affine': D_affine(F_of(int(L))), 'diff': h['D_C3_minus_H_ms'] - D_affine(F_of(int(L)))}
obs = {int(k): v for k, v in inp['sweep_obs_C3_minus_H'].items()}
res = {L: {'stored_residual (pred-obs)': var['res'][str(L)], 'affine_pred': D_affine(F_of(L)), 'obs': obs[L], 'recomputed_residual': D_affine(F_of(L)) - obs[L]} for L in sorted(obs)}
out['held_out_reproduction'] = rep; out['in_sample'] = res
out['max_abs_reproduction_error_ms'] = max(abs(v['diff']) for v in rep.values()) if rep else None
# calibration nonmonotonicity and the best affine fit to the four calibration values (least max-abs, by LP on 4 points via brute-force over pairs)
xs = [F_of(L) for L in sorted(obs)]; ys = [obs[L] for L in sorted(obs)]
out['calibration_points'] = dict(zip([str(L) for L in sorted(obs)], ys))
out['calibration_monotone'] = all(ys[i] >= ys[i + 1] for i in range(len(ys) - 1)) or all(ys[i] <= ys[i + 1] for i in range(len(ys) - 1))
# minimax affine fit: the optimum for 4 points is attained where three points have equal |residual| with alternating signs; brute force candidates
import itertools
best = None
for (i, j, k) in itertools.combinations(range(4), 3):
    for s in (1, -1):
        # solve a + b x_i - y_i = s e, a + b x_j - y_j = -s e, a + b x_k - y_k = s e
        import numpy as np
        A = np.array([[1, xs[i], -s], [1, xs[j], s], [1, xs[k], -s]], dtype=float)
        rhs = np.array([ys[i], ys[j], ys[k]], dtype=float)
        try:
            aa, bb, e = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            continue
        err = max(abs(aa + bb * x - y) for x, y in zip(xs, ys))
        if best is None or err < best[0]:
            best = (err, aa, bb)
out['calibration_only_minimax_affine'] = {'a_ms': best[1], 'b_ms_per_frame': best[2], 'max_abs_error_ms': best[0],
    'review_counterexample': {'a': 1.244551392405063, 'b': -0.001807088607595, 'max_abs_error_ms': 0.288073417721519},
    'note': 'retrospective comparator: a line CAN meet the 0.4 ms tolerance on the four calibration values; it was not the registered fit and is not evaluated at 4 s/6 s here'}
# piecewise-linear interpolation baseline through the sweep (retrospective; its 4 s/6 s values are shown for comparison only, labelled)
def pl(L):
    Ls = sorted(obs)
    for L0, L1 in zip(Ls, Ls[1:]):
        if L0 <= L <= L1:
            t = (L - L0) / (L1 - L0); return obs[L0] + t * (obs[L1] - obs[L0])
    return None
out['piecewise_linear_baseline_retrospective'] = {str(L): pl(L) for L in (176400, 264600)}
out['registered_predictions'] = {L: h['D_C3_minus_H_ms'] for L, h in var['held_out_pred'].items()}
json.dump(out, open(os.path.join(ROOT, 'docs/phase4/e9_affine_check.json'), 'w'), indent=1)
print(json.dumps({k: out[k] for k in ('affine_form', 'max_abs_reproduction_error_ms', 'calibration_monotone', 'calibration_only_minimax_affine', 'piecewise_linear_baseline_retrospective', 'registered_predictions')}, indent=1))
print('in-sample:', {L: round(v['recomputed_residual'], 4) for L, v in res.items()})
