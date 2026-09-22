"""E9 retrospective (Phase 4 §2): the registered affine predictor D(F) = a + b·F (parameters as stored in
docs/phase4/e9_affine_check.json; never refitted) against the session-4 matched C3 − H differences at
1 s / 4 s / 6 s / 8 s on one GPU. Retrospective only: E9's registered test (Phase 3) stays FAILED; the
session-4 lengths 4 s / 6 s were the registered held-out points and 8 s / 100 ms were calibration points,
and 1 s is an additional retrospective length, so no value here is an out-of-sample validation of a new model.
Usage: python e9_retrospective_session4.py <stats_v2_session4.json> <e9_affine_check.json> <out.json>"""
import json, sys
stats, aff, out = sys.argv[1:4]
S = json.load(open(stats))['pairs']
A = json.load(open(aff))['affine_form']
a, b = A['a_ms'], A['b_ms_per_frame']
TOL = 0.4
rows = {}
for L in (44100, 176400, 264600, 352800):
    F = L // 441 + 1
    pred = a + b * F
    p = S[f'C3:H:{L}']
    obs = p['point_estimate_ms (median of restart medians)']
    ci = p['ci95_cluster_bootstrap_ms (revised; same statistic)']
    rows[str(L)] = {'frames': F, 'affine_prediction_ms': round(pred, 4), 'session4_observed_ms': round(obs, 4),
                    'session4_ci95': [round(ci[0], 4), round(ci[1], 4)], 'error_pred_minus_obs_ms': round(pred - obs, 4),
                    'within_registered_tolerance_0.4ms': abs(pred - obs) <= TOL, 'sign_agrees': (pred > 0) == (obs > 0),
                    'role_in_phase3': {176400: 'registered held-out point (Phase 3, other GPU)', 264600: 'registered held-out point (Phase 3, other GPU)', 352800: 'calibration point (Phase 3 calibration set 4,410 / 22,050 / 88,200 / 352,800; other GPU/session)', 44100: 'additional retrospective length (neither calibration nor registered held-out)'}[L]}
rec = {'predictor': {'a_ms': a, 'b_ms_per_frame': b, 'F': 'L // 441 + 1', 'source': 'docs/phase4/e9_affine_check.json (stored Phase-3 parameters; not refitted)'},
       'tolerance_ms': TOL, 'rows': rows,
       'reading': ('retrospective comparison on a third GPU (session 4); the registered E9 verdict (Phase 3, FAIL) is unchanged; '
                   'signs: ' + ', '.join(f"{L}: {'agree' if r['sign_agrees'] else 'DISAGREE'}" for L, r in rows.items()) + '; '
                   'within 0.4 ms: ' + ', '.join(f"{L}: {'yes' if r['within_registered_tolerance_0.4ms'] else 'no'}" for L, r in rows.items()))}
json.dump(rec, open(out, 'w'), indent=1)
for L, r in rows.items():
    print(L, 'F', r['frames'], 'pred', r['affine_prediction_ms'], 'obs', r['session4_observed_ms'], r['session4_ci95'], 'err', r['error_pred_minus_obs_ms'], 'tol', r['within_registered_tolerance_0.4ms'], 'sign', r['sign_agrees'])
