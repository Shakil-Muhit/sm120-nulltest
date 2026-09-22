"""Phase 4 §2B — analysis revision of the matched-pair statistics (historical outputs preserved).

Estimand: for a matched pair (a, b) at one operating point, the per-repeat paired difference
d = median(latency_a over 50 passes) − median(latency_b over 50 passes), where a and b blocks
alternate within one process (repeat = one a-block and the following b-block). Experimental
unit for inference: the process restart (each restart is an independent compile + settle +
5 repeats); repeats within a restart are treated as clustered, not independent.

Point estimate (unchanged from the historical report): median over restarts of the per-restart
median of d ("median of restart medians").
Interval (revised): a two-level (cluster) bootstrap that resamples restarts with replacement and,
within each drawn restart, repeats with replacement, recomputing the same statistic; 10,000
draws, seed 20260921, percentile 2.5/97.5. With three restarts this interval is coarse and
cannot be narrower than the spread of the three restart medians; the restart medians and their
range are reported beside it. The historical interval (pooled-repeat median bootstrap) is kept
for comparison and labelled as estimating a different statistic.
Clock-flagged repeats are excluded exactly as the audit flags them (e_tools.drift_context).
"""
import argparse
import glob
import json
import os
import random
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import e_tools  # noqa: E402

BOOT_N, BOOT_SEED = 10000, 20260921


def cluster_bootstrap(restart_diffs, n=BOOT_N, seed=BOOT_SEED):
    """restart_diffs: list (restarts) of lists (per-repeat differences). Returns (lo, hi) of the
    median-of-restart-medians under a two-level bootstrap."""
    rnd = random.Random(seed)
    k = len(restart_diffs)
    vals = []
    for _ in range(n):
        meds = []
        for _ in range(k):
            r = restart_diffs[rnd.randrange(k)]
            meds.append(statistics.median([r[rnd.randrange(len(r))] for _ in range(len(r))]))
        vals.append(statistics.median(meds))
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def pair_stats(ev, a, b, O, manifest_id, flags):
    per = []
    files = []
    for f in sorted(glob.glob(os.path.join(ev, f'matched_{a}_{b}_c{O}_m*.json'))):
        m = e_tools._load(f)
        r, why = e_tools.pair_reduce(m, manifest_id, flags)
        if r is None:
            continue
        per.append(r['diffs']); files.append(os.path.basename(f))
    if not per:
        return None
    rmeds = [statistics.median(d) for d in per]
    pooled = [x for d in per for x in d]
    lo, hi = cluster_bootstrap(per)
    plo, phi = e_tools._boot_ci(pooled)
    return {'files': files, 'restarts': len(per), 'repeats_used': len(pooled),
            'restart_medians_ms': rmeds, 'restart_median_range_ms': [min(rmeds), max(rmeds)],
            'point_estimate_ms (median of restart medians)': statistics.median(rmeds),
            'ci95_cluster_bootstrap_ms (revised; same statistic)': [lo, hi],
            'historical': {'pooled_repeat_median_ms': statistics.median(pooled), 'ci95_pooled_repeat_bootstrap_ms': [plo, phi],
                           'note': 'the historical interval resampled pooled repeats (a different statistic from the reported point estimate)'},
            'inference_scope': f'this pod, this session, this software configuration; {len(rmeds)} restart(s) carry the between-restart information — the interval resamples restarts and then repeats within them, so it is not the restart-median range'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence-dir', required=True)
    ap.add_argument('--pairs', nargs='+', default=['C3:H:176400', 'C3:H:264600', 'C3:C3M:352800', 'C3:C3M:4410', 'C3:C3G:352800', 'C3:C3G:4410', 'C3:C3P:352800', 'C3:C3P:4410',
                                                  'C3:C3F:352800', 'C3:C3F:4410', 'C1:C1P:352800', 'C3M:C3N:352800', 'C3M:C3N:4410', 'C3:C3:352800', 'C3:C3:4410', 'C3:C3:176400', 'C3:C3:264600'])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    manifest_id, flags = e_tools.drift_context(a.evidence_dir)
    out = {'estimand_and_method': __doc__, 'pairs': {}}
    for spec in a.pairs:
        x, y, O = spec.split(':')
        r = pair_stats(a.evidence_dir, x, y, int(O), manifest_id, flags)
        out['pairs'][spec] = r or {'status': 'UNRESOLVED: no eligible pair'}
        if r:
            print(f"{spec:18s} point {r['point_estimate_ms (median of restart medians)']:+.4f}  cluster CI {r['ci95_cluster_bootstrap_ms (revised; same statistic)'][0]:+.4f}..{r['ci95_cluster_bootstrap_ms (revised; same statistic)'][1]:+.4f}  restart medians {[round(v, 3) for v in r['restart_medians_ms']]}  | historical pooled {r['historical']['pooled_repeat_median_ms']:+.4f} CI {r['historical']['ci95_pooled_repeat_bootstrap_ms'][0]:+.4f}..{r['historical']['ci95_pooled_repeat_bootstrap_ms'][1]:+.4f}")
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
