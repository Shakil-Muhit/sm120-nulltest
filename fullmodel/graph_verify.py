"""Graph-deployment verification (plan 0.3; prereg_amendments.md §3), v2 after
gate 02: replay membership by CorrId correlation (this nsys version's GPU
trace carries no graph-id column even under --cuda-graph-trace=node).

Per outer NVTX pass range i (CPU `Orig Start/Duration` for API calls,
GPU-projected `Projected Start/Duration` for kernels):
  API rows  : cudaGraphLaunch/cuGraphLaunch (replay launches; their distinct
              CorrIds = graphs visited), cudaLaunchKernel*/cuLaunchKernel*
              (eager launches), memcpy/memset API calls
  GPU rows  : a kernel is CAPTURED iff its CorrId belongs to a graph launch;
              EAGER iff it belongs to a kernel-launch API call; UNRESOLVED
              otherwise. Eager kernels are split into external input staging
              (cudagraph-trees `_foreach_copy_`/multi_tensor_apply copies and
              ATen direct-copy kernels) vs uncaptured model computation.
Verdict `fully_graphed` requires, in EVERY pass: >= 1 graph launch, zero
uncaptured model kernels, zero unresolved kernels. Anything else is reported
as not fully graphed with the offending kernel names.
"""

import argparse
import csv
import json
import os
import re
import statistics

# cudagraph-trees input staging = the _foreach_copy_ (multi_tensor_apply ...
# Copy<...>) or direct-copy kernels; a multi_tensor_apply with any OTHER
# functor (Add, Mul, ...) is model arithmetic, not staging
def is_staging(name):
    return (('multi_tensor_apply' in name and 'Copy<' in name) or
            'direct_copy_kernel' in name or 'copy_kernel' in name)


def read_csv(path):
    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def col(head, *names, contains=None):
    for name in names:
        for i, h in enumerate(head):
            if h.strip().strip('"') == name:
                return i
    if contains:
        for i, h in enumerate(head):
            if contains in h:
                return i
    return None


def pass_windows(nvtx_csv):
    head, rows = read_csv(nvtx_csv)
    c_name = col(head, 'Name', 'Range', contains='Name')
    c_ps, c_pd = col(head, 'Projected Start (ns)', contains='Projected Start'), col(head, 'Projected Duration (ns)', contains='Projected Duration')
    c_os, c_od = col(head, 'Orig Start (ns)', contains='Orig Start'), col(head, 'Orig Duration (ns)', contains='Orig Duration')
    if c_ps is None:
        c_ps, c_pd = col(head, contains='Start'), col(head, contains='Duration')
    wins = {}
    for r in rows:
        m = re.fullmatch(r'pass_(\d+)', r[c_name].strip().strip('"').split(':')[-1])
        if not m:
            continue
        if int(m.group(1)) in wins:
            raise SystemExit(f'duplicate NVTX pass id {m.group(1)}')
        try:
            gs, gd = int(float(r[c_ps].replace(',', ''))), int(float(r[c_pd].replace(',', '')))
            if c_os is not None:
                cs, cd = int(float(r[c_os].replace(',', ''))), int(float(r[c_od].replace(',', '')))
            else:
                cs, cd = gs, gd
        except ValueError:
            continue
        wins[int(m.group(1))] = {'gpu': (gs, gs + gd), 'cpu': (cs, cs + cd)}
    ids = sorted(wins)
    if ids != list(range(len(ids))):
        raise SystemExit(f'NVTX pass ids are not contiguous from 0: {ids[:5]}...')
    return [wins[i] for i in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='export basename (…/trace_X)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--arm', default='')
    ap.add_argument('--expected-passes', type=int, default=None)
    args = ap.parse_args()
    wins = pass_windows(args.base + '_nvtx_gpu_proj_trace.csv')
    assert wins, 'no pass_<i> NVTX windows in the projection export'

    head, rows = read_csv(args.base + '_cuda_api_trace.csv')
    c_start, c_name = col(head, 'Start (ns)', contains='Start'), col(head, 'Name')
    c_corr = col(head, 'CorrId', 'CorrID', contains='Corr')
    api = []
    for r in rows:
        try:
            api.append({'start': int(float(r[c_start].replace(',', ''))), 'name': r[c_name],
                        'corr': r[c_corr].strip() if c_corr is not None else ''})
        except (ValueError, IndexError):
            continue
    graph_corrs = {a['corr'] for a in api if 'GraphLaunch' in a['name'] and a['corr']}
    launch_corrs = {a['corr'] for a in api if 'LaunchKernel' in a['name'] and a['corr']}

    head, rows = read_csv(args.base + '_cuda_gpu_trace.csv')
    g_start, g_name = col(head, 'Start (ns)', contains='Start'), col(head, 'Name')
    g_corr = col(head, 'CorrId', 'CorrID', contains='Corr')
    kern = []
    for r in rows:
        try:
            kern.append({'start': int(float(r[g_start].replace(',', ''))), 'name': r[g_name],
                         'corr': r[g_corr].strip() if g_corr is not None else ''})
        except (ValueError, IndexError):
            continue

    per_pass = []
    for w in wins:
        cs, ce = w['cpu']
        gs, ge = w['gpu']
        a_in = [a for a in api if cs <= a['start'] < ce]
        k_in = [k for k in kern if gs <= k['start'] < ge]
        graph_launches = [a for a in a_in if 'GraphLaunch' in a['name']]
        eager_launch = sum(1 for a in a_in if 'LaunchKernel' in a['name'])
        memcpy_api = sum(1 for a in a_in if 'Memcpy' in a['name'] or 'Memset' in a['name'])
        # nsys lists memcpy/memset operations as GPU-trace rows ('[CUDA memcpy ...]',
        # '[CUDA memset]'); they correlate to cudaMemcpyAsync/cudaMemsetAsync, are
        # not kernels, and are reported in their own category (never 'unresolved')
        memops = [k for k in k_in if k['name'].startswith('[CUDA mem')]
        kernels_only = [k for k in k_in if not k['name'].startswith('[CUDA mem')]
        memops_in_graph = [k for k in memops if k['corr'] in graph_corrs]
        captured = [k for k in kernels_only if k['corr'] in graph_corrs]
        eager = [k for k in kernels_only if k['corr'] in launch_corrs and k['corr'] not in graph_corrs]
        unresolved = [k for k in kernels_only if k['corr'] not in graph_corrs and k['corr'] not in launch_corrs]
        staging = [k for k in eager if is_staging(k['name'])]
        uncaptured = [k for k in eager if k not in staging]
        per_pass.append({'graph_launch_calls': len(graph_launches),
                         'graph_launch_instances': len({a['corr'] for a in graph_launches}),
                         'eager_kernel_launch_calls': eager_launch,
                         'memcpy_memset_api_calls': memcpy_api,
                         'kernels_total': len(kernels_only),
                         'memops_total': len(memops),
                         'memops_inside_replay': len(memops_in_graph),
                         'memops_eager': len(memops) - len(memops_in_graph),
                         'memop_names': sorted({k['name'] for k in memops}),
                         'captured_model_kernels': len(captured),
                         'input_staging_kernels': len(staging),
                         'uncaptured_model_kernels': len(uncaptured),
                         'unresolved_kernels': len(unresolved),
                         'uncaptured_names': sorted({k['name'][:90] for k in uncaptured}),
                         'unresolved_names': sorted({k['name'][:90] for k in unresolved})})
    keys = ('graph_launch_calls', 'graph_launch_instances', 'eager_kernel_launch_calls',
            'memcpy_memset_api_calls', 'kernels_total', 'memops_total', 'memops_inside_replay',
            'memops_eager', 'captured_model_kernels', 'input_staging_kernels',
            'uncaptured_model_kernels', 'unresolved_kernels')
    summary = {'arm': args.arm, 'passes': len(per_pass), 'expected_passes': args.expected_passes,
               'method': 'CorrId correlation: kernel captured iff its CorrId belongs to a '
                         'cudaGraphLaunch/cuGraphLaunch API call; CPU NVTX windows for API rows, '
                         'GPU-projected windows for kernels',
               'median': {k: statistics.median(p[k] for p in per_pass) for k in keys},
               'max_uncaptured_in_any_pass': max(p['uncaptured_model_kernels'] for p in per_pass),
               'max_unresolved_in_any_pass': max(p['unresolved_kernels'] for p in per_pass),
               'min_graph_launches_in_any_pass': min(p['graph_launch_calls'] for p in per_pass),
               'uncaptured_names_union': sorted({n for p in per_pass for n in p['uncaptured_names']}),
               'unresolved_names_union': sorted({n for p in per_pass for n in p['unresolved_names']}),
               'per_pass': per_pass}
    complete = args.expected_passes is None or len(per_pass) == args.expected_passes
    exports = args.base + '_exports.json'
    exports_ok = (json.load(open(exports)).get('complete') is True) if os.path.exists(exports) else None
    summary['pass_count_complete'] = complete
    summary['exports_complete'] = exports_ok
    summary['min_captured_in_any_pass'] = min(p['captured_model_kernels'] for p in per_pass)
    summary['fully_graphed'] = bool(complete and exports_ok is not False
                                    and summary['min_graph_launches_in_any_pass'] >= 1
                                    and summary['min_captured_in_any_pass'] > 0
                                    and summary['max_uncaptured_in_any_pass'] == 0
                                    and summary['max_unresolved_in_any_pass'] == 0)
    summary['note'] = ('graph_launch_instances counts launch API instances by CorrId, not '
                       'distinct graph identities (not exported by nsys)')
    summary['eager_memop_names_union'] = sorted({n for p in per_pass for n in p['memop_names']})
    summary['verdict'] = ('FULLY_GRAPHED' if summary['fully_graphed'] else
                          'NOT_FULLY_GRAPHED' if summary['min_graph_launches_in_any_pass'] >= 1 else
                          'NO_REPLAY_EVIDENCE')
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=1)
    m = summary['median']
    print(f"[graph_verify] {args.arm}: {summary['verdict']} | graph launches/pass {m['graph_launch_calls']} "
          f"(launch instances {m['graph_launch_instances']}) | captured {m['captured_model_kernels']} staging "
          f"{m['input_staging_kernels']} uncaptured {m['uncaptured_model_kernels']} unresolved "
          f"{m['unresolved_kernels']} (max uncaptured {summary['max_uncaptured_in_any_pass']})")


if __name__ == '__main__':
    main()
