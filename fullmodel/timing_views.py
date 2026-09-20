"""Three timing views for multi-stream arms (plan 1.7; prereg_amendments §5).

From a node-level cuda_gpu_trace export with outer NVTX pass windows:
  (i)   summed kernel durations per pass
  (ii)  union of kernel intervals per pass (busy union — overlap collapsed)
  (iii) OBSERVED-ORDER critical-path estimate per pass. DEPENDENCY MODEL
        (stated; gate-02 f27): the trace exports no event/graph edges, so the
        only dependencies that can be asserted are observed orderings —
        edge a -> b iff a ENDED before b STARTED and (same stream) or
        (different streams: the fork/join structure of F/G means a side-stream
        kernel can only depend on main-stream work that finished before it,
        and vice versa). Overlapping kernels never depend on each other.
        Critical path = longest chain of pairwise non-overlapping kernels
        (sum of durations). By construction critical_path <= union <= span
        (asserted); this is an UPPER BOUND on the true dependency path, labelled
        as such — never the true event graph.
Every view is reported per pass and as the median over passes; "busy" is
never reported without its view.
"""

import argparse
import csv
import json
import re
import statistics


def read_rows(path):
    with open(path, newline='') as f:
        rd = csv.reader(f)
        head = next(rd)
        def col(*names, contains=None):
            for n in names:
                for i, h in enumerate(head):
                    if h.strip().strip('"') == n:
                        return i
            if contains:
                for i, h in enumerate(head):
                    if contains in h:
                        return i
            return None
        c_s, c_d = col('Start (ns)', contains='Start'), col('Duration (ns)', contains='Duration')
        c_n, c_st = col('Name'), col('Strm', 'Stream', contains='Strm')
        rows = []
        for r in rd:
            try:
                rows.append({'start': int(float(r[c_s].replace(',', ''))),
                             'dur': int(float(r[c_d].replace(',', ''))),
                             'name': r[c_n], 'stream': r[c_st] if c_st is not None else '0'})
            except (ValueError, IndexError):
                continue
    rows.sort(key=lambda x: x['start'])
    return rows


def windows(path):
    with open(path, newline='') as f:
        rd = csv.reader(f)
        head = next(rd)
        c_name = next(i for i, h in enumerate(head) if h.strip().strip('"') in ('Name', 'Range'))
        c_start = next(i for i, h in enumerate(head) if 'Start' in h)
        c_dur = next(i for i, h in enumerate(head) if 'Duration' in h)
        wins = {}
        for r in rd:
            m = re.fullmatch(r'pass_(\d+)', r[c_name].strip().strip('"').split(':')[-1])
            if m:
                try:
                    s = int(float(r[c_start].replace(',', '')))
                    d = int(float(r[c_dur].replace(',', '')))
                except ValueError:
                    continue
                wins[int(m.group(1))] = (s, s + d)
    return [wins[i] for i in sorted(wins)]


def union_ns(rows):
    total, cur_s, cur_e = 0, None, None
    for r in rows:
        s, e = r['start'], r['start'] + r['dur']
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def critical_path_ns(rows):
    """Longest chain of pairwise non-overlapping kernels (ended-before-started
    edges); O(n^2) per pass is fine for ~1-2k kernels."""
    if not rows:
        return 0, '-'
    order = sorted(rows, key=lambda x: (x['start'], x['start'] + x['dur']))
    ends = [r['start'] + r['dur'] for r in order]
    best = [0] * len(order)
    for i, r in enumerate(order):
        pred = 0
        for j in range(i):
            if ends[j] <= r['start'] and best[j] > pred:
                pred = best[j]
        best[i] = pred + r['dur']
    by_stream = {}
    for r in rows:
        by_stream[r['stream']] = by_stream.get(r['stream'], 0) + 1
    dominant = max(by_stream, key=by_stream.get)
    return max(best), dominant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='export basename (…/trace_X)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--arm', default='')
    args = ap.parse_args()
    rows = read_rows(args.base + '_cuda_gpu_trace.csv')
    wins = windows(args.base + '_nvtx_gpu_proj_trace.csv')
    assert wins, 'no pass windows'
    per = []
    for s, e in wins:
        p = [r for r in rows if s <= r['start'] < e]
        cp, dom = critical_path_ns(p)
        u = union_ns(p)
        span = (max(r['start'] + r['dur'] for r in p) - min(r['start'] for r in p)) if p else 0
        assert cp <= u + 1 and u <= span + 1, (cp, u, span)   # observed-order bound
        per.append({'kernels': len(p), 'streams': sorted({r['stream'] for r in p}),
                    'sum_ns': sum(r['dur'] for r in p), 'union_ns': u,
                    'critical_path_observed_order_ns': cp, 'span_ns': span,
                    'window_ns': e - s, 'dominant_stream': dom})
    def med(k):
        return statistics.median(x[k] for x in per)
    out = {'arm': args.arm, 'passes': len(per),
           'median_us': {k: med(k) / 1e3 for k in ('sum_ns', 'union_ns', 'critical_path_observed_order_ns', 'span_ns', 'window_ns')},
           'median_kernels_per_pass': med('kernels'),
           'streams_seen': sorted({s for x in per for s in x['streams']}),
           'dependency_model': ('observed-order: edge a->b iff a ended before b started; '
                                'longest chain of non-overlapping kernels; an UPPER BOUND on the '
                                'true dependency path (no event/graph edges are exported)'),
           'per_pass': per}
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    m = out['median_us']
    print(f"[views] {args.arm}: sum {m['sum_ns']:.1f} us | union {m['union_ns']:.1f} us | "
          f"observed-order critical path {m['critical_path_observed_order_ns']:.1f} us | span "
          f"{m['span_ns']:.1f} us | streams {out['streams_seen']}")


if __name__ == '__main__':
    main()
