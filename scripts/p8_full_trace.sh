#!/bin/bash
# P8 session opener (prereg P8): ONE nsys pass of the FULL public
# MelBandRoformer forward at the 8 s chunk, BEFORE any arm runs
# (scripts/run_all.sh enforces the ordering).
#
# Extracted from the trace, landing in evidence/ as the P8 result:
#   (1) front-end share of GPU time — via the NVTX boundaries that
#       scripts/p8_trace.py installs with external hooks (the public path
#       exposes no NVTX stages, so this realizes the kernel-sequence
#       boundary; p8_run.json:boundary_mode documents which split ran)
#   (2) the attention stack's kernel count; gap structure is inspected from
#       the timeline itself (evidence/trace_p8_full.nsys-rep)
#
# Provenance guard (AUDIT-MAP[5]): refuses to run unless the msst clone's HEAD
# equals the pinned commit in configs/msst_commit.txt, and the checkpoint
# passes sha256 verification (fetch_checkpoint.sh — skip-if-present).
set -u
set -o pipefail   # the extraction pipes through tee — its failure must be fatal

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# EVID_DIR override exists for scripts/smoke.sh ONLY (shakedown evidence must
# never mix with session evidence); the session always uses the default
EVID="${EVID_DIR:-$REPO_DIR/evidence}"
MSST_DIR="${MSST_DIR:-$HOME/work/msst}"
CKPT_DIR="${CKPT_DIR:-$HOME/work/ckpt}"
mkdir -p "$EVID"

[ -d "$MSST_DIR/.git" ] || { echo "FATAL: $MSST_DIR missing — run setup.sh first"; exit 1; }
PIN="$(grep -v '^[[:space:]]*#' "$REPO_DIR/configs/msst_commit.txt" | grep -m1 '[^[:space:]]' | tr -d '[:space:]')"
HEAD="$(git -C "$MSST_DIR" rev-parse HEAD)"
if [ -z "$PIN" ] || [ "$HEAD" != "$PIN" ]; then
    echo "FATAL: msst clone HEAD ($HEAD) != pinned commit ('$PIN')."
    echo "       Re-run setup.sh (section [5/6]) — the P8 trace only runs"
    echo "       against the pinned public commit."
    exit 1
fi

bash "$REPO_DIR/scripts/fetch_checkpoint.sh" || exit 1
# manifest.env provides CKPT_FILE, CONFIG_FILE, MODEL_TYPE
. "$CKPT_DIR/manifest.env"

echo "[p8] tracing full forward: model_type=$MODEL_TYPE ckpt=$CKPT_FILE"
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
    --force-overwrite=true -o "$EVID/trace_p8_full" \
    python "$REPO_DIR/scripts/p8_trace.py" \
        --msst-dir "$MSST_DIR" --ckpt "$CKPT_FILE" --config "$CONFIG_FILE" \
        --model-type "$MODEL_TYPE" --chunk 352800 \
        --warmup "${P8_WARMUP:-3}" --iters "${P8_ITERS:-3}" \
        --json "$EVID/p8_run.json" \
    || { echo "FATAL: P8 trace run failed"; exit 1; }

# NVTX-range kernel summary (new report name first, legacy alias fallback)
nsys stats --force-export=true --force-overwrite=true \
        -r nvtx_kern_sum --format csv \
        --output "$EVID/p8" "$EVID/trace_p8_full.nsys-rep" >/dev/null 2>&1 \
    || nsys stats --force-export=true --force-overwrite=true \
        -r nvtxkernsum --format csv \
        --output "$EVID/p8" "$EVID/trace_p8_full.nsys-rep" >/dev/null \
    || { echo "FATAL: nsys stats (nvtx kernel summary) failed"; exit 1; }

python3 - "$EVID" <<'PYEOF' | tee "$EVID/p8_result.txt"
import csv, glob, json, os, re, sys
evid = sys.argv[1]
paths = sorted(glob.glob(os.path.join(evid, 'p8*kern*sum*.csv')))
if not paths:
    sys.exit('no p8 NVTX kernel-summary csv found')
with open(paths[-1], newline='') as f:
    rows = list(csv.reader(f))
head = rows[0]
def col(*names):
    # exact header match first (any candidate), else guarded substring.
    # nvtx kernel-summary headers drift across nsys versions: modern emits
    # 'Kern Inst' / 'NVTX Inst' / 'Total Proj Time'; legacy emits 'Instances'
    for name in names:
        for i, h in enumerate(head):
            if h.strip().strip('"') == name:
                return i
    key = names[0].split(' ')[0]
    return next(i for i, h in enumerate(head)
                if key in h and 'Range' not in h and 'Proj' not in h
                and 'NVTX' not in h)
r_col = next(i for i, h in enumerate(head) if 'Range' in h)
t_col = col('Total Time (ns)')
i_col = col('Kern Inst', 'Instances')
ranges = {}
subs = {}   # per-submodule sub-ranges (fix-pass 2026-08-18): nested rows carry
            # the full range path (':p8_attention/:p8_L0_time_attn'), so each
            # kernel row is counted once under its outer name AND once under
            # its sub-kind — the outer totals are unchanged by nesting
SUB_RE = re.compile(r'p8_L(\d+)_(linear_attn|time_attn|rearrange|freq_attn)')
for row in rows[1:]:
    if len(row) <= max(r_col, t_col, i_col) or not row[t_col].strip():
        continue
    for name in ('p8_frontend', 'p8_attention', 'p8_rest', 'p8_tail'):
        if name in row[r_col]:
            t, n = ranges.setdefault(name, [0, 0])
            ranges[name][0] = t + int(float(row[t_col].replace(',', '')))
            ranges[name][1] = n + int(float(row[i_col].replace(',', '')))
    m = SUB_RE.search(row[r_col])
    if m:
        li, kind = m.group(1), m.group(2)
        t = int(float(row[t_col].replace(',', '')))
        n = int(float(row[i_col].replace(',', '')))
        d = subs.setdefault(kind, {'ns': 0, 'inst': 0, 'layers': {}})
        d['ns'] += t
        d['inst'] += n
        d['layers'][li] = d['layers'].get(li, 0) + t
with open(os.path.join(evid, 'p8_run.json')) as f:
    run = json.load(f)
iters = run['iters']
total_ns = sum(t for t, _ in ranges.values())
if not total_ns:
    sys.exit('P8 NVTX ranges not found in the kernel summary — inspect '
             + paths[-1] + ' and the timeline manually (README troubleshooting)')
fe_ns = ranges.get('p8_frontend', [0, 0])[0]
att = ranges.get('p8_attention')
result = {
    'frontend_share_gpu_time': fe_ns / total_ns,
    'boundary_mode': run['boundary_mode'],
    'iters': iters,
    'per_range': {k: {'gpu_time_ms': t / 1e6,
                      'gpu_time_ms_per_pass': t / 1e6 / iters,
                      'kernels_per_pass': n / iters}
                  for k, (t, n) in ranges.items()},
    'attention_kernels_per_pass': (att[1] / iters) if att else None,
    'wall_ms_per_pass': run['wall_ms_per_pass'],
    'source_csv': os.path.basename(paths[-1]),
    'note': ('gap structure of the attention stack: inspect '
             'trace_p8_full.nsys-rep in the nsys GUI (prereg P8; '
             'candidate-(b) first evidence)'),
}
# attention substages (fix-pass 2026-08-18 — the (b)-candidate numbers):
# per-kind totals + per-layer split; whatever the sub-ranges do NOT cover
# (each layer's pre-time pack/rearrange, post-freq unpack, skip-connection
# sums) is the residual attention_other_ms_per_pass
if subs:
    att_ns = ranges.get('p8_attention', [0, 0])[0]
    result['attention_substages'] = {
        k: {'gpu_time_ms_per_pass': d['ns'] / 1e6 / iters,
            'kernels_per_pass': d['inst'] / iters,
            'per_layer_ms_per_pass': {
                li: ns / 1e6 / iters
                for li, ns in sorted(d['layers'].items(), key=lambda kv: int(kv[0]))}}
        for k, d in subs.items()}
    result['attention_other_ms_per_pass'] = (
        (att_ns - sum(d['ns'] for d in subs.values())) / 1e6 / iters)
with open(os.path.join(evid, 'p8_result.json'), 'w') as f:
    json.dump(result, f, indent=1)
print('P8 RESULT (prereg P8 — prediction: front-end share ABOVE ~5%, ~8-15%)')
print(f"  front-end share of GPU time : {result['frontend_share_gpu_time']:.1%}")
for k, v in result['per_range'].items():
    print(f"  {k:<13}: {v['gpu_time_ms_per_pass']:9.3f} ms/pass, "
          f"{v['kernels_per_pass']:8.1f} kernels/pass")
if subs:
    print('  attention substages (per pass, summed over layers):')
    for k in ('linear_attn', 'time_attn', 'rearrange', 'freq_attn'):
        if k in result['attention_substages']:
            v = result['attention_substages'][k]
            print(f"    {k:<12}: {v['gpu_time_ms_per_pass']:9.3f} ms, "
                  f"{v['kernels_per_pass']:8.1f} kernels")
    print(f"    other/pack  : {result['attention_other_ms_per_pass']:9.3f} ms "
          f"(pre-time pack + post-freq unpack + skip sums)")
print(f"  boundary mode              : {result['boundary_mode']}")
print('  -> evidence/p8_result.json')
PYEOF
