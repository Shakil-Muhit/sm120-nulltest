#!/bin/bash
# PHASE-3 SMOKE (engineering only, never evidence): builds every new arm once at 4410
# with --no-manifest, one A/A matched pair, one E5 variant, the boundary bench with
# the E7 additions, and proves the E9 held-out guard refuses off-pod runs.
set -u
. "$HOME/axial/fullmodel_smoke/env.sh"
REPO="$HOME/axial/sm120-nulltest"
EVID="${EVID:-$HOME/axial/fullmodel_smoke/phase3_$(date +%Y%m%d_%H%M)}"
mkdir -p "$EVID"
cd "$REPO"
exec > >(tee -a "$EVID/smoke.log") 2>&1
echo "PHASE3 SMOKE START $(date -Is) evid=$EVID gpu=$CUDA_VISIBLE_DEVICES"
$PY -c "import torch;print('torch',torch.__version__, torch.cuda.get_device_name(0))"
run() { echo "--- $* $(date -Is)"; timeout -k 30 1500 "$@"; echo "    exit $?"; }
export TORCH_LOGS="graph_breaks,recompiles,cudagraphs"
run $PY fullmodel/run_full_arm.py --arm A --chunk 4410 --tag ref --save-reference --sdr --skip-timing --no-manifest --evidence-dir "$EVID" --clips-dir "$CLIPS_DIR"
for arm in C3F C3P C3M C1P C3N; do
    run $PY fullmodel/run_full_arm.py --arm "$arm" --chunk 4410 --tag smoke --no-manifest --evidence-dir "$EVID" --clips-dir "$CLIPS_DIR"
done
run $PY fullmodel/run_full_arm.py --arm C3N --chunk 4410 --tag smoke_sdr --sdr --skip-timing --no-manifest --evidence-dir "$EVID" --clips-dir "$CLIPS_DIR"
run $PY fullmodel/matched_run.py --pair C3 C3 --chunk 4410 --tag aa --no-manifest --evidence-dir "$EVID"
run $PY fullmodel/run_full_arm.py --arm A --chunk 4410 --tag v_d1_mquarter_ref --variant d1_mquarter --save-reference --skip-timing --no-manifest --evidence-dir "$EVID"
run $PY fullmodel/run_full_arm.py --arm C3 --chunk 4410 --tag v_d1_mquarter_r1 --variant d1_mquarter --no-manifest --evidence-dir "$EVID"
run $PY fullmodel/run_full_arm.py --arm H --chunk 4410 --tag v_d1_mquarter_r1 --variant d1_mquarter --no-manifest --evidence-dir "$EVID"
run $PY fullmodel/boundary_bench.py --out "$EVID/boundary_bench.json"
echo "--- held-out guard (must refuse)"; $PY fullmodel/run_full_arm.py --arm A --chunk 176400 --tag guard --no-manifest --evidence-dir "$EVID" --skip-timing; echo "    exit $? (non-zero expected)"
run $PY fullmodel/e_tools.py aa_stats --evidence-dir "$EVID" --out "$EVID/aa_stats.json"
run $PY fullmodel/e_tools.py e5_summary --evidence-dir "$EVID" --out "$EVID/e5_summary.json"
echo "=== summary"
$PY - "$EVID" <<'PYEOF'
import glob, json, os, sys
ev = sys.argv[1]
for p in sorted(glob.glob(os.path.join(ev, 'full_*.json')) + glob.glob(os.path.join(ev, 'matched_*.json'))):
    r = json.load(open(p))
    if 'pair' in r:
        print(os.path.basename(p), r.get('status'), 'eligible', r.get('pair_eligible'), r.get('ineligibility_reasons'), (r.get('matched_delta_ms') or {}).get('median'))
    else:
        print(os.path.basename(p), r.get('status'), (r.get('validation') or {}).get('status'), 'tier', r.get('tier'), 'mode', r.get('compile_mode'),
              'gate', (r.get('graph_gate') or {}).get('status'), 'breaks', (r.get('graph_gate') or {}).get('graph_break_count'),
              'mom', r.get('min_of_medians_ms'), 'sdr', (r.get('sdr_gate') or {}).get('status'), (r.get('sdr_gate') or {}).get('mean_delta_sdr_ref_db'),
              'cfg', r.get('inductor_config'), 'notes', (r.get('notes') or [])[:2], 'err', (r.get('error') or '')[-300:])
bb = json.load(open(os.path.join(ev, 'boundary_bench.json')))
print('pdl_ptx', bb.get('pdl_ptx'), bb.get('pdl_ptx_prologue'), 'sass', {k: v for k, v in (bb.get('pdl_sass') or {}).items() if k != 'matching_lines'})
print('e7', json.dumps(bb.get('e7_pdl_saving'), indent=1))
print('attrs', bb.get('pdl_launch_attributes'))
PYEOF
echo "PHASE3 SMOKE DONE $(date -Is)"
