"""grouped_mm arch-gate probe — replicates the 2026-08-14 probe for the
version matrix (prereg P7 / "SM120 x latest stable" cell).

Checks, mirroring the original probe's format:
  - API surface: torch.nn.functional.grouped_mm and torch._grouped_mm presence
  - EMPIRICAL ragged-M (MoE pattern: 2D a, 3D b, offs on rows)
  - EMPIRICAL ragged-K (2D x 2D, offs on contraction dim — the band case)
  - dtypes: bfloat16, float16, float32 (all dtypes, both patterns — P7)

P7 verdicts printed at the end:
  - all REJECTED            -> P7 CONFIRMED (gate persists)
  - ragged-M runs only      -> gate partially lifted; claim sentence re-scopes
  - ragged-K 2D x 2D runs   -> P7 FALSIFIED + MAJOR-REVISION FLAG (prereg:
                               "escalate immediately")

Runs inside the isolated latest-stable venv (scripts/version_matrix_check.sh)
— NEVER against the arm environment (prereg environment rule).
"""

import argparse
import json
from datetime import datetime, timezone

import torch
import torch.nn.functional as F

DTYPES = (torch.bfloat16, torch.float16, torch.float32)

# band-flavored ragged sizes (K_i in the measured ~[48, 800] range)
RAGGED_K_OFFS = [48, 348, 700, 800]          # cumulative ends on the K dim
RAGGED_M_OFFS = [128, 384, 512, 1024]        # cumulative ends on the M dim


def try_call(fn, *tensors, offs):
    """RAN / REJECTED (device/dtype gate) / API_MISMATCH (our call no longer
    matches the op's signature — must NOT be counted as a persisting gate)."""
    try:
        try:
            out = fn(*tensors, offs=offs)
        except TypeError:
            out = fn(*tensors, offs)     # older/newer signature: positional offs
        torch.cuda.synchronize()
        return 'RAN', f'output shape {tuple(out.shape)} dtype {out.dtype}'
    except TypeError as e:
        return 'API_MISMATCH', f'{type(e).__name__}: {e}'
    except Exception as e:
        return 'REJECTED', f'{type(e).__name__}: {e}'


def probe_pattern(fn, pattern, dtype, device='cuda'):
    if pattern == 'ragged_M':
        # MoE: 2D a (M_total, K), 3D b (G, K, N), offs on rows
        a = torch.randn(RAGGED_M_OFFS[-1], 256, device=device, dtype=dtype)
        b = torch.randn(len(RAGGED_M_OFFS), 256, 384, device=device, dtype=dtype)
        offs = torch.tensor(RAGGED_M_OFFS, device=device, dtype=torch.int32)
    else:
        # ragged_K: 2D a (M, K_total), 2D b (K_total, N), offs on contraction
        a = torch.randn(801, RAGGED_K_OFFS[-1], device=device, dtype=dtype)
        b = torch.randn(RAGGED_K_OFFS[-1], 384, device=device, dtype=dtype)
        offs = torch.tensor(RAGGED_K_OFFS, device=device, dtype=torch.int32)
    outcome, detail = try_call(fn, a, b, offs=offs)
    return {'pattern': pattern, 'dtype': str(dtype), 'outcome': outcome,
            'detail': detail,
            'shapes': {'a': list(a.shape), 'b': list(b.shape),
                       'offs': offs.tolist()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=None, help='write results here')
    args = ap.parse_args()

    sep = '=' * 64
    print(sep + f'\nPROBE RUN {datetime.now(timezone.utc).isoformat()}\n' + sep)
    env = {
        'torch': torch.__version__,
        'cuda_build': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'capability': list(torch.cuda.get_device_capability(0))
                      if torch.cuda.is_available() else None,
        'arch_list': torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
    }
    for k, v in env.items():
        print(f'{k:<21}: {v}')

    print('\n' + sep + '\nAPI SURFACE\n' + sep)
    ops = {}
    f_op = getattr(F, 'grouped_mm', None)
    u_op = getattr(torch, '_grouped_mm', None)
    print(f'torch.nn.functional.grouped_mm : '
          f'{"PRESENT" if f_op else "ABSENT"}')
    print(f'torch._grouped_mm              : '
          f'{"PRESENT" if u_op else "ABSENT"}')
    if f_op:
        ops['F.grouped_mm'] = f_op
    if u_op:
        ops['torch._grouped_mm'] = u_op

    results = []
    if not env['cuda_available']:
        print('\nCUDA UNAVAILABLE — empirical section skipped '
              '(default wheel may lack this arch; version_matrix_check.sh '
              'retries with the cu128 index)')
    elif not ops:
        print('\nno grouped_mm op present — empirical section skipped')
    else:
        for op_name, op in ops.items():
            for pattern, title in (
                    ('ragged_M', 'RAGGED-M (MoE pattern: 2D a, 3D b, offs on rows)'),
                    ('ragged_K', 'RAGGED-K (2D x 2D, offs on contraction dim — the band case)')):
                print('\n' + sep + f'\nEMPIRICAL [{op_name}]: {title}\n' + sep)
                for dtype in DTYPES:
                    r = probe_pattern(op, pattern, dtype)
                    r['op'] = op_name
                    results.append(r)
                    print(f'--- dtype = {dtype} ---')
                    print(f'    {r["outcome"]}: {r["detail"]}')

    # P7 verdict
    print('\n' + sep + '\nP7 VERDICT\n' + sep)
    ran_k = [r for r in results if r['pattern'] == 'ragged_K' and r['outcome'] == 'RAN']
    ran_m = [r for r in results if r['pattern'] == 'ragged_M' and r['outcome'] == 'RAN']
    mismatched = [r for r in results if r['outcome'] == 'API_MISMATCH']
    if not results:
        verdict = 'INCONCLUSIVE: empirical section did not run'
    elif mismatched and not (ran_k or ran_m):
        verdict = ('INCONCLUSIVE: probe call signature no longer matches the '
                   'op (API_MISMATCH) — update the probe before drawing a P7 '
                   'conclusion; do NOT read this as a persisting gate')
    elif ran_k:
        verdict = ('P7 FALSIFIED + MAJOR-REVISION FLAG: 2D x 2D offs-on-K RAN '
                   'on this device — prereg says ESCALATE IMMEDIATELY')
    elif ran_m:
        verdict = ('P7 partially falsified: gate lifted for ragged-M only — '
                   'in-tree op joins the baseline table, claim sentence re-scopes')
    else:
        verdict = 'P7 CONFIRMED: arch gate persists (all dtypes, both patterns REJECTED)'
    print(verdict)

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'env': env,
                       'api_surface': {'F.grouped_mm': bool(f_op),
                                       'torch._grouped_mm': bool(u_op)},
                       'results': results,
                       'verdict': verdict}, f, indent=1)
        print(f'-> {args.json}')


if __name__ == '__main__':
    main()
