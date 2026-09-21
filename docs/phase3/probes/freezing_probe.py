"""Engineering probe (never evidence): does Inductor freezing constant-fold the autocast
fp32->fp16 parameter casts under torch 2.8.0 with cudagraph trees, and under which
settings? Each variant runs in a fresh subprocess with TORCH_COMPILE_DEBUG set BEFORE
torch is imported (the debug dump is configured at import time). A tiny stack of Linear
layers with fp32 parameters runs under autocast(fp16) + no_grad, compiled with the same
options as arm C3F; the generated wrapper is scanned for *_to_copy* kernel launches
whose inputs are graph arguments (the X6a cast set).
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

VARIANTS = [  # (tag, requires_grad, inline_inbuilt_nn_modules, freezing)
    ('baseline (as C3)', True, True, False),
    ('C3F settings', True, True, True),
    ('freezing, requires_grad=False', False, True, True),
    ('freezing, inline_inbuilt_nn_modules=False', True, False, True),
    ('freezing, no grad, no inlining', False, False, True),
]

CHILD = r'''
import os, sys, torch, torch._dynamo
requires_grad, inline, freezing = (sys.argv[1] == '1'), (sys.argv[2] == '1'), (sys.argv[3] == '1')
torch._dynamo.config.inline_inbuilt_nn_modules = inline
torch.manual_seed(0)
m = torch.nn.Sequential(*[torch.nn.Linear(1536, 1536) for _ in range(4)]).cuda().eval()
for p in m.parameters():
    p.requires_grad_(requires_grad)
opts = {'triton.cudagraphs': True}
if freezing:
    opts['freezing'] = True
c = torch.compile(m, options=opts, dynamic=False)
x = torch.randn(64, 1536, device='cuda')
with torch.no_grad(), torch.autocast('cuda', enabled=True):
    for _ in range(6):
        y = c(x)
torch.cuda.synchronize()
print('OK', y.dtype)
'''


def count_casts(dump_dir):
    files = glob.glob(os.path.join(dump_dir, '**', 'output_code.py'), recursive=True)
    n = 0
    names = set()
    for p in files:
        for line in open(p):
            m = re.match(r'\s*(triton_\w*to_copy\w*)\.run\((.*)', line)
            if m:
                n += 1
                names.add(m.group(1))
    return n, len(files), sorted(names)[:6]


def main():
    out = {}
    for tag, rg, inl, fr in VARIANTS:
        dump = tempfile.mkdtemp(prefix='freeze_probe_')
        env = dict(os.environ, TORCH_COMPILE_DEBUG='1', TORCH_COMPILE_DEBUG_DIR=dump,
                   TORCHINDUCTOR_FORCE_DISABLE_CACHES='1', TORCH_LOGS='')
        env.pop('TORCH_LOGS', None)
        r = subprocess.run([sys.executable, '-c', CHILD, '1' if rg else '0', '1' if inl else '0', '1' if fr else '0'],
                           capture_output=True, text=True, env=env, timeout=600)
        n, nfiles, names = count_casts(dump)
        out[tag] = {'requires_grad': rg, 'inline_inbuilt_nn_modules': inl, 'freezing': fr,
                    'cast_launches_in_wrapper': n, 'output_code_files': nfiles, 'cast_kernel_names': names,
                    'child_ok': 'OK' in r.stdout, 'stderr_tail': r.stderr[-300:] if 'OK' not in r.stdout else ''}
        print(f"{tag:42s} rg={rg} inline={inl} freezing={fr}: cast launches = {n} (wrappers {nfiles}) {names}")
        shutil.rmtree(dump, ignore_errors=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'freezing_probe.json'), 'w'), indent=1)


if __name__ == '__main__':
    import torch
    print('torch', torch.__version__)
    main()
