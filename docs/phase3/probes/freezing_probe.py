"""Engineering probe (never evidence): does Inductor freezing constant-fold the autocast
fp32->fp16 parameter casts under torch 2.8.0 with cudagraph trees, and under which
settings? Builds a tiny stack of Linear layers with fp32 parameters, runs under
autocast(fp16) + no_grad, compiles with the same options as arm C3F, and inspects the
generated wrapper (TORCH_COMPILE_DEBUG dump) for *_to_copy* kernels reading arg*
inputs (the X6a cast set). Variants: requires_grad True/False,
dynamo.inline_inbuilt_nn_modules True/False. Run on a GPU with nothing else timing.
"""
import glob
import os
import re
import shutil
import sys
import tempfile

import torch


def count_casts(dump_dir):
    n = 0
    for p in glob.glob(os.path.join(dump_dir, '**', 'output_code.py'), recursive=True):
        src = open(p).read()
        n += len(re.findall(r'^\s*triton_\w*_to_copy\w*\.run\(', src, flags=re.M))
    return n


def run(requires_grad, inline, freezing, tag):
    import torch._dynamo
    torch._dynamo.reset()
    torch._dynamo.config.inline_inbuilt_nn_modules = inline
    dump = tempfile.mkdtemp(prefix=f'freeze_{tag}_')
    os.environ['TORCH_COMPILE_DEBUG'] = '1'
    os.environ['TORCH_COMPILE_DEBUG_DIR'] = dump
    os.environ['TORCHINDUCTOR_FORCE_DISABLE_CACHES'] = '1'
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
    n = count_casts(dump)
    print(f'{tag:40s} requires_grad={requires_grad} inline={inline} freezing={freezing}: cast kernels in wrapper = {n}')
    shutil.rmtree(dump, ignore_errors=True)
    return n


if __name__ == '__main__':
    print('torch', torch.__version__)
    run(True, True, False, 'baseline (as C3)')
    run(True, True, True, 'C3F settings')
    run(False, True, True, 'freezing, requires_grad=False')
    run(True, False, True, 'freezing, inline_inbuilt_nn_modules=False')
    run(False, False, True, 'freezing, no grad, no inlining')
