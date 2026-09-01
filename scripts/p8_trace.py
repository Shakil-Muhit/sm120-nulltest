"""P8 full-forward trace driver (prereg P8 — session opener, before any arm).

Provenance (AUDIT-MAP[5]): the full model comes ONLY from the public
Music-Source-Separation-Training clone (pinned commit — setup.sh [5/6]) via
the repo's OWN construction path, utils.settings.get_model_from_config, plus
the hash-verified public checkpoint (setup.sh [6/6] / fetch_checkpoint.sh).
Loading mirrors the public repo's inference.py (lines 211-216 at the pinned
commit): get_model_from_config -> torch.load(..., weights_only=False,
map_location='cpu') -> load_start_checkpoint(..., type_='inference').
No other model source exists anywhere in this repo (scope guard: arms A-G
never import any of this — they use frontend.py only).

NVTX stage boundaries: the public path exposes no NVTX stages, so the
front-end boundary is the KERNEL-SEQUENCE boundary made explicit with
EXTERNAL forward hooks (public torch API — no repo code is modified):

    model pre-hook                  -> push 'p8_frontend'
    model.band_split post-hook      -> pop,  push 'p8_attention' (or 'p8_rest')
    last-transformer post-hook      -> pop,  push 'p8_tail'      (if resolvable)
    model post-hook                 -> pop

The attention-stack END boundary assumes the final module of the final
model.layers block executes last in forward() (true at the pinned commit); if
that submodule cannot be resolved, the script falls back to a two-range split
(frontend / rest) and says so. 'boundary_mode' in the run JSON records which
split was realized — documented per the session-owner instruction ("document
which").

Invoked by scripts/p8_full_trace.sh under nsys with
--capture-range=cudaProfilerApi (frontend.py's capture-range pattern):
warmups run OUTSIDE the range, then torch.cuda.profiler.start(), the measured
passes, stop.
"""

import argparse
import json
import subprocess
import sys
import time

import torch


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--msst-dir', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--model-type', default='mel_band_roformer')
    ap.add_argument('--chunk', type=int, default=352800,
                    help='PREREG P8: the 8 s chunk')
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--json', required=True, help='run-metadata output path')
    return ap.parse_args()


def install_nvtx_hooks(model):
    """External NVTX boundary hooks; returns (handles, boundary_mode).

    Outer ranges (unchanged): p8_frontend / p8_attention / p8_tail.

    Sub-ranges (fix-pass 2026-08-18 — instruments the (b)-candidate so the
    registered questions / P-B1/P-B2 are answerable): nested inside
    p8_attention, one set per transformer block i:
        p8_L<i>_linear_attn  the linear transformer (only when the block has 3
                             modules — linear_transformer_depth > 0)
        p8_L<i>_time_attn    the time-axis Transformer call
        p8_L<i>_rearrange    the unpack -> rearrange -> pack between the time
                             and freq calls (bracketed by time's post-hook and
                             freq's pre-hook — module hooks cannot see inside
                             forward(), but they CAN bracket the gap between
                             two submodule calls)
        p8_L<i>_freq_attn    the freq-axis Transformer call
    Each block's pre-time pack/rearrange and post-freq unpack stay in the
    parent p8_attention range; extraction reports that residual as
    attention_other. boundary_mode records the realized granularity, e.g.
    'frontend/attention/tail;sub=per_layer(time_attn,rearrange,freq_attn)x6'
    or ';sub=none' when the block structure was not recognized.

    Registration-order invariant: sub-hooks on layers[-1][-1] are registered
    BEFORE after_last_transformer, so on the final freq module the sub-range
    pop fires first and the attention->tail transition stays balanced
    (forward hooks on one module fire in registration order).
    """
    nvtx = torch.cuda.nvtx
    handles = []

    # can we mark the end of the attention stack?
    last_tx = None
    try:
        last_tx = model.layers[-1][-1]
        if not isinstance(last_tx, torch.nn.Module):
            last_tx = None
    except (AttributeError, IndexError, TypeError):
        last_tx = None
    outer_mode = ('frontend/attention/tail' if last_tx is not None
                  else 'frontend/rest')
    mid_label = 'p8_attention' if last_tx is not None else 'p8_rest'

    # hooks MUST return None: a non-None return from a forward (pre-)hook
    # replaces the module's output (input), and nvtx range_push/range_pop
    # return ints — so these are plain functions, never bare lambdas
    def open_frontend(module, inputs):
        nvtx.range_push('p8_frontend')

    def after_band_split(module, inputs, output):
        nvtx.range_pop()
        nvtx.range_push(mid_label)

    def after_last_transformer(module, inputs, output):
        nvtx.range_pop()
        nvtx.range_push('p8_tail')

    def close_forward(module, inputs, output):
        nvtx.range_pop()

    # None-returning hook factories for the sub-ranges
    def mk_push(name):
        def pre_hook(module, inputs):
            nvtx.range_push(name)
        return pre_hook

    def mk_pop():
        def post_hook(module, inputs, output):
            nvtx.range_pop()
        return post_hook

    def mk_pop_push(name):
        def post_hook(module, inputs, output):
            nvtx.range_pop()
            nvtx.range_push(name)
        return post_hook

    def mk_pre_pop_push(name):
        def pre_hook(module, inputs):
            nvtx.range_pop()          # closes p8_L<i>_rearrange
            nvtx.range_push(name)
        return pre_hook

    handles.append(model.register_forward_pre_hook(open_frontend))
    handles.append(model.band_split.register_forward_hook(after_band_split))

    # sub-range hooks — only when the outer 3-range split is realized AND every
    # block matches the pinned commit's structure: nn.ModuleList of
    # [time, freq] or [linear, time, freq] (forward executes them in order,
    # with rearrange/pack between time and freq)
    sub_mode = 'none'
    if last_tx is not None:
        try:
            blocks = list(model.layers)
            shapes_ok = bool(blocks) and all(
                isinstance(b, torch.nn.ModuleList) and len(b) in (2, 3)
                and all(isinstance(m, torch.nn.Module) for m in b)
                for b in blocks)
        except (AttributeError, TypeError):
            blocks, shapes_ok = [], False
        if shapes_ok:
            has_linear = False
            for i, block in enumerate(blocks):
                if len(block) == 3:
                    has_linear = True
                    handles.append(block[0].register_forward_pre_hook(
                        mk_push(f'p8_L{i}_linear_attn')))
                    handles.append(block[0].register_forward_hook(mk_pop()))
                time_tx, freq_tx = block[-2], block[-1]
                handles.append(time_tx.register_forward_pre_hook(
                    mk_push(f'p8_L{i}_time_attn')))
                handles.append(time_tx.register_forward_hook(
                    mk_pop_push(f'p8_L{i}_rearrange')))
                handles.append(freq_tx.register_forward_pre_hook(
                    mk_pre_pop_push(f'p8_L{i}_freq_attn')))
                handles.append(freq_tx.register_forward_hook(mk_pop()))
            kinds = 'linear_attn,time_attn,rearrange,freq_attn' if has_linear \
                    else 'time_attn,rearrange,freq_attn'
            sub_mode = f'per_layer({kinds})x{len(blocks)}'

    # AFTER the sub-hooks: on the last freq module the sub pop must fire first
    if last_tx is not None:
        handles.append(last_tx.register_forward_hook(after_last_transformer))
    handles.append(model.register_forward_hook(close_forward))
    boundary_mode = f'{outer_mode};sub={sub_mode}'
    return handles, boundary_mode


def main():
    args = parse_args()
    assert torch.cuda.is_available(), 'P8 trace needs the GPU'

    # the public repo's own construction path — no local model code
    sys.path.insert(0, args.msst_dir)
    from utils.settings import get_model_from_config
    from utils.model_utils import load_start_checkpoint

    model, config = get_model_from_config(args.model_type, args.config)
    # mirrors the public repo's inference.py checkpoint-load lines exactly
    ckpt_ns = argparse.Namespace(
        start_check_point=args.ckpt, model_type=args.model_type,
        lora_checkpoint='', lora_checkpoint_loralib='')
    checkpoint = torch.load(args.ckpt, weights_only=False, map_location='cpu')
    load_start_checkpoint(ckpt_ns, model, checkpoint, type_='inference')
    model = model.cuda().eval()

    # Precision: mirror the public repo's OWN inference path — demix() in
    # utils/model_utils.py (lines 137-139 at the pinned commit) runs the
    # forward inside torch.cuda.amp.autocast(enabled=config.training.use_amp),
    # default True. Plain fp32 would ABORT here: the public config sets
    # flash_attn=true and attend.py forces the flash SDPA backend, which has
    # no fp32 kernels ("No available kernel" — 2026-08-17 smoke finding).
    use_amp = bool(getattr(getattr(config, 'training', object()), 'use_amp', True))

    stereo = bool(getattr(model, 'stereo', False))
    x = torch.randn(1, 2 if stereo else 1, args.chunk, device='cuda')
    print(f'[p8] model={args.model_type} stereo={stereo} '
          f'input={tuple(x.shape)} warmup={args.warmup} iters={args.iters} '
          f'use_amp={use_amp} (repo inference default)')

    handles, boundary_mode = install_nvtx_hooks(model)
    print(f'[p8] NVTX boundary mode: {boundary_mode}')

    wall_ms = []
    # same autocast wrapper as the public repo's demix() — see note above
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        for _ in range(args.warmup):                 # outside the capture range
            model(x)
        torch.cuda.synchronize()

        torch.cuda.profiler.start()                  # nsys capture range: open
        for _ in range(args.iters):
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            wall_ms.append((time.perf_counter() - t0) * 1e3)
        torch.cuda.profiler.stop()                   # nsys capture range: close
    for h in handles:
        h.remove()

    try:
        msst_commit = subprocess.run(
            ['git', '-C', args.msst_dir, 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        msst_commit = 'unavailable'
    if not msst_commit.strip():
        # git exited nonzero without raising (no check=True): not-a-repo,
        # dubious-ownership refusal — empty stdout is equally "unresolved"
        msst_commit = 'unavailable'
    if msst_commit == 'unavailable':
        # fix-pass 2026-08-18: a quiet JSON string is not enough — a trace
        # whose upstream identity is unknown defeats this script's purpose
        print('=' * 72, file=sys.stderr)
        print('WARNING: could not resolve the msst clone commit hash — the\n'
              'upstream identity of this P8 run is UNKNOWN. The trace is NOT\n'
              'provenance-clean until the pin is recorded by hand '
              '(git -C <msst> rev-parse HEAD).', file=sys.stderr)
        print('=' * 72, file=sys.stderr)

    meta = {
        'chunk': args.chunk, 'warmup': args.warmup, 'iters': args.iters,
        'use_amp': use_amp,
        'boundary_mode': boundary_mode,
        'model_type': args.model_type,
        'ckpt': args.ckpt, 'config': args.config,
        'msst_dir': args.msst_dir, 'msst_commit': msst_commit,
        'stereo': stereo,
        'wall_ms_per_pass': wall_ms,
        'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name(0),
    }
    with open(args.json, 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'[p8] passes: {[round(w, 2) for w in wall_ms]} ms -> {args.json}')


if __name__ == '__main__':
    main()
