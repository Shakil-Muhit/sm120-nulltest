# build_spec — standing rules

Standing engineering rules for this repo (session owner, fix-pass 2026-08-18).
These are binding on every current and future change; reviewers should check
any new code against them. Each rule is stated verbatim, followed by where the
existing code obeys it.

## Rule 1 — Borrow discipline

> Borrow discipline: any tensor returned from a graph-replay path is a borrowed
> reference into a reusable buffer. Acquire borrowed references last, convert to
> owned immediately (clone() after the sync), and keep the code window between
> acquisition and clone empty. Never retain replay outputs across iterations
> without cloning.

Where the code obeys it today:
- `arms/common.py` — `validate()`: `out = step(x)` → `torch.cuda.synchronize()`
  → `out = out.clone()`, with nothing in between; the clone happens before any
  comparison and before the next pass.
- `arms/common.py` — `ReplayBandSplit.forward()` returns the static output
  buffer directly and its docstring marks it as reused; every consumer treats
  it as borrowed.
- The timing loops never retain step outputs at all — only wall times.

## Rule 2 — Capture is final

> Capture is final: graph capture is the last thing that happens to a module —
> no .to(), dtype change, or state_dict load after capture; parameter storage
> must not move once a graph holds its pointers.

Where the code obeys it today:
- `arms/run_arm.py` — order of phases: models are built with
  `build_stock_frontend()` (`.to(device)` + `.eval()` inside), the state_dict
  equality assertion runs on the built models, and only then does
  `build_arm_step()` perform capture (B/E/G). Nothing mutates the module after.
- `arms/common.py` — `_capture_with_fallback()` is the last step of arm
  construction; validation and timing only call the returned step closure.

A violation of either rule is a correctness bug even if every test passes —
the failure modes (stale borrowed buffers; a graph replaying into freed or
moved storage) are silent.
