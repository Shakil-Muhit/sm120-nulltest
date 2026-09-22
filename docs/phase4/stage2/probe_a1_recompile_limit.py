"""Probe A1 (Stage 2, engineering): run the matched runner with dynamo's recompile limit set from the environment
(PROBE_RECOMPILE_LIMIT, default 8 = torch 2.8 default) so that the second-built C1-family arm's fallback can be
switched off by intervention. Everything else is the unchanged matched runner (--no-manifest: never evidence)."""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(_ROOT, 'fullmodel'))
import torch
import torch._dynamo
lim = int(os.environ.get('PROBE_RECOMPILE_LIMIT', '8'))
torch._dynamo.config.recompile_limit = lim
torch._dynamo.config.cache_size_limit = lim
torch._dynamo.config.accumulated_recompile_limit = max(256, 4 * lim)
print(f'[probe_a1] recompile_limit={torch._dynamo.config.recompile_limit} cache_size_limit={torch._dynamo.config.cache_size_limit} accumulated={torch._dynamo.config.accumulated_recompile_limit}', flush=True)
import matched_run
sys.argv = ['matched_run.py'] + sys.argv[1:]
matched_run.main()
