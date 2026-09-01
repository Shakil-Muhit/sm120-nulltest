"""Multi-stream band-loop machinery for arms F and G (prereg arms table).

Fork/join with CUDA events:

    main stream ── record fork_event ──┬─ stream 0: bands 0, N, 2N, ...
                                       ├─ stream 1: bands 1, N+1, ...
                                       │      ...
    main stream ◄─ wait join_event[i] ─┴─ stream N-1: record join_event[i]
    main stream:   torch.stack(outs)

Correctness is unaffected by stream count: band i always runs the band-i
modules on its input slice, and outputs are collected in band order before the
stack (prereg / session-owner requirement for arm F).

The same object works eagerly (arm F, and arm G's warmup) and inside a CUDA
Graph capture (arm G): cudaEventRecord / cudaStreamWaitEvent become graph
nodes, so the recorded graph replays the fork/join stream structure.
Allocator note: Tensor.record_stream() is the documented guard for tensors
allocated on one stream and consumed on another; during graph capture it is
skipped — capture uses the graph's private memory pool, whose blocks are not
returned while the graph is alive, so the guard is unnecessary there.
"""

import torch


class MultiStreamBandLoop:
    """num_streams side streams + fork/join events, reused across iterations."""

    def __init__(self, num_streams, device):
        assert num_streams >= 1
        self.num_streams = num_streams
        self.streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
        self.fork_event = torch.cuda.Event()
        self.join_events = [torch.cuda.Event() for _ in range(num_streams)]

    def run(self, band_fns, parts):
        """Run band_fns[i](parts[i]) with band i on stream i % num_streams.

        Returns outputs in band order. Caller stacks them on the main stream.
        """
        assert len(band_fns) == len(parts)
        main = torch.cuda.current_stream()
        capturing = torch.cuda.is_current_stream_capturing()

        # fork: every side stream orders itself after main's work so far
        self.fork_event.record(main)
        for st in self.streams:
            st.wait_event(self.fork_event)

        outs = [None] * len(band_fns)
        for i, (fn, part) in enumerate(zip(band_fns, parts)):
            st = self.streams[i % self.num_streams]
            if not capturing:
                # part is a view of the folded tensor (allocated on main);
                # mark its storage as used on the side stream
                part.record_stream(st)
            with torch.cuda.stream(st):
                outs[i] = fn(part)

        # join: main waits for every side stream before consuming outputs
        for j, st in enumerate(self.streams):
            self.join_events[j].record(st)
            main.wait_event(self.join_events[j])
        if not capturing:
            for out in outs:
                # allocated on a side stream, consumed (stacked/freed) on main
                out.record_stream(main)
        return outs


class MultiStreamBandSplit(torch.nn.Module):
    """Drop-in replacement for frontend.BandSplit that runs the per-band
    RMSNorm+Linear loop across streams. Holds the ORIGINAL BandSplit (same
    parameter tensors — no copies), so weights match the eager reference.

    band_fns: one callable per band, taking that band's input slice. Arm F
    passes the stock nn.Sequential(RMSNorm, Linear) modules; arm G passes
    fused closures built in common.py.
    """

    def __init__(self, orig_band_split, band_fns, loop):
        super().__init__()
        self.orig = orig_band_split          # keeps parameters registered
        self.band_fns = band_fns
        self.loop = loop

    def forward(self, x):
        parts = x.split(self.orig.dim_inputs, dim=-1)
        outs = self.loop.run(self.band_fns, parts)
        return torch.stack(outs, dim=-2)     # same as frontend.BandSplit.forward
