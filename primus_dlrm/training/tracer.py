"""Training tracer — wraps torch.profiler to dump Chrome traces for Perfetto.

Supports two modes:

1. Single window (legacy): ``Tracer("dir", wait=500, active=10)``
   profiles once starting at step 500.

2. Multi-window: ``Tracer("dir", trace_steps=[500, 1000], active=10)``
   profiles around each specified step independently.

Trace filenames embed the (optional) global ``rank`` so multi-rank captures
don't collide::

    tracer = Tracer("results/run/trace", trace_steps=[500, 1000], rank=3)
    # -> results/run/trace/trace_step500_rank3.json
    #    results/run/trace/trace_step1000_rank3.json

Multi-rank capture is orchestrated by the trainer: instantiate one Tracer per
participating rank with that rank's id, all writing to the same ``trace_dir``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from torch.profiler import ProfilerAction, ProfilerActivity, profile, schedule


def _multi_window_schedule(
    trace_steps: Sequence[int],
    warmup: int,
    active: int,
):
    """Custom schedule that profiles at each specified step.

    For each step s in trace_steps, the profiler does:
        [s - warmup, s)        -> WARMUP
        [s, s + active - 1)    -> RECORD
        s + active - 1         -> RECORD_AND_SAVE  (triggers on_trace_ready)

    Steps outside any window return NONE (zero overhead).
    """
    windows = []
    for s in sorted(trace_steps):
        warmup_start = s - warmup
        active_end = s + active
        windows.append((warmup_start, s, active_end))

    def schedule_fn(step: int) -> ProfilerAction:
        for warmup_start, active_start, active_end in windows:
            if warmup_start <= step < active_start:
                return ProfilerAction.WARMUP
            if active_start <= step < active_end - 1:
                return ProfilerAction.RECORD
            if step == active_end - 1:
                return ProfilerAction.RECORD_AND_SAVE
        return ProfilerAction.NONE

    return schedule_fn


class Tracer:
    """Thin wrapper around torch.profiler with step-based schedule.

    Args:
        trace_dir: directory to write Chrome trace JSON files.
        trace_steps: list of steps at which to start profiling windows.
            If provided, overrides ``wait`` and ``repeat``.
        wait: steps to skip before warmup (single-window mode).
        warmup: profiler warmup steps before each active window.
        active: steps to actively profile in each window.
        repeat: number of (wait+warmup+active) cycles (single-window mode).
        rank: optional global rank used in trace filenames so multi-rank
            captures don't collide. ``None`` omits the rank suffix (legacy).
    """

    def __init__(
        self,
        trace_dir: str | Path,
        trace_steps: list[int] | None = None,
        wait: int = 0,
        warmup: int = 5,
        active: int = 10,
        repeat: int = 1,
        rank: int | None = None,
    ):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._trace_count = 0
        self._trace_steps = trace_steps
        self._rank = rank

        if trace_steps:
            sched = _multi_window_schedule(trace_steps, warmup, active)
        else:
            sched = schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)

        self._prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=sched,
            on_trace_ready=self._export,
            record_shapes=True,
            with_stack=False,
        )
        self._prof.__enter__()

    def step(self):
        self._prof.step()

    def stop(self):
        self._prof.__exit__(None, None, None)

    def _export(self, prof):
        suffix = f"_rank{self._rank}" if self._rank is not None else ""
        if self._trace_steps and self._trace_count < len(self._trace_steps):
            step = self._trace_steps[self._trace_count]
            path = self.trace_dir / f"trace_step{step}{suffix}.json"
        else:
            path = self.trace_dir / f"trace_{self._trace_count}{suffix}.json"
        prof.export_chrome_trace(str(path))
        self._trace_count += 1

    @property
    def trace_files(self) -> list[Path]:
        return sorted(self.trace_dir.glob("trace_*.json"))


def parse_trace_ranks(spec: str | None) -> set[int] | None:
    """Parse a ``--trace-ranks`` spec into a set of global ranks (or ``None``).

    Accepts ``None``/``""``/``"0"`` (default — only rank 0), a comma-separated
    list of ints (``"0,3,5"``), or the literal ``"all"`` (returns ``None``,
    meaning *every* rank should trace).
    """
    if spec is None or spec == "":
        return {0}
    s = spec.strip().lower()
    if s == "all":
        return None
    out: set[int] = set()
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.add(int(tok))
    return out or {0}
