"""Training tracer — wraps torch.profiler to dump Chrome traces for Perfetto.

Supports two modes:

1. Single window (legacy): ``Tracer("dir", wait=500, active=10)``
   profiles once starting at step 500.

2. Multi-window: ``Tracer("dir", trace_steps=[500, 1000], active=10)``
   profiles around each specified step independently.

Usage::

    tracer = Tracer("results/run/trace", trace_steps=[500, 1000])
    for step, batch in enumerate(loader):
        # ... training step ...
        tracer.step()
    tracer.stop()
    # -> results/run/trace/trace_0.json  (step 500 window)
    #    results/run/trace/trace_1.json  (step 1000 window)
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
    """

    def __init__(
        self,
        trace_dir: str | Path,
        trace_steps: list[int] | None = None,
        wait: int = 0,
        warmup: int = 5,
        active: int = 10,
        repeat: int = 1,
    ):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._trace_count = 0
        self._trace_steps = trace_steps

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
        if self._trace_steps and self._trace_count < len(self._trace_steps):
            step = self._trace_steps[self._trace_count]
            path = self.trace_dir / f"trace_step{step}.json"
        else:
            path = self.trace_dir / f"trace_{self._trace_count}.json"
        prof.export_chrome_trace(str(path))
        self._trace_count += 1

    @property
    def trace_files(self) -> list[Path]:
        return sorted(self.trace_dir.glob("trace_*.json"))
