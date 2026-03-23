"""Training tracer — wraps torch.profiler to dump Chrome traces for Perfetto.

Usage::

    tracer = Tracer("results/run/trace", warmup=5, active=10)
    for batch in loader:
        # ... training step ...
        tracer.step()
    tracer.stop()
    # -> results/run/trace/trace_0.json  (loadable in ui.perfetto.dev)
"""
from __future__ import annotations

from pathlib import Path

from torch.profiler import ProfilerActivity, profile, schedule


class Tracer:
    """Thin wrapper around torch.profiler with step-based schedule.

    Args:
        trace_dir: directory to write Chrome trace JSON files.
        warmup: profiler warmup steps (collecting overhead, not recorded).
        active: steps to actively profile after warmup.
        repeat: number of warmup+active cycles (default 1).
    """

    def __init__(
        self,
        trace_dir: str | Path,
        warmup: int = 5,
        active: int = 10,
        repeat: int = 1,
    ):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._trace_count = 0

        self._prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=0, warmup=warmup, active=active, repeat=repeat),
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
        path = self.trace_dir / f"trace_{self._trace_count}.json"
        prof.export_chrome_trace(str(path))
        self._trace_count += 1

    @property
    def trace_files(self) -> list[Path]:
        return sorted(self.trace_dir.glob("trace_*.json"))
