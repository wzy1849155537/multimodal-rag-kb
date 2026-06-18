"""Context manager for timing operations."""

import time
from typing import Optional


class Timer:
    """Context manager for measuring elapsed time.

    Usage:
        with Timer() as t:
            do_work()
        print(f"Took {t.elapsed_ms:.0f}ms")
    """

    def __init__(self, label: Optional[str] = None):
        self.label = label
        self._start: float = 0.0
        self._end: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        end = self._end if self._end > 0 else time.perf_counter()
        return end - self._start

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed * 1000

    def __repr__(self) -> str:
        label_str = f" [{self.label}]" if self.label else ""
        return f"Timer{label_str}: {self.elapsed_ms:.1f}ms"
