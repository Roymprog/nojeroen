"""Exponential backoff helper for reconnect loops."""


class Backoff:
    """1s → 2s → 4s → 8s → ... capped at `cap`. `reset()` returns to `initial`."""

    def __init__(self, initial: float = 1.0, cap: float = 30.0):
        self._initial = initial
        self._cap = cap
        self._delay = initial

    def next(self) -> float:
        delay = self._delay
        self._delay = min(self._delay * 2, self._cap)
        return delay

    def reset(self) -> None:
        self._delay = self._initial
