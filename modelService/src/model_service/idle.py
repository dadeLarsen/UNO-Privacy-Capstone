"""Keep one model in memory at a time, and none once the service has been idle (NFR-03).

The classifier and the rewriter never share memory. Before a request uses one model, the
other is released, and a request waits while the other model still has a request under
way, so that no model is released in the middle of one. After the idle time with no
request, the loaded model is released too, and the next request loads it again.

A release runs on another thread than the one that loaded the model, and through
llama.cpp that returns the memory all the same: in eight load and release cycles of each
model, two of them with the load on a thread that had ended, the process stayed within
40 MiB of its level after the first release.
"""

from contextlib import contextmanager
import logging
from threading import Condition
import time
from typing import Protocol

logger = logging.getLogger(__name__)


class Releasable(Protocol):
    def release(self) -> None:
        """Free the model's memory. The next load brings the model back."""


class IdleRelease:
    def __init__(self, models, idle_seconds: float, *, clock=time.monotonic):
        self.idle_seconds = idle_seconds
        self._clock = clock
        # One lock over every model's count: a request starts only while no other model is in use.
        self._changed = Condition()
        self._in_use = {model: 0 for model in models}
        # The one model that a request has used since the last release. Every load happens inside in_use().
        self._loaded: Releasable | None = None
        self._idle_since = clock()

    @contextmanager
    def in_use(self, model: Releasable):
        """Keep the model loaded for the length of the block, after the other model has been released.

        The idle time starts over when the block ends.
        """
        with self._changed:
            # shortcut: a waiting request is not served in order of arrival, so a steady stream of requests for
            # one model can keep the other waiting; add a queue if the service ever gets more than one user.
            while any(count for other, count in self._in_use.items() if other is not model):
                self._changed.wait()
            if self._loaded is not None and self._loaded is not model:
                self._loaded.release()
                logger.info("Released %s before a request for %s", type(self._loaded).__name__, type(model).__name__)
            self._loaded = model
            self._in_use[model] += 1
        try:
            yield
        finally:
            with self._changed:
                self._in_use[model] -= 1
                self._idle_since = self._clock()
                self._changed.notify_all()

    def check(self) -> float:
        """Release the loaded model if its idle time has passed. Returns the seconds until the next check is due."""
        # The lock is held throughout, so no request is under way and none can start before the release ends.
        with self._changed:
            if self._loaded is None or any(self._in_use.values()):
                return self.idle_seconds
            remaining = self._idle_since + self.idle_seconds - self._clock()
            if remaining > 0:
                return remaining
            self._loaded.release()
            self._loaded = None
            logger.info("Released the model after %s s without a request", self.idle_seconds)
            return self.idle_seconds
