"""Free the model's memory once the service has been idle (NFR-03).

A loaded model holds memory that other programs could use: GPU memory on the GPU, system
RAM on the CPU. After the idle time with no request, the model is released, and the next
request loads it again. The check runs on another thread than the one that loaded the
model, and through llama.cpp that returns the memory all the same: in eight load and
release cycles of each default condition, two of them with the load on a thread that had
ended, the process stayed within 40 MiB of its level after the first release.
"""

from contextlib import contextmanager
import logging
from threading import Lock
import time

from .classifier import Classifier

logger = logging.getLogger(__name__)


class IdleRelease:
    def __init__(self, classifier: Classifier, idle_seconds: float, *, clock=time.monotonic):
        self.idle_seconds = idle_seconds
        self._classifier = classifier
        self._clock = clock
        self._lock = Lock()
        self._in_use = 0
        # When the model was last in use, or None once check() has acted on that.
        self._idle_since: float | None = clock()

    @contextmanager
    def in_use(self):
        """Keep the model loaded for the length of the block; the idle time starts over when it ends."""
        with self._lock:
            self._in_use += 1
        try:
            yield
        finally:
            with self._lock:
                self._in_use -= 1
                self._idle_since = self._clock()

    def check(self) -> float:
        """Release the model if its idle time has passed. Returns the seconds until the next check is due."""
        # The lock is held throughout, so no request is under way and none can start before the release ends.
        with self._lock:
            if self._in_use or self._idle_since is None:
                return self.idle_seconds
            remaining = self._idle_since + self.idle_seconds - self._clock()
            if remaining > 0:
                return remaining
            self._idle_since = None
            if self._classifier.metadata() is not None:
                self._classifier.release()
                logger.info("Released the model after %s s without a request", self.idle_seconds)
            return self.idle_seconds
