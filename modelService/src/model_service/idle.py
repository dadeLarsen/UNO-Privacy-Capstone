"""Free the model's GPU memory once the service has been idle (NFR-03).

A loaded model holds GPU memory that other programs could use. After the idle time with no
request, the model is released, and the next request loads it again. On the CPU the model
stays loaded: NFR-03 names GPU memory, a computer without a GPU has the least time to
spare for a reload, and a release there gives little back. Measured with the default CPU
condition, each load and release left the process about 0.5 GiB larger.
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
            metadata = self._classifier.metadata()
            if metadata is not None and str(metadata.get("device")).startswith("cuda"):
                self._classifier.release()
                logger.info("Released the model's GPU memory after %s s without a request", self.idle_seconds)
            return self.idle_seconds
