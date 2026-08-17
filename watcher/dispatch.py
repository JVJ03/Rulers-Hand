"""Send confirmed gestures to the VS Code extension.

Fire-and-forget over plain HTTP to 127.0.0.1. Two things matter here:

- **It must never block the capture loop.** A gesture POST that hangs would
  freeze the preview and drop frames, so the timeout is deliberately tiny and
  the send happens on a worker thread.
- **It must never crash the watcher.** If the extension isn't running, that's
  a normal state, not an error — you'll often have the watcher up before you
  press F5. Failures are counted and reported, not raised.
"""

from __future__ import annotations

import threading
from queue import Empty, Queue

import requests

import config

# Short enough that a wedged server can't stall gesture handling.
TIMEOUT_S = 0.5


class Dispatcher:
    """Posts gestures on a background thread so capture never waits on I/O."""

    def __init__(self, url: str | None = None) -> None:
        self.url = url or config.SERVER_URL
        self._queue: Queue[dict] = Queue(maxsize=32)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="dispatch", daemon=True)

        self.sent = 0
        self.failed = 0
        self.last_error: str | None = None

        self._thread.start()

    def send(self, gesture: str, held_ms: int) -> None:
        """Queue a gesture. Returns immediately; never raises."""
        payload = {"gesture": gesture, "held_ms": held_ms}
        try:
            self._queue.put_nowait(payload)
        except Exception:
            # Queue full means the server is wedged. Dropping is the right
            # response — a backlog of stale interrupts helps nobody.
            self.failed += 1
            self.last_error = "queue full"

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                payload = self._queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                requests.post(self.url, json=payload, timeout=TIMEOUT_S)
                self.sent += 1
                self.last_error = None
            except requests.exceptions.ConnectionError:
                self.failed += 1
                self.last_error = "extension not running"
            except requests.exceptions.Timeout:
                self.failed += 1
                self.last_error = "timed out"
            except Exception as exc:
                self.failed += 1
                self.last_error = type(exc).__name__

    @property
    def status(self) -> str:
        """One-line summary for the overlay."""
        if self.sent == 0 and self.failed == 0:
            return "idle"
        if self.last_error:
            return f"{self.last_error} ({self.failed} failed)"
        return f"sent {self.sent}"

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
