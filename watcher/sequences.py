"""Detect gestures that are a *sequence* of shapes, not a single held pose.

"Not quite my tempo" is an open palm circling, then a closed fist. No single
frame identifies it — an open palm on its own is just an open palm, and a fist
on its own is just a fist. What makes it the gesture is one following the other.

So `gestures.classify_shape()` names the shape in the current frame, and this
watches that stream for an ordered pattern.

Two tolerances matter, both learned from real video:

- **Frames where no hand is found do not reset progress.** A hand moving fast
  enough to motion-blur drops out of detection for a few frames. In the
  reference clip the quick repetition lost its entire open-palm phase that way.
- **Unrelated shapes in between are ignored.** Going from open palm to fist
  passes through half-closed shapes; insisting on adjacency would never match.

Only the window timing out resets the sequence.
"""

from __future__ import annotations

import time


class SequenceDetector:
    """Fires when a list of shapes is seen in order, inside a time window."""

    def __init__(
        self,
        name: str,
        steps: list[str],
        window_ms: int,
        cooldown_ms: int,
    ) -> None:
        if len(steps) < 2:
            raise ValueError("a sequence needs at least two steps")
        self.name = name
        self.steps = steps
        self.window_ms = window_ms
        self.cooldown_ms = cooldown_ms

        self._index = 0  # how many steps matched so far
        self._started: float = 0.0
        self._last_fire_at: float = -1e9

    def update(self, shape: str | None, now: float | None = None) -> str | None:
        """Feed one frame's shape. Returns the gesture name when it fires."""
        now = time.monotonic() if now is None else now

        # Time out a half-finished sequence.
        if self._index > 0 and (now - self._started) * 1000.0 > self.window_ms:
            self._index = 0

        if shape is None:
            return None  # no hand this frame — hold position, don't reset

        # Waiting for the first step.
        if self._index == 0:
            if shape == self.steps[0]:
                self._index = 1
                self._started = now
            return None

        # Re-seeing the current step is fine — you hold the palm up for a while.
        if shape == self.steps[self._index - 1]:
            return None

        # The next step in the sequence?
        if shape == self.steps[self._index]:
            self._index += 1
            if self._index < len(self.steps):
                return None
            # Complete.
            self._index = 0
            if (now - self._last_fire_at) * 1000.0 < self.cooldown_ms:
                return None
            self._last_fire_at = now
            return self.name

        # Some other shape: ignore it, the transition passes through junk.
        return None

    @property
    def matched(self) -> int:
        """How many steps of the sequence are done. Drives the step pips."""
        return self._index

    def progress(self, now: float | None = None) -> float:
        """Fraction of the sequence completed, for the progress bar."""
        if self._index == 0:
            return 0.0
        return self._index / len(self.steps)

    def describe(self) -> str:
        """e.g. 'OPEN_PALM > FIST', with the matched part marked."""
        parts = []
        for i, step in enumerate(self.steps):
            parts.append(f"[{step}]" if i < self._index else step)
        return " > ".join(parts)
