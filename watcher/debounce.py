"""Hold-to-fire debounce.

A gesture has to be held steadily for HOLD_DURATION_MS before it counts. Two
reasons: hands pass through all sorts of shapes on the way to the one you mean,
and a single-frame misread would otherwise interrupt Claude mid-answer.

After firing, the same gesture needs *both* things before it fires again: you
have to drop it and make it afresh, **and** COOLDOWN_MS has to have passed
since the last firing. Dropping and re-making alone isn't enough — that's what
stops a hand wobbling in and out of the pose from machine-gunning interrupts.

So holding a flat palm up for ten seconds sends exactly one interrupt, not two
hundred; and a deliberate second interrupt means waiting out the cooldown.
"""

from __future__ import annotations

import time


class HoldDebouncer:
    """Turns a per-frame stream of gesture names into occasional firings."""

    def __init__(self, hold_ms: int, cooldown_ms: int) -> None:
        self.hold_ms = hold_ms
        self.cooldown_ms = cooldown_ms

        self._candidate: str | None = None  # gesture currently being held
        self._since: float = 0.0  # when the hold started
        self._fired_this_hold = False  # already fired without letting go?
        self._last_fire_at: float = -1e9
        self._last_fired: str | None = None

    def update(self, gesture: str | None, now: float | None = None) -> str | None:
        """Feed one frame's classification.

        Returns the gesture name on the single frame it fires, otherwise None.
        """
        now = time.monotonic() if now is None else now

        # Gesture changed (or vanished) — start a fresh hold.
        if gesture != self._candidate:
            self._candidate = gesture
            self._since = now
            self._fired_this_hold = False
            return None

        if gesture is None or self._fired_this_hold:
            return None

        held_ms = (now - self._since) * 1000.0
        if held_ms < self.hold_ms:
            return None

        # Long enough. Respect the cooldown only when it's the same gesture
        # again — switching to a different gesture should feel immediate.
        since_last_ms = (now - self._last_fire_at) * 1000.0
        if gesture == self._last_fired and since_last_ms < self.cooldown_ms:
            return None

        self._fired_this_hold = True
        self._last_fire_at = now
        self._last_fired = gesture
        return gesture

    @property
    def charging(self) -> str | None:
        """The gesture currently building toward a fire, if any."""
        return None if self._fired_this_hold else self._candidate

    def progress(self, now: float | None = None) -> float:
        """How far along the current hold is, 0.0 to 1.0. Drives the fill bar."""
        if self._candidate is None or self._fired_this_hold:
            return 0.0
        now = time.monotonic() if now is None else now
        held_ms = (now - self._since) * 1000.0
        return min(1.0, held_ms / self.hold_ms) if self.hold_ms > 0 else 1.0
