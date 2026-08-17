"""Track where the hand is going, not just what shape it's in.

"Not quite my tempo" isn't only palm-then-fist — the palm *circles* first. Shape
alone can't tell a circling palm from a palm held still, so this keeps a short
history of where the palm has been and measures the path.

Circularity is measured as **total angle swept around the path's own centre**.
Take the recent palm positions, find their centroid, and sum how far the
position vector rotates from frame to frame. A full loop accumulates 360
degrees; a straight swipe accumulates almost none, because the vector barely
rotates. Waving side to side cancels out, since the rotation reverses sign.

Positions come from image space, with x scaled by the aspect ratio — otherwise
a real circle traces an ellipse on a 16:9 frame and the maths sees a shape
that isn't there.
"""

from __future__ import annotations

import math
from collections import deque


class MotionTracker:
    """Rolling history of palm position, with a circularity measure."""

    def __init__(self, history: int, aspect: float = 1.0) -> None:
        self.aspect = aspect
        self._points: deque[tuple[float, float, float]] = deque(maxlen=history)

    def update(self, centre: tuple[float, float] | None, now: float) -> None:
        """Add this frame's palm position, or None if no hand was seen."""
        if centre is None:
            # Don't clear the history — a hand lost to motion blur for a few
            # frames shouldn't erase the loop it was halfway through.
            return
        self._points.append((centre[0] * self.aspect, centre[1], now))

    def reset(self) -> None:
        self._points.clear()

    @property
    def samples(self) -> int:
        return len(self._points)

    @property
    def path_length(self) -> float:
        """Total distance travelled, in aspect-corrected image units."""
        total = 0.0
        for (x0, y0, _), (x1, y1, _) in zip(self._points, list(self._points)[1:]):
            total += math.hypot(x1 - x0, y1 - y0)
        return total

    @property
    def displacement(self) -> float:
        """Straight-line distance from the oldest sample to the newest."""
        if len(self._points) < 2:
            return 0.0
        x0, y0, _ = self._points[0]
        x1, y1, _ = self._points[-1]
        return math.hypot(x1 - x0, y1 - y0)

    @property
    def swept_deg(self) -> float:
        """Signed angle swept around the path's centroid, in degrees.

        Sign gives direction: positive one way round, negative the other. Take
        `abs()` unless you actually care which way the circle went.
        """
        if len(self._points) < 5:
            return 0.0
        cx = sum(p[0] for p in self._points) / len(self._points)
        cy = sum(p[1] for p in self._points) / len(self._points)

        total = 0.0
        previous: float | None = None
        for x, y, _ in self._points:
            dx, dy = x - cx, y - cy
            if math.hypot(dx, dy) < 1e-6:
                continue  # sitting on the centroid, angle is meaningless
            theta = math.atan2(dy, dx)
            if previous is not None:
                delta = theta - previous
                # Unwrap: keep each step in (-pi, pi] so crossing the +/-pi
                # boundary doesn't register as a full turn backwards.
                while delta > math.pi:
                    delta -= 2 * math.pi
                while delta < -math.pi:
                    delta += 2 * math.pi
                total += delta
            previous = theta
        return math.degrees(total)

    def is_circling(self, min_swept_deg: float, min_path: float) -> bool:
        """Has the hand gone round enough, far enough, to count as circling?

        Both tests are needed. A hand jittering in place can accumulate swept
        angle from noise alone, so it must also have actually travelled.
        """
        return abs(self.swept_deg) >= min_swept_deg and self.path_length >= min_path

    def describe(self) -> str:
        return (f"swept {self.swept_deg:+6.0f}deg  path {self.path_length:.2f}  "
                f"disp {self.displacement:.2f}")
