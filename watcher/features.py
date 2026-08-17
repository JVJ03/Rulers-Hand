"""Turn 21 raw landmarks into a handful of facts a human can reason about.

This is the layer the gesture classifier reads. The point is that `gestures.py`
should say things like

    if f.extended == (False, True, True, True, True) and f.is_upright:

instead of doing trigonometry inline. If you want to add a gesture and find
yourself needing a new geometric fact, add it here as a field and it becomes
available to every gesture at once.

All distances are normalised by `scale` (wrist -> middle knuckle), so they mean
the same thing whether you're leaning into the camera or sitting back.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import hand_landmarks as hl

# Finger name order used everywhere in this file and the overlay.
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

# (tip, pip-equivalent) per finger. The thumb has no PIP so its IP joint stands
# in — it's the same idea, just one joint closer to the palm.
_TIP_AND_PIP = (
    (hl.THUMB_TIP, hl.THUMB_IP),
    (hl.INDEX_TIP, hl.INDEX_PIP),
    (hl.MIDDLE_TIP, hl.MIDDLE_PIP),
    (hl.RING_TIP, hl.RING_PIP),
    (hl.PINKY_TIP, hl.PINKY_PIP),
)


def _dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


@dataclass(frozen=True)
class HandFeatures:
    """Everything the classifier is allowed to care about, for one hand."""

    handedness: str  # "Left" or "Right", as you see it in the mirrored preview
    extended: tuple[bool, ...]  # one per finger, in FINGER_NAMES order
    scale: float  # wrist -> middle knuckle, in normalised units
    tilt_deg: float  # 0 = fingers point straight up, +/-180 = straight down
    spread: float  # mean fingertip gap, in units of `scale`
    centre: tuple[float, float]  # palm centre, normalised image coords

    @property
    def extended_count(self) -> int:
        return sum(self.extended)

    @property
    def is_upright(self) -> bool:
        """Fingers pointing roughly up — within 30 degrees of vertical."""
        return abs(self.tilt_deg) <= 30.0

    @property
    def is_flat_palm(self) -> bool:
        """All four fingers out. The thumb is deliberately ignored: whether it
        tucks or not varies hugely between people and is not worth policing."""
        return all(self.extended[1:])

    @property
    def is_fist(self) -> bool:
        """All four fingers curled. Ignores the thumb for the same reason
        `is_flat_palm` does — plenty of people rest it outside the fist."""
        return not any(self.extended[1:])

    def describe(self) -> str:
        """One-line summary, for the console and the overlay."""
        flags = "".join(
            name[0].upper() if ext else "-" for name, ext in zip(FINGER_NAMES, self.extended)
        )
        return f"{self.handedness:<5} [{flags}] tilt {self.tilt_deg:+6.1f}deg spread {self.spread:.2f}"


def extract(landmarks, handedness: str) -> HandFeatures:
    """Compute features from one hand's landmarks."""
    wrist = landmarks[hl.WRIST]
    middle_mcp = landmarks[hl.MIDDLE_MCP]

    # Everything scales against the palm length. Guard against a degenerate
    # zero so a bad frame can't produce infinities downstream.
    scale = max(_dist(wrist, middle_mcp), 1e-6)

    # A finger is extended when its tip is further from the wrist than its
    # middle joint. Comparing distances rather than y-coordinates means this
    # still works when the hand is rotated or upside down.
    extended = tuple(
        _dist(wrist, landmarks[tip]) > _dist(wrist, landmarks[pip]) for tip, pip in _TIP_AND_PIP
    )

    # Tilt of the wrist -> knuckle vector. Screen y grows downward, so negate it
    # to get a normal maths angle, then measure from straight up.
    dx = middle_mcp.x - wrist.x
    dy = -(middle_mcp.y - wrist.y)
    tilt_deg = math.degrees(math.atan2(dx, dy))

    # How far apart the four fingertips sit — separates a flat open palm from
    # fingers held together.
    tips = [landmarks[i] for i in hl.FINGER_TIPS]
    gaps = [_dist(a, b) for a, b in zip(tips, tips[1:])]
    spread = (sum(gaps) / len(gaps)) / scale if gaps else 0.0

    centre = ((wrist.x + middle_mcp.x) / 2, (wrist.y + middle_mcp.y) / 2)

    return HandFeatures(
        handedness=handedness,
        extended=extended,
        scale=scale,
        tilt_deg=tilt_deg,
        spread=spread,
        centre=centre,
    )
