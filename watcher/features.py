"""Turn 21 landmarks into a handful of facts a human can reason about.

This is the layer the gesture classifier reads. The point is that `gestures.py`
should say things like

    if f.is_flat_palm and f.spread > 0.22:

instead of doing trigonometry inline. If you want to add a gesture and find
yourself needing a new geometric fact, add it here as a field and it becomes
available to every gesture at once.

**Geometry uses MediaPipe's world landmarks, not the image ones.** This matters
enormously. Image landmarks put x in [0,1] across the frame *width* and y in
[0,1] across the *height*, so at 16:9 everything is horizontally stretched and
all depth is thrown away. Measured against the reference clip, an angle
computed that way differed from the true angle **by up to 84 degrees**, and it
was why a hand read differently depending on whether the palm or the back faced
the camera — the hand hadn't changed, only its projection had.

World landmarks are metric, roughly in meters, with the origin at the hand's
geometric centre. Angles and distances from them are real.

Image landmarks are still used for anything that is genuinely about the frame:
where the hand sits on screen, and drawing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import config
import hand_landmarks as hl

# Finger name order used everywhere in this file and the overlay.
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

# (tip, pip-equivalent) per finger. The thumb has no PIP so its IP joint stands
# in — same idea, one joint closer to the palm.
_TIP_AND_PIP = (
    (hl.THUMB_TIP, hl.THUMB_IP),
    (hl.INDEX_TIP, hl.INDEX_PIP),
    (hl.MIDDLE_TIP, hl.MIDDLE_PIP),
    (hl.RING_TIP, hl.RING_PIP),
    (hl.PINKY_TIP, hl.PINKY_PIP),
)

# Joints used to measure how bent each finger is: (base, middle, tip). The
# angle at the middle joint approaches 180 for a straight finger.
_CURL_JOINTS = (
    (hl.THUMB_CMC, hl.THUMB_MCP, hl.THUMB_TIP),
    (hl.INDEX_MCP, hl.INDEX_PIP, hl.INDEX_TIP),
    (hl.MIDDLE_MCP, hl.MIDDLE_PIP, hl.MIDDLE_TIP),
    (hl.RING_MCP, hl.RING_PIP, hl.RING_TIP),
    (hl.PINKY_MCP, hl.PINKY_PIP, hl.PINKY_TIP),
)


def _dist3(a, b) -> float:
    dz = getattr(a, "z", 0.0) - getattr(b, "z", 0.0)
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + dz * dz)


def _angle_at(base, middle, tip) -> float:
    """Interior angle at `middle`, in degrees, in 3D. 180 = perfectly straight."""
    ax, ay = base.x - middle.x, base.y - middle.y
    az = getattr(base, "z", 0.0) - getattr(middle, "z", 0.0)
    bx, by = tip.x - middle.x, tip.y - middle.y
    bz = getattr(tip, "z", 0.0) - getattr(middle, "z", 0.0)
    na = math.sqrt(ax * ax + ay * ay + az * az)
    nb = math.sqrt(bx * bx + by * by + bz * bz)
    if na < 1e-12 or nb < 1e-12:
        return 180.0
    cos = max(-1.0, min(1.0, (ax * bx + ay * by + az * bz) / (na * nb)))
    return math.degrees(math.acos(cos))


@dataclass(frozen=True)
class HandFeatures:
    """Everything the classifier is allowed to care about, for one hand."""

    handedness: str  # "Left" or "Right", as you see it in the mirrored preview
    extended: tuple[bool, ...]  # one per finger, in FINGER_NAMES order
    straightness: tuple[float, ...]  # per finger, degrees; ~170 = dead straight
    scale: float  # wrist -> middle knuckle, in metres
    tilt_deg: float  # 0 = fingers point straight up, +/-180 = straight down
    spread: float  # mean fingertip gap, in units of `scale`
    centre: tuple[float, float]  # palm centre in *image* coords, 0..1

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
        tucks or not varies hugely between people and isn't worth policing."""
        return all(self.extended[1:])

    @property
    def is_fist(self) -> bool:
        """All four fingers curled. Ignores the thumb, same reasoning."""
        return not any(self.extended[1:])

    def describe(self) -> str:
        """One-line summary, for the console and the overlay."""
        flags = "".join(
            name[0].upper() if ext else "-" for name, ext in zip(FINGER_NAMES, self.extended)
        )
        return f"{self.handedness:<5} [{flags}] tilt {self.tilt_deg:+6.1f}deg spread {self.spread:.2f}"


def extract(landmarks, world_landmarks=None, handedness: str = "?") -> HandFeatures:
    """Compute features for one hand.

    `landmarks` are image-space (used for on-screen position). `world_landmarks`
    are metric and used for all geometry — pass them whenever you have them. If
    omitted, image landmarks are used for both, which is only good enough for
    synthetic test hands built in a flat plane.
    """
    world = world_landmarks if world_landmarks is not None else landmarks

    wrist = world[hl.WRIST]
    middle_mcp = world[hl.MIDDLE_MCP]

    # Everything scales against palm length. Guard a degenerate zero so one bad
    # frame can't produce infinities downstream.
    scale = max(_dist3(wrist, middle_mcp), 1e-9)

    straightness = tuple(
        _angle_at(world[a], world[b], world[c]) for a, b, c in _CURL_JOINTS
    )

    # A finger counts as extended when it's both straight *and* reaching away
    # from the wrist.
    #
    # Neither test alone is enough. Distance alone reports the thumb extended
    # almost always, because the thumb tip sits further from the wrist than its
    # IP joint even when tucked. Angle alone accepts a finger folded flat across
    # the palm, which stays fairly straight. Requiring both covers each one's
    # blind spot.
    thresholds = (config.THUMB_STRAIGHT_DEG,) + (config.FINGER_STRAIGHT_DEG,) * 4
    extended = tuple(
        straightness[i] >= thresholds[i]
        and _dist3(wrist, world[tip]) > _dist3(wrist, world[pip])
        for i, (tip, pip) in enumerate(_TIP_AND_PIP)
    )

    # Tilt of the wrist -> knuckle vector, in the image plane, because "which
    # way is up" is a question about the screen rather than about the hand.
    img_wrist = landmarks[hl.WRIST]
    img_mcp = landmarks[hl.MIDDLE_MCP]
    dx = img_mcp.x - img_wrist.x
    dy = -(img_mcp.y - img_wrist.y)  # screen y grows downward
    tilt_deg = math.degrees(math.atan2(dx, dy))

    # How far apart the four fingertips sit — separates a spread open palm from
    # fingers held together.
    tips = [world[i] for i in hl.FINGER_TIPS]
    gaps = [_dist3(a, b) for a, b in zip(tips, tips[1:])]
    spread = (sum(gaps) / len(gaps)) / scale if gaps else 0.0

    centre = ((img_wrist.x + img_mcp.x) / 2, (img_wrist.y + img_mcp.y) / 2)

    return HandFeatures(
        handedness=handedness,
        extended=extended,
        straightness=straightness,
        scale=scale,
        tilt_deg=tilt_deg,
        spread=spread,
        centre=centre,
    )
