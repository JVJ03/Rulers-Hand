"""The gesture classifier. This is the file you'll tinker with most.

There are two layers, and it matters which one you're editing.

**Shapes** are what a hand looks like in one frame — an open palm, a fist.
`classify_shape()` names them. Adding one is a `_is_<name>` predicate reading
the named facts on `features.HandFeatures`, plus a line in `classify_shape()`.

**Gestures** are what you actually mean, and some of them span time.
"Not quite my tempo" is an open palm circling, then a fist — no single frame
contains it. Those live in SEQUENCES below as an ordered list of shapes.

If a gesture needs a geometric fact that doesn't exist — palm facing, finger
angle, whatever — add it as a field in `features.py` rather than doing
trigonometry here. Then every gesture gets it and this file stays a list of
plain statements about hands.
"""

from __future__ import annotations

import config
from features import HandFeatures
from sequences import SequenceDetector

# --- Shapes -----------------------------------------------------------------

OPEN_PALM = "OPEN_PALM"
CIRCLING_PALM = "CIRCLING_PALM"
FIST = "FIST"


def _is_open_palm(f: HandFeatures) -> bool:
    """All four fingers out and spread apart.

    Requiring spread is what separates a deliberate open palm from a hand that
    merely has its fingers straight. Measured from the reference clip: the
    open-palm phase sat at 0.32-0.36, the fist at 0.09-0.14.
    """
    return f.is_flat_palm and f.spread >= config.OPEN_PALM_MIN_SPREAD


def _is_fist(f: HandFeatures) -> bool:
    """All four fingers curled. The thumb is ignored — it rides outside a fist
    as often as inside, and checking it only causes misses."""
    return f.is_fist


def classify_shape(hand_features: list[HandFeatures], tracker=None) -> str | None:
    """Name what the hand is doing right now, or None.

    `tracker` carries the recent motion path. An open palm that has circled far
    enough is a different thing from one held still, and only the moving one is
    part of "not quite my tempo" — so it gets its own name.

    Only the first hand is considered. Two-handed gestures will need their own
    path here; that's milestone 7's problem.
    """
    if not hand_features:
        return None
    f = hand_features[0]

    if config.REQUIRE_HANDEDNESS and f.handedness != config.GESTURE_HAND:
        return None

    if _is_open_palm(f):
        circling = tracker is not None and tracker.is_circling(
            config.CIRCLE_MIN_SWEPT_DEG, config.CIRCLE_MIN_PATH
        )
        return CIRCLING_PALM if circling else OPEN_PALM
    if _is_fist(f):
        return FIST
    return None


# --- Gestures ---------------------------------------------------------------

NOT_QUITE_MY_TEMPO = "NOT_QUITE_MY_TEMPO"

# Ordered shapes that make up each gesture. Intervening shapes are ignored and
# frames with no hand don't reset progress — see sequences.py for why.
#
# The first step is CIRCLING_PALM, not OPEN_PALM: the circle is what makes this
# the gesture rather than someone simply holding a hand up and then relaxing it.
SEQUENCES = [
    SequenceDetector(
        name=NOT_QUITE_MY_TEMPO,
        steps=[CIRCLING_PALM, FIST],
        window_ms=config.SEQUENCE_WINDOW_MS,
        cooldown_ms=config.COOLDOWN_MS,
    ),
]


def update(
    hand_features: list[HandFeatures], now: float, tracker=None
) -> tuple[str | None, str | None]:
    """Advance every sequence by one frame.

    Returns `(shape_now, fired_gesture)` — the shape so the overlay can show
    what's being seen, and the gesture name on the frame it completes.
    """
    shape = classify_shape(hand_features, tracker)
    fired = None
    for sequence in SEQUENCES:
        result = sequence.update(shape, now)
        if result and not fired:
            fired = result
    return shape, fired
