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

from collections import deque

import config
from features import HandFeatures
from sequences import SequenceDetector

# --- Shapes -----------------------------------------------------------------

OPEN_PALM = "OPEN_PALM"
CIRCLING_PALM = "CIRCLING_PALM"
FIST = "FIST"


def _is_open_palm(f: HandFeatures) -> bool:
    """A deliberately open hand: every finger out, thumb included, fanned apart.

    Stricter than `is_flat_palm`, which ignores the thumb. The thumb is
    included here precisely because this gesture starts the sequence — without
    it, a hand with one or two fingers loosely straight could pass, and with
    FINGER_STRAIGHT_DEG at 110 a lazily-curled finger sometimes does. The
    calibration supports the extra check: the thumb measured 169 during the
    open palm against 129 during the fist, so it separates cleanly.

    Spread is what separates a deliberate open palm from a hand that merely has
    straight fingers. Measured on the reference clip: open palm 0.37-0.43,
    fist 0.19-0.20.
    """
    if f.spread < config.OPEN_PALM_MIN_SPREAD:
        return False
    if config.OPEN_PALM_REQUIRE_THUMB:
        return all(f.extended)  # all five, thumb included
    return f.is_flat_palm


def _is_fist(f: HandFeatures) -> bool:
    """All four fingers curled. The thumb is ignored — it rides outside a fist
    as often as inside, and checking it only causes misses."""
    return f.is_fist


# Recent history of whether the gesturing hand was identifiable. MediaPipe's
# handedness wobbles during fast motion — measured on the reference clip, it
# reported both Left and Right for the same hand, which cost one repetition in
# six once right-hand-only was enforced.
_hand_seen: deque[bool] = deque(maxlen=config.HANDEDNESS_VOTE_FRAMES)


def _pick_gesturing_hand(hand_features: list[HandFeatures]) -> HandFeatures | None:
    """Choose which visible hand drives gestures.

    Both hands are tracked and drawn regardless; this only decides who gets
    listened to. When the label wobbles for a frame or two but only one hand is
    on screen, there's nothing it could be confused with, so the recent history
    is trusted over the current frame.
    """
    if not config.REQUIRE_HANDEDNESS:
        return hand_features[0]

    match = next((h for h in hand_features if h.handedness == config.GESTURE_HAND), None)
    _hand_seen.append(match is not None)
    if match is not None:
        return match

    # Label lost this frame. Accept a lone hand if the label was mostly right
    # across the recent window — but never when both hands are visible, since
    # then it really could be the other one.
    if len(hand_features) == 1 and _hand_seen and sum(_hand_seen) >= len(_hand_seen) / 2:
        return hand_features[0]
    return None


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

    f = _pick_gesturing_hand(hand_features)
    if f is None:
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
