"""The gesture classifier. This is the file you'll tinker with most.

Adding a gesture is meant to be two small edits:

  1. write a `_is_<name>` function that answers one yes/no question about a
     hand, reading the named facts from `features.HandFeatures`
  2. add a line to `classify()`

Keep each check readable in one breath. If a check needs a geometric fact that
doesn't exist yet — palm facing, finger angle, whatever — add it as a field in
`features.py` rather than doing trigonometry here. That way every gesture gets
it, and this file stays a list of plain statements about hand shape.

Order matters in `classify()`: the first match wins. Put the most specific
gestures first so a loose one doesn't shadow them.
"""

from __future__ import annotations

from features import HandFeatures

# Gesture names. These strings are the contract with the VS Code extension —
# they're what gets POSTed, and what gesture-map.json keys off. Don't rename
# one without changing the map too.
STOP_CHOP = "STOP_CHOP"


def _is_stop_chop(f: HandFeatures) -> bool:
    """Fletcher's "stop, not quite my tempo": right hand up, palm open, flat.

    Deliberately loose. It's the interrupt, so it should fire when you mean it
    even if your hand is a bit crooked — `is_upright` allows 30 degrees of tilt
    either way. The thumb is ignored entirely; whether it sticks out or tucks
    varies per person and policing it only causes misses.
    """
    return f.handedness == "Right" and f.is_flat_palm and f.is_upright


def classify(hand_features: list[HandFeatures]) -> str | None:
    """Name the gesture currently being made, or None.

    Takes every visible hand, because later gestures (TWO_HAND_SLAM) need to
    look at more than one at a time.
    """
    for f in hand_features:
        if _is_stop_chop(f):
            return STOP_CHOP

    return None
