"""Check shape classification and sequence detection without a camera.

Run after changing a threshold in `gestures.py` or a timing in `config.py`:

    .venv\\Scripts\\python.exe watcher\\selftest.py

Hands are synthetic, built from landmark coordinates directly. The geometry is
calibrated against the reference clip — the synthetic open palm measures
spread ~0.35 and the fist ~0.10, matching what the real gesture produced.

When you add a gesture, add a row to SHAPE_CASES for the shape that should
match plus a near-miss that shouldn't, and a sequence case for the timing.
"""

from __future__ import annotations

import math
import sys

sys.path.insert(0, "watcher")
import config
import features
import gestures
import hand_landmarks as hl
from sequences import SequenceDetector


class P:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


def _rotate(lm, degrees):
    a = math.radians(degrees)
    wx, wy = lm[hl.WRIST].x, lm[hl.WRIST].y
    out = []
    for p in lm:
        dx, dy = p.x - wx, p.y - wy
        out.append(P(wx + dx * math.cos(a) - dy * math.sin(a),
                     wy + dx * math.sin(a) + dy * math.cos(a)))
    return out


def open_palm(rotate_deg=0.0):
    """Fingers straight and fanned out. Spread lands ~0.35, as in the clip."""
    lm = [P(0.5, 0.9) for _ in range(21)]
    lm[hl.WRIST] = P(0.50, 0.90)
    lm[hl.THUMB_CMC] = P(0.44, 0.86)
    lm[hl.THUMB_MCP] = P(0.40, 0.81)
    lm[hl.THUMB_IP] = P(0.36, 0.76)
    lm[hl.THUMB_TIP] = P(0.32, 0.71)
    # Knuckles stay close together; tips fan apart. That's what spread measures.
    for i, (mcp_x, tip_x) in enumerate(
        ((0.47, 0.42), (0.52, 0.50), (0.57, 0.58), (0.62, 0.66))
    ):
        b = i * 4
        lm[hl.INDEX_MCP + b] = P(mcp_x, 0.68)
        lm[hl.INDEX_PIP + b] = P(mcp_x + (tip_x - mcp_x) * 0.35, 0.60)
        lm[hl.INDEX_DIP + b] = P(mcp_x + (tip_x - mcp_x) * 0.70, 0.54)
        lm[hl.INDEX_TIP + b] = P(tip_x, 0.48)
    return _rotate(lm, rotate_deg) if rotate_deg else lm


def fist():
    """Fingers folded back toward the palm — angle at the PIP collapses."""
    lm = open_palm()
    for i in range(4):
        b = i * 4
        x = lm[hl.INDEX_MCP + b].x
        lm[hl.INDEX_PIP + b] = P(x, 0.62)
        lm[hl.INDEX_DIP + b] = P(x, 0.68)
        lm[hl.INDEX_TIP + b] = P(x, 0.72)
    lm[hl.THUMB_IP] = P(0.44, 0.79)
    lm[hl.THUMB_TIP] = P(0.48, 0.76)
    return lm


def one_finger():
    """Only the pinky out. This is the case that used to report 'two fingers'."""
    lm = fist()
    b = 3 * 4  # pinky
    x = lm[hl.PINKY_MCP].x
    lm[hl.PINKY_PIP] = P(x, 0.60)
    lm[hl.PINKY_DIP] = P(x, 0.54)
    lm[hl.PINKY_TIP] = P(x, 0.48)
    return lm


def feat(landmarks, hand="Right"):
    return features.extract(landmarks, hand)


ok = True

print("=== features ===")
for name, lm in (("open palm", open_palm()), ("fist", fist()), ("pinky only", one_finger())):
    f = feat(lm)
    angles = " ".join(f"{a:3.0f}" for a in f.straightness)
    print(f"  {name:<12} {f.describe()}  straightness [{angles}]")

# The bug that started this: a relaxed thumb made a single raised finger read
# as two, because the old test only asked whether the tip was far from the wrist.
pinky = feat(one_finger())
good = pinky.extended_count == 1
ok &= good
print(f"  {'PASS' if good else 'FAIL'}  pinky alone counts as "
      f"{pinky.extended_count} finger(s), expected 1")

print("\n=== classify_shape() ===")
SHAPE_CASES = [
    ("open palm", [feat(open_palm())], gestures.OPEN_PALM),
    ("open palm rotated 40", [feat(open_palm(rotate_deg=40))], gestures.OPEN_PALM),
    ("open palm rotated 90", [feat(open_palm(rotate_deg=90))], gestures.OPEN_PALM),
    ("fist", [feat(fist())], gestures.FIST),
    ("pinky only", [feat(one_finger())], None),
    ("no hands", [], None),
]
for name, feats, expected in SHAPE_CASES:
    got = gestures.classify_shape(feats)
    good = got == expected
    ok &= good
    extra = f"  spread={feats[0].spread:.2f}" if feats else ""
    print(f"  {'PASS' if good else 'FAIL'}  {name:<22} -> {got}{extra}")

print("\n  (rotation must NOT matter — the gesture circles the palm around)")

print("\n=== sequence: OPEN_PALM > FIST ===")


def sequence_run(steps_over_time, window_ms=2500, cooldown_ms=1200):
    """steps_over_time: list of (shape, seconds). Returns fire times."""
    det = SequenceDetector("TEST", [gestures.OPEN_PALM, gestures.FIST],
                           window_ms, cooldown_ms)
    t, fires = 0.0, []
    for shape, seconds in steps_over_time:
        for _ in range(max(1, int(seconds / 0.04))):
            t += 0.04
            if det.update(shape, t):
                fires.append(round(t, 2))
    return fires


CASES = [
    ("palm 0.7s then fist", [("OPEN_PALM", 0.7), ("FIST", 0.5)], 1),
    ("palm only", [("OPEN_PALM", 1.5)], 0),
    ("fist only", [("FIST", 1.5)], 0),
    ("fist then palm (backwards)", [("FIST", 0.7), ("OPEN_PALM", 0.7)], 0),
    ("palm, hand lost, fist", [("OPEN_PALM", 0.6), (None, 0.4), ("FIST", 0.4)], 1),
    ("palm, junk shape, fist", [("OPEN_PALM", 0.6), ("SOMETHING", 0.3), ("FIST", 0.4)], 1),
    ("palm, long gap, fist", [("OPEN_PALM", 0.5), (None, 3.0), ("FIST", 0.5)], 0),
    ("two reps, spaced out",
     [("OPEN_PALM", 0.6), ("FIST", 0.5), (None, 1.5), ("OPEN_PALM", 0.6), ("FIST", 0.5)], 2),
    ("two reps, back to back",
     [("OPEN_PALM", 0.5), ("FIST", 0.3), ("OPEN_PALM", 0.3), ("FIST", 0.3)], 1),
]
for name, script, expected in CASES:
    fires = sequence_run(script)
    good = len(fires) == expected
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  {name:<28} fired {len(fires)}x "
          f"(expect {expected})  {fires}")

print(f"\nthresholds: OPEN_PALM_MIN_SPREAD={config.OPEN_PALM_MIN_SPREAD}  "
      f"SEQUENCE_WINDOW_MS={config.SEQUENCE_WINDOW_MS}")
print(f"\n{'ALL PASS' if ok else 'FAILURES ABOVE'}")
sys.exit(0 if ok else 1)
