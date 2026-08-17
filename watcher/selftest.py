"""Check the classifier and the hold-debounce without a camera.

Run this after changing a threshold in `gestures.py`, or the timings in
`config.py`, to see what you broke before going to look for it on video:

    .venv\\Scripts\\python.exe watcher\\selftest.py

Hands are synthetic — built from landmark coordinates directly, then rotated
about the wrist to test tilt limits. Nothing here touches the webcam.

When you add a gesture, add a row to `cases` for the shape that should match
and at least one near-miss that shouldn't.
"""
import sys

sys.path.insert(0, "watcher")
import features
import gestures
import hand_landmarks as hl
from debounce import HoldDebouncer


class P:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


def flat_palm(rotate_deg=0.0):
    """Open hand, fingers up, optionally rotated about the wrist."""
    lm = [P(0.5, 0.9) for _ in range(21)]
    lm[hl.WRIST] = P(0.50, 0.90)
    lm[hl.THUMB_CMC] = P(0.44, 0.86)
    lm[hl.THUMB_MCP] = P(0.40, 0.81)
    lm[hl.THUMB_IP] = P(0.37, 0.76)
    lm[hl.THUMB_TIP] = P(0.34, 0.71)
    for i, base_x in enumerate((0.46, 0.51, 0.56, 0.61)):
        b = i * 4
        lm[hl.INDEX_MCP + b] = P(base_x, 0.68)
        lm[hl.INDEX_PIP + b] = P(base_x, 0.60)
        lm[hl.INDEX_DIP + b] = P(base_x, 0.54)
        lm[hl.INDEX_TIP + b] = P(base_x, 0.48)

    if rotate_deg:
        import math
        a = math.radians(rotate_deg)
        wx, wy = lm[hl.WRIST].x, lm[hl.WRIST].y
        for i, p in enumerate(lm):
            dx, dy = p.x - wx, p.y - wy
            lm[i] = P(wx + dx * math.cos(a) - dy * math.sin(a),
                      wy + dx * math.sin(a) + dy * math.cos(a))
    return lm


def fist():
    lm = flat_palm()
    for i in range(4):
        b = i * 4
        x = lm[hl.INDEX_PIP + b].x
        lm[hl.INDEX_PIP + b] = P(x, 0.62)
        lm[hl.INDEX_DIP + b] = P(x, 0.68)
        lm[hl.INDEX_TIP + b] = P(x, 0.72)
    return lm


def feat(landmarks, hand):
    return features.extract(landmarks, hand)


print("=== classify() ===")
cases = [
    ("right flat palm upright", [feat(flat_palm(), "Right")], gestures.STOP_CHOP),
    ("left flat palm upright", [feat(flat_palm(), "Left")], None),
    ("right fist", [feat(fist(), "Right")], None),
    ("right palm rotated 20deg", [feat(flat_palm(rotate_deg=20), "Right")], gestures.STOP_CHOP),
    ("right palm rotated 50deg", [feat(flat_palm(rotate_deg=50), "Right")], None),
    ("right palm upside down", [feat(flat_palm(rotate_deg=180), "Right")], None),
    ("no hands", [], None),
    ("left palm + right palm", [feat(flat_palm(), "Left"), feat(flat_palm(), "Right")],
     gestures.STOP_CHOP),
]
ok = True
for name, feats, expected in cases:
    got = gestures.classify(feats)
    good = got == expected
    ok &= good
    tilt = f"  tilt={feats[0].tilt_deg:+.0f}" if feats else ""
    print(f"  {'PASS' if good else 'FAIL'}  {name:<26} -> {got}{tilt}")

print("\n=== hold debounce (hold=400ms, cooldown=1200ms) ===")
d = HoldDebouncer(hold_ms=400, cooldown_ms=1200)
t = 0.0


def run(gesture, seconds, sink):
    """Feed `gesture` at 25fps for `seconds`, collecting any firings."""
    global t
    for _ in range(int(seconds / 0.04)):
        t += 0.04
        got = d.update(gesture, t)
        if got:
            sink.append(round(t, 2))


a = []
run("STOP_CHOP", 1.0, a)
print(f"  held 1.0s continuously      -> fired at {a}  (expect one, ~0.44s)")
ok &= len(a) == 1

# Release briefly, re-make, and complete the hold while still inside cooldown.
b = []
run(None, 0.08, b)
run("STOP_CHOP", 0.5, b)  # hold completes ~t=1.6, cooldown ends t=1.68
print(f"  re-made, held inside cooldown -> fired at {b}  (expect none)")
ok &= len(b) == 0

# Drop it, idle well past the cooldown, then make it again.
c = []
run(None, 1.5, c)
run("STOP_CHOP", 0.6, c)
print(f"  re-made after cooldown      -> fired at {c}  (expect one)")
ok &= len(c) == 1

# flicker: gesture appearing for 2 frames should never fire
d2 = HoldDebouncer(hold_ms=400, cooldown_ms=1200)
t = 0.0
flicker = []
for i in range(40):
    t += 0.04
    g = "STOP_CHOP" if i % 10 < 2 else None
    if d2.update(g, t):
        flicker.append(round(t, 2))
print(f"  2-frame flickers        -> fired at {flicker}  (expect none)")
ok &= len(flicker) == 0

print(f"\n{'ALL PASS' if ok else 'FAILURES ABOVE'}")
sys.exit(0 if ok else 1)
