"""The debug view: what the program thinks your hands are doing, drawn live.

Everything here is drawn on the *preview-sized* frame, not the full-resolution
capture. Landmarks are normalised to [0,1], so drawing at preview size costs a
fraction as much and looks identical. At 2560x1440 that difference is large.

Panels are alpha-blended over just their own rectangle rather than the whole
frame — `cv2.addWeighted` on a 200x300 ROI is cheap, on a full frame it is not.
"""

from __future__ import annotations

import cv2

import config
import hand_landmarks as hl
from features import FINGER_NAMES, HandFeatures

FONT = cv2.FONT_HERSHEY_SIMPLEX

PANEL_BG = (28, 28, 30)
PANEL_ALPHA = 0.72
ON = (90, 230, 120)  # finger extended
OFF = (95, 95, 105)  # finger curled


def _panel(frame, x: int, y: int, w: int, h: int) -> None:
    """Darken a rectangle so text on top of video stays readable."""
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    tile = roi.copy()
    tile[:] = PANEL_BG
    cv2.addWeighted(tile, PANEL_ALPHA, roi, 1 - PANEL_ALPHA, 0, dst=roi)


def _text(frame, s: str, x: int, y: int, scale=0.46, color=(238, 238, 238), weight=1) -> None:
    cv2.putText(frame, s, (x, y), FONT, scale, color, weight, cv2.LINE_AA)


def draw_hands(frame, hands_landmarks) -> None:
    """Skeleton overlay for every detected hand."""
    for landmarks in hands_landmarks:
        hl.draw_hand(frame, landmarks)


def draw_top_bar(
    frame,
    fps: float,
    capture_size: tuple[int, int],
    hand_count: int,
    link_status: str = "",
) -> None:
    w = frame.shape[1]
    _panel(frame, 0, 0, w, 34)
    cw, ch = capture_size
    _text(frame, "CLAUDELASH", 12, 22, scale=0.5, color=(120, 200, 255))
    _text(frame, f"{fps:5.1f} fps", 130, 22)
    _text(frame, f"{cw}x{ch}", 215, 22, color=(170, 170, 178))
    _text(frame, f"hands {hand_count}", 320, 22, color=(170, 170, 178))
    if link_status:
        # Red when the extension isn't reachable — that's the usual reason a
        # gesture fires but nothing happens in VS Code.
        bad = "not running" in link_status or "failed" in link_status
        _text(frame, link_status, 425, 22, color=(90, 90, 235) if bad else (150, 200, 150))


def draw_footer(frame, keys: str) -> None:
    h, w = frame.shape[:2]
    _panel(frame, 0, h - 26, w, 26)
    _text(frame, keys, 12, h - 8, scale=0.42, color=(165, 165, 172))


def draw_hand_panel(frame, feature: HandFeatures, slot: int) -> None:
    """One readout panel per hand, stacked down the left edge.

    `slot` is 0 for the first hand, 1 for the second — so two hands don't
    draw on top of each other.
    """
    x, y = 12, 46 + slot * 176
    w, h = 224, 166
    _panel(frame, x, y, w, h)

    _text(frame, f"{feature.handedness.upper()} HAND", x + 12, y + 22, scale=0.5,
          color=(120, 200, 255))

    # One row per finger: a filled dot when extended, hollow when curled.
    for i, name in enumerate(FINGER_NAMES):
        row_y = y + 44 + i * 17
        extended = feature.extended[i]
        colour = ON if extended else OFF
        cv2.circle(frame, (x + 18, row_y - 4), 4, colour, -1 if extended else 1, cv2.LINE_AA)
        _text(frame, name, x + 32, row_y, scale=0.42, color=colour)
        _text(frame, "ext" if extended else "curl", x + 96, row_y, scale=0.42, color=colour)

    # Continuous values — these are what you tune thresholds against.
    _text(frame, f"tilt   {feature.tilt_deg:+6.1f}", x + 12, y + 146, scale=0.42,
          color=(200, 200, 206))
    _text(frame, f"spread {feature.spread:5.2f}", x + 118, y + 146, scale=0.42,
          color=(200, 200, 206))


def _shape_summary(features_list: list[HandFeatures]) -> tuple[str, tuple[int, int, int]]:
    """Plain-language description of the hand, for when no gesture matches.

    Useful when a gesture *should* be firing and isn't — it tells you which
    part of the shape the classifier disagrees with you about.
    """
    if not features_list:
        return "no hands", (150, 150, 158)
    f = features_list[0]
    if f.is_flat_palm and not f.is_upright:
        return f"flat palm, tilted {f.tilt_deg:+.0f}", (230, 200, 90)
    if f.is_flat_palm:
        return "flat palm", (190, 190, 198)
    if f.is_fist:
        return "fist", (190, 190, 198)
    return f"{f.extended_count} fingers out", (190, 190, 198)


def draw_gesture_panel(
    frame,
    features_list: list[HandFeatures],
    shape: str | None,
    sequences,
    fired_ago: float,
    tracker=None,
) -> None:
    """Top-right: the shape being seen now, and how far each sequence has got.

    `fired_ago` is seconds since the last firing — used to flash the panel so a
    fire is visible even though it only lasts one frame.
    """
    w = frame.shape[1]
    x, y = w - 336, 46
    ph = 56 + len(sequences) * 34 + (22 if tracker is not None else 0)
    _panel(frame, x, y, 324, ph)

    if shape:
        label, colour = shape, (90, 230, 120)
    else:
        label, colour = _shape_summary(features_list)

    if fired_ago < 0.5:
        cv2.rectangle(frame, (x, y), (x + 324, y + ph), (90, 230, 120), 2, cv2.LINE_AA)

    _text(frame, label, x + 14, y + 30, scale=0.6, color=colour, weight=2)

    # Circular-motion readout. The swept figure is what decides whether an open
    # palm counts as circling, so seeing it live is how you tune the threshold.
    row_offset = 0
    if tracker is not None:
        import config
        swept = tracker.swept_deg
        enough = abs(swept) >= config.CIRCLE_MIN_SWEPT_DEG
        far = tracker.path_length >= config.CIRCLE_MIN_PATH
        _text(frame, f"swept {swept:+5.0f} / {config.CIRCLE_MIN_SWEPT_DEG:.0f}",
              x + 14, y + 48, scale=0.42,
              color=(90, 230, 120) if enough else (185, 185, 192))
        _text(frame, f"path {tracker.path_length:.2f}", x + 170, y + 48, scale=0.42,
              color=(90, 230, 120) if far else (185, 185, 192))
        row_offset = 22

    # One row per sequence: its name, then a pip per step. Pips light up as
    # each shape in the sequence is seen.
    for i, seq in enumerate(sequences):
        row_y = y + 54 + row_offset + i * 34
        done = seq.matched
        name_colour = (215, 215, 222) if done == 0 else (235, 190, 80)
        _text(frame, seq.name, x + 14, row_y + 10, scale=0.44, color=name_colour)

        for step_i, step in enumerate(seq.steps):
            cx = x + 190 + step_i * 30
            lit = step_i < done
            cv2.circle(frame, (cx, row_y + 5), 8, (90, 230, 120) if lit else (75, 75, 84),
                       -1 if lit else 1, cv2.LINE_AA)
            _text(frame, step[0], cx - 4, row_y + 10, scale=0.38,
                  color=(20, 20, 20) if lit else (150, 150, 158))
            if step_i < len(seq.steps) - 1:
                cv2.line(frame, (cx + 9, row_y + 5), (cx + 21, row_y + 5),
                         (100, 100, 110), 1, cv2.LINE_AA)


def scale_for_preview(frame):
    """Downscale for display. INTER_AREA is the right filter for shrinking."""
    if config.PREVIEW_SCALE == 1.0:
        return frame
    return cv2.resize(
        frame, None,
        fx=config.PREVIEW_SCALE, fy=config.PREVIEW_SCALE,
        interpolation=cv2.INTER_AREA,
    )
