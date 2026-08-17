"""Record labelled gesture samples, so thresholds come from your hand.

Guessing thresholds from a description doesn't work — "open hand" and the
gesture you actually mean can differ in ways neither of us would think to
describe. So: hold a gesture, press its number key, and this captures a burst
of landmark frames under that label.

    .venv\\Scripts\\python.exe watcher\\record.py

Keys:
    1-9       record a burst for that label (see RECORD_LABELS in config.py)
    backspace drop the most recent burst, if you fumbled it
    c         clear ALL samples for every label (asks first, in the console)
    q / Esc   quit

Recording a *negative* example matters as much as a positive one. If the
classifier is firing on a plain open hand when you meant something else,
record both — the difference between them is exactly what the rule needs.

**No images are written.** The CSV holds only landmark coordinates: 21 points
per frame, as MediaPipe reports them. Nothing leaves the machine.
"""

from __future__ import annotations

import csv
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp

import camera
import config
import features
import gesture_watcher as gw
import hand_landmarks as hl
import overlay

SAMPLES_PATH = Path(__file__).parent.parent / "data" / "samples.csv"
BURST_FRAMES = 45  # ~2 seconds at 25fps
COUNTDOWN_S = 1.0  # grace period so you can settle into the pose


def csv_header() -> list[str]:
    cols = ["label", "handedness"]
    for i in range(21):
        cols += [f"x{i}", f"y{i}", f"z{i}"]
    return cols


def row_for(label: str, handedness: str, landmarks) -> list:
    row: list = [label, handedness]
    for p in landmarks:
        row += [round(p.x, 5), round(p.y, 5), round(getattr(p, "z", 0.0), 5)]
    return row


def load_counts() -> Counter:
    counts: Counter = Counter()
    if not SAMPLES_PATH.exists():
        return counts
    with SAMPLES_PATH.open(newline="", encoding="utf8") as fh:
        for row in csv.DictReader(fh):
            counts[row["label"]] += 1
    return counts


def append_rows(rows: list[list]) -> None:
    SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not SAMPLES_PATH.exists()
    with SAMPLES_PATH.open("a", newline="", encoding="utf8") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(csv_header())
        writer.writerows(rows)


def drop_label_burst(label: str, count: int) -> int:
    """Remove the last `count` rows for `label`. Returns how many went."""
    if not SAMPLES_PATH.exists():
        return 0
    with SAMPLES_PATH.open(newline="", encoding="utf8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)

    keep, removed = [], 0
    for row in reversed(rows):
        if row and row[0] == label and removed < count:
            removed += 1
            continue
        keep.append(row)
    keep.reverse()

    with SAMPLES_PATH.open("w", newline="", encoding="utf8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(keep)
    return removed


def draw_recorder_hud(view, labels: dict[str, str], counts: Counter,
                      active: str | None, remaining: int, countdown: float) -> None:
    """Right-hand column: what each key records, and how much you've got."""
    w = view.shape[1]
    x, y = w - 300, 130
    height = 34 + len(labels) * 20
    overlay._panel(view, x, y, 288, height)
    overlay._text(view, "RECORD  (press a number)", x + 12, y + 22,
                  scale=0.46, color=(120, 200, 255))

    for i, (key, name) in enumerate(sorted(labels.items())):
        row_y = y + 44 + i * 20
        n = counts.get(name, 0)
        if name == active:
            colour = (90, 230, 120)
        elif n == 0:
            colour = (140, 140, 148)
        else:
            colour = (215, 215, 222)
        overlay._text(view, f"{key}  {name}", x + 12, row_y, scale=0.44, color=colour)
        overlay._text(view, f"{n:4d}", x + 236, row_y, scale=0.44, color=colour)

    if active:
        if countdown > 0:
            msg, colour = f"GET READY  {countdown:.1f}s", (235, 190, 80)
        else:
            msg, colour = f"RECORDING {active}  {remaining} left", (90, 230, 120)
        overlay._panel(view, 12, view.shape[0] - 70, 340, 34)
        overlay._text(view, msg, 24, view.shape[0] - 48, scale=0.56, color=colour, weight=2)


def main() -> int:
    labels: dict[str, str] = config.RECORD_LABELS
    if not labels:
        print("No RECORD_LABELS configured in watcher/config.py", file=sys.stderr)
        return 1

    counts = load_counts()
    print(f"Samples file: {SAMPLES_PATH}")
    print("Labels:", ", ".join(f"[{k}] {v}" for k, v in sorted(labels.items())))
    print("Hold the gesture, press its number, hold still through the countdown.\n")

    cap = camera.open_camera()
    capture_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    active: str | None = None
    pending: list[list] = []
    remaining = 0
    countdown_until = 0.0
    start = time.perf_counter()

    with gw.create_landmarker() as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if config.MIRROR_PREVIEW:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(
                image, int((time.perf_counter() - start) * 1000)
            )
            hands = result.hand_landmarks or []
            hand_features = [
                features.extract(lm, gw.read_handedness(result, i))
                for i, lm in enumerate(hands)
            ]

            now = time.perf_counter()
            countdown = max(0.0, countdown_until - now)

            # Capture a frame if we're mid-burst and a hand is actually visible.
            if active and countdown == 0 and remaining > 0:
                if hands:
                    pending.append(row_for(active, hand_features[0].handedness, hands[0]))
                    remaining -= 1
                    if remaining == 0:
                        append_rows(pending)
                        counts[active] += len(pending)
                        print(f"  saved {len(pending)} frames of {active} "
                              f"(total {counts[active]})")
                        pending = []
                        active = None

            view = overlay.scale_for_preview(frame)
            overlay.draw_hands(view, hands)
            for slot, feature in enumerate(hand_features[:2]):
                overlay.draw_hand_panel(view, feature, slot)
            overlay.draw_top_bar(view, 0.0, capture_size, len(hands))
            draw_recorder_hud(view, labels, counts, active, remaining, countdown)
            overlay.draw_footer(view, "1-9 record   backspace undo   c clear all   q quit")
            cv2.imshow("Claudelash - recorder", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            ch = chr(key) if 32 <= key < 127 else ""
            if ch in labels and active is None:
                active = labels[ch]
                remaining = BURST_FRAMES
                pending = []
                countdown_until = now + COUNTDOWN_S
                print(f"recording {active} ...")
            elif key == 8 and active is None:  # backspace
                if counts:
                    last = max(counts, key=lambda k: counts[k])
                    gone = drop_label_burst(last, BURST_FRAMES)
                    counts[last] = max(0, counts[last] - gone)
                    print(f"  dropped {gone} frames of {last}")
            elif ch == "c" and active is None:
                if SAMPLES_PATH.exists():
                    SAMPLES_PATH.unlink()
                    counts = Counter()
                    print("  cleared all samples")

    cap.release()
    cv2.destroyAllWindows()
    print("\nTotals:", dict(counts))
    print(f"Now run:  .venv\\Scripts\\python.exe watcher\\analyse.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
