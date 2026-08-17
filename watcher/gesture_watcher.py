"""Gesture watcher — MILESTONE 2.

Shows the webcam with hand landmarks and a live readout of what the program
thinks your hands are doing, classifies STOP_CHOP, and prints to the console
when it fires after being held.

Nothing is dispatched anywhere yet — no HTTP, no VS Code, no keystrokes. The
point of this milestone is to confirm the gesture triggers when you mean it and
stays quiet when you don't.

Run:
    .venv\\Scripts\\python.exe watcher\\gesture_watcher.py

Keys:
    q / Esc   quit
    d         landmark overlay on/off
    p         readout panels on/off
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

import camera
import config
import features
import gestures
import overlay
from debounce import HoldDebouncer
from dispatch import Dispatcher

MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"

KEY_HINTS = "q quit    d landmarks    p panels"


def create_landmarker() -> vision.HandLandmarker:
    """Build the MediaPipe Tasks hand landmarker in VIDEO mode.

    VIDEO mode is synchronous and wants a monotonically increasing timestamp,
    which keeps the main loop straightforward — no async callback to juggle.
    """
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Missing model file: {MODEL_PATH}\n"
            f"Re-download it from:\n"
            f"  https://storage.googleapis.com/mediapipe-models/hand_landmarker"
            f"/hand_landmarker/float16/1/hand_landmarker.task"
        )
    options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=config.MAX_HANDS,
        min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=config.MIN_PRESENCE_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )
    return vision.HandLandmarker.create_from_options(options)


def read_handedness(result, i: int) -> str:
    """Which hand this is, as *you* see it in the mirrored preview.

    MediaPipe is looking at an already-mirrored frame, so it reports the
    opposite hand from the real one. Flipping the label back means "RIGHT HAND"
    on screen is the hand you'd call your right hand.
    """
    try:
        name = result.handedness[i][0].category_name
    except (IndexError, AttributeError):
        return "?"
    if config.MIRROR_PREVIEW:
        return {"Left": "Right", "Right": "Left"}.get(name, name)
    return name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Claudelash gesture watcher")
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="pin a camera index, overriding CAMERA_NAME in config.py",
    )
    args = parser.parse_args(argv)

    print(f"mediapipe {mp.__version__} | opencv {cv2.__version__}")
    print("Opening camera... (first frame can take a second)")

    cap = camera.open_camera(args.camera)
    capture_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    frame_times: deque[float] = deque(maxlen=30)
    show_landmarks = True
    show_panels = True
    start = time.perf_counter()

    debouncer = HoldDebouncer(config.HOLD_DURATION_MS, config.COOLDOWN_MS)
    dispatcher = Dispatcher()
    last_fire_at = -1e9
    fire_count = 0

    print(f"Ready at {capture_size[0]}x{capture_size[1]}. {KEY_HINTS}")

    with create_landmarker() as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Dropped frame from camera, retrying...", file=sys.stderr)
                continue

            if config.MIRROR_PREVIEW:
                frame = cv2.flip(frame, 1)

            # Inference runs on the full-resolution frame.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.perf_counter() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            hands = result.hand_landmarks or []
            hand_features = [
                features.extract(landmarks, read_handedness(result, i))
                for i, landmarks in enumerate(hands)
            ]

            now = time.monotonic()
            gesture = gestures.classify(hand_features)
            fired = debouncer.update(gesture, now)
            if fired:
                fire_count += 1
                last_fire_at = now
                dispatcher.send(fired, config.HOLD_DURATION_MS)
                print(f"[{fire_count:3d}] FIRED  {fired}  -> {dispatcher.url}", flush=True)

            frame_times.append(time.perf_counter())
            fps = 0.0
            if len(frame_times) > 1:
                span = frame_times[-1] - frame_times[0]
                if span > 0:
                    fps = (len(frame_times) - 1) / span

            # Shrink *before* drawing. Landmarks are normalised, so the skeleton
            # lands in the same place either way, and drawing on the smaller
            # frame is far cheaper at 2560x1440.
            view = overlay.scale_for_preview(frame)

            if show_landmarks:
                overlay.draw_hands(view, hands)
            if show_panels:
                for slot, feature in enumerate(hand_features[:2]):
                    overlay.draw_hand_panel(view, feature, slot)
                overlay.draw_gesture_panel(
                    view,
                    hand_features,
                    gesture,
                    debouncer.progress(now),
                    now - last_fire_at,
                )

            overlay.draw_top_bar(view, fps, capture_size, len(hands), dispatcher.status)
            overlay.draw_footer(view, KEY_HINTS)
            cv2.imshow("Claudelash - gesture watcher", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                show_landmarks = not show_landmarks
            if key == ord("p"):
                show_panels = not show_panels

    dispatcher.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Sent {dispatcher.sent}, failed {dispatcher.failed}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
