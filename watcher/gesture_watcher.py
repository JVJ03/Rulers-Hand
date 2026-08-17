"""Gesture watcher — MILESTONE 1: webcam feed + MediaPipe hand landmarks.

No gesture classification, no HTTP dispatch, no arming. This exists purely to
prove MediaPipe sees your hands on this machine at a usable frame rate.

Run:
    .venv\\Scripts\\python.exe watcher\\gesture_watcher.py

Keys:
    q / Esc   quit
    d         toggle the landmark overlay (useful for judging lighting)
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
import hand_landmarks as hl

MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"


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


def draw_hud(frame, fps: float, hand_count: int, overlay_on: bool) -> None:
    """Milestone-1 HUD: just enough to tell whether tracking is healthy."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 34), (25, 25, 25), thickness=-1)
    cv2.putText(
        frame,
        f"MILESTONE 1  |  {fps:5.1f} fps  |  hands: {hand_count}",
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        config.COLOR_TEXT,
        1,
        cv2.LINE_AA,
    )
    hint = "q quit   d overlay" + ("" if overlay_on else "   [overlay off]")
    cv2.putText(
        frame, hint, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_MUTED, 1, cv2.LINE_AA
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Claudelash gesture watcher (milestone 1)")
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
    frame_times: deque[float] = deque(maxlen=30)
    overlay_on = True
    start = time.perf_counter()

    print("Ready. Press q or Esc in the preview window to quit.")

    with create_landmarker() as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Dropped frame from camera, retrying...", file=sys.stderr)
                continue

            if config.MIRROR_PREVIEW:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.perf_counter() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            hands = result.hand_landmarks or []
            if overlay_on:
                for landmarks in hands:
                    hl.draw_hand(frame, landmarks)

            frame_times.append(time.perf_counter())
            fps = 0.0
            if len(frame_times) > 1:
                span = frame_times[-1] - frame_times[0]
                if span > 0:
                    fps = (len(frame_times) - 1) / span

            draw_hud(frame, fps, len(hands), overlay_on)
            cv2.imshow("Claudelash - gesture watcher", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                overlay_on = not overlay_on

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
