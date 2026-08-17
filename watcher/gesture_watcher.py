"""Gesture watcher — MILESTONE 2.

Shows the webcam with hand landmarks and a live readout of what the program
thinks your hands are doing, classifies NOT_QUITE_MY_TEMPO, and prints to the console
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
from dispatch import Dispatcher
from motion import MotionTracker


def hand_features_from(result) -> list[features.HandFeatures]:
    """Build features for every detected hand, pairing image and world landmarks."""
    hands = result.hand_landmarks or []
    world = getattr(result, "hand_world_landmarks", None) or []
    out = []
    for i, landmarks in enumerate(hands):
        out.append(
            features.extract(
                landmarks,
                world[i] if i < len(world) else None,
                read_handedness(result, i),
            )
        )
    return out

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

    # flush=True throughout: when stdout is piped (a log file, a background
    # task) Python fully buffers it, and a watcher that prints nothing for
    # minutes looks hung rather than idle.
    print(f"mediapipe {mp.__version__} | opencv {cv2.__version__}", flush=True)
    print("Opening camera... (first frame can take a second)", flush=True)

    cap = camera.open_camera(args.camera)
    capture_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    frame_times: deque[float] = deque(maxlen=30)
    read_times: deque[float] = deque(maxlen=30)
    infer_times: deque[float] = deque(maxlen=30)
    show_landmarks = True
    show_panels = True
    start = time.perf_counter()

    dispatcher = Dispatcher()
    tracker = MotionTracker(
        config.MOTION_HISTORY,
        aspect=capture_size[0] / max(1, capture_size[1]),
        max_age_s=config.MOTION_MAX_AGE_S,
    )
    last_fire_at = -1e9
    last_fired_name: str | None = None
    fire_count = 0

    print(f"Ready at {capture_size[0]}x{capture_size[1]}. {KEY_HINTS}", flush=True)

    # The Dell module has a habit of streaming perfectly-black frames at higher
    # resolutions while reporting itself healthy — a bandwidth-starved link
    # rather than a covered lens. Say so plainly instead of leaving a dark
    # window and no explanation.
    black_streak = 0
    black_check_done = False

    with create_landmarker() as landmarker:
        while True:
            t_read = time.perf_counter()
            ok, frame = cap.read()
            t_read_done = time.perf_counter()
            if not ok:
                print("Dropped frame from camera, retrying...", file=sys.stderr)
                continue

            # Only until the question is settled. `frame.max()` scans every
            # pixel — 6 million of them at 1080p — so leaving it in the hot
            # loop costs real frame rate. One frame with light in it proves the
            # camera works, and the check switches off for good.
            if not black_check_done:
                if frame.max() > 0:
                    black_check_done = True
                    black_streak = 0
                else:
                    black_streak += 1
                if black_streak >= 30:
                    black_check_done = True
                    print(
                        f"\n  WARNING: 30 consecutive all-black frames at "
                        f"{capture_size[0]}x{capture_size[1]}.\n"
                        f"  The camera is streaming but no light is getting "
                        f"through. On the Dell module this\n"
                        f"  usually means the link can't sustain this "
                        f"resolution — try a lower one:\n"
                        f"    FRAME_WIDTH = 640 / FRAME_HEIGHT = 480 in "
                        f"watcher/config.py\n",
                        flush=True,
                    )

            if config.MIRROR_PREVIEW:
                frame = cv2.flip(frame, 1)

            # Inference runs on a downscaled copy; the preview keeps the full
            # frame. Landmarks come back normalised, so they overlay correctly
            # on the big image regardless of what the model was fed.
            infer_frame = frame
            if config.INFERENCE_WIDTH and frame.shape[1] > config.INFERENCE_WIDTH:
                scaled_h = int(frame.shape[0] * config.INFERENCE_WIDTH / frame.shape[1])
                infer_frame = cv2.resize(
                    frame, (config.INFERENCE_WIDTH, scaled_h),
                    interpolation=cv2.INTER_AREA,
                )
            rgb = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.perf_counter() - start) * 1000)
            t_infer = time.perf_counter()
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            t_infer_done = time.perf_counter()

            # Rolling averages, so the HUD says where the time actually goes.
            # Capture climbing while inference stays flat means the camera link
            # is the problem, not the code.
            read_times.append(t_read_done - t_read)
            infer_times.append(t_infer_done - t_infer)

            hands = result.hand_landmarks or []
            hand_features = hand_features_from(result)

            now = time.monotonic()
            tracker.update(hand_features[0].centre if hand_features else None, now)
            shape, fired = gestures.update(hand_features, now, tracker)
            if fired:
                tracker.reset()  # start the next circle from scratch
                fire_count += 1
                last_fire_at = now
                last_fired_name = fired
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
                overlay.draw_motion_trail(view, tracker)
                overlay.draw_hands(view, hands)
            if show_panels:
                for slot, feature in enumerate(hand_features[:2]):
                    overlay.draw_hand_panel(view, feature, slot)
                overlay.draw_gesture_panel(
                    view,
                    hand_features,
                    shape,
                    gestures.SEQUENCES,
                    now - last_fire_at,
                    tracker,
                )

            overlay.draw_last_fired(view, last_fired_name, now - last_fire_at, fire_count)
            timing = ""
            if read_times:
                timing = (f"cap {1000 * sum(read_times) / len(read_times):.0f}ms  "
                          f"inf {1000 * sum(infer_times) / len(infer_times):.0f}ms")
            overlay.draw_top_bar(view, fps, capture_size, len(hands),
                                 dispatcher.status, timing)
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
