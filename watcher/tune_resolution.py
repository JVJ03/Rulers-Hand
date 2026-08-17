"""Measure which capture resolution actually works best, at your desk.

Raw frame rate is only half the question. The other half is whether your hand
lands on enough pixels for MediaPipe to read it — and that depends on how far
you sit from the camera, which no benchmark on my side can know.

Hold your hand up in a natural gesture position and run:

    .venv\\Scripts\\python.exe watcher\\tune_resolution.py

Each mode is sampled for a few seconds. Keep your hand roughly still and at a
consistent distance so the comparison is fair. Nothing is saved to disk.

What the columns mean:
    detect%   frames where a hand was found. Below ~95% is unusable.
    hand px   palm length in pixels. MediaPipe's landmark model works on a
              224px input, so a palm well under ~150px is losing real detail.
    fps       end-to-end, capture + inference.
"""

from __future__ import annotations

import statistics
import sys
import time

import cv2
import mediapipe as mp

import camera
import gesture_watcher as gw
import hand_landmarks as hl

MODES = [(640, 480), (1280, 720), (1920, 1080), (2560, 1440)]
SECONDS_PER_MODE = 4.0
MJPG = cv2.VideoWriter_fourcc(*"MJPG")


def palm_pixels(landmarks, width: int, height: int) -> float:
    """Wrist -> middle knuckle, in pixels. A stable proxy for apparent size."""
    wrist, mcp = landmarks[hl.WRIST], landmarks[hl.MIDDLE_MCP]
    dx = (wrist.x - mcp.x) * width
    dy = (wrist.y - mcp.y) * height
    return (dx * dx + dy * dy) ** 0.5


def sample(landmarker, index: int, w: int, h: int, stamp_base: int):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None, stamp_base
    try:
        cap.set(cv2.CAP_PROP_FOURCC, MJPG)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, 30)
        for _ in range(8):
            cap.read()

        got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = hits = 0
        sizes: list[float] = []
        stamp = stamp_base
        t_end = time.perf_counter() + SECONDS_PER_MODE
        t0 = time.perf_counter()

        while time.perf_counter() < t_end:
            ok, frame = cap.read()
            if not ok:
                continue
            frames += 1
            rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            stamp += 33
            result = landmarker.detect_for_video(img, stamp)
            if result.hand_landmarks:
                hits += 1
                sizes.append(palm_pixels(result.hand_landmarks[0], got_w, got_h))

        elapsed = time.perf_counter() - t0
        if frames == 0:
            return None, stamp
        return {
            "asked": (w, h),
            "got": (got_w, got_h),
            "fps": frames / elapsed,
            "detect": 100.0 * hits / frames,
            "palm": statistics.median(sizes) if sizes else 0.0,
        }, stamp
    finally:
        cap.release()


def main() -> int:
    index, explanation = camera.resolve_index()
    print(f"Camera: {explanation}")
    print("Hold your hand up, roughly where you'd gesture, and keep still.\n")

    landmarker = gw.create_landmarker()
    stamp = 0
    rows = []
    try:
        for w, h in MODES:
            print(f"  sampling {w}x{h} ...", end="", flush=True)
            row, stamp = sample(landmarker, index, w, h, stamp)
            print(" done" if row else " failed")
            if row:
                rows.append(row)
    finally:
        landmarker.close()

    if not rows:
        print("\nNo modes produced frames. Is the camera free?", file=sys.stderr)
        return 1

    print(f"\n{'mode':>11}  {'fps':>6}  {'detect%':>8}  {'hand px':>8}")
    for r in rows:
        gw_, gh_ = r["got"]
        fallback = "" if r["got"] == r["asked"] else "  (fell back)"
        print(
            f"{gw_}x{gh_:<6}  {r['fps']:6.1f}  {r['detect']:7.1f}%  "
            f"{r['palm']:7.0f}{fallback}"
        )

    usable = [r for r in rows if r["detect"] >= 95.0 and r["palm"] >= 150.0]
    if usable:
        best = min(usable, key=lambda r: r["got"][0] * r["got"][1])
        w, h = best["got"]
        print(
            f"\nRecommended: {w}x{h} — the cheapest mode that still detects "
            f"reliably with a large enough hand.\nSet FRAME_WIDTH/FRAME_HEIGHT "
            f"in watcher/config.py."
        )
    else:
        print(
            "\nNo mode cleared both thresholds. If detect% is low everywhere, "
            "check lighting.\nIf hand px is small everywhere, sit closer or "
            "keep the highest resolution."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
