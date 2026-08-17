"""Probe camera indices and report which ones actually deliver frames.

Run this when the watcher can't open a camera, or when you're not sure which
index is the webcam you mean:

    .venv\\Scripts\\python.exe watcher\\list_cameras.py

Opens each index in turn, grabs a couple of frames, prints what came back, and
releases it. No image is written to disk and nothing leaves the machine — the
only thing kept from each frame is its resolution and mean brightness.
"""

from __future__ import annotations

import argparse

import cv2

# CAP_DSHOW is the DirectShow backend. On Windows the default (MSMF) can take
# 5-10s per index and reports failures inconsistently, which makes probing slow
# and unreliable. CAP_MSMF is included as a fallback because a few UVC cameras
# only enumerate under it.
BACKENDS = [("CAP_DSHOW", cv2.CAP_DSHOW), ("CAP_MSMF", cv2.CAP_MSMF)]


def probe(index: int, backend_name: str, backend_id: int) -> str | None:
    """Try one index on one backend. Returns a description, or None if dead."""
    cap = cv2.VideoCapture(index, backend_id)
    try:
        if not cap.isOpened():
            return None
        # First frame off a cold camera is often garbage or black; take a few.
        frame = None
        for _ in range(5):
            ok, candidate = cap.read()
            if ok and candidate is not None:
                frame = candidate
        if frame is None:
            return f"opens but delivers no frames ({backend_name})"

        h, w = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        brightness = float(frame.mean())
        note = "  <-- all black, lens cover or privacy shutter?" if brightness < 2.0 else ""
        return f"{w}x{h} @ {fps:.0f}fps  mean-brightness {brightness:5.1f}  ({backend_name}){note}"
    finally:
        cap.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="List working camera indices")
    parser.add_argument("--max-index", type=int, default=4, help="highest index to try")
    args = parser.parse_args()

    print(f"opencv {cv2.__version__}\n")

    import camera as cam

    names = cam.list_devices()
    if names:
        print("DirectShow reports these devices:")
        for i, n in enumerate(names):
            print(f"  [{i}] {n}")
        print()

    print(f"Probing indices 0..{args.max_index}\n")
    found: list[int] = []

    for index in range(args.max_index + 1):
        label = f" {names[index]!r}" if index < len(names) else ""
        for backend_name, backend_id in BACKENDS:
            result = probe(index, backend_name, backend_id)
            if result:
                print(f"  [{index}]{label}  {result}")
                if index not in found:
                    found.append(index)
                break  # first backend that works is the one the watcher will use
        else:
            busy = "  (enumerated but won't open — something else is holding it?)" if label else ""
            print(f"  [{index}]{label}  -{busy}")

    print()
    if not found:
        print("No cameras found. Check Windows Settings > Privacy & security > Camera,")
        print("and close anything holding the webcam (Teams is the usual culprit).")
        return 1

    print(f"Working indices: {found}")
    print(f"Use the watcher with:  --camera {found[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
