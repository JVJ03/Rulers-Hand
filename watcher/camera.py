"""Camera selection and opening.

Indices are not stable on a docked laptop — plug the monitor in or out and the
DirectShow ordering can change underneath you. So the watcher picks a camera by
*name* and only falls back to an index.

`pygrabber` is used purely to read DirectShow's device-name list; capture itself
is still plain OpenCV. It's Windows-only, which is fine — so is this project.
"""

from __future__ import annotations

import cv2

import config


def list_devices() -> list[str]:
    """DirectShow capture device names, in index order. Empty if unavailable."""
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        return []
    try:
        return list(FilterGraph().get_input_devices())
    except Exception:
        return []


def resolve_index() -> tuple[int, str]:
    """Work out which camera index to open. Returns (index, human explanation)."""
    if config.CAMERA_INDEX is not None:
        return config.CAMERA_INDEX, f"index {config.CAMERA_INDEX} (pinned in config.py)"

    devices = list_devices()
    if not devices:
        return 0, "index 0 (could not enumerate device names, falling back)"

    wanted = config.CAMERA_NAME.lower()
    for index, name in enumerate(devices):
        if wanted in name.lower():
            return index, f"index {index} = {name!r}"

    available = ", ".join(f"[{i}] {n}" for i, n in enumerate(devices))
    raise SystemExit(
        f"No camera matching CAMERA_NAME={config.CAMERA_NAME!r}.\n"
        f"Available: {available}\n"
        f"Fix CAMERA_NAME in watcher/config.py, or set CAMERA_INDEX to pin one."
    )


def open_camera(index: int | None = None) -> cv2.VideoCapture:
    """Open a camera and negotiate a sane capture format.

    Three settings matter here, and all three were found the hard way:

    - CAP_DSHOW: the default MSMF backend on Windows can take 5-10s to open and
      reports failures inconsistently.
    - FOURCC=MJPG *before* the resolution: without it the driver may fall back
      to an uncompressed mode that USB can only sustain at a few frames/sec.
    - CAP_PROP_FPS: if left alone the driver may pick its slowest advertised
      mode. Measured on this machine: unset gave 1.0 fps, set gave 10.0 fps.
    """
    if index is None:
        index, explanation = resolve_index()
        print(f"Camera: {explanation}")

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera index {index}.\n"
            f"Most likely something else is holding it. Close the Windows\n"
            f"Settings > Bluetooth & devices > Cameras page, Teams, or any\n"
            f"browser tab with camera access, then try again."
        )

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (actual_w, actual_h) != (config.FRAME_WIDTH, config.FRAME_HEIGHT):
        print(
            f"  note: asked for {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}, "
            f"camera gave {actual_w}x{actual_h}"
        )
    return cap
