"""Named landmark indices, the skeleton connections, and a drawing helper.

MediaPipe hands you 21 landmarks as a flat list. Indexing that list by number
in the classifier would be unreadable, so use the names from here instead:

    if lm[INDEX_TIP].y < lm[INDEX_PIP].y:   # index finger is extended

Coordinates are normalised to [0, 1] in image space. **y grows downward** —
a landmark higher on screen has a *smaller* y. This trips everyone up once.
"""

from __future__ import annotations

import cv2

# --- Landmark indices -------------------------------------------------------
WRIST = 0

THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_TIPS = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
FINGER_PIPS = (INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP)

# --- Skeleton ---------------------------------------------------------------
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    # thumb
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    # index
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    # middle
    (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    # ring
    (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    # pinky
    (WRIST, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
    # palm
    (INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, RING_MCP), (RING_MCP, PINKY_MCP),
)


def to_pixels(landmarks, width: int, height: int) -> list[tuple[int, int]]:
    """Normalised landmarks -> integer pixel coordinates, for drawing."""
    return [(int(p.x * width), int(p.y * height)) for p in landmarks]


def draw_hand(frame, landmarks, bone_color=(200, 200, 200), joint_color=(80, 220, 255)) -> None:
    """Draw the skeleton over `frame` in place.

    MediaPipe's old `solutions.drawing_utils` doesn't exist in the Tasks API,
    so we draw it ourselves. Twenty lines, and we get to pick the colours.
    """
    h, w = frame.shape[:2]
    pts = to_pixels(landmarks, w, h)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], bone_color, 2, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        radius = 5 if i in FINGER_TIPS or i == THUMB_TIP else 3
        cv2.circle(frame, pt, radius, joint_color, -1, cv2.LINE_AA)
