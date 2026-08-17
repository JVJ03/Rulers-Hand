"""Tunables for the gesture watcher.

Everything you'd want to fiddle with lives here so you don't have to read the
main loop to change the feel of the thing.
"""

# --- Camera -----------------------------------------------------------------
# Picked by name, because DirectShow indices shuffle when you dock/undock.
# Case-insensitive substring match. Run `python watcher/list_cameras.py` to see
# what's attached.
CAMERA_NAME = "DELL Display 4MP Webcam"
CAMERA_INDEX = None  # set to an int to ignore CAMERA_NAME and pin an index

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30  # must be set explicitly or the driver may pick its slowest mode
MIRROR_PREVIEW = True  # flip horizontally so it behaves like a mirror

# --- MediaPipe Hands --------------------------------------------------------
MAX_HANDS = 2  # 2 from the start so TWO_HAND_SLAM works later without a rewrite
MIN_DETECTION_CONFIDENCE = 0.7  # how sure it must be to find a hand at all
MIN_PRESENCE_CONFIDENCE = 0.5  # how sure it must be the hand is still there
MIN_TRACKING_CONFIDENCE = 0.5  # below this it re-runs full detection

# --- Dispatch (milestone 3+) ------------------------------------------------
SERVER_URL = "http://localhost:9247/gesture"
HOLD_DURATION_MS = 400  # how long a gesture must persist before it fires
COOLDOWN_MS = 1200  # ignore repeat fires of the same gesture within this window

# --- Arming (milestone 5) ---------------------------------------------------
ARM_HOTKEY = "ctrl+alt+g"
START_ARMED = False  # always boot disarmed; you arm it deliberately

# --- Preview colours (BGR, because OpenCV) ----------------------------------
COLOR_ARMED = (80, 220, 100)
COLOR_DISARMED = (60, 60, 235)
COLOR_TEXT = (250, 250, 250)
COLOR_MUTED = (170, 170, 170)
