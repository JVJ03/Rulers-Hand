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

# 1920x1080 is the knee of the curve, not the sensor's maximum.
#
#   640x480    29.5 fps    hand too small to read well
#   1280x720   29.4 fps    marginal detail at monitor distance
#   1920x1080  27.9 fps    <- here
#   2560x1440  19.3 fps    costs a third of the frame rate
#
# Going up from 720p is nearly free; going up from 1080p is not. MediaPipe's
# landmark model runs on a 224px input and downscales whatever it's given, so
# 1440p spends inference time on detail the model discards before it looks.
# Re-check with `python watcher/tune_resolution.py` if you move the camera or
# change where you sit.
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
TARGET_FPS = 30  # must be set explicitly or the driver may pick its slowest mode
MIRROR_PREVIEW = True  # flip horizontally so it behaves like a mirror

# The preview window only — inference still runs on the full-resolution frame.
PREVIEW_SCALE = 0.5

# --- MediaPipe Hands --------------------------------------------------------
MAX_HANDS = 2  # 2 from the start so TWO_HAND_SLAM works later without a rewrite
MIN_DETECTION_CONFIDENCE = 0.7  # how sure it must be to find a hand at all
MIN_PRESENCE_CONFIDENCE = 0.5  # how sure it must be the hand is still there
MIN_TRACKING_CONFIDENCE = 0.5  # below this it re-runs full detection

# --- Dispatch (milestone 3+) ------------------------------------------------
# 127.0.0.1, NOT "localhost". On Windows "localhost" resolves to ::1 (IPv6)
# first, while the extension's server binds IPv4 only — so "localhost" gives a
# connection-refused that looks exactly like the extension not running.
SERVER_URL = "http://127.0.0.1:9247/gesture"
HOLD_DURATION_MS = 400  # how long a gesture must persist before it fires
COOLDOWN_MS = 1200  # ignore repeat fires of the same gesture within this window

# --- Recording (watcher/record.py) ------------------------------------------
# Number key -> label. Record the gesture you *want*, and also the shapes it
# currently fires on by mistake — the difference between them is what the
# classifier rule gets built from. Add or rename freely.
RECORD_LABELS = {
    "1": "STOP_CHOP",  # the real gesture you mean
    "2": "OPEN_HAND",  # plain open palm — currently a false positive
    "3": "RELAXED",  # hand just resting in frame, doing nothing
    "4": "SPARE_A",
    "5": "SPARE_B",
}

# --- Arming (milestone 5) ---------------------------------------------------
ARM_HOTKEY = "ctrl+alt+g"
START_ARMED = False  # always boot disarmed; you arm it deliberately

# --- Preview colours (BGR, because OpenCV) ----------------------------------
COLOR_ARMED = (80, 220, 100)
COLOR_DISARMED = (60, 60, 235)
COLOR_TEXT = (250, 250, 250)
COLOR_MUTED = (170, 170, 170)
