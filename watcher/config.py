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
# Detection confidence was swept against the reference clip, which contains
# three repetitions — two slow, one fast:
#
#   0.7  ->  2 of 3   (lost the fast rep entirely to motion blur)
#   0.5  ->  3 of 3, no false positives   <- here
#   0.3  ->  3 of 3
#   0.2  ->  3 of 3 plus a false positive
#
# Below 0.5 buys nothing and eventually starts inventing hands.
MIN_DETECTION_CONFIDENCE = 0.5  # how sure it must be to find a hand at all
MIN_PRESENCE_CONFIDENCE = 0.5  # how sure it must be the hand is still there
MIN_TRACKING_CONFIDENCE = 0.5  # below this it re-runs full detection

# --- Dispatch (milestone 3+) ------------------------------------------------
# 127.0.0.1, NOT "localhost". On Windows "localhost" resolves to ::1 (IPv6)
# first, while the extension's server binds IPv4 only — so "localhost" gives a
# connection-refused that looks exactly like the extension not running.
SERVER_URL = "http://127.0.0.1:9247/gesture"
HOLD_DURATION_MS = 400  # how long a *static* gesture must persist before firing
COOLDOWN_MS = 1200  # ignore repeat fires of the same gesture within this window

# --- Gesture shapes and sequences -------------------------------------------
# How far apart the fingertips must be to count as an open palm, in units of
# hand size, measured on world landmarks. From the reference clip: the open
# palm sat at 0.37-0.43, the fist at 0.19-0.20, so 0.28 splits them with room
# on both sides.
OPEN_PALM_MIN_SPREAD = 0.28

# How long a whole sequence may take, from its first shape to its last.
# The reference clip ran 0.72-1.48s of open palm, then ~0.24s to close, so
# 2500ms leaves room for a slow, deliberate performance.
SEQUENCE_WINDOW_MS = 2500

# Restrict gestures to one hand? MediaPipe's handedness proved unreliable
# during fast motion in the reference clip — it reported both Left and Right
# for the same hand across the same clip. Off by default because of that; turn
# it on if your other hand keeps triggering things.
REQUIRE_HANDEDNESS = False
GESTURE_HAND = "Right"

# --- Finger extension -------------------------------------------------------
# How straight a finger must be, in degrees at its middle joint, to count as
# extended. Measured on MediaPipe's *world* landmarks, where a fully straight
# finger reads about 170 rather than a clean 180.
#
# These are lower than they'd be in 2D on purpose. Measuring in image space
# gave errors of up to 84 degrees depending only on which way the hand faced,
# which is what made a single raised finger read as two when the back of the
# hand was toward the camera.
#
# Calibrated against the reference clip, comparing the open-palm phase with the
# held-fist phase (degrees at the middle joint, world landmarks):
#
#   finger    palm p10    fist p90    margin
#   thumb          169         129       40
#   index          140          61       79
#   middle         150          41      109
#   ring           150          51       99
#   pinky          147          80       67
#
# The thresholds sit midway between the two, so both a lazily-held finger and
# a loosely-curled one land on the right side. Lower them if fingers you're
# holding out read as curled; raise them if slack fingers read as extended.
FINGER_STRAIGHT_DEG = 110.0
THUMB_STRAIGHT_DEG = 150.0

# --- Motion -----------------------------------------------------------------
# How many recent palm positions to keep. At ~25fps, 30 frames is about 1.2s —
# roughly one deliberate circle.
MOTION_HISTORY = 30

# Degrees swept around the path's own centre before it counts as circling.
# A full loop is 360; requiring less means a half-circle still registers, which
# matters because the palm phase is often cut short by closing into the fist.
CIRCLE_MIN_SWEPT_DEG = 200.0

# Minimum distance travelled, in aspect-corrected image widths. Stops a hand
# jittering in place from accumulating swept angle out of pure noise.
CIRCLE_MIN_PATH = 0.25

# --- Recording (watcher/record.py) ------------------------------------------
# Number key -> label. Record the gesture you *want*, and also the shapes it
# currently fires on by mistake — the difference between them is what the
# classifier rule gets built from. Add or rename freely.
RECORD_LABELS = {
    "1": "NOT_QUITE_MY_TEMPO",  # the real gesture you mean
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
