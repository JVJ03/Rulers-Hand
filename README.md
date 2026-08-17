# Claudelash

Wave at your webcam, and Claude Code does what you told it to.

A Python watcher reads hand landmarks from the webcam, classifies them into
named gestures, and POSTs them to a tiny local VS Code extension that turns
each gesture into a keystroke aimed at the terminal running Claude Code.

Personal toy. Runs entirely on `localhost` — no cloud, no telemetry, no auth.
The extension is never published; it runs in the Extension Development Host.

```
webcam ──▶ OpenCV ──▶ MediaPipe Hands ──▶ classify() ──▶ hold-debounce
                                                              │
                                              POST localhost:9247/gesture
                                                              │
                                      VS Code extension ──▶ Claude Code terminal
```

## Gesture map

| Gesture | Action | Status |
| --- | --- | --- |
| `STOP_CHOP` | Escape — interrupt Claude Code | milestone 2–4 |
| `FIST_HOLD` | pause (behaviour TBD) | milestone 6 |
| `MODE_SWITCH` | Shift+Tab — cycle mode | milestone 6 |
| `MODEL_POINT` | type `/model` + Enter | milestone 6 |
| `TWO_HAND_SLAM` | kill the session (destructive) | milestone 7 |

Nothing fires unless the watcher is **armed** (`Ctrl+Alt+G`). It boots disarmed.

## Setup

### Python watcher

Needs Python 3.13 (3.14 is untested; MediaPipe 0.10.35 has 3.13 wheels).

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Deps: `mediapipe`, `opencv-contrib-python`, `requests`, `keyboard`. The hand
landmarker model is committed in `watcher/models/`, so nothing is downloaded at
runtime.

In VS Code, run **Python: Select Interpreter** and pick `.venv` — otherwise the
editor will underline every import.

### VS Code extension

```powershell
cd extension
npm install
npm run compile
```

## Running it

**1. Start the watcher.**

```powershell
.venv\Scripts\python.exe watcher\gesture_watcher.py
```

A preview window opens. Keys: `q` or `Esc` to quit, `d` to toggle the landmark
overlay. If the camera doesn't open, try `--camera 1` — and check nothing else
is holding the webcam (Teams is the usual culprit).

**2. Load the extension.**

Open the `extension/` folder in VS Code and press `F5`. That launches an
Extension Development Host window with the extension active. Open the
**Output** panel and pick **Claudelash** from the dropdown to watch gestures
arrive.

The gesture → action mapping lives in [`extension/gesture-map.json`](extension/gesture-map.json)
and is re-read at runtime, so you can retune it without recompiling.

## Tuning the classifier

Everything you'd want to adjust is in [`watcher/config.py`](watcher/config.py) —
hold duration, cooldown, detection confidence, the arm hotkey, preview colours.

The gestures themselves live in [`watcher/gestures.py`](watcher/gestures.py).
Each one is a small named function doing plain geometry on the 21 landmarks,
with the shape described in a comment. Adding a gesture means writing one
function and adding one line to `classify()`. The landmark index reference is
in [CLAUDE.md](CLAUDE.md).

## Build order

Built and hand-tested one milestone at a time — see the checklist in
[CLAUDE.md](CLAUDE.md). Current state: **milestone 1 complete** (webcam +
landmark overlay, no classification yet).

## Known tradeoffs

- **`MODEL_POINT` interrupts.** Typing into Claude Code while it's generating
  corrupts the input buffer, so the extension sends Escape first, waits 150ms,
  then types. Switching model therefore also stops whatever was in flight.
- **`keyboard` is unmaintained.** Fine on Windows without admin; would need
  root on Linux.
- **Gestures are geometric, not learned.** They'll misfire in bad lighting or
  at odd hand angles. Raising `HOLD_DURATION_MS` costs responsiveness but buys
  a lot of reliability.
