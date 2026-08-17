# CLAUDE.md — Claudelash

Hand gestures over a webcam drive a Claude Code session running in a VS Code
terminal. Personal toy, local-only, vibe-coded. Optimise for a fast working
loop and for **the gesture classifier being readable and easy to extend by
hand** — that's the part the owner tinkers with.

## Non-negotiable constraints

- **Localhost only.** No cloud services, no telemetry, no auth, no network
  egress beyond `127.0.0.1`. If a change would send anything off the machine,
  don't make it — ask first.
- **The extension is never published.** It runs in the Extension Development
  Host. No publisher id, no marketplace metadata, no `vsce publish`.
- **`classify()` stays simple.** Plain geometry on landmark coordinates, no ML
  layer, no gesture-recognition library. Each gesture is a small named helper
  with a comment explaining the geometry in words. If a change makes
  `gestures.py` harder to read, it's the wrong change.

## Repo layout

```
Claudelash/
├── watcher/
│   ├── gesture_watcher.py   main loop: capture → landmarks → classify → dispatch
│   ├── gestures.py          classify() + per-gesture helpers   ← the tinkering file
│   ├── hand_landmarks.py    named landmark indices, skeleton, draw_hand()
│   ├── config.py            all tunables (thresholds, timings, colours, hotkey)
│   └── models/              hand_landmarker.task (committed, 7.8 MB)
├── extension/
│   ├── src/extension.ts     HTTP server + terminal targeting
│   ├── gesture-map.json     gesture name → action, editable without recompiling
│   └── package.json
├── requirements.txt
└── README.md
```

## Commands

```powershell
# Python watcher
.venv\Scripts\python.exe watcher\gesture_watcher.py
.venv\Scripts\python.exe watcher\gesture_watcher.py --camera 1   # other webcam

# Extension
cd extension; npm install; npm run compile
# then F5 in VS Code → Extension Development Host
```

There is no test suite and no CI. Don't add either unless asked — the feedback
loop here is "wave at the camera and see what happens".

## Camera

Target is the **DELL Display 4MP Webcam** (the pop-up module on the P2724DEB
monitor), selected by name in `config.py`. Never hardcode an index: DirectShow
ordering shifts with docking state. Observed both ways round in one session —
the Dell was index 1, then index 0 after a reconnect.

The laptop's `Integrated Camera` is effectively blind while docked (measured
mean brightness 2.4/255, lid closed). It only exists as a fallback.

Running at **1920x1080**, which is the knee of the curve — *not* the sensor's
2560x1440 ceiling. Measured on the Dell, MJPG, `CAP_PROP_FPS=30`:

| Mode | capture | inference | end-to-end |
| --- | --- | --- | --- |
| 640x480 | 19.9 ms | 14.1 ms | 29.5 fps |
| 1280x720 | 8.6 ms | 26.9 ms | 29.4 fps |
| **1920x1080** | 3.9 ms | 37.2 ms | **24-28 fps** |
| 2560x1440 | 6.2 ms | 50.9 ms | 19.3 fps |

Cost is entirely in inference, not capture. Going 720p -> 1080p is nearly free;
going 1080p -> 1440p costs a third of the frame rate. MediaPipe's landmark
model runs on a **224px input** and downscales whatever it's handed, so 1440p
buys detail the model discards before it looks at the frame. Extra resolution
only helps when the hand is genuinely small in frame — i.e. sitting further
back — so re-run `watcher/tune_resolution.py` if the desk setup changes.

At ~25 fps a 400 ms hold gets ~10 frames to confirm. **Threaded capture is not
needed**; don't add that complexity unless the feel demands it.

`PREVIEW_SCALE` shrinks the preview window only — inference always runs on the
full-resolution frame.

For comparison, the integrated camera manages 11.2 fps at 720p (63 ms capture).

### If the Dell camera vanishes

It presents as `PID_D003`, a *separate* USB device from the monitor's
speakerphone (`PID_C034`). The speakerphone working tells you nothing about the
camera. If `Get-PnpDevice -Class Camera` shows `Present: False` with no problem
code, it is not attached to the bus — that is a physical/monitor-side state
(module retracted, or disabled in the OSD), not something software can fix.
Reboots and driver work do not help. Checked and ruled out once already:
driver faults, disabled state, apps holding the device, stale enumeration,
and USB SuperSpeed availability.

## Environment facts (verified 2026-08-17)

- Windows 11, PowerShell. `&&` does not work in this shell — use `;`.
- Python 3.13.15 at `py -3.13`; venv at `.venv/`. Python 3.14 also present,
  don't use it.
- Node v24.19.0. No `gh` CLI installed.
- The git repo root is `Claudelash/`, one level *below* the VS Code workspace
  root (`CLaudelash/`). Files written to the workspace root are outside the repo.

## Dependency pins

- `mediapipe==0.10.35` — **uses the Tasks API, not `mp.solutions`.** The legacy
  `mp.solutions.hands` API is gone in every mediapipe version that has Python
  3.13 wheels (0.10.30+); `import mediapipe as mp; mp.solutions` raises
  `AttributeError`. Any tutorial or snippet using `mp.solutions.hands` or
  `mp.solutions.drawing_utils` will not run here — translate it to
  `mediapipe.tasks.python.vision.HandLandmarker` instead.
- **The model file is committed** at `watcher/models/hand_landmarker.task`
  (7.8 MB, float16). Downloaded once from
  `storage.googleapis.com/mediapipe-models/...`; nothing fetches it at runtime.
  `.gitattributes` marks `*.task binary` so `text=auto` can't corrupt it.
- **Landmark drawing is hand-rolled** in `watcher/hand_landmarks.py`, because
  the Tasks API ships no drawing utilities. That module also holds the named
  landmark indices — use `hl.INDEX_TIP`, never `lm[8]`.
- `opencv-contrib-python==4.12.0.88` — mediapipe depends on *contrib*, and if
  left unpinned pip resolves it to 5.0.0.93. Stay on 4.12: **do not also install
  `opencv-python`**. Both packages own the `cv2` namespace, whichever landed
  last wins, and uninstalling one leaves the other broken in a confusing way
  (`cv2.line` vanishes while `cv2.CAP_DSHOW` still resolves). If cv2 ever goes
  weird, uninstall *both* and reinstall contrib alone with `--no-cache-dir`.
- Use `cv2.CAP_DSHOW` when opening the camera; the default MSMF backend on
  Windows can take 5–10s and may ignore resolution hints.
- `keyboard==0.13.5` — global hotkey. Works without admin on Windows (unlike
  Linux). It's unmaintained but fine for this.

## Landmark reference

MediaPipe Hands returns 21 landmarks per hand, normalised to `[0,1]` in image
space (`x` right, `y` **down**, `z` toward camera, roughly wrist-relative).

```
0  wrist
1-4    thumb   (cmc, mcp, ip, tip)
5-8    index   (mcp, pip, dip, tip)
9-12   middle
13-16  ring
17-20  pinky
```

Useful idioms: a finger is *extended* when `tip.y < pip.y` (remember y grows
downward); hand *scale* for normalising distances is `dist(wrist, middle_mcp)`.
Never compare raw pixel distances — they change with how close you sit.

## Design decisions already made

- **Preview is mirrored** (`cv2.flip`) so it reads as a mirror. Classification
  runs on the mirrored frame, so "left of screen" means "left as you see it".
- **Two hands tracked from the start** (`MAX_HANDS = 2`) so `TWO_HAND_SLAM`
  doesn't force a rewrite later.
- **Boots disarmed.** `START_ARMED = False`. Arming is a deliberate act.
- **Arming is a keyboard hotkey, never a gesture** — a gesture that arms the
  gesture system can arm itself by accident.
- **`MODEL_POINT` sends Escape first, then types.** Typing into Claude Code
  mid-response corrupts the input buffer. Escape → 150ms → type → Enter. This
  means MODEL_POINT implicitly interrupts; that's the accepted tradeoff.
- **Terminal targeting is by name, not `activeTerminal`.** A misfocused window
  must never receive keystrokes. See `gesture-map.json` / the terminal lookup
  in `extension.ts`.

## Milestones — build in order, don't skip ahead

Each milestone is tested by hand before the next one starts.

- [x] **1. Webcam + landmarks.** Preview window, overlay, FPS. No classification.
- [ ] **2. STOP_CHOP only.** `classify()` + hold-debounce. Prints to console.
- [ ] **3. Extension skeleton.** Local server logs POSTs to an output channel.
      Confirm the Python → extension round-trip before wiring real commands.
- [ ] **4. STOP_CHOP → Escape** into the Claude Code terminal, by name.
- [ ] **5. ARMED/DISARMED hotkey** + on-screen state. Disarmed must fully ignore.
- [ ] **6. FIST_HOLD, MODE_SWITCH, MODEL_POINT** — one at a time, tested alone.
- [ ] **7. TWO_HAND_SLAM** last, with extra confirmation. Destructive.

## Open questions to raise, not guess

- **What does "pause" mean for `FIST_HOLD`?** Claude Code has no documented
  pause keybinding. Confirm the intended behaviour before implementing.
- **What exactly does `TWO_HAND_SLAM` kill?** Terminal process, the whole
  terminal, or the VS Code window? Confirm before wiring — it's irreversible.
