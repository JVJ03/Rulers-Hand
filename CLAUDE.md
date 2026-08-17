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

```powershell
# Check classify() + debounce against synthetic hands, no camera needed
.venv\Scripts\python.exe watcher\selftest.py
```

No CI, and no test suite beyond `selftest.py` — don't add one unless asked. The
feedback loop here is "wave at the camera and see what happens". `selftest.py`
exists only because the classifier is the part that gets edited most, and
catching a broken threshold without standing up is worth the one file.

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

### Cost with a hand actually in frame

The numbers above are with an empty frame, which flatters them badly — the
landmark model only runs once a hand is *found*, and again for each extra
hand. Measured over 150 frames of real footage:

| MAX_HANDS | infer at | prep | inference | total | fps |
| --- | --- | --- | --- | --- | --- |
| 2 | full | 8.7 ms | 42.4 ms | 51.1 ms | 19.6 |
| 1 | full | 9.0 ms | 27.4 ms | 36.4 ms | 27.5 |
| **1** | **640w** | 4.5 ms | 30.1 ms | **34.6 ms** | **28.9** |

Two levers, both applied:

- **`MAX_HANDS = 2`** — kept at 2 so both hands are tracked. Costs about 15ms
  per frame against 1 hand; INFERENCE_WIDTH offsets part of it.
- **`INFERENCE_WIDTH = 640`** — inference runs on a downscaled copy while the
  preview keeps full resolution. Free, because MediaPipe resizes to a 192px
  input internally anyway and returns *normalised* landmarks. Detection rate
  and every derived feature measured identical at 1920/1280/960/640.

Don't "optimise" by resizing to 1280 — that measured *slower* than passing the
full frame, because the resize costs more than it saves.

Verified quality-neutral: the v2 reference clip fires 6 times at identical
timestamps before and after.

At ~25 fps a 400 ms hold gets ~10 frames to confirm. **Threaded capture is not
needed**; don't add that complexity unless the feel demands it.

`PREVIEW_SCALE` shrinks the preview window only — inference always runs on the
full-resolution frame.

For comparison, the integrated camera manages 11.2 fps at 720p (63 ms capture).

### If the Dell camera enumerates but won't open

Distinct from the failure below, and it looks healthy in every Windows view:
`Present: True`, `Status: OK`, `Problem: 0`, listed by DirectShow — yet
`cv2.VideoCapture` refuses it on **all three** backends (DSHOW, MSMF, ANY)
while the integrated camera opens fine on all three. The consent store reports
no app using a camera, and it persists with Teams closed and no Python running.

First suspect is still an app holding it — **Teams reserves this camera on a
conferencing monitor even outside a call**, and the new Teams client runs on
WebView2, so orphaned `msedgewebview2` processes can outlive it. Point Teams at
the `Integrated Camera` in its device settings to avoid the fight entirely.

When nothing holds it, the module itself is in a bad state. Power-cycle it by
pushing the pop-up down and back up. Software has no reach here — checked and
ruled out: all capture backends, driver problem codes, the consent store,
stray processes, and the biometric service.

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

## The localhost link

Watcher POSTs JSON `{"gesture": ..., "held_ms": ...}` to
`http://127.0.0.1:9247/gesture`.

**Use `127.0.0.1`, never `localhost`.** On Windows `localhost` resolves to
`::1` (IPv6) first, and the extension's server binds IPv4 only — so
`localhost` produces a connection-refused that looks identical to the
extension not being loaded. This cost real debugging time once.

`Dispatcher` posts from a worker thread with a 0.5 s timeout so a wedged
server can never stall the capture loop, and treats "extension not running"
as a normal state rather than an error — the watcher is often up before you
press F5. The top bar shows the link status in red when posts are failing.

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
- [x] **2. NOT_QUITE_MY_TEMPO only.** `classify()` + hold-debounce. Prints to console.
      Right hand, flat open palm, held upright — Fletcher's "not quite my
      tempo". Static pose, deliberately not a chopping motion.
- [x] **3. Extension skeleton.** Local server logs POSTs to an output channel.
      Round-trip verified against a stub server (3/3 delivered).
- [ ] **4. NOT_QUITE_MY_TEMPO → Escape** into the Claude Code terminal, by name.
- [ ] **5. ARMED/DISARMED hotkey** + on-screen state. Disarmed must fully ignore.
- [ ] **6. FIST_HOLD, MODE_SWITCH, MODEL_POINT** — one at a time, tested alone.
- [ ] **7. TWO_HAND_SLAM** last, with extra confirmation. Destructive.

## Open questions to raise, not guess

- **What does "pause" mean for `FIST_HOLD`?** Claude Code has no documented
  pause keybinding. Confirm the intended behaviour before implementing.
- **What exactly does `TWO_HAND_SLAM` kill?** Terminal process, the whole
  terminal, or the VS Code window? Confirm before wiring — it's irreversible.
