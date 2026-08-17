"""Run the hand tracker over a recorded video and describe what it saw.

Useful when a gesture is easier to perform than to describe. Record it on your
phone or with the Camera app, then:

    .venv\\Scripts\\python.exe watcher\\analyse_video.py "path\\to\\clip.mp4"

Prints a timeline collapsed into segments — consecutive frames sharing a finger
pattern become one row — plus per-finger straightness so you can see exactly
where the extension thresholds are landing.

Options:
    --mirror     flip frames, if the clip wasn't saved mirrored
    --every N    process every Nth frame (default 1)
    --frames     print every frame instead of collapsing into segments
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import cv2
import mediapipe as mp

import features
import gesture_watcher as gw
from features import FINGER_NAMES


def pattern_of(f: features.HandFeatures) -> str:
    return "".join(n[0].upper() if e else "-" for n, e in zip(FINGER_NAMES, f.extended))


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse a gesture video")
    parser.add_argument("path")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--frames", action="store_true")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="run the real classifier over the clip and report every firing",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"No such file: {path}", file=sys.stderr)
        return 1

    # A folder runs every clip in it, which is how you check a batch of takes
    # against the current thresholds in one go.
    if path.is_dir():
        clips = sorted(
            p for p in path.iterdir()
            if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".wmv"}
        )
        if not clips:
            print(f"No video files in {path}", file=sys.stderr)
            return 1
        print(f"Found {len(clips)} clips in {path}\n")
        worst = 0
        for clip in clips:
            print("=" * 70)
            sys.argv = [sys.argv[0], str(clip)] + (
                ["--simulate"] if args.simulate else []
            ) + (["--mirror"] if args.mirror else [])
            worst = max(worst, main())
        return worst

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"Could not open {path}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{path.name}: {w}x{h}, {fps:.1f} fps, {total} frames, "
          f"{total / fps:.1f}s\n")

    import config
    from motion import MotionTracker

    tracker = MotionTracker(config.MOTION_HISTORY, aspect=w / max(1, h))

    rows: list[tuple[float, features.HandFeatures]] = []
    fired_at: list[tuple[float, str]] = []
    shape_timeline: list[tuple[float, str | None]] = []
    swept_peak = 0.0
    index = 0
    with gw.create_landmarker() as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
            if index % args.every:
                continue
            if args.mirror:
                frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, int(index / fps * 1000))

            t = index / fps
            hand_features = gw.hand_features_from(result)
            if hand_features:
                rows.append((t, hand_features[0]))

            tracker.update(hand_features[0].centre if hand_features else None, t)
            swept_peak = max(swept_peak, abs(tracker.swept_deg))

            if args.simulate:
                # Feed the real pipeline using video time, so sequence windows
                # and cooldowns behave exactly as they would live.
                import gestures
                shape, fired = gestures.update(hand_features, t, tracker)
                shape_timeline.append((t, shape))
                if fired:
                    fired_at.append((t, fired))
                    tracker.reset()
    cap.release()

    if args.simulate:
        print("=== simulated firings (real classifier, video timing) ===")
        if fired_at:
            for t, name in fired_at:
                print(f"  {t:6.2f}s  {name}")
        else:
            print("  none — the gesture never completed")
        seen = [s for _, s in shape_timeline if s]
        counts: dict[str, int] = {}
        for s in seen:
            counts[s] = counts.get(s, 0) + 1
        print(f"  shapes recognised: {counts or '(none)'}")
        print(f"  peak swept angle: {swept_peak:.0f} deg "
              f"(threshold {config.CIRCLE_MIN_SWEPT_DEG:.0f})\n")

    if not rows:
        print("No hands detected anywhere in the clip.", file=sys.stderr)
        return 1

    detected = 100.0 * len(rows) * args.every / max(1, total)
    print(f"Hand detected in {len(rows)} sampled frames ({detected:.0f}% of clip)\n")

    if args.frames:
        print(f"{'t':>7}  {'pattern':<7} {'tilt':>7} {'spread':>7}  straightness")
        for t, f in rows:
            angles = " ".join(f"{a:3.0f}" for a in f.straightness)
            print(f"{t:7.2f}  {pattern_of(f):<7} {f.tilt_deg:7.1f} {f.spread:7.2f}  {angles}")
        return 0

    # Collapse consecutive frames that share a finger pattern.
    print("=== timeline (consecutive frames with the same finger pattern) ===")
    print(f"{'from':>7} {'to':>7} {'dur':>6}  {'pattern':<7} {'tilt':>14} {'spread':>7}")
    segments: list[tuple[float, float, str, list[float], list[float]]] = []
    for t, f in rows:
        p = pattern_of(f)
        if segments and segments[-1][2] == p:
            start, _, _, tilts, spreads = segments[-1]
            tilts.append(f.tilt_deg)
            spreads.append(f.spread)
            segments[-1] = (start, t, p, tilts, spreads)
        else:
            segments.append((t, t, p, [f.tilt_deg], [f.spread]))

    for start, end, p, tilts, spreads in segments:
        dur = end - start
        if dur < 0.10:  # ignore single-frame flickers
            continue
        print(f"{start:7.2f} {end:7.2f} {dur:6.2f}  {p:<7} "
              f"{min(tilts):+6.0f}..{max(tilts):+6.0f} {statistics.mean(spreads):7.2f}")

    print("\n=== per-finger straightness (degrees, 180 = straight) ===")
    print(f"{'finger':<8} {'min':>6} {'mean':>6} {'max':>6}   "
          f"threshold  reads-extended")
    import config
    for i, name in enumerate(FINGER_NAMES):
        vals = [f.straightness[i] for _, f in rows]
        thr = config.THUMB_STRAIGHT_DEG if i == 0 else config.FINGER_STRAIGHT_DEG
        pct = 100.0 * sum(1 for _, f in rows if f.extended[i]) / len(rows)
        print(f"{name:<8} {min(vals):6.0f} {statistics.mean(vals):6.0f} "
              f"{max(vals):6.0f}   {thr:9.0f}  {pct:13.0f}%")

    print("\n=== overall ===")
    hands = {f.handedness for _, f in rows}
    tilts = [f.tilt_deg for _, f in rows]
    print(f"handedness reported: {hands}")
    print(f"tilt range: {min(tilts):+.0f} .. {max(tilts):+.0f}")
    counts: dict[str, int] = {}
    for _, f in rows:
        counts[pattern_of(f)] = counts.get(pattern_of(f), 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:6]
    print("most common patterns: " +
          ", ".join(f"{p} {100 * c / len(rows):.0f}%" for p, c in top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
