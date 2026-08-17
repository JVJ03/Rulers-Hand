"""Read recorded samples and work out what actually separates the gestures.

Run after `record.py`:

    .venv\\Scripts\\python.exe watcher\\analyse.py

For every feature it prints the range each label occupies, then scores how well
that feature separates the labels. A feature whose ranges don't overlap is one
you can write a clean threshold against; a feature whose ranges sit on top of
each other is useless no matter how obvious the difference looks to you.

The finger-pattern table at the end is usually the most immediately useful
part — if two gestures differ by which fingers are out, that's your rule and
you don't need thresholds at all.
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import features
from features import FINGER_NAMES

SAMPLES_PATH = Path(__file__).parent.parent / "data" / "samples.csv"


class Point:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x, self.y, self.z = x, y, z


def load() -> dict[str, list[features.HandFeatures]]:
    if not SAMPLES_PATH.exists():
        print(f"No samples at {SAMPLES_PATH}.", file=sys.stderr)
        print("Record some first:  .venv\\Scripts\\python.exe watcher\\record.py",
              file=sys.stderr)
        raise SystemExit(1)

    by_label: dict[str, list[features.HandFeatures]] = defaultdict(list)
    with SAMPLES_PATH.open(newline="", encoding="utf8") as fh:
        for row in csv.DictReader(fh):
            landmarks = [
                Point(float(row[f"x{i}"]), float(row[f"y{i}"]), float(row[f"z{i}"]))
                for i in range(21)
            ]
            by_label[row["label"]].append(
                features.extract(landmarks, row["handedness"])
            )
    return dict(by_label)


def separation(a: list[float], b: list[float]) -> float:
    """How cleanly two value sets split, 0 = identical, 1+ = no overlap.

    Distance between the means, in units of their combined spread. Anything
    above ~2 is a threshold you can rely on; below ~1 will misfire.
    """
    if len(a) < 2 or len(b) < 2:
        return 0.0
    spread = statistics.pstdev(a) + statistics.pstdev(b)
    if spread < 1e-9:
        return 99.0 if abs(statistics.mean(a) - statistics.mean(b)) > 1e-9 else 0.0
    return abs(statistics.mean(a) - statistics.mean(b)) / spread


def main() -> int:
    by_label = load()
    labels = sorted(by_label)
    print(f"Loaded {sum(len(v) for v in by_label.values())} samples "
          f"across {len(labels)} labels\n")
    for label in labels:
        print(f"  {label:<12} {len(by_label[label]):5d} frames")

    print("\n=== finger patterns (how often each combination appeared) ===")
    print("    T=thumb I=index M=middle R=ring P=pinky, '-' = curled\n")
    for label in labels:
        patterns = Counter()
        for f in by_label[label]:
            patterns["".join(
                n[0].upper() if e else "-" for n, e in zip(FINGER_NAMES, f.extended)
            )] += 1
        total = sum(patterns.values())
        top = ", ".join(f"{p} {100 * c / total:.0f}%" for p, c in patterns.most_common(3))
        print(f"  {label:<12} {top}")

    print("\n=== continuous features ===")
    numeric = {
        "tilt_deg": lambda f: f.tilt_deg,
        "spread": lambda f: f.spread,
        "scale": lambda f: f.scale,
        "extended_count": lambda f: float(f.extended_count),
    }
    for name, getter in numeric.items():
        print(f"\n  {name}")
        values = {label: [getter(f) for f in by_label[label]] for label in labels}
        for label in labels:
            v = values[label]
            if not v:
                continue
            print(f"    {label:<12} min {min(v):8.2f}  mean {statistics.mean(v):8.2f}  "
                  f"max {max(v):8.2f}")
        # Pairwise separation, best pairs first.
        pairs = [
            (separation(values[a], values[b]), a, b)
            for i, a in enumerate(labels) for b in labels[i + 1:]
            if values[a] and values[b]
        ]
        for score, a, b in sorted(pairs, reverse=True):
            verdict = "CLEAN" if score >= 2 else "usable" if score >= 1 else "overlaps"
            print(f"      {a} vs {b}: separation {score:5.2f}  {verdict}")

    print("\n=== handedness ===")
    for label in labels:
        hands = Counter(f.handedness for f in by_label[label])
        print(f"  {label:<12} {dict(hands)}")

    print("\nLook for CLEAN separations, or a finger pattern that's ~100% for one")
    print("label and absent from the others. That's what the rule should test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
