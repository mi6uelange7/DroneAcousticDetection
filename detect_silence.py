# detect_silence.py -- flags clips that are pure digital silence, adds a
# silent column to manifest.csv. matters because ~301 synthetic_noise clips
# are exactly RMS 0.0 (deliberate silence samples on purpose) and a chunk of
# esc50 negatives are just quiet lead-in/lead-out from a 5s recording getting
# cut into 1s pieces. no drone clip comes anywhere near silent though, the
# quietest one is still RMS ~637, so a low threshold cant ever catch a real
# drone by accident.

import csv
import sys
import wave
from collections import Counter
from pathlib import Path

import numpy as np


def load_manifest(manifest_path):
    with open(manifest_path, newline="") as f:
        return list(csv.DictReader(f))


# raw pcm rms, one pass, straight off the wav. dont need to go through the
# cached spectrograms for this, the gap between silent and real audio is
# wide enough that a dumb threshold on raw energy is plenty.
def rms(wav_path):
    with wave.open(str(wav_path)) as w:
        frames = w.readframes(w.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples ** 2)))


def main(root, manifest_path, threshold):
    root = Path(root)
    entries = load_manifest(manifest_path)

    for r in entries:
        r["silent"] = int(rms(root / r["path"]) < threshold)

    # silent might already be a column from a previous run, keep it out of
    # the keys() order and stick it back on at the end so reruns dont
    # duplicate it
    fieldnames = [f for f in entries[0].keys() if f != "silent"] + ["silent"]
    with open(manifest_path, "w", newline="") as f:
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(entries)

    silent = [r for r in entries if r["silent"]]
    print(f"{len(silent)} / {len(entries)} clips silent ({100 * len(silent) / len(entries):.1f}%), threshold rms < {threshold}")

    print("by family:")
    fam_counter = Counter(r["family"] for r in silent)
    for fam in sorted(fam_counter):
        fam_total = sum(1 for r in entries if r["family"] == fam)
        print(f"  {fam:16s} {fam_counter[fam]:6d} / {fam_total:6d}")

    print("by label:")
    label_counter = Counter(int(r["label"]) for r in silent)
    for label in sorted(label_counter):
        print(f"  label {label}: {label_counter[label]}")

    # a silent positive means a drone clip with no drone in it somehow,
    # thats either a bad source file or a bug up above, not something
    # that should just happen on its own
    positives = [r for r in silent if int(r["label"]) == 1]
    if positives:
        print(f"WARNING: {len(positives)} positive clip(s) flagged silent, this should not happen:")
        for r in positives:
            print(f"  {r['path']}")


if __name__ == "__main__":
    root = sys.argv[1]
    manifest_path = sys.argv[2] if len(sys.argv) > 2 else "manifest.csv"
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    main(root, manifest_path, threshold)
