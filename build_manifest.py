# build_manifest.py -- slaps together a csv manifest from the raw drone audio folders.
# group_id is the whole point: every 1-second clip gets the id of its parent recording
# so the splitter can keep neighbouring seconds together and not leak across train/test.

import csv
import re
import sys
import wave
from pathlib import Path
import os

# esc50 filename pattern: fold, source clip id, take letter, then esc50 target + segment index.
ESC50 = re.compile(r"^(\d+)-(\d+)-([A-Z]+)-(\d+)\.wav$")
# drone pattern: stem, then model, then segment index, trailing underscore.
DRONE = re.compile(r"^(.+?)-(bebop|membo)_(\d+)_\.wav$")
# leftover pattern: anything with a number at the end but no segment suffix.
FLAT = re.compile(r"^(.*?)(\d+)\.wav$")


# spits back (recording_id, family, model, augmented) for esc50-style names, or None.
def parse_esc50(name):
    match_obj = ESC50.match(name)
    if not match_obj:
        return None
    fold, clip, take, _ = match_obj.groups()
    # one esc50 source recording = fold + clip + take, five segments each
    return f"esc50_{fold}-{clip}-{take}", "esc50", None, False


# spits back tuple for drone-style names, or None.
def parse_drone(name):
    match_obj = DRONE.match(name)
    if not match_obj:
        return None
    stem, model, _ = match_obj.groups()
    # mixed_12-bebop is the same drone recording with noise on top, so it's its own group but flagged.
    augmented = stem.startswith("mixed")
    return f"drone_{stem}", "mixed" if augmented else "clean", model, augmented


# guesstimates recording_id, family, model, augmented. recording_id is the only thing that matters.
def classify(name, label):
    # try esc50 first
    result = parse_esc50(name)
    if result:
        return result

    # then drone
    result = parse_drone(name)
    if result:
        return result

    # leftovers: flat names
    match_obj = FLAT.match(name)
    if match_obj:
        stem, _ = match_obj.groups()
        stem = stem.rstrip("_")
        model = "membo" if "membo" in stem else None
        # no recording boundaries here, so the whole pile is one group. on purpose.
        # if silence leaks, it leaks all at once and shows up, not scattered.
        family = "extra_drone" if label == 1 else "synthetic_noise"
        return f"flat_{stem}", family, model, False

    # couldn't parse, toss it into unparsed bucket so we see it later
    return f"unparsed_{name}", "unparsed", None, False


# reads wav header and returns seconds as float, rounded to 4 decimals.
def duration(path):
    with wave.open(str(path)) as w:
        return round(w.getnframes() / w.getframerate(), 4)


def main(root, out):
    root = Path(root)
    entries = []  # renamed from rows

    # scan yes_drone with os.scandir, unknown with glob, just to be inconsistent.
    # yes_drone
    target_path = root / "yes_drone"
    if not target_path.is_dir():
        sys.exit(f"missing {target_path} -- point this at Binary_Drone_Audio")
    with os.scandir(target_path) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith(".wav"):
                wav_path = Path(entry.path)
                recording_id, family, model, aug = classify(wav_path.name, 1)
                entries.append({
                    "path": str(wav_path.relative_to(root)),
                    "label": 1,
                    "group_id": recording_id,
                    "family": family,
                    "drone_model": model or "",
                    "augmented": int(aug),
                    "seconds": duration(wav_path),
                })

    # unknown
    target_path = root / "unknown"
    if not target_path.is_dir():
        sys.exit(f"missing {target_path} -- point this at Binary_Drone_Audio")
    for wav_path in sorted(target_path.glob("*.wav")):
        recording_id, family, model, aug = classify(wav_path.name, 0)
        entries.append({
            "path": str(wav_path.relative_to(root)),
            "label": 0,
            "group_id": recording_id,
            "family": family,
            "drone_model": model or "",
            "augmented": int(aug),
            "seconds": duration(wav_path),
        })

    # write csv
    with open(out, "w", newline="") as f:
        csv_writer = csv.DictWriter(f, fieldnames=list(entries[0].keys()))
        csv_writer.writeheader()
        csv_writer.writerows(entries)

    # sanity check: if unparsed isn't zero, regexes drifted and the split is a lie
    unparsed = sum(1 for r in entries if r["family"] == "unparsed")
    groups = len({r["group_id"] for r in entries})
    pos = sum(r["label"] for r in entries)
    print(f"{len(entries)} clips -> {groups} groups")
    print(f"positives {pos} / negatives {len(entries) - pos}")
    print(f"unparsed {unparsed}")

    for fam in sorted({r["family"] for r in entries}):
        sub = [r for r in entries if r["family"] == fam]
        g = len({r["group_id"] for r in sub})
        print(f"  {fam:16s} {len(sub):6d} clips  {g:5d} groups")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "manifest.csv")
