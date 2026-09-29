# domain_manifest.py -- builds a manifest for the Acoustic-UAV-Identification
# corpus, same shape as manifest.csv so it drops straight into baseline.py's
# existing functions. this is the domain generalization test, a completely
# separate mic, room, and drone recording setup from Binary_Drone_Audio, so
# scoring well here means the model learned drone sound and not this one
# room's fingerprint.

import csv
import sys
from pathlib import Path


# subset instead of family, same idea though: which bucket a clip came from,
# so results can be broken down later instead of trusting one aggregate
# number. drone+helicopter is the hard case on purpose, same background
# track as everything else with a helicopter layered in too
def classify(path, label):
    name = path.stem
    if label == 1 and "helicopter" in name:
        return "drone_helicopter"
    if label == 1:
        return "drone_traffic"
    return "no_drone"


def main(root, out):
    root = Path(root)
    entries = []

    for folder, label in (("drone", 1), ("no drone", 0)):
        d = root / "Real World Testing" / folder
        if not d.is_dir():
            sys.exit(f"missing {d} -- point this at the extracted corpus root")

        for wav_path in sorted(d.glob("*.wav")):
            entries.append({
                "path": str(wav_path.relative_to(root)),
                "label": label,
                "subset": classify(wav_path, label),
            })

    with open(out, "w", newline="") as f:
        csv_writer = csv.DictWriter(f, fieldnames=list(entries[0].keys()))
        csv_writer.writeheader()
        csv_writer.writerows(entries)

    pos = sum(r["label"] for r in entries)
    print(f"{len(entries)} clips -> positives {pos} / negatives {len(entries) - pos}")
    for subset in sorted({r["subset"] for r in entries}):
        n = sum(1 for r in entries if r["subset"] == subset)
        print(f"  {subset:16s} {n:4d} clips")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "domain_manifest.csv")
