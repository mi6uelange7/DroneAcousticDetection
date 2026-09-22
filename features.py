# features.py -- turns each clip into a mel-spectrogram and caches it as .npy
# so training never has to recompute them. run once, reuse forever.

import csv
import sys
from pathlib import Path

import librosa
import numpy as np

SR = 16000              # native rate, dont let librosa resample
N_FFT = 1024             # 64ms window
HOP_LENGTH = 256          # 16ms step
N_MELS = 64
TARGET_SAMPLES = 16384  # 1.024s at 16khz, most clips are exactly this


def load_manifest(manifest_path):
    with open(manifest_path, newline="") as f:
        return list(csv.DictReader(f))


def fixed_length(y):
    # some clips run short (the last segment of a recording is often 0.896s
    # instead of 1.024s), pad those with zeros so every spectrogram comes out
    # the same shape. truncate the rare long one too, just in case.
    if len(y) < TARGET_SAMPLES:
        return np.pad(y, (0, TARGET_SAMPLES - len(y)))
    return y[:TARGET_SAMPLES]


# mel first cause its the standard starting point, power_to_db after cause
# raw power values are tiny and swing over a huge range, db compresses that
# down to something a model can actually learn from
def spectrogram(wav_path):
    y, _ = librosa.load(wav_path, sr=SR)
    y = fixed_length(y)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    return librosa.power_to_db(mel).astype(np.float32)


def main(root, manifest_path, cache_dir):
    root = Path(root)
    cache_dir = Path(cache_dir)
    entries = load_manifest(manifest_path)

    index_rows = []
    for i, r in enumerate(entries):
        wav_path = root / r["path"]
        spec = spectrogram(wav_path)

        npy_path = cache_dir / Path(r["path"]).with_suffix(".npy")
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, spec)

        index_rows.append({
            "path": r["path"],
            "npy_path": str(npy_path.relative_to(cache_dir)),
            "shape": "x".join(map(str, spec.shape)),
        })

        if (i + 1) % 2000 == 0:
            print(f"{i + 1}/{len(entries)} done")

    with open(cache_dir / "index.csv", "w", newline="") as f:
        csv_writer = csv.DictWriter(f, fieldnames=["path", "npy_path", "shape"])
        csv_writer.writeheader()
        csv_writer.writerows(index_rows)

    print(f"{len(entries)} spectrograms cached to {cache_dir}, shape {index_rows[0]['shape']}")

    # normalization isnt baked in here on purpose. per-clip normalization
    # (subtract each clips own mean) is fine to do later since it never
    # touches other clips. dataset-wide mean/std is not fine unless its
    # computed from the train split only, otherwise test data leaks into
    # the stats the model gets normalized against


if __name__ == "__main__":
    root = sys.argv[1]
    manifest_path = sys.argv[2] if len(sys.argv) > 2 else "manifest.csv"
    cache_dir = sys.argv[3] if len(sys.argv) > 3 else "data/spectrograms"
    main(root, manifest_path, cache_dir)
