# save_model.py -- trains the CNN once, fits the recalibration correction on
# a slice of the domain corpus, and saves both together as one checkpoint.
# thats the bundle live inference actually needs, the trained weights plus
# the calibration fix. a lucky seed by itself was never the reliable part,
# recalibration is what proved stable across all 8 seeds in the sweep.

import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from manifest_io import load_manifest
from cnn_domain_test import domain_probs
from train import SEED, train_on_everything

CALIB_SPLIT_SEED = 42  # same split recalibrate.py uses, so results line up


def main(cache_dir, manifest_path, domain_root, domain_manifest_path, epochs, calib_frac, out_path):
    entries = load_manifest(manifest_path)
    domain_entries = load_manifest(domain_manifest_path)
    labels = np.array([int(r["label"]) for r in domain_entries])

    model, mean, std = train_on_everything(entries, cache_dir, epochs, seed=SEED)
    raw_probs = domain_probs(model, domain_entries, domain_root, mean, std)

    idx = np.arange(len(domain_entries))
    calib_idx, test_idx = train_test_split(
        idx, test_size=1 - calib_frac, stratify=labels, random_state=CALIB_SPLIT_SEED
    )

    calibrator = LogisticRegression()
    calibrator.fit(raw_probs[calib_idx].reshape(-1, 1), labels[calib_idx])

    # same check recalibrate.py does, run it here too so a bad checkpoint
    # cant get saved without anyone noticing
    corrected = calibrator.predict_proba(raw_probs[test_idx].reshape(-1, 1))[:, 1]
    preds = (corrected >= 0.5).astype(int)
    recall = preds[labels[test_idx] == 1].mean()
    print(f"sanity check before saving: recall on held-out domain slice {recall:.1%}")

    # coef/intercept instead of the sklearn object itself, so loading this
    # checkpoint later for live inference doesnt need sklearn installed,
    # just sigmoid(coef * raw_prob + intercept)
    torch.save({
        "model_state": model.state_dict(),
        "mean": mean,
        "std": std,
        "calib_coef": float(calibrator.coef_[0, 0]),
        "calib_intercept": float(calibrator.intercept_[0]),
    }, out_path)
    print(f"saved checkpoint to {out_path}")


if __name__ == "__main__":
    cache_dir = sys.argv[1] if len(sys.argv) > 1 else "data/spectrograms"
    manifest_path = sys.argv[2] if len(sys.argv) > 2 else "manifest.csv"
    domain_root = sys.argv[3] if len(sys.argv) > 3 else "data/domain_corpus/extracted"
    domain_manifest_path = sys.argv[4] if len(sys.argv) > 4 else "domain_manifest.csv"
    epochs = int(sys.argv[5]) if len(sys.argv) > 5 else 15
    calib_frac = float(sys.argv[6]) if len(sys.argv) > 6 else 0.2
    out_path = sys.argv[7] if len(sys.argv) > 7 else "model.pt"
    main(cache_dir, manifest_path, domain_root, domain_manifest_path, epochs, calib_frac, out_path)
