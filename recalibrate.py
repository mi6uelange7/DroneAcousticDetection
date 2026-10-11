# recalibrate.py -- fixes the calibration instability the seed sweep found.
# splits the domain corpus into a small calibration slice and a held-out
# test slice, fits a correction on the calibration slice only (calibrating
# and testing on the same clips would just be a new leak), then checks
# whether recall on the held-out slice gets both higher and more stable
# across different random model seeds, not just higher for one lucky run.

import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from baseline import load_manifest
from cnn_domain_test import domain_probs
from train import train_on_everything

MODEL_SEEDS = range(8)  # same seeds the sweep used that found the instability
CALIB_SPLIT_SEED = 42  # fixes which domain clips become the calibration slice, same slice every run, only the model seed changes


def recall_at(probs, labels, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return preds[labels == 1].mean()


def main(cache_dir, manifest_path, domain_root, domain_manifest_path, epochs, calib_frac):
    entries = load_manifest(manifest_path)
    domain_entries = load_manifest(domain_manifest_path)
    labels = np.array([int(r["label"]) for r in domain_entries])

    # same calibration/test split every run, only the model changes. this is
    # the domain corpus split into its own two pieces, same train/test
    # discipline as everywhere else, calibrating on data youre about to
    # score defeats the point
    idx = np.arange(len(domain_entries))
    calib_idx, test_idx = train_test_split(
        idx, test_size=1 - calib_frac, stratify=labels, random_state=CALIB_SPLIT_SEED
    )
    test_labels = labels[test_idx]

    raw_recalls, corrected_recalls = [], []

    for seed in MODEL_SEEDS:
        model, mean, std = train_on_everything(entries, cache_dir, epochs, seed=seed)
        raw_probs = domain_probs(model, domain_entries, domain_root, mean, std)

        calibrator = LogisticRegression()
        calibrator.fit(raw_probs[calib_idx].reshape(-1, 1), labels[calib_idx])
        corrected_probs = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]

        raw_test, corrected_test = raw_probs[test_idx], corrected_probs[test_idx]
        raw_recall = recall_at(raw_test, test_labels)
        corrected_recall = recall_at(corrected_test, test_labels)
        raw_recalls.append(raw_recall)
        corrected_recalls.append(corrected_recall)

        # PR-AUC shouldnt move much here, recalibration is just a reordering
        # of where the 0.5 line falls, not a change to the ranking itself.
        # printing both is the check that this held
        raw_pr = average_precision_score(test_labels, raw_test)
        corrected_pr = average_precision_score(test_labels, corrected_test)
        print(
            f"seed {seed}: recall {raw_recall:.1%} -> {corrected_recall:.1%}  "
            f"(PR-AUC {raw_pr:.4f} -> {corrected_pr:.4f})"
        )

    def summarize(label, values):
        values = np.array(values)
        print(f"{label}: min {values.min():.1%}  max {values.max():.1%}  mean {values.mean():.1%}  std {values.std():.1%}")

    print()
    summarize("recall before recalibration", raw_recalls)
    summarize("recall after recalibration ", corrected_recalls)


if __name__ == "__main__":
    cache_dir = sys.argv[1] if len(sys.argv) > 1 else "data/spectrograms"
    manifest_path = sys.argv[2] if len(sys.argv) > 2 else "manifest.csv"
    domain_root = sys.argv[3] if len(sys.argv) > 3 else "data/domain_corpus/extracted"
    domain_manifest_path = sys.argv[4] if len(sys.argv) > 4 else "domain_manifest.csv"
    epochs = int(sys.argv[5]) if len(sys.argv) > 5 else 15
    calib_frac = float(sys.argv[6]) if len(sys.argv) > 6 else 0.2
    main(cache_dir, manifest_path, domain_root, domain_manifest_path, epochs, calib_frac)
