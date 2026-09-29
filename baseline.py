# baseline.py -- non-neural baseline. 20 mfccs per clip, mean+std over time,
# logistic regression on the resulting 40 numbers. point isnt the best score,
# its finding out how far you get before writing a line of pytorch.

import csv
import sys
from pathlib import Path

import librosa
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

SR = 16000
N_MFCC = 20


def load_manifest(manifest_path):
    with open(manifest_path, newline="") as f:
        return list(csv.DictReader(f))


def load_splits(splits_path):
    with open(splits_path, newline="") as f:
        return {r["path"]: r["split"] for r in csv.DictReader(f)}


def mfcc_features(wav_path):
    y, _ = librosa.load(wav_path, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    # mean and std per coefficient across time, 20 + 20 = 40 numbers per clip
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def build_dataset(root, entries):
    X = np.array([mfcc_features(root / r["path"]) for r in entries])
    y = np.array([int(r["label"]) for r in entries])
    return X, y


# family accuracy instead of family PR-AUC, on purpose. family is always
# single label here, clean/mixed/extra_drone are all label 1, esc50 and
# synthetic_noise are all label 0, so theres no positive/negative mix to
# rank within one family and PR-AUC just isnt defined on it. accuracy at
# the plain 0.5 cutoff still answers the real question though, is one
# family (synthetic_noise especially, its mostly silence) trivially easy
# and dragging the whole number up on its own
def family_breakdown(test_entries, preds, key="family"):
    groups = {}
    for r, pred in zip(test_entries, preds):
        groups.setdefault(r[key], []).append(pred == int(r["label"]))
    for name in sorted(groups):
        correct = groups[name]
        print(f"    {name:16s} {sum(correct):5d} / {len(correct):5d} correct")


def run_split(entries, X, y, split_map, split_name, exclude_silent):
    keep = np.array([r["silent"] == "0" for r in entries]) if exclude_silent else np.ones(len(entries), dtype=bool)
    paths = [r["path"] for r in entries]
    train_mask = keep & np.array([split_map[p] == "train" for p in paths])
    test_mask = keep & np.array([split_map[p] == "test" for p in paths])

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X[train_mask], y[train_mask])
    probs = clf.predict_proba(X[test_mask])[:, 1]
    preds = (probs >= 0.5).astype(int)

    # PR-AUC not accuracy, 11.4% positives means "always say no drone" scores
    # 88.6% and looks great while learning nothing
    pr_auc = average_precision_score(y[test_mask], probs)
    tag = "silence excluded" if exclude_silent else "silence included"
    print(f"{split_name} ({tag}): train {train_mask.sum()} clips, test {test_mask.sum()} clips, PR-AUC {pr_auc:.4f}")

    test_entries = [r for r, k in zip(entries, test_mask) if k]
    family_breakdown(test_entries, preds)
    return pr_auc


def main(root, manifest_path, splits_path, naive_splits_path):
    root = Path(root)
    entries = load_manifest(manifest_path)

    print(f"extracting mfccs for {len(entries)} clips...")
    X, y = build_dataset(root, entries)

    group_map = load_splits(splits_path)
    naive_map = load_splits(naive_splits_path)

    # runs each split twice, silence in and silence out, so the two numbers
    # sit next to each other and the gap between them is easy to see instead
    # of buried in two separate runs from two separate days
    for exclude_silent in (False, True):
        run_split(entries, X, y, group_map, "group split", exclude_silent)
        run_split(entries, X, y, naive_map, "naive split", exclude_silent)


if __name__ == "__main__":
    root = sys.argv[1]
    manifest_path = sys.argv[2] if len(sys.argv) > 2 else "manifest.csv"
    splits_path = sys.argv[3] if len(sys.argv) > 3 else "splits.csv"
    naive_splits_path = sys.argv[4] if len(sys.argv) > 4 else "splits_naive.csv"
    main(root, manifest_path, splits_path, naive_splits_path)
