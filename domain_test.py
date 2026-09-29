# domain_test.py -- trains on the whole Binary_Drone_Audio manifest, tests on
# the Acoustic-UAV-Identification corpus. group split only proves the model
# didnt cheat within one dataset, this is the harder question: does it know
# anything about drone sound that transfers to a different mic, room, and
# recording setup entirely. reuses mfcc_features and build_dataset straight
# from baseline.py, no reason to extract audio features two different ways.

import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from baseline import build_dataset, family_breakdown, load_manifest


def main(train_root, train_manifest_path, domain_root, domain_manifest_path):
    train_root = Path(train_root)
    domain_root = Path(domain_root)

    train_entries = load_manifest(train_manifest_path)
    domain_entries = load_manifest(domain_manifest_path)

    print(f"extracting mfccs for {len(train_entries)} training clips...")
    X_train, y_train = build_dataset(train_root, train_entries)

    print(f"extracting mfccs for {len(domain_entries)} domain clips...")
    X_domain, y_domain = build_dataset(domain_root, domain_entries)

    # train on everything, none of this data is held out for anything else
    # and none of it overlaps with the domain corpus to begin with
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_domain)[:, 1]
    preds = (probs >= 0.5).astype(int)

    pr_auc = average_precision_score(y_domain, probs)
    print(f"\ndomain test: train {len(train_entries)} clips, test {len(domain_entries)} clips, PR-AUC {pr_auc:.4f}")
    family_breakdown(domain_entries, preds, key="subset")


if __name__ == "__main__":
    train_root = sys.argv[1]
    train_manifest_path = sys.argv[2] if len(sys.argv) > 2 else "manifest.csv"
    domain_root = sys.argv[3] if len(sys.argv) > 3 else "data/domain_corpus/extracted"
    domain_manifest_path = sys.argv[4] if len(sys.argv) > 4 else "domain_manifest.csv"
    main(train_root, train_manifest_path, domain_root, domain_manifest_path)
