# cnn_domain_test.py -- same domain generalization test as domain_test.py,
# but for the trained CNN instead of the logistic regression baseline.
# domain corpus spectrograms get computed on the fly (reusing features.py's
# spectrogram function) since theres only 546 of them, not worth building
# out a whole separate cache for a one-off test.

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from baseline import family_breakdown, load_manifest
from features import spectrogram
from train import DEVICE, SpectrogramDataset, train_model, train_norm_stats


def domain_probs(model, entries, root, mean, std):
    specs = np.stack([spectrogram(Path(root) / r["path"]) for r in entries])
    specs = (specs.astype(np.float32) - mean) / std
    specs = torch.from_numpy(specs).unsqueeze(1).to(DEVICE)

    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(specs)).cpu().numpy()


def main(cache_dir, manifest_path, domain_root, domain_manifest_path, epochs):
    entries = load_manifest(manifest_path)
    domain_entries = load_manifest(domain_manifest_path)

    print(f"computing normalization stats from all {len(entries)} training clips...")
    mean, std = train_norm_stats(entries, cache_dir)

    # train on everything, same reasoning as domain_test.py, none of this
    # data is held out for anything else and none of it overlaps with the
    # domain corpus to begin with
    train_ds = SpectrogramDataset(entries, cache_dir, mean, std)
    print(f"training on {DEVICE}, {len(entries)} clips, {epochs} epochs...")
    model = train_model(train_ds, epochs)

    print(f"\nscoring {len(domain_entries)} domain clips...")
    probs = domain_probs(model, domain_entries, domain_root, mean, std)
    preds = (probs >= 0.5).astype(int)

    labels = np.array([int(r["label"]) for r in domain_entries])
    pr_auc = average_precision_score(labels, probs)
    print(f"domain test: train {len(entries)} clips, test {len(domain_entries)} clips, PR-AUC {pr_auc:.4f}")
    family_breakdown(domain_entries, preds, key="subset")


if __name__ == "__main__":
    cache_dir = sys.argv[1] if len(sys.argv) > 1 else "data/spectrograms"
    manifest_path = sys.argv[2] if len(sys.argv) > 2 else "manifest.csv"
    domain_root = sys.argv[3] if len(sys.argv) > 3 else "data/domain_corpus/extracted"
    domain_manifest_path = sys.argv[4] if len(sys.argv) > 4 else "domain_manifest.csv"
    epochs = int(sys.argv[5]) if len(sys.argv) > 5 else 15
    main(cache_dir, manifest_path, domain_root, domain_manifest_path, epochs)
