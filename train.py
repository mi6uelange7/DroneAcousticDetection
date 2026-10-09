# train.py -- trains a small CNN on the cached mel-spectrograms from features.py.
# same honest-eval philosophy as baseline.py: group split is the real number,
# naive split sits next to it for comparison, per-family breakdown so one
# easy family cant hide behind a good aggregate score.

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score

from baseline import family_breakdown, load_manifest, load_splits

# nvidia gpu if theres one, apple silicon gpu (mps) on a mac, otherwise plain cpu
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# loads every cached spectrogram for the given entries into memory up front.
# 11704 clips at 64x65 float32 is only ~190mb, way cheaper than hitting disk
# once per sample every single epoch
class SpectrogramDataset(Dataset):
    def __init__(self, entries, cache_dir, mean, std):
        self.entries = entries
        # for every clip, swap .wav for .npy and load its saved spectrogram from the cache folder.
        # the / here glues folder + filename together, its not division
        specs = [np.load(Path(cache_dir) / Path(r["path"]).with_suffix(".npy")) for r in entries]
        # stack every 64x65 grid into one big block, then (x - mean) / std on every number so
        # everything sits around 0 in a small range. networks learn way better that way
        self.specs = (np.stack(specs).astype(np.float32) - mean) / std
        self.labels = np.array([int(r["label"]) for r in entries], dtype=np.float32)

    def __len__(self):
        return len(self.entries)

    # pytorch calls __len__ and __getitem__ itself, names are fixed. [None, :, :] adds a channel
    # dimension, 64x65 -> 1x64x65, cause conv layers want channels first and a spectrogram is mono
    def __getitem__(self, i):
        return self.specs[i][None, :, :], self.labels[i]


# computes mean/std off train spectrograms only. doing this on the full
# dataset would leak test statistics into the normalization, same mistake
# features.py already warned about
def train_norm_stats(entries, cache_dir):
    specs = [np.load(Path(cache_dir) / Path(r["path"]).with_suffix(".npy")) for r in entries]
    stacked = np.stack(specs).astype(np.float32)
    return stacked.mean(), stacked.std()


# three conv blocks, each halving the spatial size with a maxpool, then one
# linear layer on top. 64x65 input is small, a dataset this size doesnt earn
# anything fancier than this
class DroneCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # each row is one block. conv slides a 3x3 window over the image looking for patterns
        # (16, then 32, then 64 of em), relu keeps the strong matches and zeros the rest,
        # maxpool halves the size. so it goes small details first, then bigger shapes
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        # 64 "how much of each pattern" scores -> 1 drone score
        self.classifier = nn.Linear(64, 1)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x).squeeze(1)


def split_entries(entries, split_map):
    train = [r for r in entries if split_map[r["path"]] == "train"]
    test = [r for r in entries if split_map[r["path"]] == "test"]
    return train, test


def train_model(train_ds, epochs):
    model = DroneCNN().to(DEVICE)
    # adam adjusts the weights after every batch, lr is how big each nudge is
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    # scores how wrong a yes/no guess was. takes the raw output and does the sigmoid itself
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for specs, labels in loader:
            specs, labels = specs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()                          # clear last batchs notes
            loss = loss_fn(model(specs), labels)     # guess, then score how wrong it was
            loss.backward()                          # work out which weights caused the error
            opt.step()                               # nudge them a little to be less wrong
            total_loss += loss.item() * len(labels)
        print(f"  epoch {epoch + 1}/{epochs}  loss {total_loss / len(train_ds):.4f}")

    return model


@torch.no_grad()
def evaluate(model, test_entries, cache_dir, mean, std, split_name):
    model.eval()
    ds = SpectrogramDataset(test_entries, cache_dir, mean, std)
    loader = DataLoader(ds, batch_size=256)

    probs = []
    for specs, _ in loader:
        logits = model(specs.to(DEVICE))
        # sigmoid squishes the raw output into 0-1, so it reads as a drone probability
        probs.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.concatenate(probs)
    preds = (probs >= 0.5).astype(int)  # the 50% line, 1 = drone

    labels = np.array([int(r["label"]) for r in test_entries])
    pr_auc = average_precision_score(labels, probs)
    print(f"{split_name}: test {len(test_entries)} clips, PR-AUC {pr_auc:.4f}")
    family_breakdown(test_entries, preds)
    return pr_auc


# trains its own model under its own split, same way baseline.py does it.
# the naive model has to actually be trained on leaky data for the
# comparison to mean anything, testing a group-trained model on naive's
# test set wouldnt reproduce the leak at all
def run_split(entries, split_map, cache_dir, epochs, split_name):
    train_entries, test_entries = split_entries(entries, split_map)

    mean, std = train_norm_stats(train_entries, cache_dir)
    train_ds = SpectrogramDataset(train_entries, cache_dir, mean, std)

    print(f"\ntraining {split_name} model on {DEVICE}, {len(train_entries)} clips, {epochs} epochs...")
    model = train_model(train_ds, epochs)

    return evaluate(model, test_entries, cache_dir, mean, std, split_name)


def main(cache_dir, manifest_path, splits_path, naive_splits_path, epochs):
    entries = load_manifest(manifest_path)
    group_map = load_splits(splits_path)
    naive_map = load_splits(naive_splits_path)

    run_split(entries, group_map, cache_dir, epochs, "group split")
    run_split(entries, naive_map, cache_dir, epochs, "naive split")


if __name__ == "__main__":
    cache_dir = sys.argv[1] if len(sys.argv) > 1 else "data/spectrograms"
    manifest_path = sys.argv[2] if len(sys.argv) > 2 else "manifest.csv"
    splits_path = sys.argv[3] if len(sys.argv) > 3 else "splits.csv"
    naive_splits_path = sys.argv[4] if len(sys.argv) > 4 else "splits_naive.csv"
    epochs = int(sys.argv[5]) if len(sys.argv) > 5 else 15
    main(cache_dir, manifest_path, splits_path, naive_splits_path, epochs)
