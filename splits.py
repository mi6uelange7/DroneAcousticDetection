# splits.py -- chops manifest into train/test csvs.
# the real one respects group_id; naive one pretends group_id doesn't exist.
# having both under same seed makes the leak obvious when you compare numbers.

import csv
import sys
from collections import Counter

from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def load_manifest(manifest_file):
    # reads csv
    with open(manifest_file, newline="") as f:
        return list(csv.DictReader(f))


def dump_splits(out_file, entries, assign_map):
    # writes a stupid simple csv: path, split
    with open(out_file, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["path", "split"])
        for r in entries:
            csv_writer.writerow([r["path"], assign_map[r["path"]]])


# real split. StratifiedGroupKFold keeps groups whole and also stratifies on label, because positive groups (257 of em) are
# outnumbered like 8 to 1 by negative groups. a plain group random split could easily
#  just starve test of drones for no reason. fold 0 is test, rest is train, works out to roughly 80/20 by group count.
def compute_group_split(entries, rng_seed, num_folds=5):
    group_ids = [r["group_id"] for r in entries]
    label_list = [int(r["label"]) for r in entries]

    splitter_obj = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=rng_seed)
    train_indices, test_indices = next(splitter_obj.split(entries, label_list, group_ids))

    assign_map = {}
    for i in train_indices:
        assign_map[entries[i]["path"]] = "train"
    for i in test_indices:
        assign_map[entries[i]["path"]] = "test"
    return assign_map


# the wrong baseline on purpose. plain stratified random split over
# rows, doesn't know group_id exists, so the five segments of one
# recording can scatter across both sides no problem. 

#basically what this dataset looks like before you notice the leak, and test_no_leakage.py is checking this stays broken.
def compute_naive_split(entries, rng_seed, test_frac=0.2):
    label_list = [int(r["label"]) for r in entries]
    train_entries, test_entries = train_test_split(
        entries, test_size=test_frac, random_state=rng_seed, stratify=label_list
    )

    assign_map = {}
    for r in train_entries:
        assign_map[r["path"]] = "train"
    for r in test_entries:
        assign_map[r["path"]] = "test"
    return assign_map


def print_summary(title, entries, assign_map):
    # dumps group count, clip count, pos/neg, silent count, family breakdown
    # per split. just so you can eyeball if test ended up with barely any
    # drones, or loaded up on silence
    print(f"\n{title}")
    for split in ("train", "test"):
        split_entries = [r for r in entries if assign_map[r["path"]] == split]
        num_groups = len({r["group_id"] for r in split_entries})
        positive_count = sum(int(r["label"]) for r in split_entries)
        negative_count = len(split_entries) - positive_count
        # silent count here is the actual thing to check: if its close to even
        # across train and test, silent clips arent giving either side an edge
        silent_count = sum(int(r["silent"]) for r in split_entries)
        silent_pct = 100 * silent_count / len(split_entries)
        print(
            f"  {split:5s} {len(split_entries):6d} clips  {num_groups:5d} groups  "
            + f"positives {positive_count:5d}  negatives {negative_count:6d}  "
            + f"silent {silent_count:5d} ({silent_pct:.1f}%)"
        )

        fam_counter = Counter(r["family"] for r in split_entries)
        for fam in sorted(fam_counter):
            print(f"    {fam:16s} {fam_counter[fam]:6d} clips")


def main(manifest_file, real_out_file, naive_out_file, rng_seed):
    entries = load_manifest(manifest_file)

    real_assign = compute_group_split(entries, rng_seed)
    dump_splits(real_out_file, entries, real_assign)
    print_summary(f"group split (real, seed={rng_seed}) -> {real_out_file}", entries, real_assign)

    naive_assign = compute_naive_split(entries, rng_seed)
    dump_splits(naive_out_file, entries, naive_assign)
    print_summary(
        f"naive split (broken on purpose, seed={rng_seed}) -> {naive_out_file}", entries, naive_assign
    )


if __name__ == "__main__":
    manifest_path = sys.argv[1]
    real_out_path = sys.argv[2] if len(sys.argv) > 2 else "splits.csv"
    naive_out_path = sys.argv[3] if len(sys.argv) > 3 else "splits_naive.csv"
    rng_seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    main(manifest_path, real_out_path, naive_out_path, rng_seed)
