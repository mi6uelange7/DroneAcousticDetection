# tests/test_no_leakage.py -- makes sure splits.py didnt mix recordings across train/test
# group_id in manifest.csv means same source recording, so no group should ever land on both sides
# last test does the same check but on splits_naive.csv and expects the opposite, cause naive is
# suppose to leak. if it ever stops leaking the control is broken and we gotta toss it

import csv
import os
from collections import Counter

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_csv(filepath):
    with open(filepath, newline="") as fh:
        return list(csv.DictReader(fh))


def load_manifest():
    p = os.path.join(ROOT, "manifest.csv")
    if not os.path.exists(p):
        pytest.skip(f"{p} not found, run build_manifest.py first")
    return read_csv(p)


def load_splits(name):
    p = os.path.join(ROOT, name)
    if not os.path.exists(p):
        pytest.skip(f"{p} not found, run splits.py first")
    return read_csv(p)


def attach_metadata(split_rows, manifest_entries):
    # splits.csv only got path and split, group_id and label still live in manifest
    # so join on path and put everything on one row
    gid_lookup = {r["path"]: r["group_id"] for r in manifest_entries}
    label_lookup = {r["path"]: r["label"] for r in manifest_entries}
    return [
        {
            "path": r["path"],
            "split": r["split"],
            "group_id": gid_lookup[r["path"]],
            "label": label_lookup[r["path"]],
        }
        for r in split_rows
    ]


def groups_in(rows, split):
    return {r["group_id"] for r in rows if r["split"] == split}


def test_no_group_overlap():
    manifest = load_manifest()
    rows = attach_metadata(load_splits("splits.csv"), manifest)

    train_groups = groups_in(rows, "train")
    test_groups = groups_in(rows, "test")
    overlap = sorted(train_groups & test_groups)

    assert not overlap, (
        f"{len(overlap)} group_id(s) appear in both train and test, which means "
        f"a source recording got cut across the split: "
        f"{overlap[:10]}{' ...' if len(overlap) > 10 else ''}"
    )


def test_both_splits_nonempty_and_have_positives():
    manifest = load_manifest()
    rows = attach_metadata(load_splits("splits.csv"), manifest)

    for split in ("train", "test"):
        split_rows = [r for r in rows if r["split"] == split]
        assert split_rows, f"{split} split is empty"

        positives = sum(1 for r in split_rows if int(r["label"]) == 1)
        assert positives > 0, (
            f"{split} split has zero positive clips, model never sees a "
            f"drone during {split}"
        )


def test_every_manifest_path_appears_exactly_once():
    manifest = load_manifest()
    split_rows = load_splits("splits.csv")

    manifest_paths = {r["path"] for r in manifest}
    split_paths = [r["path"] for r in split_rows]
    split_path_set = set(split_paths)

    missing = sorted(manifest_paths - split_path_set)
    assert not missing, (
        f"{len(missing)} manifest path(s) missing from splits.csv: "
        f"{missing[:10]}{' ...' if len(missing) > 10 else ''}"
    )

    extra = sorted(split_path_set - manifest_paths)
    assert not extra, (
        f"{len(extra)} splits.csv path(s) not found in manifest.csv: "
        f"{extra[:10]}{' ...' if len(extra) > 10 else ''}"
    )

    # if a path shows up twice that means one clip got put in both splits somehow
    counts = Counter(split_paths)
    dupes = sorted(p for p, c in counts.items() if c > 1)
    assert not dupes, (
        f"{len(dupes)} path(s) appear more than once in splits.csv: "
        f"{dupes[:10]}{' ...' if len(dupes) > 10 else ''}"
    )


# control test. splits_naive.csv doesnt know group_id exists so it should
# always leak groups across train and test. if this test stops failing
# that means naive baseline isnt useful as the wrong comparison no more
def test_naive_split_still_leaks():
    manifest = load_manifest()
    rows = attach_metadata(load_splits("splits_naive.csv"), manifest)

    train_groups = groups_in(rows, "train")
    test_groups = groups_in(rows, "test")
    overlap = train_groups & test_groups

    assert overlap, (
        "splits_naive.csv has zero group overlap, but it's supposed to leak "
        "since it splits on rows not groups. if it's not leaking the naive "
        "baseline stopped being broken and isn't a valid control anymore"
    )
