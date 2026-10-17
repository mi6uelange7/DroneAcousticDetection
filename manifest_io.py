# manifest_io.py -- just the manifest reader, split out because splits.py,
# detect_silence.py, and baseline.py all had their own copy of this exact
# function. pulling load_manifest from baseline.py instead wouldve dragged
# librosa and sklearn into scripts that dont touch either, so this stays a
# one-function file on purpose

import csv


def load_manifest(manifest_path):
    with open(manifest_path, newline="") as f:
        return list(csv.DictReader(f))
