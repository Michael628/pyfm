import json
import os

import pandas as pd

from pyfm.nanny.aggregator import (
    _manifest_path,
    _read_skip_manifest,
    _write_skip_manifest,
    _formatted_input_filenames,
)


def test_manifest_path_derivation():
    assert _manifest_path("out/corr_{series}.h5") == "out/corr_{series}.manifest.json"


def test_manifest_round_trip(tmp_path):
    stem = str(tmp_path / "agg")
    files = ["a/0/corr.h5", "a/1/corr.h5", "a/2/corr.h5"]
    _write_skip_manifest(stem, files)
    skip_set, exists = _read_skip_manifest(stem)
    assert exists and skip_set == set(files)


def test_missing_manifest_returns_empty(tmp_path):
    """Missing manifest -> (set(), False), triggering warn + aggregate-all (Q4->C)."""
    skip_set, exists = _read_skip_manifest(str(tmp_path / "nonexistent"))
    assert skip_set == set() and not exists


def test_formatted_input_filenames_uses_format_map():
    fs = "correlators/{series}/{cfg}/corr.h5"
    repls = [("a/0", {"series": "a", "cfg": "0"}), ("a/1", {"series": "a", "cfg": "1"})]
    out = _formatted_input_filenames(fs, repls)
    assert out == ["correlators/a/0/corr.h5", "correlators/a/1/corr.h5"]


def test_process_data_writes_manifest(tmp_path):
    """process_data writes a manifest sidecar after write_files for each run key."""
    from unittest.mock import patch

    from pyfm.nanny.aggregator import process_data

    out_stem = str(tmp_path / "out_{series}")
    agg_params = {
        "key1": {
            "load_files": {
                "filestem": "in/{series}/corr.h5",
                "replacements": {"series": ["a", "b"]},
            },
            "out_files": {"filestem": out_stem},
            "actions": {},
        }
    }
    df = pd.DataFrame({"corr": [1.0]}, index=pd.Index([0], name="t"))
    with patch("pyfm.nanny.aggregator.dio") as mock_dio:
        process_data({"key1": df}, agg_params, format="csv")
    manifest = _manifest_path(out_stem)
    assert os.path.exists(manifest)
    with open(manifest) as f:
        assert set(json.load(f)) == {"in/a/corr.h5", "in/b/corr.h5"}
