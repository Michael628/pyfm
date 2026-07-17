import json
import os

import pandas as pd
import pytest

from pyfm.nanny.aggregator import (
    _manifest_path,
    _read_skip_manifest,
    _write_skip_manifest,
    _formatted_input_filenames,
    _split_series_cfg,
    _recover_series_cfg_values,
    _present_tsources,
    _reconstruct_manifest_inputs,
    generate_manifests,
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
    """process_data writes a manifest sidecar after write_files for each run key.

    With a templated output filestem, the write system emits one manifest per
    output file (one per resolved group) alongside each file, using the same
    keyword replacement. Here ``out_{series}`` with ``series in {a, b}`` yields
    ``out_a.manifest.json`` and ``out_b.manifest.json``, each holding only its
    own group's inputs.
    """
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

    # One manifest per resolved output group, at brace-free paths.
    man_a = _manifest_path(str(tmp_path / "out_a"))
    man_b = _manifest_path(str(tmp_path / "out_b"))
    assert os.path.exists(man_a) and os.path.exists(man_b)
    with open(man_a) as f:
        assert set(json.load(f)) == {"in/a/corr.h5"}
    with open(man_b) as f:
        assert set(json.load(f)) == {"in/b/corr.h5"}

    # No braced (un-resolved) manifest path is written.
    assert not os.path.exists(_manifest_path(out_stem))


def test_read_skip_manifest_globs_per_output_file(tmp_path):
    """A templated output filestem globs all per-output-file manifests and unions them."""
    from pyfm.nanny.aggregator import _read_skip_manifest

    out_stem = str(tmp_path / "out_{series}/m{mass}/corr")
    # Hand-write two per-output-file manifests as process_data would.
    _write_skip_manifest(str(tmp_path / "out_a/m_l/corr"), ["in/a/l/corr.h5"])
    _write_skip_manifest(str(tmp_path / "out_b/m_u/corr"), ["in/b/u/corr.h5"])

    skip, exists = _read_skip_manifest(out_stem, format="csv")
    assert exists
    assert skip == {"in/a/l/corr.h5", "in/b/u/corr.h5"}


def test_read_skip_manifest_is_format_scoped(tmp_path):
    """The {format} key resolves to the current format so other formats are ignored."""
    from pyfm.nanny.aggregator import _read_skip_manifest

    out_stem = str(tmp_path / "out_{format}/corr")
    _write_skip_manifest(str(tmp_path / "out_csv/corr"), ["in/csv/corr.h5"])
    _write_skip_manifest(str(tmp_path / "out_hdf5/corr"), ["in/hdf5/corr.h5"])

    skip_csv, _ = _read_skip_manifest(out_stem, format="csv")
    assert skip_csv == {"in/csv/corr.h5"}


def test_write_manifests_partitions_inputs_by_output_group(tmp_path):
    """Each output group's manifest lists only the inputs mapped to it."""
    from pyfm.nanny.aggregator import _write_manifests

    out_stem = str(tmp_path / "out_{format}/m{mass}/corr_{mass}")
    load_files_cfg = {
        "filestem": "in/{series}/m{mass}/corr.h5",
        "replacements": {"series": ["a", "b"], "mass": ["l", "u"]},
    }
    _write_manifests(load_files_cfg, out_stem, "csv")

    man_l = _manifest_path(str(tmp_path / "out_csv/ml/corr_l"))
    man_u = _manifest_path(str(tmp_path / "out_csv/mu/corr_u"))
    assert os.path.exists(man_l) and os.path.exists(man_u)
    with open(man_l) as f:
        assert set(json.load(f)) == {"in/a/ml/corr.h5", "in/b/ml/corr.h5"}
    with open(man_u) as f:
        assert set(json.load(f)) == {"in/a/mu/corr.h5", "in/b/mu/corr.h5"}


# --- Phase 1: reconstruction & extraction helpers ---


def test_split_series_cfg_recovers_series_and_cfg():
    assert _split_series_cfg("a.1000") == ("a", "1000")
    assert _split_series_cfg("b.20") == ("b", "20")


def test_recover_series_cfg_values_from_column():
    df = pd.DataFrame({"series_cfg": ["a.0", "a.1", "b.0"], "corr": [1.0, 2.0, 3.0]})
    assert sorted(_recover_series_cfg_values(df)) == ["a.0", "a.1", "b.0"]


def test_recover_series_cfg_values_from_index_level():
    idx = pd.MultiIndex.from_tuples(
        [("a.0", "g", 0), ("a.1", "g", 0)], names=["series_cfg", "gamma", "t"]
    )
    df = pd.DataFrame({"corr": [1.0, 2.0]}, index=idx)
    assert sorted(_recover_series_cfg_values(df)) == ["a.0", "a.1"]


def test_recover_series_cfg_values_missing_raises():
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="series_cfg"):
        _recover_series_cfg_values(df)


def test_present_tsources_empty_when_averaged():
    df = pd.DataFrame({"series_cfg": ["a.0"], "corr": [1.0]})
    assert _present_tsources(df) == {}


def test_present_tsources_groups_per_config():
    df = pd.DataFrame(
        {"series_cfg": ["a.0", "a.0", "a.1"], "tsource": ["0", "1", "0"]}
    )
    assert _present_tsources(df) == {"a.0": {"0", "1"}, "a.1": {"0"}}


def test_reconstruct_manifest_inputs_gates_incomplete_config():
    load_files = {
        "filestem": "in/{series}/{cfg}/t{tsource}.h5",
        "replacements": {"tsource": ["0", "1"]},
    }
    df = pd.DataFrame(
        {"series_cfg": ["a.0", "a.0", "a.1"], "tsource": ["0", "1", "0"]}
    )
    out = _reconstruct_manifest_inputs(df, load_files, {})
    assert set(out) == {"in/a/0/t0.h5", "in/a/0/t1.h5"}


def test_reconstruct_manifest_inputs_averaged_includes_all():
    load_files = {
        "filestem": "in/{series}/{cfg}/t{tsource}.h5",
        "replacements": {"tsource": ["0", "1"]},
    }
    df = pd.DataFrame({"series_cfg": ["a.0", "a.1"]})
    out = _reconstruct_manifest_inputs(df, load_files, {})
    assert set(out) == {
        "in/a/0/t0.h5",
        "in/a/0/t1.h5",
        "in/a/1/t0.h5",
        "in/a/1/t1.h5",
    }


def test_reconstruct_manifest_inputs_merges_path_repl():
    load_files = {
        "filestem": "root/e{eigs}/{series}/{cfg}/t{tsource}.h5",
        "replacements": {"tsource": ["0"]},
    }
    path_repl = {"eigs": "10"}
    df = pd.DataFrame({"series_cfg": ["a.0"], "tsource": ["0"]})
    assert _reconstruct_manifest_inputs(df, load_files, path_repl) == [
        "root/e10/a/0/t0.h5"
    ]


def test_reconstruct_manifest_inputs_skips_unresolvable(caplog):
    load_files = {
        "filestem": "in/{series}/{cfg}/{unknown}.h5",
        "replacements": {"tsource": ["0"]},
    }
    df = pd.DataFrame({"series_cfg": ["a.0"], "tsource": ["0"]})
    assert _reconstruct_manifest_inputs(df, load_files, {}) == []


def test_reconstruct_manifest_inputs_without_tsource_dim():
    load_files = {"filestem": "in/{series}/{cfg}/corr.h5", "replacements": {}}
    df = pd.DataFrame({"series_cfg": ["a.0", "a.1"]})
    assert set(_reconstruct_manifest_inputs(df, load_files, {})) == {
        "in/a/0/corr.h5",
        "in/a/1/corr.h5",
    }


# --- Phase 2: generate_manifests orchestrator ---


def test_generate_manifests_writes_manifest_for_resolved_files(tmp_path):
    from unittest.mock import patch

    out_stem = str(tmp_path / "agg_{mass}")
    agg_params = {
        "run": ["r1"],
        "r1": {
            "load_files": {
                "filestem": "in/{series}/{cfg}/t{tsource}.h5",
                "replacements": {"mass": "m1", "tsource": ["0", "1"]},
                "regex": {"series": "[a-z]", "cfg": "[0-9]+"},
            },
            "out_files": {"filestem": out_stem},
        },
    }
    (tmp_path / "agg_m1.csv").write_text("")  # exists so process_files finds it

    fake_df = pd.DataFrame(
        {"series_cfg": ["a.0", "a.0"], "tsource": ["0", "1"]}  # complete config
    )
    with patch("pyfm.nanny.aggregator.dio") as mock_dio:
        mock_dio.load_files_chunked.return_value = fake_df
        generate_manifests(agg_params, format="csv")

    manifest = _manifest_path(str(tmp_path / "agg_m1"))
    assert os.path.exists(manifest)
    with open(manifest) as f:
        assert set(json.load(f)) == {"in/a/0/t0.h5", "in/a/0/t1.h5"}


def test_generate_manifests_warns_when_no_agg_files(tmp_path):
    from unittest.mock import patch

    out_stem = str(tmp_path / "agg_{mass}")
    agg_params = {
        "run": ["r1"],
        "r1": {
            "load_files": {
                "filestem": "in/{series}/{cfg}/t{tsource}.h5",
                "replacements": {"mass": "m1", "tsource": ["0", "1"]},
            },
            "out_files": {"filestem": out_stem},
        },
    }
    # No agg file created on disk.
    with patch("pyfm.nanny.aggregator.dio") as mock_dio:
        generate_manifests(agg_params, format="csv")
    mock_dio.load_files_chunked.assert_not_called()
    assert not os.path.exists(_manifest_path(str(tmp_path / "agg_m1")))


def test_generate_manifests_averaged_includes_all(tmp_path):
    from unittest.mock import patch

    out_stem = str(tmp_path / "agg_{mass}")
    agg_params = {
        "run": ["r1"],
        "r1": {
            "load_files": {
                "filestem": "in/{series}/{cfg}/t{tsource}.h5",
                "replacements": {"mass": "m1", "tsource": ["0", "1"]},
            },
            "out_files": {"filestem": out_stem},
        },
    }
    (tmp_path / "agg_m1.csv").write_text("")
    # Averaged: no tsource column -> assume complete, include all present configs.
    fake_df = pd.DataFrame({"series_cfg": ["a.0", "a.1"]})
    with patch("pyfm.nanny.aggregator.dio") as mock_dio:
        mock_dio.load_files_chunked.return_value = fake_df
        generate_manifests(agg_params, format="csv")

    manifest = _manifest_path(str(tmp_path / "agg_m1"))
    with open(manifest) as f:
        assert set(json.load(f)) == {
            "in/a/0/t0.h5",
            "in/a/0/t1.h5",
            "in/a/1/t0.h5",
            "in/a/1/t1.h5",
        }
