"""Unit tests for pyfm/nanny/aggregator.py: manifest/skip logic and convert path."""
import json
import os
from unittest.mock import MagicMock, patch

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


from pyfm.nanny import aggregator
from pyfm.nanny.taskbuilder import Task


def _make_task(agg_params):
    handler = MagicMock()
    handler.build_aggregator_params.return_value = agg_params
    return Task(handler=handler, config=MagicMock(), key="test_key")


def _agg_single(filestem="processed/{format}/pion", load_files=None):
    return {
        "run": ["diagram"],
        "diagram": {
            "out_files": {"filestem": filestem},
            "load_files": load_files or {"regex": {"cfg": "[0-9]+"}},
        },
    }


def _make_split_task(raw_params, avg_params):
    """Fake task whose handler returns raw params for average=False and avg
    params for average=True (mirrors the real handler contract)."""
    handler = MagicMock()
    handler.build_aggregator_params.side_effect = (
        lambda config, average: avg_params if average else raw_params
    )
    return Task(handler=handler, config=MagicMock(), key="test_key")


def _avg_single(raw_filestem="processed/{format}/pion", actions=None):
    """Raw (average=False) and avg (average=True) param pairs for one run key."""
    raw = {
        "run": ["diagram"],
        "diagram": {
            "out_files": {"filestem": raw_filestem},
            "load_files": {"regex": {"cfg": "[0-9]+"}},
        },
    }
    avg = {
        "run": ["diagram"],
        "diagram": {
            "out_files": {"filestem": raw_filestem + "_avg"},
            "load_files": {"regex": {"cfg": "[0-9]+"}},
            "actions": actions if actions is not None else {"time_average": ["t1", "t4"]},
        },
    }
    return raw, avg


def _average_of(call):
    """Extract the `average` build argument from either args or kwargs."""
    if "average" in call.kwargs:
        return call.kwargs["average"]
    return call.args[1] if len(call.args) > 1 else False


def test_convert_loads_and_writes_each_run_key():
    task = _make_task(_agg_single())
    df = pd.DataFrame({"corr": [1.0, 2.0], "cfg": ["c0", "c1"]})
    with (
        patch("pyfm.nanny.aggregator.create_task", return_value=task),
        patch("pyfm.nanny.aggregator.dio") as mock_dio,
    ):
        mock_dio.load_files.return_value.agg.return_value = df
        aggregator.convert_task_data(
            "job", {"tasks": []}, input_format="csv", output_format="hdf5"
        )

    _, lkw = mock_dio.load_files.call_args
    assert lkw["filestem"] == "processed/{format}/pion.csv"
    assert lkw["replacements"]["format"] == "csv"
    _, wkw = mock_dio.write_files.call_args
    assert wkw["format"] == "hdf5"
    assert wkw["filestem"] == "processed/{format}/pion"
    written = mock_dio.write_files.call_args.args[0]
    assert written["format"].eq("hdf5").all()


def test_set_format_col_with_placeholder():
    df = pd.DataFrame({"corr": [1.0], "format": ["csv"]})
    out = aggregator.set_format_col(df, "processed/{format}/x", "hdf5")
    assert (out["format"] == "hdf5").all()


def test_set_format_col_without_placeholder_drops_format():
    df = pd.DataFrame({"corr": [1.0], "format": ["csv"]})
    out = aggregator.set_format_col(df, "plain/x", "hdf5")
    assert "format" not in out.columns


def test_convert_forwards_dict_metadata():
    load_files = {
        "dict_labels": ["perm", "gamma"],
        "array_order": ["t1", "t2"],
        "array_labels": {"t1": "0..1", "t2": "0..1"},
        "regex": {"cfg": "[0-9]+"},
    }
    task = _make_task(_agg_single(load_files=load_files))
    with (
        patch("pyfm.nanny.aggregator.create_task", return_value=task),
        patch("pyfm.nanny.aggregator.dio") as mock_dio,
    ):
        mock_dio.load_files.return_value.agg.return_value = pd.DataFrame({"corr": [1.0]})
        aggregator.convert_task_data("job", {}, input_format="dict", output_format="csv")

    _, lkw = mock_dio.load_files.call_args
    assert lkw["dict_labels"] == ["perm", "gamma"]
    assert lkw["array_order"] == ["t1", "t2"]
    assert lkw["array_labels"] == {"t1": "0..1", "t2": "0..1"}


def test_load_convert_data_preserves_restored_index_levels():
    """Regression guard: index-restoring readers (hdf5/parquet) hand back the
    written table's index already rebuilt, so leftover columns must extend it —
    parity with the flat csv read — not silently replace the restored levels
    via set_index."""
    run_params = {
        "out_files": {"filestem": "processed/{format}/pion"},
        "load_files": {},
    }
    idx = pd.MultiIndex.from_product([["c0", "c1"], ["t1", "t2"]], names=["cfg", "t"])
    restored = pd.DataFrame(
        {"corr": [1.0, 2.0, 3.0, 4.0], "extra": list("abcd")}, index=idx
    )
    flat = restored.reset_index()  # csv read: everything comes back as columns

    with patch("pyfm.nanny.aggregator.dio") as mock_dio:
        mock_dio.load_files.return_value.agg.return_value = restored
        out_restored = aggregator.load_convert_data(run_params, "hdf5")
        mock_dio.load_files.return_value.agg.return_value = flat
        out_flat = aggregator.load_convert_data(run_params, "csv")

    assert out_restored.index.names == ["cfg", "t", "extra"]
    assert list(out_restored.columns) == ["corr"]
    assert out_flat.index.names == ["cfg", "t", "extra"]
    assert out_restored.sort_index().equals(out_flat.sort_index())


def test_convert_output_single_key_is_exact_stem():
    task = _make_task(_agg_single())
    with (
        patch("pyfm.nanny.aggregator.create_task", return_value=task),
        patch("pyfm.nanny.aggregator.dio") as mock_dio,
    ):
        mock_dio.load_files.return_value.agg.return_value = pd.DataFrame({"corr": [1.0]})
        aggregator.convert_task_data("job", {}, output="/tmp/out.csv")
    assert mock_dio.write_files.call_args.kwargs["filestem"] == "/tmp/out"


def test_convert_output_multi_key_joins_under_base():
    agg = {
        "run": ["d1", "d2"],
        "d1": {"out_files": {"filestem": "processed/{format}/a"}, "load_files": {}},
        "d2": {"out_files": {"filestem": "processed/{format}/b"}, "load_files": {}},
    }
    task = _make_task(agg)
    stems = []
    with (
        patch("pyfm.nanny.aggregator.create_task", return_value=task),
        patch("pyfm.nanny.aggregator.dio") as mock_dio,
    ):
        mock_dio.load_files.return_value.agg.return_value = pd.DataFrame({"corr": [1.0]})
        mock_dio.write_files.side_effect = lambda df, **kw: stems.append(kw["filestem"])
        aggregator.convert_task_data("job", {}, output="/tmp/base")
    assert stems == ["/tmp/base/processed/{format}/a", "/tmp/base/processed/{format}/b"]


def test_convert_skips_empty_run_key():
    task = _make_task(_agg_single())
    with (
        patch("pyfm.nanny.aggregator.create_task", return_value=task),
        patch("pyfm.nanny.aggregator.dio") as mock_dio,
    ):
        mock_dio.load_files.return_value.agg.return_value = pd.DataFrame(
            {"corr": pd.array([], dtype="float64")}
        )
        aggregator.convert_task_data("job", {})
    mock_dio.write_files.assert_not_called()


def test_convert_no_agg_params_raises():
    task = Task(handler=MagicMock(), config=MagicMock(), key="k")
    task.handler.build_aggregator_params.return_value = {}
    with patch("pyfm.nanny.aggregator.create_task", return_value=task):
        with pytest.raises(ValueError, match="No aggregator parameters"):
            aggregator.convert_task_data("job", {})


def test_convert_average_builds_both_param_sets_and_writes_avg_stem():
    """Averaging on convert mirrors aggregate_task_data's raw/avg split (regressed
    twice via commits 4ec1283, 7b5d439): builds params for both the agg load
    (average=False) and the averaged output (average=True), loads the existing
    non-averaged agg files, routes the averaging actions through the processor,
    and writes to the ``_avg`` output filestem."""
    raw, avg = _avg_single(actions={"time_average": ["t1", "t4"], "real": True})
    task = _make_split_task(raw, avg)

    captured = {}

    def fake_execute(df, actions):
        captured["actions"] = dict(actions)
        return df

    with (
        patch("pyfm.nanny.aggregator.create_task", return_value=task),
        patch("pyfm.nanny.aggregator.dio") as mock_dio,
        patch("pyfm.nanny.aggregator.pc.execute", side_effect=fake_execute),
    ):
        mock_dio.load_files.return_value.agg.return_value = pd.DataFrame(
            {"corr": [1.0, 2.0]}
        )
        # Stub both loader entry points (convert uses load_files today; the
        # chunked stub keeps the test robust if the loader switches).
        mock_dio.load_files_chunked.return_value = pd.DataFrame(
            {"corr": [1.0, 2.0]}
        )
        aggregator.convert_task_data(
            "job", {}, input_format="csv", output_format="csv", average=True
        )

    built = [
        _average_of(c) for c in task.handler.build_aggregator_params.call_args_list
    ]
    assert {True, False} == set(built)

    # Load targets the existing (non-averaged) agg files.
    _, lkw = mock_dio.load_files.call_args
    assert lkw["filestem"] == "processed/{format}/pion.csv"
    assert lkw["replacements"]["format"] == "csv"

    # Averaging actions reach the processor.
    assert captured["actions"]["time_average"] == ["t1", "t4"]
    assert captured["actions"]["real"] is True

    # Write targets the _avg stem; same in/out format is fine when averaging.
    _, wkw = mock_dio.write_files.call_args
    assert wkw["filestem"] == "processed/{format}/pion_avg"
    assert wkw["format"] == "csv"


def test_convert_average_restores_complex_dtype_from_csv_strings():
    """CSV agg files store complex corr values as strings; the averaging step must
    restore numeric dtype before pc.execute or the numeric actions operate on
    strings (prototype scripts/locate_agg_files.py:53-54)."""
    raw, avg = _avg_single(actions={"average": ["tsource"]})
    task = _make_split_task(raw, avg)

    captured = {}

    def fake_execute(df, actions):
        captured["corr_dtype"] = str(df["corr"].dtype)
        return df

    with (
        patch("pyfm.nanny.aggregator.create_task", return_value=task),
        patch("pyfm.nanny.aggregator.dio") as mock_dio,
        patch("pyfm.nanny.aggregator.pc.execute", side_effect=fake_execute),
    ):
        mock_dio.load_files.return_value.agg.return_value = pd.DataFrame(
            {"corr": ["(1+2j)", "(3+4j)"]}
        )
        mock_dio.load_files_chunked.return_value = pd.DataFrame(
            {"corr": ["(1+2j)", "(3+4j)"]}
        )
        aggregator.convert_task_data("job", {}, input_format="csv", average=True)

    assert captured["corr_dtype"] == "complex128"


def test_convert_average_output_overrides_stem():
    """--output composes with averaging: single run key -> exact stem."""
    raw, avg = _avg_single()
    task = _make_split_task(raw, avg)
    with (
        patch("pyfm.nanny.aggregator.create_task", return_value=task),
        patch("pyfm.nanny.aggregator.dio") as mock_dio,
    ):
        mock_dio.load_files.return_value.agg.return_value = pd.DataFrame({"corr": [1.0]})
        mock_dio.load_files_chunked.return_value = pd.DataFrame({"corr": [1.0]})
        aggregator.convert_task_data("job", {}, average=True, output="/tmp/out.csv")
    assert mock_dio.write_files.call_args.kwargs["filestem"] == "/tmp/out"


def test_convert_without_average_runs_no_actions():
    """Parity guard: without --average, no processor actions run on the convert path."""
    task = _make_task(_agg_single())
    with (
        patch("pyfm.nanny.aggregator.create_task", return_value=task),
        patch("pyfm.nanny.aggregator.dio") as mock_dio,
        patch("pyfm.nanny.aggregator.pc.execute") as mock_exec,
    ):
        mock_dio.load_files.return_value.agg.return_value = pd.DataFrame({"corr": [1.0]})
        mock_dio.load_files_chunked.return_value = pd.DataFrame({"corr": [1.0]})
        aggregator.convert_task_data("job", {}, input_format="csv", output_format="hdf5")
    mock_exec.assert_not_called()


def test_aggregate_task_data_average_writes_to_avg_stem_and_runs_actions():
    """Regression guard for the averaging path (Precedent 2 / Composite Lesson 1),
    which the dataio refactor regressed twice (commits 4ec1283, 7b5d439). Pins that
    aggregation with averaging builds params for both the raw load (average=False) and
    the averaged output (average=True), routes the averaging actions through the
    processor, and writes to the ``_avg`` output filestem.
    """
    handler = MagicMock()

    def build_params(config, average):
        suffix = "_avg" if average else ""
        return {
            "run": ["diagram"],
            "diagram": {
                "out_files": {"filestem": f"processed/{{format}}/pion{suffix}"},
                "load_files": {
                    "filestem": "in/{series}/{cfg}/pion.h5",
                    "regex": {"series": "[a-z]", "cfg": "[0-9]+"},
                },
                "actions": {"time_average": ["t1", "t4"]} if average else {},
            },
        }

    handler.build_aggregator_params.side_effect = build_params
    task = Task(handler=handler, config=MagicMock(), key="diagram_key")

    captured = {}

    def fake_execute(df, actions):
        captured["actions"] = dict(actions)
        return df

    with (
        patch("pyfm.nanny.aggregator.create_task", return_value=task),
        patch("pyfm.nanny.aggregator.dio") as mock_dio,
        patch("pyfm.nanny.aggregator.pc.execute", side_effect=fake_execute),
    ):
        # The merged load path uses the chunked loader (feature/dataio-refactor);
        # convert still uses load_files, so stub both entry points.
        mock_dio.load_files.return_value.agg.return_value = pd.DataFrame({"corr": [1.0]})
        mock_dio.load_files_chunked.return_value = pd.DataFrame({"corr": [1.0]})
        aggregator.aggregate_task_data("job", {}, format="csv", average=True)

    built = [c.args[1] for c in handler.build_aggregator_params.call_args_list]
    assert {True, False} == set(built)
    assert captured["actions"]["time_average"] == ["t1", "t4"]
    assert mock_dio.write_files.call_args.kwargs["filestem"] == "processed/{format}/pion_avg"
