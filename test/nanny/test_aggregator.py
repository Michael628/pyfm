"""Unit tests for pyfm/nanny/aggregator.py convert path."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

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
                "load_files": {"filestem": "in/{series}/{cfg}/pion.h5"},
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
        mock_dio.load_files.return_value.agg.return_value = pd.DataFrame({"corr": [1.0]})
        aggregator.aggregate_task_data("job", {}, format="csv", average=True)

    built = [c.args[1] for c in handler.build_aggregator_params.call_args_list]
    assert {True, False} == set(built)
    assert captured["actions"]["time_average"] == ["t1", "t4"]
    assert mock_dio.write_files.call_args.kwargs["filestem"] == "processed/{format}/pion_avg"
