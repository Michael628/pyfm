"""Unit tests for pyfm CLI export corr subcommand dispatch logic."""
from unittest.mock import patch

from pyfm.cli import cli


FAKE_PARAMS = {"tasks": []}


def test_corr_dispatches_defaults(runner):
    with (
        patch("pyfm.utils.io.load_param", return_value=FAKE_PARAMS),
        patch("pyfm.nanny.aggregator.aggregate_task_data") as mock_agg,
        patch("pyfm.utils.set_logging_level"),
    ):
        result = runner.invoke(cli, ["export", "corr", "-j", "hadrons_lmi"])
        assert result.exit_code == 0, result.output
        mock_agg.assert_called_once_with(
            "hadrons_lmi", FAKE_PARAMS, format="csv", average=False, skip_existing=False
        )


def test_corr_average_flag(runner):
    with (
        patch("pyfm.utils.io.load_param", return_value=FAKE_PARAMS),
        patch("pyfm.nanny.aggregator.aggregate_task_data") as mock_agg,
        patch("pyfm.utils.set_logging_level"),
    ):
        result = runner.invoke(cli, ["export", "corr", "-j", "hadrons_lmi", "--average"])
        assert result.exit_code == 0, result.output
        mock_agg.assert_called_once_with(
            "hadrons_lmi", FAKE_PARAMS, format="csv", average=True, skip_existing=False
        )


def test_corr_skip_existing_flag(runner):
    with (
        patch("pyfm.utils.io.load_param", return_value=FAKE_PARAMS),
        patch("pyfm.nanny.aggregator.aggregate_task_data") as mock_agg,
        patch("pyfm.utils.set_logging_level"),
    ):
        result = runner.invoke(cli, ["export", "corr", "-j", "hadrons_lmi", "--skip-existing"])
        assert result.exit_code == 0, result.output
        mock_agg.assert_called_once_with(
            "hadrons_lmi", FAKE_PARAMS, format="csv", average=False, skip_existing=True
        )


def test_corr_custom_format(runner):
    with (
        patch("pyfm.utils.io.load_param", return_value=FAKE_PARAMS),
        patch("pyfm.nanny.aggregator.aggregate_task_data") as mock_agg,
        patch("pyfm.utils.set_logging_level"),
    ):
        result = runner.invoke(cli, ["export", "corr", "-j", "hadrons_lmi", "-f", "hdf5"])
        assert result.exit_code == 0, result.output
        mock_agg.assert_called_once_with(
            "hadrons_lmi", FAKE_PARAMS, format="hdf5", average=False, skip_existing=False
        )


def test_corr_rejects_invalid_format(runner):
    result = runner.invoke(cli, ["export", "corr", "-j", "hadrons_lmi", "-f", "xml"])
    assert result.exit_code != 0
    assert "is not one of" in result.output


def test_corr_missing_job_fails(runner):
    result = runner.invoke(cli, ["export", "corr"])
    assert result.exit_code != 0
