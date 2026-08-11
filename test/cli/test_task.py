"""Unit tests for pyfm CLI task subcommand dispatch logic (deprecated shims)."""
from unittest.mock import patch

from pyfm.cli import cli


FAKE_PARAMS = {"tasks": []}


def test_generate_delegates_to_nanny_generate(runner):
    with (
        patch("pyfm.utils.io.load_param", return_value=FAKE_PARAMS),
        patch("pyfm.nanny.write_input_file", return_value="output.xml") as mock_wif,
        patch("pyfm.utils.get_logger"),
    ):
        result = runner.invoke(cli, ["task", "generate", "-j", "hadrons_lmi", "-s", "a", "-n", "100"])
        assert result.exit_code == 0, result.output
        assert "deprecated" in result.output.lower()
        mock_wif.assert_called_once_with("hadrons_lmi", FAKE_PARAMS, "a", "100")


def test_generate_uses_custom_param_file(runner):
    with (
        patch("pyfm.utils.io.load_param", return_value=FAKE_PARAMS) as mock_load,
        patch("pyfm.nanny.write_input_file", return_value="output.xml"),
        patch("pyfm.utils.get_logger"),
    ):
        result = runner.invoke(
            cli, ["task", "generate", "-j", "hadrons_lmi", "-s", "a", "-n", "100", "-p", "custom.yaml"]
        )
        assert result.exit_code == 0, result.output
        mock_load.assert_called_once_with("custom.yaml")


def test_generate_missing_required_arg_fails(runner):
    result = runner.invoke(cli, ["task", "generate", "-s", "a", "-n", "100"])
    assert result.exit_code != 0


def test_aggregate_delegates_to_export_corr(runner):
    with (
        patch("pyfm.utils.io.load_param", return_value=FAKE_PARAMS),
        patch("pyfm.nanny.aggregator.aggregate_task_data") as mock_agg,
        patch("pyfm.utils.set_logging_level"),
    ):
        result = runner.invoke(cli, ["task", "aggregate", "-j", "hadrons_lmi"])
        assert result.exit_code == 0, result.output
        assert "deprecated" in result.output.lower()
        mock_agg.assert_called_once_with("hadrons_lmi", FAKE_PARAMS, format="csv", average=False, skip_existing=False)


def test_aggregate_average_flag(runner):
    with (
        patch("pyfm.utils.io.load_param", return_value=FAKE_PARAMS),
        patch("pyfm.nanny.aggregator.aggregate_task_data") as mock_agg,
        patch("pyfm.utils.set_logging_level"),
    ):
        result = runner.invoke(cli, ["task", "aggregate", "-j", "hadrons_lmi", "--average"])
        assert result.exit_code == 0, result.output
        mock_agg.assert_called_once_with("hadrons_lmi", FAKE_PARAMS, format="csv", average=True, skip_existing=False)


def test_aggregate_skip_existing_flag(runner):
    with (
        patch("pyfm.utils.io.load_param", return_value=FAKE_PARAMS),
        patch("pyfm.nanny.aggregator.aggregate_task_data") as mock_agg,
        patch("pyfm.utils.set_logging_level"),
    ):
        result = runner.invoke(cli, ["task", "aggregate", "-j", "hadrons_lmi", "--skip-existing"])
        assert result.exit_code == 0, result.output
        mock_agg.assert_called_once_with("hadrons_lmi", FAKE_PARAMS, format="csv", average=False, skip_existing=True)


def test_aggregate_custom_format(runner):
    with (
        patch("pyfm.utils.io.load_param", return_value=FAKE_PARAMS),
        patch("pyfm.nanny.aggregator.aggregate_task_data") as mock_agg,
        patch("pyfm.utils.set_logging_level"),
    ):
        result = runner.invoke(cli, ["task", "aggregate", "-j", "hadrons_lmi", "-f", "hdf5"])
        assert result.exit_code == 0, result.output
        mock_agg.assert_called_once_with("hadrons_lmi", FAKE_PARAMS, format="hdf5", average=False, skip_existing=False)


def test_aggregate_missing_job_fails(runner):
    result = runner.invoke(cli, ["task", "aggregate"])
    assert result.exit_code != 0
