"""Unit tests for pyfm CLI task subcommand dispatch logic."""
from unittest.mock import MagicMock, patch

from pyfm.cli import cli


FAKE_PARAMS = {"tasks": []}


def test_generate_dispatches_with_required_args(runner):
    with (
        patch("pyfm.cli.task.utils") as mock_utils,
        patch("pyfm.cli.task.write_input_file") as mock_wif,
    ):
        mock_utils.io.load_param.return_value = FAKE_PARAMS
        mock_wif.return_value = "output.xml"
        result = runner.invoke(cli, ["task", "generate", "-j", "hadrons_lmi", "-s", "a", "-n", "100"])
        assert result.exit_code == 0, result.output
        mock_wif.assert_called_once_with("hadrons_lmi", FAKE_PARAMS, "a", "100")


def test_generate_uses_custom_param_file(runner):
    with (
        patch("pyfm.cli.task.utils") as mock_utils,
        patch("pyfm.cli.task.write_input_file") as mock_wif,
    ):
        mock_utils.io.load_param.return_value = FAKE_PARAMS
        mock_wif.return_value = "output.xml"
        result = runner.invoke(
            cli, ["task", "generate", "-j", "hadrons_lmi", "-s", "a", "-n", "100", "-p", "custom.yaml"]
        )
        assert result.exit_code == 0, result.output
        mock_utils.io.load_param.assert_called_once_with("custom.yaml")


def test_generate_missing_required_arg_fails(runner):
    result = runner.invoke(cli, ["task", "generate", "-s", "a", "-n", "100"])
    assert result.exit_code != 0


def test_aggregate_dispatches_defaults(runner):
    with (
        patch("pyfm.cli.task.utils") as mock_utils,
        patch("pyfm.cli.task.aggregator.aggregate_task_data") as mock_agg,
    ):
        mock_utils.io.load_param.return_value = FAKE_PARAMS
        result = runner.invoke(cli, ["task", "aggregate", "-j", "hadrons_lmi"])
        assert result.exit_code == 0, result.output
        mock_agg.assert_called_once_with("hadrons_lmi", FAKE_PARAMS, format="csv", average=False, skip_existing=False, max_workers=1)


def test_aggregate_average_flag(runner):
    with (
        patch("pyfm.cli.task.utils") as mock_utils,
        patch("pyfm.cli.task.aggregator.aggregate_task_data") as mock_agg,
    ):
        mock_utils.io.load_param.return_value = FAKE_PARAMS
        result = runner.invoke(cli, ["task", "aggregate", "-j", "hadrons_lmi", "--average"])
        assert result.exit_code == 0, result.output
        mock_agg.assert_called_once_with("hadrons_lmi", FAKE_PARAMS, format="csv", average=True, skip_existing=False, max_workers=1)


def test_aggregate_skip_existing_flag(runner):
    with (
        patch("pyfm.cli.task.utils") as mock_utils,
        patch("pyfm.cli.task.aggregator.aggregate_task_data") as mock_agg,
    ):
        mock_utils.io.load_param.return_value = FAKE_PARAMS
        result = runner.invoke(cli, ["task", "aggregate", "-j", "hadrons_lmi", "--skip-existing"])
        assert result.exit_code == 0, result.output
        mock_agg.assert_called_once_with("hadrons_lmi", FAKE_PARAMS, format="csv", average=False, skip_existing=True, max_workers=1)


def test_aggregate_custom_format(runner):
    with (
        patch("pyfm.cli.task.utils") as mock_utils,
        patch("pyfm.cli.task.aggregator.aggregate_task_data") as mock_agg,
    ):
        mock_utils.io.load_param.return_value = FAKE_PARAMS
        result = runner.invoke(cli, ["task", "aggregate", "-j", "hadrons_lmi", "-f", "hdf5"])
        assert result.exit_code == 0, result.output
        mock_agg.assert_called_once_with("hadrons_lmi", FAKE_PARAMS, format="hdf5", average=False, skip_existing=False, max_workers=1)


def test_aggregate_missing_job_fails(runner):
    result = runner.invoke(cli, ["task", "aggregate"])
    assert result.exit_code != 0


def test_aggregate_max_workers_flag(runner):
    """--max-workers N forwards max_workers=N to aggregate_task_data."""
    with (
        patch("pyfm.cli.task.utils") as mock_utils,
        patch("pyfm.cli.task.aggregator.aggregate_task_data") as mock_agg,
    ):
        mock_utils.io.load_param.return_value = FAKE_PARAMS
        result = runner.invoke(
            cli, ["task", "aggregate", "-j", "hadrons_lmi", "--max-workers", "4"]
        )
        assert result.exit_code == 0, result.output
        mock_agg.assert_called_once_with(
            "hadrons_lmi", FAKE_PARAMS, format="csv", average=False, skip_existing=False, max_workers=4
        )


def test_aggregate_max_workers_must_be_int(runner):
    """Non-int --max-workers is rejected by Click's type validation."""
    with (
        patch("pyfm.cli.task.utils") as mock_utils,
        patch("pyfm.cli.task.aggregator.aggregate_task_data") as mock_agg,
    ):
        mock_utils.io.load_param.return_value = FAKE_PARAMS
        result = runner.invoke(
            cli, ["task", "aggregate", "-j", "hadrons_lmi", "--max-workers", "abc"]
        )
        assert result.exit_code != 0
