"""Unit tests for pyfm CLI contract subcommand dispatch logic."""
from unittest.mock import MagicMock, patch

from pyfm.cli import cli


def _fake_config():
    cfg = MagicMock()
    cfg.diagrams = {}
    cfg.hardware = "cpu"
    cfg.logging_level = "INFO"
    cfg.comm_size = 1
    cfg.rank = 0
    cfg.overwrite = False
    return cfg


def test_contract_run_dispatches(runner):
    with (
        patch("pyfm.cli.contract.utils") as mock_utils,
        patch("pyfm.cli.contract.build_config") as mock_bc,
        patch("pyfm.cli.contract.execute") as mock_execute,
    ):
        mock_utils.io.load_param.return_value = {}
        mock_utils.set_logging_level.return_value = MagicMock()
        mock_bc.return_value = _fake_config()
        result = runner.invoke(cli, ["contract", "run", "-p", "params.yaml"])
        assert result.exit_code == 0, result.output
        mock_bc.assert_called_once()
        # diagrams is empty so execute is never called — that is correct
        mock_execute.assert_not_called()


def test_contract_run_missing_param_file_fails(runner):
    result = runner.invoke(cli, ["contract", "run"])
    assert result.exit_code != 0


def test_contract_run_uses_param_file(runner):
    with (
        patch("pyfm.cli.contract.utils") as mock_utils,
        patch("pyfm.cli.contract.build_config") as mock_bc,
        patch("pyfm.cli.contract.execute"),
    ):
        mock_utils.io.load_param.return_value = {}
        mock_utils.set_logging_level.return_value = MagicMock()
        mock_bc.return_value = _fake_config()
        result = runner.invoke(cli, ["contract", "run", "-p", "my_params.yaml"])
        assert result.exit_code == 0, result.output
        mock_utils.io.load_param.assert_called_once_with("my_params.yaml")
