"""Unit tests for pyfm CLI 'nanny add' subcommand."""
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from pyfm.cli import cli

NANNY_MOD = "pyfm.cli.nanny"


@pytest.fixture
def runner():
    return CliRunner()


def _fake_params(job_setup_keys=("smear", "hadrons")):
    return {
        "nanny": {"todo_file": "/tmp/todo"},
        "job_setup": {k: {} for k in job_setup_keys},
    }


class TestNannyAddCfg:
    def test_cfg_dispatches_to_add_entries(self, runner):
        fake_params = _fake_params()
        with (
            patch(f"{NANNY_MOD}.utils") as mock_utils,
            patch(f"{NANNY_MOD}.validate_steps") as mock_vs,
            patch(f"{NANNY_MOD}.parse_cfgs", return_value=["200", "220"]) as mock_pc,
            patch(f"{NANNY_MOD}.add_entries") as mock_ae,
        ):
            mock_utils.io.load_param.return_value = fake_params
            result = runner.invoke(
                cli, ["nanny", "add", "-s", "a", "smear", "hadrons", "--config", "200", "--config", "220"]
            )
        assert result.exit_code == 0, result.output
        mock_vs.assert_called_once_with(("smear", "hadrons"), fake_params["job_setup"])
        mock_pc.assert_called_once_with(cfg=(200, 220), cfg_range=None)
        mock_ae.assert_called_once_with("/tmp/todo", "a", ["200", "220"], ("smear", "hadrons"))

    def test_cfg_range_dispatches_correctly(self, runner):
        fake_params = _fake_params()
        with (
            patch(f"{NANNY_MOD}.utils") as mock_utils,
            patch(f"{NANNY_MOD}.validate_steps"),
            patch(f"{NANNY_MOD}.parse_cfgs", return_value=["200", "220"]) as mock_pc,
            patch(f"{NANNY_MOD}.add_entries") as mock_ae,
        ):
            mock_utils.io.load_param.return_value = fake_params
            result = runner.invoke(
                cli, ["nanny", "add", "-s", "a", "smear", "--config-range", "200", "400", "20"]
            )
        assert result.exit_code == 0, result.output
        mock_pc.assert_called_once_with(cfg=None, cfg_range=(200, 400, 20))
        mock_ae.assert_called_once_with("/tmp/todo", "a", ["200", "220"], ("smear",))


class TestNannyAddValidation:
    def test_invalid_step_prints_error_and_exits_nonzero(self, runner):
        fake_params = _fake_params()
        with (
            patch(f"{NANNY_MOD}.utils") as mock_utils,
            patch(f"{NANNY_MOD}.validate_steps", side_effect=ValueError("Invalid steps: ['bogus']")),
            patch(f"{NANNY_MOD}.parse_cfgs"),
            patch(f"{NANNY_MOD}.add_entries"),
        ):
            mock_utils.io.load_param.return_value = fake_params
            result = runner.invoke(
                cli, ["nanny", "add", "-s", "a", "bogus", "--config", "200"]
            )
        assert result.exit_code != 0
        assert "Invalid steps" in result.output

    def test_both_cfg_and_cfg_range_exits_nonzero(self, runner):
        fake_params = _fake_params()
        with patch(f"{NANNY_MOD}.utils") as mock_utils:
            mock_utils.io.load_param.return_value = fake_params
            result = runner.invoke(
                cli,
                ["nanny", "add", "-s", "a", "smear", "--config", "200", "--config-range", "200", "400", "20"],
            )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower() or "mutually exclusive" in (result.output + "").lower()

    def test_neither_cfg_nor_cfg_range_exits_nonzero(self, runner):
        fake_params = _fake_params()
        with patch(f"{NANNY_MOD}.utils") as mock_utils:
            mock_utils.io.load_param.return_value = fake_params
            result = runner.invoke(cli, ["nanny", "add", "-s", "a", "smear"])
        assert result.exit_code != 0
        assert "required" in result.output.lower()


class TestNannyAddParamFile:
    def test_uses_custom_param_file(self, runner):
        fake_params = _fake_params()
        with (
            patch(f"{NANNY_MOD}.utils") as mock_utils,
            patch(f"{NANNY_MOD}.validate_steps"),
            patch(f"{NANNY_MOD}.parse_cfgs", return_value=["200"]),
            patch(f"{NANNY_MOD}.add_entries"),
        ):
            mock_utils.io.load_param.return_value = fake_params
            result = runner.invoke(
                cli, ["nanny", "add", "-s", "a", "smear", "--config", "200", "-p", "custom.yaml"]
            )
        assert result.exit_code == 0, result.output
        mock_utils.io.load_param.assert_called_once_with("custom.yaml")

    def test_defaults_to_params_yaml(self, runner):
        fake_params = _fake_params()
        with (
            patch(f"{NANNY_MOD}.utils") as mock_utils,
            patch(f"{NANNY_MOD}.validate_steps"),
            patch(f"{NANNY_MOD}.parse_cfgs", return_value=["200"]),
            patch(f"{NANNY_MOD}.add_entries"),
        ):
            mock_utils.io.load_param.return_value = fake_params
            result = runner.invoke(
                cli, ["nanny", "add", "-s", "a", "smear", "--config", "200"]
            )
        assert result.exit_code == 0, result.output
        mock_utils.io.load_param.assert_called_once_with("params.yaml")
