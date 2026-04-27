"""Unit tests for pyfm CLI nanny subcommand dispatch logic."""
from unittest.mock import MagicMock, patch

from pyfm.cli import cli


def test_nanny_run_defaults(runner, fake_yaml_params):
    with (
        patch("pyfm.cli.nanny.utils") as mock_utils,
        patch("pyfm.cli.nanny.nanny_loop") as mock_loop,
        patch("pyfm.cli.nanny.os.system"),
    ):
        result = runner.invoke(cli, ["nanny", "run"])
        assert result.exit_code == 0
        mock_utils.set_logging_level.assert_called_once_with("INFO")
        mock_loop.assert_called_once_with("params.yaml", require_step=None)


def test_nanny_run_custom_params(runner, fake_yaml_params):
    with (
        patch("pyfm.cli.nanny.utils") as mock_utils,
        patch("pyfm.cli.nanny.nanny_loop") as mock_loop,
        patch("pyfm.cli.nanny.os.system"),
    ):
        result = runner.invoke(
            cli,
            ["nanny", "run", "-p", "custom.yaml", "-j", "hadrons_lmi", "--logging-level", "DEBUG"],
        )
        assert result.exit_code == 0
        mock_utils.set_logging_level.assert_called_once_with("DEBUG")
        mock_loop.assert_called_once_with("custom.yaml", require_step="hadrons_lmi")


def test_nanny_submit_required_args(runner, fake_yaml_params):
    fake_nanny_cfg = MagicMock()
    fake_job_cfg = MagicMock()
    with (
        patch("pyfm.cli.nanny.utils") as mock_utils,
        patch("pyfm.cli.nanny.get_nanny_config", return_value=fake_nanny_cfg) as mock_gnc,
        patch("pyfm.cli.nanny.get_job_config", return_value=fake_job_cfg) as mock_gjc,
        patch("pyfm.cli.nanny.submit_job") as mock_submit,
        patch("pyfm.cli.nanny.os.environ", {}) as mock_env,
    ):
        mock_utils.io.load_param.return_value = fake_yaml_params
        result = runner.invoke(
            cli,
            ["nanny", "submit", "-p", "params.yaml", "-i", "/path/to/todo", "-j", "hadrons_lmi"],
        )
        assert result.exit_code == 0
        mock_utils.io.load_param.assert_called_once_with("params.yaml")
        mock_gnc.assert_called_once_with(fake_yaml_params)
        mock_gjc.assert_called_once_with("hadrons_lmi", fake_yaml_params)
        assert mock_env["INPUTLIST"] == "/path/to/todo"
        mock_submit.assert_called_once_with(fake_nanny_cfg, fake_job_cfg, 1)


def test_nanny_submit_missing_input_fails(runner):
    result = runner.invoke(cli, ["nanny", "submit", "-j", "hadrons_lmi"])
    assert result.exit_code != 0


def test_nanny_submit_missing_job_fails(runner):
    result = runner.invoke(cli, ["nanny", "submit", "-i", "/path/to/todo"])
    assert result.exit_code != 0


def test_nanny_check_full_todo_mode(runner, fake_yaml_params):
    with (
        patch("pyfm.cli.nanny.utils") as mock_utils,
        patch("pyfm.cli.nanny.check_jobs") as mock_cj,
        patch("pyfm.cli.nanny.audit_outfiles") as mock_ao,
    ):
        mock_utils.io.load_param.return_value = fake_yaml_params
        result = runner.invoke(cli, ["nanny", "check"])
        assert result.exit_code == 0
        mock_utils.io.load_param.assert_called_once_with("params.yaml")
        mock_cj.assert_called_once_with(fake_yaml_params)
        mock_ao.assert_not_called()


def test_nanny_check_audit_mode(runner, fake_yaml_params):
    with (
        patch("pyfm.cli.nanny.utils") as mock_utils,
        patch("pyfm.cli.nanny.check_jobs") as mock_cj,
        patch("pyfm.cli.nanny.audit_outfiles") as mock_ao,
    ):
        mock_utils.io.load_param.return_value = fake_yaml_params
        result = runner.invoke(
            cli, ["nanny", "check", "-j", "hadrons_lmi", "-s", "48I", "-n", "1000"]
        )
        assert result.exit_code == 0
        mock_ao.assert_called_once_with("hadrons_lmi", fake_yaml_params, "48I", "1000", verbose=False)
        mock_cj.assert_not_called()


def test_nanny_check_audit_mode_verbose(runner, fake_yaml_params):
    with (
        patch("pyfm.cli.nanny.utils") as mock_utils,
        patch("pyfm.cli.nanny.audit_outfiles") as mock_ao,
        patch("pyfm.cli.nanny.check_jobs"),
    ):
        mock_utils.io.load_param.return_value = fake_yaml_params
        result = runner.invoke(
            cli, ["nanny", "check", "-j", "hadrons_lmi", "-s", "48I", "-n", "1000", "-v"]
        )
        assert result.exit_code == 0
        mock_ao.assert_called_once_with("hadrons_lmi", fake_yaml_params, "48I", "1000", verbose=True)


def test_nanny_check_partial_args_falls_back_to_check_jobs(runner, fake_yaml_params):
    """Job specified but no series or config -> AND logic falls through to check_jobs."""
    with (
        patch("pyfm.cli.nanny.utils") as mock_utils,
        patch("pyfm.cli.nanny.check_jobs") as mock_cj,
        patch("pyfm.cli.nanny.audit_outfiles") as mock_ao,
    ):
        mock_utils.io.load_param.return_value = fake_yaml_params
        result = runner.invoke(cli, ["nanny", "check", "-j", "hadrons_lmi"])
        assert result.exit_code == 0
        mock_cj.assert_called_once_with(fake_yaml_params)
        mock_ao.assert_not_called()
