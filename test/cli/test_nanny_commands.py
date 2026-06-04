"""Unit tests for pyfm CLI nanny subcommands dispatch logic."""
from unittest.mock import MagicMock, call, patch

import pytest
from click.testing import CliRunner

from pyfm.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


class TestNannyRun:
    def test_run_defaults(self, runner):
        with (
            patch("pyfm.cli.nanny.utils") as mock_utils,
            patch("pyfm.cli.nanny.nanny_loop") as mock_loop,
            patch("pyfm.cli.nanny.os.system") as mock_sys,
        ):
            result = runner.invoke(cli, ["nanny", "run"])
            assert result.exit_code == 0
            mock_sys.assert_called_once_with("umask 022")
            mock_utils.set_logging_level.assert_called_once_with("INFO")
            mock_loop.assert_called_once_with("params.yaml", require_step=None)

    def test_run_custom_args(self, runner):
        with (
            patch("pyfm.cli.nanny.utils") as mock_utils,
            patch("pyfm.cli.nanny.nanny_loop") as mock_loop,
            patch("pyfm.cli.nanny.os.system"),
        ):
            result = runner.invoke(
                cli,
                ["nanny", "run", "-p", "custom.yaml", "-j", "myjob", "--logging-level", "DEBUG"],
            )
            assert result.exit_code == 0
            mock_utils.set_logging_level.assert_called_once_with("DEBUG")
            mock_loop.assert_called_once_with("custom.yaml", require_step="myjob")

    def test_run_calls_umask(self, runner):
        with (
            patch("pyfm.cli.nanny.utils"),
            patch("pyfm.cli.nanny.nanny_loop"),
            patch("pyfm.cli.nanny.os.system") as mock_sys,
        ):
            runner.invoke(cli, ["nanny", "run"])
            mock_sys.assert_called_once_with("umask 022")


class TestNannySubmit:
    def test_submit_requires_input(self, runner):
        result = runner.invoke(cli, ["nanny", "submit", "-j", "step1"])
        assert result.exit_code != 0
        assert "input" in result.output.lower() or "Missing" in result.output

    def test_submit_requires_job(self, runner):
        result = runner.invoke(cli, ["nanny", "submit", "-i", "todo.txt"])
        assert result.exit_code != 0

    def test_submit_dispatches_correctly(self, runner):
        fake_params = {"key": "val"}
        fake_nanny_cfg = MagicMock()
        fake_job_cfg = MagicMock()

        with (
            patch("pyfm.cli.nanny.utils") as mock_utils,
            patch("pyfm.cli.nanny.get_nanny_config", return_value=fake_nanny_cfg) as mock_gnc,
            patch("pyfm.cli.nanny.get_job_config", return_value=fake_job_cfg) as mock_gjc,
            patch("pyfm.cli.nanny.submit_job") as mock_submit,
            patch("pyfm.cli.nanny.os.environ", {}) as mock_env,
        ):
            mock_utils.io.load_param.return_value = fake_params
            result = runner.invoke(
                cli, ["nanny", "submit", "-p", "p.yaml", "-i", "todo.txt", "-j", "step1"]
            )
            assert result.exit_code == 0
            mock_utils.set_logging_level.assert_called_once_with("INFO")
            mock_utils.io.load_param.assert_called_once_with("p.yaml")
            mock_gnc.assert_called_once_with(fake_params)
            mock_gjc.assert_called_once_with("step1", fake_params)
            assert mock_env["INPUTLIST"] == "todo.txt"
            mock_submit.assert_called_once_with(fake_nanny_cfg, fake_job_cfg, 1)

    def test_submit_custom_logging_level(self, runner):
        with (
            patch("pyfm.cli.nanny.utils") as mock_utils,
            patch("pyfm.cli.nanny.get_nanny_config", return_value=MagicMock()),
            patch("pyfm.cli.nanny.get_job_config", return_value=MagicMock()),
            patch("pyfm.cli.nanny.submit_job"),
            patch("pyfm.cli.nanny.os.environ", {}),
        ):
            mock_utils.io.load_param.return_value = {}
            runner.invoke(
                cli,
                ["nanny", "submit", "-i", "t.txt", "-j", "step", "--logging-level", "WARNING"],
            )
            mock_utils.set_logging_level.assert_called_once_with("WARNING")


class TestNannyCheck:
    def _fake_params(self):
        return {"key": "val"}

    def test_check_all_none_calls_check_jobs(self, runner):
        fake_params = self._fake_params()
        with (
            patch("pyfm.cli.nanny.utils") as mock_utils,
            patch("pyfm.cli.nanny.check_jobs") as mock_cj,
            patch("pyfm.cli.nanny.audit_outfiles") as mock_ao,
        ):
            mock_utils.io.load_param.return_value = fake_params
            result = runner.invoke(cli, ["nanny", "check"])
            assert result.exit_code == 0
            mock_cj.assert_called_once_with(fake_params)
            mock_ao.assert_not_called()

    def test_check_partial_args_calls_check_jobs(self, runner):
        """Only job specified (no series/config) -> check_jobs branch."""
        fake_params = self._fake_params()
        with (
            patch("pyfm.cli.nanny.utils") as mock_utils,
            patch("pyfm.cli.nanny.check_jobs") as mock_cj,
            patch("pyfm.cli.nanny.audit_outfiles") as mock_ao,
        ):
            mock_utils.io.load_param.return_value = fake_params
            result = runner.invoke(cli, ["nanny", "check", "-j", "step1"])
            assert result.exit_code == 0
            mock_cj.assert_called_once_with(fake_params)
            mock_ao.assert_not_called()

    def test_check_all_three_calls_audit_outfiles(self, runner):
        fake_params = self._fake_params()
        with (
            patch("pyfm.cli.nanny.utils") as mock_utils,
            patch("pyfm.cli.nanny.check_jobs") as mock_cj,
            patch("pyfm.cli.nanny.create_task", return_value="fake-task") as mock_ct,
            patch("pyfm.cli.nanny.audit_outfiles") as mock_ao,
        ):
            mock_utils.io.load_param.return_value = fake_params
            result = runner.invoke(
                cli, ["nanny", "check", "-j", "step1", "-s", "A", "-n", "42"]
            )
            assert result.exit_code == 0
            mock_ct.assert_called_once_with("step1", fake_params, "A", "42")
            mock_ao.assert_called_once_with("fake-task", verbose=False)
            mock_cj.assert_not_called()

    def test_check_verbose_flag_passed_to_audit(self, runner):
        fake_params = self._fake_params()
        with (
            patch("pyfm.cli.nanny.utils") as mock_utils,
            patch("pyfm.cli.nanny.create_task", return_value="fake-task") as mock_ct,
            patch("pyfm.cli.nanny.audit_outfiles") as mock_ao,
            patch("pyfm.cli.nanny.check_jobs"),
        ):
            mock_utils.io.load_param.return_value = fake_params
            runner.invoke(cli, ["nanny", "check", "-j", "step1", "-s", "A", "-n", "42", "-v"])
            mock_ct.assert_called_once_with("step1", fake_params, "A", "42")
            mock_ao.assert_called_once_with("fake-task", verbose=True)

    def test_check_series_and_config_no_job_calls_check_jobs(self, runner):
        """Series+config without job -> check_jobs (all three must be non-None)."""
        fake_params = self._fake_params()
        with (
            patch("pyfm.cli.nanny.utils") as mock_utils,
            patch("pyfm.cli.nanny.check_jobs") as mock_cj,
            patch("pyfm.cli.nanny.audit_outfiles") as mock_ao,
        ):
            mock_utils.io.load_param.return_value = fake_params
            result = runner.invoke(cli, ["nanny", "check", "-s", "A", "-n", "42"])
            assert result.exit_code == 0
            mock_cj.assert_called_once_with(fake_params)
            mock_ao.assert_not_called()
