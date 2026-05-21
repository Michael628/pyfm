from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from pyfm.cli.systems import build, workspace


@pytest.fixture
def runner():
    return CliRunner()


class TestBuildRun:
    def test_help(self, runner):
        result = runner.invoke(build, ["run", "--help"])
        assert result.exit_code == 0
        assert "--system" in result.output
        assert "--threads" in result.output

    def test_default_args(self, runner):
        with patch("pyfm.cli.systems.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = runner.invoke(build, ["run"])
        assert result.exit_code == 0
        call_args = mock_run.call_args[0][0]
        assert call_args[0].endswith("build.sh")
        assert "--system" in call_args
        assert "scalar" in call_args
        assert "--threads" in call_args
        assert "4" in call_args

    def test_boolean_flags_passed(self, runner):
        with patch("pyfm.cli.systems.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = runner.invoke(build, ["run", "--gmp", "--grid", "--force"])
        assert result.exit_code == 0
        call_args = mock_run.call_args[0][0]
        assert "--gmp" in call_args
        assert "--grid" in call_args
        assert "--force" in call_args

    def test_custom_system_and_threads(self, runner):
        with patch("pyfm.cli.systems.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = runner.invoke(build, ["run", "--system", "perlmutter", "--threads", "8"])
        assert result.exit_code == 0
        call_args = mock_run.call_args[0][0]
        assert "perlmutter" in call_args
        assert "8" in call_args

    def test_ext_option(self, runner):
        with patch("pyfm.cli.systems.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = runner.invoke(build, ["run", "--ext", "cuda"])
        assert result.exit_code == 0
        call_args = mock_run.call_args[0][0]
        assert "--ext" in call_args
        assert "cuda" in call_args

    def test_script_path_is_package_relative(self, runner):
        with patch("pyfm.cli.systems.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            runner.invoke(build, ["run"])
        call_args = mock_run.call_args[0][0]
        assert "systems/build.sh" in call_args[0]

    def test_inactive_flags_not_passed(self, runner):
        with patch("pyfm.cli.systems.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            runner.invoke(build, ["run"])
        call_args = mock_run.call_args[0][0]
        assert "--gmp" not in call_args
        assert "--hadrons" not in call_args


class TestWorkspaceSetup:
    def test_help(self, runner):
        result = runner.invoke(workspace, ["setup", "--help"])
        assert result.exit_code == 0
        assert "--workspace" in result.output
        assert "--scheduler" in result.output

    def test_no_args(self, runner):
        with patch("pyfm.cli.systems.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = runner.invoke(workspace, ["setup"])
        assert result.exit_code == 0
        call_args = mock_run.call_args[0][0]
        assert call_args[0].endswith("setup-workspace.sh")

    def test_all_options(self, runner):
        with patch("pyfm.cli.systems.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = runner.invoke(workspace, [
                "setup",
                "--workspace", "/work/me",
                "--storage", "/scratch/me",
                "--scheduler", "slurm",
                "--lattice", "3248",
                "--system", "perlmutter",
            ])
        assert result.exit_code == 0
        call_args = mock_run.call_args[0][0]
        assert "--workspace" in call_args and "/work/me" in call_args
        assert "--storage" in call_args and "/scratch/me" in call_args
        assert "--scheduler" in call_args and "slurm" in call_args
        assert "--lattice" in call_args and "3248" in call_args
        assert "--system" in call_args and "perlmutter" in call_args

    def test_unset_options_not_passed(self, runner):
        with patch("pyfm.cli.systems.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            runner.invoke(workspace, ["setup", "--system", "scalar"])
        call_args = mock_run.call_args[0][0]
        assert "--workspace" not in call_args
        assert "--storage" not in call_args


class TestWorkspaceEnv:
    def test_help(self, runner):
        result = runner.invoke(workspace, ["env", "--help"])
        assert result.exit_code == 0
        assert "--system" in result.output
        assert "--ext" in result.output
        assert "--runtime-env" in result.output

    def test_system_required(self, runner):
        result = runner.invoke(workspace, ["env"])
        assert result.exit_code != 0
        assert "Missing option '--system'" in result.output

    def test_prints_export_statements(self, runner):
        subshell_output = (
            "PATH=/new/path:/usr/bin\0"
            "NEWVAR=hello\0"
            "_PRIV=skip\0"
            "BASH_FUNC_foo%%=skip\0"
        )
        with patch("pyfm.cli.systems.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=subshell_output)
            with patch("pyfm.cli.systems.os.environ", {"PATH": "/usr/bin"}):
                result = runner.invoke(workspace, ["env", "--system", "perlmutter"])
        assert result.exit_code == 0
        assert "export NEWVAR=hello" in result.output
        assert "export PATH=/new/path:/usr/bin" in result.output
        assert "export _PRIV=skip" in result.output
        assert "BASH_FUNC" not in result.output

    def test_unchanged_vars_skipped(self, runner):
        subshell_output = "UNCHANGED=same_value\0CHANGED=new_value\0"
        with patch("pyfm.cli.systems.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=subshell_output)
            with patch("pyfm.cli.systems.os.environ", {"UNCHANGED": "same_value", "CHANGED": "old_value"}):
                result = runner.invoke(workspace, ["env", "--system", "perlmutter"])
        assert "export UNCHANGED" not in result.output
        assert "export CHANGED=new_value" in result.output

    def test_bash_subshell_command_contains_source(self, runner):
        with patch("pyfm.cli.systems.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="")
            runner.invoke(workspace, ["env", "--system", "perlmutter", "--ext", "cuda"])
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "bash"
        assert call_args[1] == "-c"
        cmd_str = call_args[2]
        assert "source" in cmd_str
        assert "--system perlmutter" in cmd_str
        assert "--ext cuda" in cmd_str
        assert "&& env" in cmd_str

    def test_runtime_env_default(self, runner):
        with patch("pyfm.cli.systems.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="")
            runner.invoke(workspace, ["env", "--system", "perlmutter"])
        cmd_str = mock_run.call_args[0][0][2]
        assert "--runtime-env true" in cmd_str
