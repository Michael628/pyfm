import pytest

from pyfm.cli import cli, completion


class TestCompletionHelp:
    def test_help(self, runner):
        result = runner.invoke(completion, ["--help"])
        assert result.exit_code == 0
        assert "--shell" in result.output
        assert "--prog" in result.output

    def test_registered_on_root_cli(self, runner):
        """The completion command is reachable via the root pyfm group."""
        result = runner.invoke(cli, ["completion", "--help"])
        assert result.exit_code == 0
        assert "--shell" in result.output

    def test_listed_in_root_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "completion" in result.output


class TestShellRequired:
    def test_missing_shell_errors(self, runner):
        result = runner.invoke(completion, [])
        assert result.exit_code != 0
        assert "Missing option '--shell'" in result.output

    def test_invalid_shell_rejected(self, runner):
        result = runner.invoke(completion, ["--shell", "tcsh"])
        assert result.exit_code != 0
        assert "Invalid value for '--shell'" in result.output
        assert "tcsh" in result.output


class TestScriptGeneration:
    def test_bash_output(self, runner):
        result = runner.invoke(completion, ["--shell", "bash"])
        assert result.exit_code == 0
        out = result.output
        # The completion function is derived from prog_name (pyfm).
        assert "_pyfm_completion()" in out
        # Bash wiring: registers the function against the prog name.
        assert "complete -o nosort -F _pyfm_completion pyfm" in out
        # The env var Click reads at completion time.
        assert "_PYFM_COMPLETE=bash_complete" in out

    def test_zsh_output(self, runner):
        result = runner.invoke(completion, ["--shell", "zsh"])
        assert result.exit_code == 0
        out = result.output
        assert out.startswith("#compdef pyfm")
        assert "_pyfm_completion()" in out
        assert "_PYFM_COMPLETE=zsh_complete" in out

    def test_fish_output(self, runner):
        result = runner.invoke(completion, ["--shell", "fish"])
        assert result.exit_code == 0
        out = result.output
        assert "function _pyfm_completion;" in out
        assert "_PYFM_COMPLETE=fish_complete" in out

    def test_generated_bash_matches_env_var_bootstrap(self, runner):
        """The script we emit must match what `_PYFM_COMPLETE=bash_source` produces,
        so a user sourcing our output gets the same completion Click expects."""
        from click.shell_completion import BashComplete

        bootstrap = BashComplete(
            cli=cli, ctx_args={}, prog_name="pyfm", complete_var="_PYFM_COMPLETE"
        ).source()
        result = runner.invoke(completion, ["--shell", "bash"])
        assert result.exit_code == 0
        assert result.output.rstrip("\n") == bootstrap.rstrip("\n")


class TestProgOverride:
    def test_prog_changes_env_var_and_complete_target(self, runner):
        result = runner.invoke(completion, ["--shell", "bash", "--prog", "mypyfm"])
        assert result.exit_code == 0
        out = result.output
        # Env var is derived from prog_name (uppercased).
        assert "_MYPYFM_COMPLETE=bash_complete" in out
        # The `complete` line binds the function to the prog name.
        assert "complete -o nosort -F _mypyfm_completion mypyfm" in out
        # Original name must not leak when overridden.
        assert "_PYFM_COMPLETE=bash_complete" not in out
