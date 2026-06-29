import click
from click.shell_completion import BashComplete, FishComplete, ZshComplete

from pyfm.cli.audit import audit
from pyfm.cli.contract import contract
from pyfm.cli.nanny import nanny
from pyfm.cli.systems import build, workspace
from pyfm.cli.task import task

# Supported shells mapped to their Click completion-script generators.
_SHELL_COMPLETIONS = {
    "bash": BashComplete,
    "zsh": ZshComplete,
    "fish": FishComplete,
}


def _complete_var(prog_name: str) -> str:
    """Derive the completion env var exactly as Click does internally.

    Mirrors ``click.core.BaseCommand._main_shell_completion`` so the script we
    emit is byte-for-byte what the entry point would expect at completion time.
    """
    complete_name = prog_name.replace("-", "_").replace(".", "_")
    return f"_{complete_name}_COMPLETE".upper()


@click.group()
def cli():
    """PyFM - lattice QCD workflow toolkit."""
    pass


@click.command()
@click.option(
    "--shell",
    type=click.Choice(list(_SHELL_COMPLETIONS)),
    required=True,
    help="Target shell to generate a completion script for.",
)
@click.option(
    "--prog",
    "prog_name",
    type=str,
    default="pyfm",
    show_default=True,
    help="Executable name as installed on PATH (override for aliased installs).",
)
def completion(shell, prog_name):
    """Print a shell completion script for pyfm.

    Install once by sourcing the output. For example, for bash:

    \b
        pyfm completion --shell bash >> ~/.bashrc

    For zsh, redirect to a file on your ``fpath`` (e.g. ``~/.zsh/_pyfm``);
    for fish, to ``~/.config/fish/completions/pyfm.fish``. The script assumes
    the executable named by --prog (default ``pyfm``) is on your PATH.
    """
    complete_cls = _SHELL_COMPLETIONS[shell]
    completer = complete_cls(
        cli=cli,
        ctx_args={},
        prog_name=prog_name,
        complete_var=_complete_var(prog_name),
    )
    click.echo(completer.source())


cli.add_command(completion)
cli.add_command(nanny)
cli.add_command(task)
cli.add_command(contract)
cli.add_command(audit)
cli.add_command(build)
cli.add_command(workspace)
