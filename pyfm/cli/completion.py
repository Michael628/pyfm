"""Shell-completion script generation for the ``pyfm`` console script.

Extracted from the facade (``pyfm/cli/__init__.py``) so the root group reads
as "what groups exist" rather than "how shell completion is generated". The
``completion`` command is re-exported from ``pyfm.cli`` and registered on the
root group.
"""
import click
from click.shell_completion import BashComplete, FishComplete, ZshComplete

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
    # Imported lazily to avoid a circular import at module load (the facade
    # imports this module to register `completion`) and to keep this module
    # import-light for lazy subcommand loading.
    from pyfm.cli import cli

    complete_cls = _SHELL_COMPLETIONS[shell]
    completer = complete_cls(
        cli=cli,
        ctx_args={},
        prog_name=prog_name,
        complete_var=_complete_var(prog_name),
    )
    click.echo(completer.source())
