import click

from pyfm.cli._lazy import LazyGroup
from pyfm.cli.completion import completion


@click.group(cls=LazyGroup, lazy_subcommands={
    "nanny": "pyfm.cli.nanny:nanny",
    "export": "pyfm.cli.export:export",
    "task": "pyfm.cli.task:task",
    "contract": "pyfm.cli.contract:contract",
    "audit": "pyfm.cli.audit:audit",
    "build": "pyfm.cli.systems:build",
    "workspace": "pyfm.cli.systems:workspace",
})
def cli():
    """PyFM - lattice QCD workflow toolkit."""
    pass


cli.add_command(completion)
