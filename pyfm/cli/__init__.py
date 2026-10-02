import click

from pyfm.cli._lazy import LazyGroup
from pyfm.cli.completion import completion
from pyfm.version import HADRONS_MILC_COMPAT, __version__


@click.group(cls=LazyGroup, lazy_subcommands={
    "nanny": "pyfm.cli.nanny:nanny",
    "export": "pyfm.cli.export:export",
    "task": "pyfm.cli.task:task",
    "contract": "pyfm.cli.contract:contract",
    "audit": "pyfm.cli.audit:audit",
    "build": "pyfm.cli.systems:build",
    "workspace": "pyfm.cli.systems:workspace",
})
@click.version_option(
    version=__version__,
    prog_name="pyfm",
    message=f"%(prog)s, version %(version)s (HadronsMILC {HADRONS_MILC_COMPAT})",
)
def cli():
    """PyFM - lattice QCD workflow toolkit."""
    pass


cli.add_command(completion)
