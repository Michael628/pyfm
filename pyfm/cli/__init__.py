import click

from pyfm.cli.contract import contract
from pyfm.cli.nanny import nanny
from pyfm.cli.performance import performance
from pyfm.cli.systems import build, workspace
from pyfm.cli.task import task


@click.group()
def cli():
    """PyFM - lattice QCD workflow toolkit."""
    pass


cli.add_command(nanny)
cli.add_command(task)
cli.add_command(contract)
cli.add_command(performance)
cli.add_command(build)
cli.add_command(workspace)
