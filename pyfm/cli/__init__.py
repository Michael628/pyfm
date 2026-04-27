import click

from pyfm.cli.nanny import nanny
from pyfm.cli.task import task
from pyfm.cli.contract import contract
from pyfm.cli.systems import build, workspace


@click.group()
def cli():
    pass


cli.add_command(nanny)
cli.add_command(task)
cli.add_command(contract)
cli.add_command(build)
cli.add_command(workspace)
