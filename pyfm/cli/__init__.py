import click

from pyfm.cli.nanny import nanny
from pyfm.cli.task import task
from pyfm.cli.contract import contract


@click.group()
def cli():
    pass


cli.add_command(nanny)
cli.add_command(task)
cli.add_command(contract)
