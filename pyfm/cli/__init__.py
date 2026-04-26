import click

from pyfm.cli.nanny import nanny


@click.group()
def cli():
    pass


cli.add_command(nanny)
