import json

import click

from pyfm import utils
from pyfm.performance import analyze_file, benchmark_lmi_performance


@click.group()
def performance():
    """Analyze performance output logs."""
    pass


@performance.command(name="analyze")
@click.argument(
    "output_file", type=click.Path(exists=True, dir_okay=False, readable=True)
)
def analyze(output_file):
    """Print a performance summary for a Hadrons OUTPUT_FILE."""
    try:
        analyze_file(output_file)
    except ValueError as e:
        raise click.ClickException(str(e)) from e


@performance.command(name="benchmark")
@click.option("-j", "--job", type=str, required=True, help="Job step name.")
@click.option(
    "--log",
    "log_file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
    help="Hadrons/Grid performance log file.",
)
@click.option(
    "-p",
    "--param-file",
    type=str,
    default="params.yaml",
    help="Path to YAML parameter file.",
)
def benchmark(job, log_file, param_file):
    """Emit component-first JSON benchmark data for a configured Hadrons/Grid LMI LOG."""
    try:
        params = utils.io.load_param(param_file)
        result = benchmark_lmi_performance(job, log_file, params)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    click.echo(json.dumps(result, sort_keys=True))
