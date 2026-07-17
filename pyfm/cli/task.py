import click

from pyfm import utils
from pyfm.nanny import write_input_file
from pyfm.nanny import aggregator


@click.group()
def task():
    """Generate input files and aggregate task output data."""
    pass


@task.command()
@click.option("-p", "--param-file", type=click.Path(dir_okay=False), default="params.yaml", help="Path to YAML parameter file.")
@click.option("-j", "--job", type=str, required=True, help="Job step name.")
@click.option("-s", "--series", type=str, required=True, help="Gauge field series label.")
@click.option("-n", "--config", "cfg", type=str, required=True, help="Configuration number.")
def generate(param_file, job, series, cfg):
    """Generate input files for a specific job/series/config."""
    params = utils.io.load_param(param_file)
    ifile = write_input_file(job, params, series, cfg)
    utils.get_logger().info(f"Input parameters written to {ifile}")


@task.command()
@click.option("-p", "--param-file", type=click.Path(dir_okay=False), default="params.yaml", help="Path to YAML parameter file.")
@click.option("-j", "--job", type=str, required=True, help="Job step name to aggregate.")
@click.option("-f", "--format", "fmt", type=str, default="csv", help="Output file format (csv, hdf5).")
@click.option("--average", is_flag=True, default=False, help="Average over configurations after aggregation.")
@click.option("--skip-existing", is_flag=True, default=False, help="Skip configs whose output already exists.")
@click.option("--generate-manifest", is_flag=True, default=False, help="Generate manifest sidecars from existing processed agg files instead of aggregating.")
@click.option(
    "--max-workers",
    type=int,
    default=1,
    help=(
        "Number of worker threads for loading HDF5 files "
        "(default: 1, today's behavior)."
    ),
)
@click.option("--logging-level", type=str, default="INFO", help="Logging verbosity (DEBUG, INFO, WARNING, ERROR).")
def aggregate(param_file, job, fmt, average, skip_existing, generate_manifest, max_workers, logging_level):
    """Aggregate output data across configurations into a single file."""
    if generate_manifest and skip_existing:
        raise click.UsageError(
            "--generate-manifest and --skip-existing are mutually exclusive."
        )
    params = utils.io.load_param(param_file)
    utils.set_logging_level(logging_level)
    aggregator.aggregate_task_data(
        job, params,
        format=fmt,
        average=average,
        skip_existing=skip_existing,
        generate_manifest=generate_manifest,
        max_workers=max_workers,
    )
