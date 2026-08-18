import click

from pyfm.cli._options import (
    format_option,
    job_option,
    logging_level_option,
    param_file_option,
)


@click.group()
def task():
    """Generate input files and aggregate task output (deprecated aliases)."""
    pass


@task.command(
    "generate",
    deprecated="Use 'pyfm nanny generate' instead.",
)
@param_file_option()
@job_option(required=True, help="Job step name.")
@click.option("-s", "--series", type=str, required=True, help="Gauge field series label.")
@click.option("-n", "--config", "cfg", type=str, required=True, help="Configuration number.")
@click.pass_context
def generate(ctx, param_file, job, series, cfg):
    """Generate input files for a specific job/series/config (deprecated alias)."""
    from pyfm.cli.nanny import nanny

    ctx.forward(nanny.commands["generate"])


@task.command(
    "aggregate",
    deprecated="Use 'pyfm export corr' instead.",
)
@param_file_option()
@job_option(required=True, help="Job step name to aggregate.")
@format_option()
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
@logging_level_option()
@click.pass_context
def aggregate(ctx, param_file, job, fmt, average, skip_existing, generate_manifest, max_workers, logging_level):
    """Aggregate output data across configurations into a single file (deprecated alias)."""
    from pyfm.cli.export import export

    ctx.forward(export.commands["corr"])
