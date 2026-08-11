"""The ``pyfm export`` command group.

``export`` is the home for format-based data export. Its first member,
``export corr``, aggregates the correlator (``corr``) output produced by a job
step across configurations into a single file — the operation previously
exposed as ``pyfm task aggregate``. Future siblings (e.g. ``export tar``) are
documented but not yet implemented.

The module is import-light (only ``click`` + the shared option decorators):
``utils`` and ``aggregator`` are imported inside the ``corr`` callback so that
``pyfm export --help`` does not pull in pandas/h5py (see Phase 4 lazy loading).
"""
import click

from pyfm.cli._options import (
    format_option,
    job_option,
    logging_level_option,
    param_file_option,
)


@click.group()
def export():
    """Export aggregated task output data."""
    pass


@export.command()
@param_file_option()
@job_option(required=True, help="Job step name to aggregate.")
@format_option()
@click.option("--average", is_flag=True, default=False, help="Average over configurations after aggregation.")
@click.option("--skip-existing", is_flag=True, default=False, help="Skip configs whose output already exists.")
@logging_level_option()
def corr(param_file, job, fmt, average, skip_existing, logging_level):
    """Aggregate output data across configurations into a single file."""
    from pyfm import utils
    from pyfm.nanny import aggregator

    params = utils.io.load_param(param_file)
    utils.set_logging_level(logging_level)
    aggregator.aggregate_task_data(
        job, params, format=fmt, average=average, skip_existing=skip_existing
    )
