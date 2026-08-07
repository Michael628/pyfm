import os

import click

from pyfm import utils
from pyfm.nanny import (
    add_entries,
    audit_outfiles,
    check_jobs,
    create_task,
    get_job_config,
    get_nanny_config,
    nanny_loop,
    parse_cfgs,
    submit_job,
    validate_steps,
)


@click.group()
def nanny():
    """Manage automated HPC job submission via todo files."""
    pass


@nanny.command()
@click.option("-p", "--param-file", type=click.Path(dir_okay=False), default="params.yaml", help="Path to YAML parameter file.")
@click.option("-j", "--job", type=str, default=None, help="Restrict nanny loop to this job step only.")
@click.option("--logging-level", type=str, default="INFO", help="Logging verbosity (DEBUG, INFO, WARNING, ERROR).")
def run(param_file, job, logging_level):
    """Run the nanny loop to submit and monitor HPC jobs."""
    os.system("umask 022")
    utils.set_logging_level(logging_level)
    nanny_loop(param_file, require_step=job)


@nanny.command()
@click.option("-p", "--param-file", type=click.Path(dir_okay=False), default="params.yaml", help="Path to YAML parameter file.")
@click.option("-i", "--input", "input_file", type=click.Path(dir_okay=False), required=True, help="Input file list to submit.")
@click.option("-j", "--job", type=str, required=True, help="Job step name to submit.")
@click.option("--logging-level", type=str, default="INFO", help="Logging verbosity (DEBUG, INFO, WARNING, ERROR).")
def submit(param_file, input_file, job, logging_level):
    """Submit a single job to the HPC scheduler."""
    utils.set_logging_level(logging_level)
    yaml_params = utils.io.load_param(param_file)
    nanny_config = get_nanny_config(yaml_params)
    job_config = get_job_config(job, yaml_params)
    os.environ["INPUTLIST"] = input_file
    submit_job(nanny_config, job_config, 1)


@nanny.command()
@click.argument("steps", nargs=-1, required=True)
@click.option("-s", "--series", type=str, required=True, help="Gauge field series to add.")
@click.option("-n","--config", "cfg_list", multiple=True, type=int, help="Individual configuration numbers to add.")
@click.option("--config-range", "cfg_range", nargs=3, type=int, default=None, help="Config range as START STOP STEP (exclusive stop).")
@click.option("-p", "--param-file", type=click.Path(dir_okay=False), default="params.yaml", help="Path to YAML parameter file.")
def add(steps, series, cfg_list, cfg_range, param_file):
    """Add todo entries for SERIES and STEPS.

    SERIES is the gauge field series label.
    STEPS are one or more job step names to add.
    Exactly one of --config or --config-range must be provided.
    """
    if cfg_list and cfg_range:
        click.echo("Error: --config and --config-range are mutually exclusive.", err=True)
        raise click.Abort()
    if not cfg_list and not cfg_range:
        click.echo("Error: one of --config or --config-range is required.", err=True)
        raise click.Abort()

    yaml_params = utils.io.load_param(param_file)
    todo_file = yaml_params["nanny"]["todo_file"]
    job_setup = yaml_params["job_setup"]

    try:
        validate_steps(steps, job_setup)
    except ValueError as e:
        click.echo(str(e), err=True)
        raise click.Abort()

    try:
        cfgnos = parse_cfgs(cfg=cfg_list if cfg_list else None, cfg_range=cfg_range)
    except ValueError as e:
        click.echo(str(e), err=True)
        raise click.Abort()

    add_entries(todo_file, series, cfgnos, steps)


@nanny.command()
@click.option("-p", "--param-file", type=click.Path(dir_okay=False), default="params.yaml", help="Path to YAML parameter file.")
@click.option("-j", "--job", type=str, default=None, help="Job step to inspect.")
@click.option("-s", "--series", type=str, default=None, help="Gauge field series to filter on.")
@click.option("-n", "--config", "config", type=str, default=None, help="Configuration number to filter on.")
@click.option("-v", "--verbose", is_flag=True, default=False, help="Print detailed output file status.")
def check(param_file, job, series, config, verbose):
    """Check job status or audit output files.

    Without --job/--series/--config, prints a summary of all job statuses.
    With all three flags, audits output files for the specified job/series/config.
    """
    yaml_params = utils.io.load_param(param_file)
    if job is not None and series is not None and config is not None:
        task = create_task(job, yaml_params, series, config)
        audit_outfiles(task, verbose=verbose)
    else:
        check_jobs(yaml_params)
