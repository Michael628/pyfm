import os

import click

from pyfm import utils
from pyfm.nanny import (
    add_entries,
    audit_outfiles,
    check_jobs,
    get_job_config,
    get_nanny_config,
    nanny_loop,
    parse_cfgs,
    submit_job,
    validate_steps,
)


@click.group()
def nanny():
    pass


@nanny.command()
@click.option("-p", "--param-file", type=str, default="params.yaml")
@click.option("-j", "--job", type=str, default=None)
@click.option("--logging-level", type=str, default="INFO")
def run(param_file, job, logging_level):
    os.system("umask 022")
    utils.set_logging_level(logging_level)
    nanny_loop(param_file, require_step=job)


@nanny.command()
@click.option("-p", "--param-file", type=str, default="params.yaml")
@click.option("-i", "--input", "input_file", type=str, required=True)
@click.option("-j", "--job", type=str, required=True)
@click.option("--logging-level", type=str, default="INFO")
def submit(param_file, input_file, job, logging_level):
    utils.set_logging_level(logging_level)
    yaml_params = utils.io.load_param(param_file)
    nanny_config = get_nanny_config(yaml_params)
    job_config = get_job_config(job, yaml_params)
    os.environ["INPUTLIST"] = input_file
    submit_job(nanny_config, job_config, 1)


@nanny.command()
@click.argument("series")
@click.argument("steps", nargs=-1, required=True)
@click.option("--cfg", "cfg_list", multiple=True, type=int)
@click.option("--cfg-range", "cfg_range", nargs=3, type=int, default=None)
@click.option("-p", "--param-file", type=str, default="params.yaml")
def add(series, steps, cfg_list, cfg_range, param_file):
    if cfg_list and cfg_range:
        click.echo("Error: --cfg and --cfg-range are mutually exclusive.", err=True)
        raise click.Abort()
    if not cfg_list and not cfg_range:
        click.echo("Error: one of --cfg or --cfg-range is required.", err=True)
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
@click.option("-p", "--param-file", type=str, default="params.yaml")
@click.option("-j", "--job", type=str, default=None)
@click.option("-s", "--series", type=str, default=None)
@click.option("-n", "--config", "config", type=str, default=None)
@click.option("-v", "--verbose", is_flag=True, default=False)
def check(param_file, job, series, config, verbose):
    yaml_params = utils.io.load_param(param_file)
    if job is not None and series is not None and config is not None:
        audit_outfiles(job, yaml_params, series, config, verbose=verbose)
    else:
        check_jobs(yaml_params)
