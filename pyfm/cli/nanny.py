import os

import click

from pyfm import utils
from pyfm.nanny import audit_outfiles, check_jobs, get_job_config, get_nanny_config, nanny_loop, submit_job


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
