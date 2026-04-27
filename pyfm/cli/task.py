import click

from pyfm import utils
from pyfm.nanny import write_input_file
from pyfm.nanny import aggregator


@click.group()
def task():
    pass


@task.command()
@click.option("-p", "--param-file", type=str, default="params.yaml")
@click.option("-j", "--job", type=str, required=True)
@click.option("-s", "--series", type=str, required=True)
@click.option("-n", "--config", "cfg", type=str, required=True)
def generate(param_file, job, series, cfg):
    params = utils.io.load_param(param_file)
    ifile = write_input_file(job, params, series, cfg)
    utils.get_logger().info(f"Input parameters written to {ifile}")


@task.command()
@click.option("-p", "--param-file", type=str, default="params.yaml")
@click.option("-j", "--job", type=str, required=True)
@click.option("-f", "--format", "fmt", type=str, default="csv")
@click.option("--average", is_flag=True, default=False)
@click.option("--skip-existing", is_flag=True, default=False)
@click.option("--logging-level", type=str, default="INFO")
def aggregate(param_file, job, fmt, average, skip_existing, logging_level):
    params = utils.io.load_param(param_file)
    utils.set_logging_level(logging_level)
    aggregator.aggregate_task_data(job, params, format=fmt, average=average, skip_existing=skip_existing)
