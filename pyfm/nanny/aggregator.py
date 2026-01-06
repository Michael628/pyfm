import typing as t
from pyfm.nanny.setup import create_task
import pyfm.dataio as dio
from pyfm.dataio import processor as pc
from pyfm import utils
import pandas as pd


@t.runtime_checkable
class AggregatorProtocol(t.Protocol):
    def build_aggregator_params(self, average: bool) -> t.Any: ...
    def format_string(self, to_format: str) -> str: ...


def aggregate_data(agg_params: t.Dict[str, t.Any], format: str = "csv") -> None:
    result = {}
    for key in agg_params["run"]:
        run_params = agg_params[key]

        data = dio.load_files(**run_params["load_files"]).agg()
        result[key] = data
        actions = run_params.get("actions", {})
        out_files = run_params.get("out_files", {})
        index = out_files.pop("index", None)

        if index:
            actions.update({"index": index})

        if "actions" in run_params:
            result[key] = pc.execute(result[key], run_params["actions"])

        keys = utils.io.format_keys(out_files["filestem"])

        if "format" in keys:
            # Add format to dataframe column so that it is picked up by dio.write for file string processing
            result[key].format = format

        if out_files:
            dio.write_files(result[key], format=format, **out_files)


def aggregate_task_data(
    job_step: str, yaml_data: t.Dict, format: str = "csv", average: bool = False
) -> None:
    task: AggregatorProtocol = create_task(job_step, yaml_data)

    agg_params = task.build_aggregator_params(average)

    if not agg_params:
        raise ValueError(f"No aggregator parameters provided for task: {job_step}.")

    aggregate_data(agg_params, format)
