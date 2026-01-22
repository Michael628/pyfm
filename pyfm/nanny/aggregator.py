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


def load_data(
    agg_params: t.Dict[str, t.Any], skip_existing: bool = False, format: str = "csv"
) -> t.Dict[str, pd.DataFrame]:
    result = {}
    logger = utils.get_logger()
    for key in agg_params["run"]:
        run_params = agg_params[key]
        out_files = run_params.get("out_files", {})
        load_files = run_params["load_files"]

        skip_file_set = set()
        old_data = pd.DataFrame()

        if skip_existing:

            ext = utils.io.get_file_ext_from_format(format)
            out_filestem = out_files["filestem"] + ext
            replacements = load_files.get("replacements", {}) | {"format": format}
            old_data = dio.load_files(
                filestem=out_filestem,
                wildcard_fill=True,
                replacements=replacements,
                regex=load_files.get("regex", {}),
            ).agg()

            logger.debug("Loaded existing agg file")
            # split up series.cfg to use for formatting file path

            old_data_entries = lambda: (
                (
                    old_data.assign(
                        series=lambda x: x.index.get_level_values(
                            "series_cfg"
                        ).str.split(".", expand=True),
                    )
                    .assign(cfg=lambda x: x.series.transform(lambda y: y[1]))
                    .assign(series=lambda x: x.series.transform(lambda y: y[0]))
                    .reset_index()
                    .drop(["series_cfg", "corr"], axis=1)
                    .to_dict(orient="records")
                )
                if not old_data.empty
                else old_data
            )
            logger.debug("built existing entries")
            skip_file_set = set(
                map(load_files["filestem"].format_map, old_data_entries())
            )
            logger.debug("built file exclude list")

        data = (
            dio.load_files(skip_file_set=skip_file_set, **load_files)
            .agg()
            .assign(format=format)
        )

        if data.empty:
            logger.debug("No new data found")
            result[key] = old_data
        elif old_data.empty:
            if skip_existing:
                logger.debug("No data loaded from existing agg file")
            result[key] = data
        else:
            result[key] = pd.concat([old_data, data])

        # Drop format column if it is not in the file stem
        # Avoids unnecessary columns in the output
        if "format" not in utils.io.format_keys(out_files["filestem"]):
            result[key].drop("format", axis=1, inplace=True)

    return result


def process_data(
    result: t.Dict[str, pd.DataFrame],
    agg_params: t.Dict[str, t.Any],
    format: str = "csv",
) -> None:

    for key, df in result.items():
        run_params = agg_params[key]
        actions = run_params.get("actions", {})

        if index := run_params.get("out_files", {}).get("index", None):
            actions["index"] = index

        if result[key].empty:
            utils.get_logger().warning(f"Empty DataFrame for {key}")
            continue

        result[key] = pc.execute(result[key], actions)

        if out_files := run_params.get("out_files", {}):
            dio.write_files(result[key], format=format, **out_files)


def aggregate_task_data(
    job_step: str,
    yaml_data: t.Dict,
    format: str = "csv",
    average: bool = False,
    skip_existing: bool = False,
) -> None:

    task: AggregatorProtocol = create_task(job_step, yaml_data)
    agg_params = task.build_aggregator_params(average)
    if not agg_params:
        raise ValueError(f"No aggregator parameters provided for task: {job_step}.")

    if average:
        agg_params_raw = task.build_aggregator_params(False)
    else:
        agg_params_raw = agg_params

    result = load_data(agg_params_raw, skip_existing, format)
    process_data(result, agg_params, format)
