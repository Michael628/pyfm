import os
import typing as t
from pyfm.nanny.taskbuilder import create_task
import pyfm.dataio as dio
from pyfm.dataio import processor as pc
from pyfm import utils
import pandas as pd


@t.runtime_checkable
class AggregatorProtocol(t.Protocol):
    def build_aggregator_params(self, average: bool) -> t.Any: ...


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

    task = create_task(job_step, yaml_data)
    agg_params = task.handler.build_aggregator_params(task.config, average)
    if not agg_params:
        raise ValueError(f"No aggregator parameters provided for task: {job_step}.")

    if average:
        agg_params_raw = task.handler.build_aggregator_params(task.config, False)
    else:
        agg_params_raw = agg_params

    result = load_data(agg_params_raw, skip_existing, format)
    process_data(result, agg_params, format)


def set_format_col(df: pd.DataFrame, out_stem: str, out_format: str) -> pd.DataFrame:
    if "format" in utils.io.format_keys(out_stem):
        return df.assign(format=out_format)
    if "format" in df.columns:
        return df.drop(columns="format")
    return df


def load_convert_data(
    run_params: t.Dict[str, t.Any], input_format: str, data_col: str = "corr"
) -> pd.DataFrame:
    """Load a run key's existing aggregated output files in ``input_format``.

    Loads against the templated ``out_files`` filestem (not resolved paths) with
    ``wildcard_fill`` so the loader recovers any ``{placeholder}`` values embedded in
    the filename. Replacement/regex hints are filtered to keys that occur in the
    filestem, and ``{format}`` is pinned to the input format. When ``input_format``
    is ``dict``, the handler-derived load metadata (``dict_labels`` /
    ``array_order`` / ``array_labels``) is forwarded so the non-self-describing
    ``.npy`` files can be reconstructed.

    Columns that are neither the data column, the ``format`` metadata, nor
    outfile_stem placeholders (the writer's groupby keys) are moved into the index,
    faithfully reconstructing the written table's index — needed for faithful
    csv/hdf5/parquet bodies and for ``dict`` output's ``frame_to_dict`` reshape.
    """
    out_filestem_base = run_params["out_files"]["filestem"]
    out_filestem = out_filestem_base + utils.io.get_file_ext_from_format(input_format)
    stem_keys = utils.io.format_keys(out_filestem_base)

    load_files = run_params.get("load_files", {})
    replacements = {
        k: v for k, v in load_files.get("replacements", {}).items() if k in stem_keys
    }
    regex = {k: v for k, v in load_files.get("regex", {}).items() if k in stem_keys}
    if "format" in stem_keys:
        replacements["format"] = input_format

    loader_kwargs: t.Dict[str, t.Any] = {}
    if input_format == "dict":
        for meta_key in ("dict_labels", "array_order", "array_labels"):
            if meta_key in load_files:
                loader_kwargs[meta_key] = load_files[meta_key]

    df = (
        dio.load_files(
            filestem=out_filestem,
            wildcard_fill=True,
            replacements=replacements,
            regex=regex,
            **loader_kwargs,
        )
        .agg()
    )

    non_index_cols = {data_col, "format"} | set(stem_keys)
    index_cols = [c for c in df.columns if c not in non_index_cols]
    if index_cols:
        df = df.set_index(index_cols)
    return df


def resolve_convert_output_stem(
    out_filestem: str, output: str | None, num_keys: int
) -> str:
    if output is None:
        return out_filestem
    base = os.path.splitext(output)[0]
    if num_keys == 1:
        return base
    return os.path.join(base, out_filestem)


def convert_task_data(
    job_step: str,
    yaml_data: t.Dict,
    input_format: str = "csv",
    output_format: str = "csv",
    output: str | None = None,
) -> None:
    """Convert a prior run's aggregated output to a different file format.

    Re-derives the aggregated output filestems via ``create_task`` ->
    ``build_aggregator_params(average=False)`` (paths are not persisted by
    ``export corr``), loads each run key's existing aggregated files in
    ``input_format``, and re-emits them in ``output_format`` by template re-emit:
    ``set_format_col`` sets the ``format`` column to the output format (driving the
    writer's ``{format}`` groupby to rewrite the directory) and ``write_files`` swaps
    the extension. ``--output`` overrides the write path (exact stem for a single run
    key; base directory for multi-run-key steps).
    """
    task = create_task(job_step, yaml_data)
    agg_params = task.handler.build_aggregator_params(task.config, average=False)
    if not agg_params:
        raise ValueError(f"No aggregator parameters provided for task: {job_step}.")

    logger = utils.get_logger()
    run_keys = agg_params["run"]
    num_keys = len(run_keys)

    for key in run_keys:
        run_params = agg_params[key]
        df = load_convert_data(run_params, input_format)

        if df.empty:
            logger.warning(f"Empty DataFrame for {key}")
            continue

        out_filestem = run_params["out_files"]["filestem"]
        stem = resolve_convert_output_stem(out_filestem, output, num_keys)

        dio.write_files(
            set_format_col(df, stem, output_format),
            filestem=stem,
            format=output_format,
        )
