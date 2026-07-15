import json
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


def _manifest_path(out_filestem: str) -> str:
    """Manifest sidecar path: ``<out_filestem>.manifest.json``.

    The manifest lives alongside the output files, keyed on the output
    filestem (before the format extension is applied).
    """
    return os.path.splitext(out_filestem)[0] + ".manifest.json"


def _read_skip_manifest(
    out_filestem: str, logger=None
) -> t.Tuple[t.Set[str], bool]:
    """Read the manifest sidecar of pre-formatted input filenames.

    Returns ``(skip_set, manifest_exists)``. A missing manifest returns
    ``(set(), False)`` so the caller can warn and aggregate all inputs (Q4 -> C).
    """
    path = _manifest_path(out_filestem)
    if not os.path.exists(path):
        return set(), False
    with open(path) as f:
        return set(json.load(f)), True


def _write_skip_manifest(out_filestem: str, input_filenames: t.List[str]) -> str:
    """Write the manifest sidecar as a sorted list[str] of pre-formatted input filenames.

    Format-agnostic, O(1) read, git-diffable. The manifest stores the exact
    pre-formatted input filenames the skip-set keys on, bypassing the reshape
    chain entirely (and immune to the latent ``series_cfg`` split bug).
    """
    path = _manifest_path(out_filestem)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(sorted(input_filenames), f)
    return path


def _formatted_input_filenames(
    filestem: str, file_repls: t.List[t.Tuple[str, t.Dict]]
) -> t.List[str]:
    """Format the input filestem with each resolved replacement dict.

    Reproduces the input-filename formatting the skip-set keys on: for each
    ``(filename, repl)`` pair from `process_files`, ``filestem.format_map(repl)``.
    """
    return [filestem.format_map(dict(r)) for _, r in file_repls]


def _enumerate_input_files(
    load_files_cfg: t.Dict[str, t.Any],
) -> t.List[t.Tuple[str, t.Dict]]:
    """Enumerate input ``(filename, repl)`` pairs for manifest writing.

    Thin wrapper over ``utils.io.process_files`` mirroring the enumeration that
    ``load_files`` / ``load_files_chunked`` perform internally, so the manifest
    records exactly the inputs that would be loaded.
    """
    return utils.io.process_files(
        load_files_cfg["filestem"],
        lambda f, r: (f, r),
        load_files_cfg.get("replacements"),
        load_files_cfg.get("regex"),
        load_files_cfg.get("wildcard_fill", False),
    )


def load_data(
    agg_params: t.Dict[str, t.Any],
    skip_existing: bool = False,
    format: str = "csv",
    max_workers: int = 1,
) -> t.Dict[str, pd.DataFrame]:
    result = {}
    logger = utils.get_logger()

    # Set HDF5 file locking off once per process before raising worker count
    # (D3). Only set when actually concurrent; default max_workers=1 leaves the
    # environment untouched (D6 conservative opt-in).
    if max_workers > 1:
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

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
            old_data = dio.load_files_chunked(
                filestem=out_filestem,
                wildcard_fill=True,
                replacements=replacements,
                regex=load_files.get("regex", {}),
                max_workers=max_workers,
            )

            logger.debug("Loaded existing agg file")

            # Read pre-formatted input filenames from the manifest sidecar (D5).
            # Replaces the reshape chain (str.split -> transform -> reset_index
            # -> drop -> to_dict -> format_map) entirely.
            skip_file_set, manifest_exists = _read_skip_manifest(
                out_files["filestem"], logger
            )
            if not manifest_exists:
                logger.warning(
                    f"No manifest found for {out_files['filestem']!r}; "
                    "aggregating all inputs (manifest will be written after "
                    "this run by process_data)."
                )

        data = (
            dio.load_files_chunked(
                skip_file_set=skip_file_set, max_workers=max_workers, **load_files
            )
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

            # Write manifest sidecar (D5): pre-formatted input filenames so the
            # next --skip-existing run reads them as a pure set-membership lookup.
            input_filenames = _formatted_input_filenames(
                run_params["load_files"]["filestem"],
                _enumerate_input_files(run_params["load_files"]),
            )
            _write_skip_manifest(out_files["filestem"], input_filenames)


def aggregate_task_data(
    job_step: str,
    yaml_data: t.Dict,
    format: str = "csv",
    average: bool = False,
    skip_existing: bool = False,
    max_workers: int = 1,
) -> None:

    task = create_task(job_step, yaml_data)
    agg_params = task.handler.build_aggregator_params(task.config, average)
    if not agg_params:
        raise ValueError(f"No aggregator parameters provided for task: {job_step}.")

    if average:
        agg_params_raw = task.handler.build_aggregator_params(task.config, False)
    else:
        agg_params_raw = agg_params

    result = load_data(agg_params_raw, skip_existing, format, max_workers=max_workers)
    process_data(result, agg_params, format)
