import glob
import json
import os
import re
import tarfile
import typing as t
from datetime import datetime

from pyfm.nanny.config import get_nanny_config
from pyfm.nanny.taskbuilder import create_task
import pyfm.dataio as dio
from pyfm.dataio import processor as pc
from pyfm import utils
from pyfm.utils.string import PartialFormatter, format_keys
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
    out_filestem: str, format: str | None = None, logger=None
) -> t.Tuple[t.Set[str], bool]:
    """Read the manifest sidecar(s) of pre-formatted input filenames.

    Returns ``(skip_set, manifest_exists)``. A missing manifest returns
    ``(set(), False)`` so the caller can warn and aggregate all inputs (Q4 -> C).

    When the output filestem carries template keys (e.g. ``{mass}``,
    ``{gamma_label}``, ``{format}``), the write system emits one manifest per
    output file alongside each file. This reader globs all of those manifests —
    resolving ``{format}`` to the current ``format`` (so other-format outputs do
    not pollute the skip set) and every remaining ``{key}`` to ``*`` — and unions
    their input-filename sets. A brace-free filestem reads a single manifest
    (backward-compatible).
    """
    path_template = _manifest_path(out_filestem)
    if "{" not in path_template:
        if not os.path.exists(path_template):
            return set(), False
        with open(path_template) as f:
            return set(json.load(f)), True

    glob_pat = _manifest_glob_pattern(path_template, format)
    manifests = sorted(glob.glob(glob_pat))
    skip_set: t.Set[str] = set()
    for mpath in manifests:
        with open(mpath) as f:
            skip_set |= set(json.load(f))
    return skip_set, bool(manifests)


def _manifest_glob_pattern(path_template: str, format: str | None) -> str:
    """Build a glob pattern for per-output-file manifests from a path template.

    ``{format}`` resolves to the current output ``format`` (keeps the skip set
    format-scoped); every other ``{key}`` becomes ``*`` (``*`` does not cross a
    path separator, so each matches exactly one path segment).
    """
    pat = path_template
    if format is not None:
        pat = pat.replace("{format}", str(format))
    return re.sub(r"\{[^}]+\}", "*", pat)


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


def _write_manifests(
    load_files_cfg: t.Dict[str, t.Any], out_filestem: str, format: str
) -> None:
    """Write per-output-file manifest sidecars alongside each output file.

    Mirrors the write system's per-group keyword replacement: each input file maps
    deterministically to exactly one output group via the output filestem's
    template keys (output keys are a subset of the input replacement keys plus
    ``format``). Inputs are partitioned by their resolved output stem, and each
    group's manifest lists only the inputs that contributed to that output file —
    so a re-run with ``--skip-existing`` skips inputs per output file, and the
    manifests land at resolved (brace-free) paths next to the data.
    """
    out_keys = format_keys(out_filestem)
    input_pairs = _enumerate_input_files(load_files_cfg)

    if not out_keys:
        # Non-templated output: a single output file -> a single manifest.
        _write_skip_manifest(
            out_filestem,
            _formatted_input_filenames(load_files_cfg["filestem"], input_pairs),
        )
        return

    # Partition inputs by their resolved output stem (per-output-file manifests).
    groups: t.Dict[str, t.List[str]] = {}
    for filename, repl in input_pairs:
        repl_map = dict(repl)
        if "format" in out_keys:
            repl_map["format"] = format
        resolved = out_filestem.format_map(PartialFormatter(repl_map))
        if "{" in resolved:
            # An output key is not resolvable from the input replacements, so the
            # input cannot be attributed to an output file. Skip it (with a
            # warning) rather than writing a braced — and therefore invalid — path.
            utils.get_logger().warning(
                f"Could not resolve output filestem {out_filestem!r} for input "
                f"{filename!r}; skipping its manifest entry."
            )
            continue
        groups.setdefault(resolved, []).append(filename)

    for resolved_stem, filenames in groups.items():
        _write_skip_manifest(resolved_stem, filenames)


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


def _split_series_cfg(series_cfg: str) -> t.Tuple[str, str]:
    """Split a ``series_cfg`` value (``"{series}.{cfg}"``) into ``(series, cfg)``.

    The ``index`` action (``processor.index``) builds ``series_cfg`` as
    ``series + "." + cfg`` with ``series`` a single ``[a-z]`` letter and ``cfg``
    ``[0-9]+``, so splitting on the first ``"."`` recovers both unambiguously.
    """
    series, cfg = series_cfg.split(".", 1)
    return series, cfg


def _recover_series_cfg_values(df: pd.DataFrame) -> t.List[str]:
    """Unique ``series_cfg`` values from an agg DataFrame, wherever they live.

    CSV round-trips the ``series_cfg`` index as a column (``read_csv``) while
    HDF5 restores it as an index level (``read_hdf``); look in both. Raise if
    absent — the agg was not built with ``series_cfg`` and cannot be mapped back
    to input filenames.
    """
    name = "series_cfg"
    if name in df.index.names:
        return list(pd.Index(df.index.get_level_values(name)).unique())
    if name in df.columns:
        return list(df[name].unique())
    raise ValueError(
        "Agg DataFrame has no 'series_cfg' index level or column; cannot recover "
        "(series, cfg) for manifest generation."
    )


def _present_tsources(df: pd.DataFrame) -> t.Dict[str, t.Set[str]]:
    """Map each ``series_cfg`` to its set of present ``tsource`` values.

    Returns ``{}`` when ``tsource`` is absent from both index and columns
    (averaged aggs), signalling the caller to skip the completeness gate and
    treat every present config as complete.
    """
    if "tsource" not in (set(df.index.names) | set(df.columns)):
        return {}
    work = df.reset_index()
    return {
        sc: set(map(str, vals))
        for sc, vals in work.groupby("series_cfg")["tsource"]
    }


def _reconstruct_manifest_inputs(
    df: pd.DataFrame,
    load_files_cfg: t.Dict[str, t.Any],
    path_repl: t.Dict[str, str],
) -> t.List[str]:
    """Reconstruct the input-filename list a manifest should hold for one agg file.

    Recovers ``(series, cfg)`` from ``series_cfg``; applies the tsource
    completeness gate when ``tsource`` is visible in the agg (include a config
    only when its present tsources equal the configured list); otherwise
    (averaged, or a task with no tsource dimension) includes every present
    config. Formats ``load_files_cfg["filestem"]`` once per included
    ``(series, cfg)`` × configured tsource, merging path-constant keys from
    ``path_repl`` (recovered from the globbed agg path) with the run's
    replacements. Filenames that still contain ``{`` (unresolvable key) are
    skipped with a warning, mirroring ``_write_manifests``.
    """
    logger = utils.get_logger()
    filestem = load_files_cfg["filestem"]
    replacements = load_files_cfg.get("replacements", {})

    raw_tsources = replacements.get("tsource")
    if raw_tsources is None:
        has_tsource_dim = False
        configured_tsources: t.List = []
    elif isinstance(raw_tsources, (list, tuple)):
        has_tsource_dim = True
        configured_tsources = list(raw_tsources)
    else:
        has_tsource_dim = True
        configured_tsources = [raw_tsources]
    configured_set = set(map(str, configured_tsources))

    present_tsources = _present_tsources(df)
    gate_active = has_tsource_dim and bool(present_tsources)

    series_cfgs = _recover_series_cfg_values(df)
    inputs: t.List[str] = []

    for sc in series_cfgs:
        if gate_active and present_tsources.get(sc, set()) != configured_set:
            logger.debug(
                f"Config {sc!r} is incomplete (tsource mismatch); "
                "omitting from generated manifest."
            )
            continue
        series, cfg = _split_series_cfg(sc)
        tsources = configured_tsources if has_tsource_dim else [None]
        for ts in tsources:
            repl = dict(path_repl)
            repl.update(
                {k: v for k, v in replacements.items() if k != "tsource"}
            )
            repl["series"] = series
            repl["cfg"] = cfg
            if has_tsource_dim:
                repl["tsource"] = ts
            filename = filestem.format_map(PartialFormatter(repl))
            if "{" in filename:
                logger.warning(
                    f"Could not resolve input filestem {filestem!r} for {sc!r}; "
                    "skipping its manifest entry."
                )
                continue
            inputs.append(filename)

    return inputs


def load_data(
    agg_params: t.Dict[str, t.Any],
    skip_existing: bool = False,
    format: str = "csv",
    max_workers: int = 1,
) -> t.Dict[str, pd.DataFrame]:
    result = {}
    logger = utils.get_logger()

    # The HDF5 concurrency guard lives in the loader: `_resolve_load_context`
    # passes locking=False to h5py opens when max_workers > 1. (An os.environ
    # set here would be a no-op — HDF5 reads HDF5_USE_FILE_LOCKING at library
    # init, which precedes any set in this process.)

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

            # Read pre-formatted input filenames from the manifest sidecar(s) (D5).
            # Replaces the reshape chain (str.split -> transform -> reset_index
            # -> drop -> to_dict -> format_map) entirely.
            skip_file_set, manifest_exists = _read_skip_manifest(
                out_files["filestem"], format=format, logger=logger
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

            # Write manifest sidecar(s) (D5): pre-formatted input filenames so the
            # next --skip-existing run reads them as a pure set-membership lookup.
            # One manifest per output file, alongside each file, using the same
            # keyword replacement as the write system.
            _write_manifests(run_params["load_files"], out_files["filestem"], format)


def generate_manifests(
    agg_params: t.Dict[str, t.Any],
    format: str = "csv",
    max_workers: int = 1,
) -> None:
    """Generate manifest sidecars from existing processed agg files.

    For each run key: enumerate the agg output files (resolving templated stems
    and recovering path-constant keys via ``wildcard_fill``), skip any that do
    not exist on disk, read each via ``load_files_chunked``, reconstruct the
    input-filename list via ``_reconstruct_manifest_inputs`` (tsource
    completeness gate applied there), and write a manifest next to each resolved
    agg file via ``_write_skip_manifest``. Run keys with no agg files are warned
    and skipped. No input files are loaded; no agg files are written.
    """
    logger = utils.get_logger()
    ext = utils.io.get_file_ext_from_format(format)

    for key in agg_params["run"]:
        run_params = agg_params[key]
        load_files = run_params["load_files"]
        out_files = run_params.get("out_files", {})
        out_filestem = out_files.get("filestem")
        if not out_filestem:
            logger.debug(f"No out_files.filestem for {key!r}; skipping.")
            continue

        # Same read-back recipe as load_data's skip-existing block: the output
        # filestem + format ext, replacements unioned with {format}, input regex,
        # wildcard_fill to recover path-constant keys (e.g. {eigs,noise,dt}).
        out_filestem_ext = out_filestem + ext
        out_replacements = load_files.get("replacements", {}) | {"format": format}
        agg_files = utils.io.process_files(
            out_filestem_ext,
            lambda f, r: (f, r),
            out_replacements,
            load_files.get("regex", {}),
            wildcard_fill=True,
        )

        if not agg_files:
            logger.warning(
                f"No processed agg files found for {key!r} at "
                f"{out_filestem_ext!r}; skipping manifest generation."
            )
            continue

        for resolved_path, path_repl in agg_files:
            # process_files yields formatted paths without existence-checking
            # when the filestem is fully resolvable (no regex/wildcard keys), so
            # check existence explicitly before reading.
            if not os.path.exists(resolved_path):
                logger.warning(
                    f"Processed agg file {resolved_path!r} does not exist; "
                    "skipping."
                )
                continue
            df = dio.load_files_chunked(
                filestem=resolved_path, max_workers=max_workers
            )
            inputs = _reconstruct_manifest_inputs(df, load_files, dict(path_repl))
            manifest_stem = os.path.splitext(resolved_path)[0]
            _write_skip_manifest(manifest_stem, inputs)
            logger.info(
                f"Wrote manifest for {resolved_path!r} with {len(inputs)} input(s)."
            )


def aggregate_task_data(
    job_step: str,
    yaml_data: t.Dict,
    format: str = "csv",
    average: bool = False,
    skip_existing: bool = False,
    generate_manifest: bool = False,
    max_workers: int = 1,
) -> None:

    task = create_task(job_step, yaml_data)
    agg_params = task.handler.build_aggregator_params(task.config, average)
    if not agg_params:
        raise ValueError(f"No aggregator parameters provided for task: {job_step}.")

    if generate_manifest:
        # Backfill manifest sidecars from existing processed agg files; do not
        # re-aggregate. agg_params already reflects `average` (targets the _avg
        # files when --average), so a single build suffices (no raw/avg split).
        generate_manifests(agg_params, format=format, max_workers=max_workers)
        return

    if average:
        agg_params_raw = task.handler.build_aggregator_params(task.config, False)
    else:
        agg_params_raw = agg_params

    result = load_data(agg_params_raw, skip_existing, format, max_workers=max_workers)
    process_data(result, agg_params, format)


def set_format_col(df: pd.DataFrame, out_stem: str, out_format: str) -> pd.DataFrame:
    if "format" in utils.io.format_keys(out_stem):
        return df.assign(format=out_format)
    if "format" in df.columns:
        return df.drop(columns="format")
    return df


def _apply_average_actions(
    df: pd.DataFrame, avg_run_params: t.Dict[str, t.Any], data_col: str = "corr"
) -> pd.DataFrame:
    """Apply a run key's averaging actions to already-aggregated data.

    Mirrors ``process_data``'s composition: pulls the averaging actions (and the
    ``_avg`` output index override when a handler sets one) and runs them through
    the processor. CSV stores complex values as strings, so the data column is
    restored to numeric dtype first — otherwise the numeric averaging actions
    (``time_average`` / ``average`` / ``real``) operate on strings. Parity with the
    ``scripts/locate_agg_files.py`` prototype's ``apply_average``.
    """
    if data_col in df.columns and not pd.api.types.is_numeric_dtype(df[data_col]):
        df[data_col] = df[data_col].apply(complex)

    actions = dict(avg_run_params.get("actions", {}))
    if index := avg_run_params.get("out_files", {}).get("index"):
        actions["index"] = index
    return pc.execute(df, actions)


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
    average: bool = False,
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

    With ``average=True``, the existing (non-averaged) aggregated files are loaded
    in ``input_format``, the averaging actions from
    ``build_aggregator_params(average=True)`` are applied in-process via
    ``_apply_average_actions``, and the result is written to the ``_avg`` output
    filestems in ``output_format``. Input and output formats may then match — the
    averaging is the transformation.
    """
    task = create_task(job_step, yaml_data)
    agg_params = task.handler.build_aggregator_params(task.config, average=False)
    if not agg_params:
        raise ValueError(f"No aggregator parameters provided for task: {job_step}.")

    logger = utils.get_logger()

    if average:
        avg_params = task.handler.build_aggregator_params(task.config, True)
    else:
        avg_params = agg_params

    run_keys = agg_params["run"]
    num_keys = len(run_keys)

    for key in run_keys:
        run_params = agg_params[key]
        df = load_convert_data(run_params, input_format)

        if df.empty:
            logger.warning(f"Empty DataFrame for {key}")
            continue

        if average:
            df = _apply_average_actions(df, avg_params[key])

        out_filestem = avg_params[key]["out_files"]["filestem"]
        stem = resolve_convert_output_stem(out_filestem, output, num_keys)

        dio.write_files(
            set_format_col(df, stem, output_format),
            filestem=stem,
            format=output_format,
        )


def _collect_source_files(agg_params: t.Dict[str, t.Any]) -> t.List[str]:
    """Collect the raw source files (aggregation inputs) for every run key.

    Globs each run key's ``load_files["filestem"]`` -- the per-config input files
    that ``export corr`` aggregates -- with the same ``process_files`` primitive the
    loader uses, forwarding the handler's ``load_files["regex"]`` hints and setting
    ``wildcard_fill`` so every remaining ``{placeholder}`` (series, cfg, mass,
    gamma_label, ...) is expanded. Multi-run-key handlers that share one filestem
    (e.g. highmode) glob overlapping sets, so results are deduplicated by absolute
    path. Returns absolute paths of files on disk.
    """
    files: t.List[str] = []
    seen: t.Set[str] = set()
    for key in agg_params["run"]:
        load_files = agg_params[key].get("load_files", {})
        filestem = load_files.get("filestem")
        if not filestem:
            continue
        regex = load_files.get("regex", {})
        for filename, _ in utils.io.process_files(
            filestem,
            lambda f, r: (f, r),
            replacements={},
            regex=regex,
            wildcard_fill=True,
        ):
            if os.path.exists(filename):
                abspath = os.path.abspath(filename)
                if abspath not in seen:
                    seen.add(abspath)
                    files.append(abspath)
    return files


def tar_task_data(
    job_step: str | None,
    yaml_data: t.Dict,
    include_dirs: t.Tuple[str, ...] = (),
    output: str | None = None,
) -> str:
    """Archive a job step's raw source files and/or extra directories into a tar.

    Locates the raw per-config source files for ``job_step`` (the inputs to
    ``export corr`` aggregation) via ``create_task`` ->
    ``build_aggregator_params(average=False)`` -> ``_collect_source_files``, adds
    any ``include_dirs``, and writes an uncompressed tar into the params ``home``
    directory. ``output`` names the tar file under ``home`` (default
    ``<ens>_<YYYYMMDD-%H%M%S>.tar``). At least one of ``job_step`` or
    ``include_dirs`` must be supplied; raises ``ValueError`` if there is nothing
    to archive. Returns the absolute tar path.
    """
    logger = utils.get_logger()

    home = os.path.expanduser(get_nanny_config(yaml_data).home)
    ens = yaml_data.get("shared_params", {}).get("ens", "export")

    if output is None:
        tar_name = f"{ens}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.tar"
    else:
        tar_name = output

    os.makedirs(home, exist_ok=True)
    tar_path = os.path.join(home, tar_name)

    source_files: t.List[str] = []
    if job_step is not None:
        task = create_task(job_step, yaml_data)
        agg_params = task.handler.build_aggregator_params(task.config, average=False)
        if agg_params:
            source_files = _collect_source_files(agg_params)

    if not source_files and not include_dirs:
        raise ValueError(
            "Nothing to archive: no source files found and no --include "
            "directories given."
        )

    logger.info(
        f"Writing tar: {tar_path} "
        f"({len(source_files)} source file(s), {len(include_dirs)} include dir(s))"
    )
    with tarfile.open(tar_path, "w") as tf:
        if source_files:
            root = os.path.commonpath([os.path.dirname(f) for f in source_files])
            for f in source_files:
                tf.add(f, arcname=os.path.relpath(f, root))
        for d in include_dirs:
            tf.add(os.path.expanduser(d), arcname=os.path.basename(os.path.normpath(d)))

    logger.info(f"Tar written: {tar_path}")
    return tar_path
