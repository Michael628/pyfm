"""Locate the aggregated output files produced by ``pyfm task aggregate``.

For a given job step this reports where ``pyfm task aggregate`` writes (or has
written) its aggregated data, by reconstructing the same ``out_files`` filestems
the aggregator uses and globbing the filesystem for the resolved paths.

Optionally the located data can be re-emitted: ``--combine`` merges every run
key into a single table, ``--format`` controls the output file format, and
``--output`` sets the output filename.

``--average`` mirrors ``pyfm task aggregate --average`` but operates on the
*existing* (non-averaged) agg files: it loads them, applies the averaging actions
in-process, and writes the averaged result to the ``_avg`` output locations.
"""

import argparse
import os

import pandas as pd

from pyfm import utils
import pyfm.dataio as dio
from pyfm.dataio import processor as pc
from pyfm.nanny.core import create_task

# Input formats understood by the loader, mapped to their file extension.
INPUT_EXTENSIONS = {"csv": ".csv", "hdf5": ".h5", "parquet": ".parquet", "dict": ".npy"}


def build_agg_params(job, params, average):
    """Return the aggregator parameter dict for ``job`` (matches the aggregator).

    With ``average=False`` this gives the locations/actions for the per-config agg
    output; with ``average=True`` it gives the ``_avg`` filestems and the averaging
    actions. We use the former to locate/load existing files and the latter to
    process and write the averaged result.
    """
    task = create_task(job, params)
    agg_params = task.handler.build_aggregator_params(task.config, average)
    if not agg_params:
        raise ValueError(f"No aggregator parameters provided for task: {job}.")
    return agg_params


def apply_average(df, avg_run_params, data_col="corr"):
    """Apply a run key's averaging actions to already-aggregated data.

    Mirrors ``aggregator.process_data``: pulls the averaging actions (and the
    ``_avg`` output index) and runs them through the processor. CSV stores complex
    values as strings, so the data column is restored to numeric dtype first --
    otherwise the numeric averaging actions operate on strings.
    """
    if data_col in df.columns and not pd.api.types.is_numeric_dtype(df[data_col]):
        df[data_col] = df[data_col].apply(complex)

    actions = dict(avg_run_params.get("actions", {}))
    if index := avg_run_params.get("out_files", {}).get("index"):
        actions["index"] = index
    return pc.execute(df, actions)


def set_format_col(df, out_stem, out_format):
    """Reconcile the ``format`` metadata column with the output stem.

    If the stem has a ``{format}`` placeholder, the column drives both the output
    directory and is consumed (not written) by the writer, so set it to the output
    format. Otherwise drop it so the metadata does not leak into the written file.
    """
    if "format" in utils.string.format_keys(out_stem):
        return df.assign(format=out_format)
    if "format" in df.columns:
        return df.drop(columns="format")
    return df


def locate_files(out_filestem, input_format):
    """Find existing aggregated files on disk for an ``out_files`` filestem.

    The filestem carries no extension and may contain ``{placeholders}`` that the
    writer fills per output group. We append the input format's extension, pin the
    ``{format}`` placeholder (if present) to the input format so we don't match
    other format directories, and let ``process_files`` glob the rest.
    """
    stem = out_filestem + INPUT_EXTENSIONS[input_format]
    stem_keys = utils.string.format_keys(stem)
    replacements = {"format": input_format} if "format" in stem_keys else {}

    found = []
    for filename, _ in utils.io.process_files(
        stem, lambda f, r: (f, r), replacements=replacements, wildcard_fill=True
    ):
        if os.path.exists(filename):
            found.append(filename)
    return found


def load_run_data(run_params, input_format, data_col="corr"):
    """Load a run key's aggregated output files into a single DataFrame.

    Loads against the *templated* ``out_files`` filestem (not resolved paths) with
    ``wildcard_fill`` so the loader recovers any ``{placeholder}`` values embedded
    in the filename. Replacement/regex hints are filtered to keys that actually
    occur in the filestem -- passing keys that don't appear leaves the loader with
    an empty replacement set and raises in ``string_replacement_gen``. The
    ``{format}`` placeholder is pinned to the input format.

    All non-data columns (except the ``format`` metadata column) are moved into the
    index, both to faithfully reconstruct the written table and to give a clean,
    fully named index for any downstream group/average actions.
    """
    out_filestem = run_params["out_files"]["filestem"] + INPUT_EXTENSIONS[input_format]
    stem_keys = utils.string.format_keys(out_filestem)

    load_files = run_params.get("load_files", {})
    replacements = {
        k: v for k, v in load_files.get("replacements", {}).items() if k in stem_keys
    }
    regex = {k: v for k, v in load_files.get("regex", {}).items() if k in stem_keys}
    if "format" in stem_keys:
        replacements["format"] = input_format

    df = dio.load_files(
        filestem=out_filestem,
        wildcard_fill=True,
        replacements=replacements,
        regex=regex,
    ).agg()

    index_cols = [c for c in df.columns if c not in (data_col, "format")]
    if index_cols:
        df = df.set_index(index_cols)
    return df


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Locate the aggregated output files produced by `pyfm task aggregate` "
            "for a job step, and optionally re-emit them in another format."
        )
    )
    parser.add_argument("-j", "--job", type=str, required=True, help="Job step name.")
    parser.add_argument(
        "-p",
        "--param-file",
        type=str,
        default="params.yaml",
        help="Path to YAML parameter file.",
    )
    parser.add_argument(
        "--average",
        action="store_true",
        default=False,
        help="Average the existing (non-averaged) agg files in-process and write "
        "the averaged result to the `_avg` locations (implies a write).",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        default=False,
        help="Combine all run keys into a single table before writing.",
    )
    parser.add_argument(
        "--input-format",
        type=str,
        default="csv",
        choices=list(INPUT_EXTENSIONS),
        help="Format of the aggregated files to locate/load (default: csv, "
        "matching `pyfm task aggregate`).",
    )
    parser.add_argument(
        "--format",
        dest="fmt",
        type=str,
        default=None,
        help="Write the located data out in this format (csv, hdf5, parquet).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (stem) for the written data.",
    )
    parser.add_argument(
        "--logging-level", type=str, default="INFO", help="Set logging level."
    )
    args = parser.parse_args()

    utils.set_logging_level(args.logging_level)
    logger = utils.get_logger()

    params = utils.io.load_param(args.param_file)

    # Existing files on disk are the non-averaged outputs: locate/load from these.
    load_agg_params = build_agg_params(args.job, params, False)
    # Output locations (and, when averaging, the processing actions) come from here.
    out_agg_params = (
        build_agg_params(args.job, params, True) if args.average else load_agg_params
    )

    # --average produces new data, so it always implies a write.
    write_requested = args.fmt is not None or args.output is not None or args.average

    # Map each run key to its located files.
    located = {}
    for key in load_agg_params["run"]:
        out_filestem = load_agg_params[key]["out_files"]["filestem"]
        located[key] = locate_files(out_filestem, args.input_format)

    # Always report the locations.
    total = 0
    for key, files in located.items():
        print(f"[{key}]")
        if not files:
            print("  (no files found)")
        for f in files:
            print(f"  {f}")
            total += 1
    print(f"\nFound {total} aggregated file(s) across {len(located)} run key(s).")

    if not write_requested:
        return

    # Default output format: infer from --output extension, else csv.
    fmt = args.fmt
    if fmt is None and args.output is not None:
        ext = os.path.splitext(args.output)[1]
        fmt = {".csv": "csv", ".h5": "hdf5", ".parquet": "parquet"}.get(ext, "csv")
    fmt = fmt or "csv"

    # Load (and, when averaging, process) data for each run key that has files.
    frames = {}
    for key, files in located.items():
        if not files:
            logger.warning(f"No files to load for run key '{key}'.")
            continue
        df = load_run_data(load_agg_params[key], args.input_format)
        if args.average:
            df = apply_average(df, out_agg_params[key])
        frames[key] = df

    if not frames:
        logger.warning("No data loaded; nothing to write.")
        return

    if args.combine:
        if args.output is None:
            raise ValueError("--combine requires --output to name the merged file.")
        stem = os.path.splitext(args.output)[0]
        combined = set_format_col(pd.concat(frames.values()), stem, fmt)
        dio.write_files(combined, filestem=stem, format=fmt)
    else:
        for key, df in frames.items():
            if args.output is not None and len(frames) == 1:
                stem = os.path.splitext(args.output)[0]
            else:
                # Re-emit at the output stem for this run key (the `_avg` stem when
                # averaging, otherwise the original aggregate stem).
                stem = out_agg_params[key]["out_files"]["filestem"]
            dio.write_files(set_format_col(df, stem, fmt), filestem=stem, format=fmt)


if __name__ == "__main__":
    main()
