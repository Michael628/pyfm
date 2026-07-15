import dataclasses
import math
import os
import typing as t

import h5py
import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from pyfm.domain import LoadDictConfig, LoadH5Config

from pyfm.dataio.converter import (
    data_to_frame,
    build_hdf5_template,
    _fill_hdf5_frame,
    Hdf5FrameTemplate,
)
from pyfm.domain import WrappedDataPipe
from pyfm import utils

from concurrent.futures import ThreadPoolExecutor
from functools import partial

dataFrameFn = t.Callable[[np.ndarray], pd.DataFrame]
loadFn = t.Callable[[str, t.Dict], pd.DataFrame]

_PYTABLES_ROOT_ATTR = "PYTABLES_FORMAT_VERSION"


@dataclasses.dataclass(frozen=True)
class ResolvedHdf5Context:
    """Per-batch HDF5 context, resolved ONCE and shared across all files.

    Built by `_resolve_load_context`. Carries the build-once index template
    (D1, for raw Grid files), the PyTables flag (D2), and the resolved
    `LoadH5Config` so every file reuses the same index and config object.
    """

    is_pytables: bool
    h5_config: t.Optional[LoadH5Config]
    template: t.Optional[Hdf5FrameTemplate]


@dataclasses.dataclass(frozen=True)
class ResolvedLoadContext:
    """Per-batch resolved context shared by legacy and chunked load paths.

    Holds the resolved `file_loader` (a `partial` bound with the kwargs needed
    for a single file) plus, for HDF5, the per-batch HDF5 context. Built once by
    `_resolve_load_context` and reused across every file in the batch (D1 win for
    both paths; D2 single format probe).
    """

    file_loader: loadFn
    h5_context: t.Optional[ResolvedHdf5Context] = None


def _detect_hdf5_format(filename: str) -> t.Literal["pytables", "raw"]:
    """Detect the internal format of an HDF5 file via a single O(1) root probe.

    PyTables output (`writer.py:116` `to_hdf(key="corr", mode="w")`, default
    `format="fixed"`) carries a root-group attribute `PYTABLES_FORMAT_VERSION`
    (empirically `b'2.1'`) and, as a fallback marker, a top-level group with a
    `pandas_type` attribute. Raw Grid files have empty root attrs (`{}`) and
    carry physics attrs on their datasets — so the discriminator tests for
    PyTables PRESENCE, not attribute absence (a naive "has attrs" check would
    falsely classify raw Grid as PyTables). Returns "pytables" if either marker
    is found, else "raw".
    """
    with h5py.File(filename) as file:
        if _PYTABLES_ROOT_ATTR in file.attrs:
            return "pytables"
        # Fallback: top-level group carrying a pandas_type attribute.
        for name in file:
            obj = file[name]
            if isinstance(obj, h5py.Group) and "pandas_type" in obj.attrs:
                return "pytables"
    return "raw"


def get_pickle_loader(filename: str, _: t.Dict, **kwargs):
    data = np.load(filename, allow_pickle=True)
    if isinstance(data, np.ndarray) and len(data.shape) == 0:
        data = data.item()

    # TODO: Debug this for when pickle file is just pure ndarray
    pickle_config = LoadDictConfig.create(**kwargs)

    return data_to_frame(data, pickle_config)


def get_csv_loader(filename: str, _: t.Dict[str, str], **kwargs):

    return pd.read_csv(filename)


def get_parquet_loader(filename: str, _: t.Dict[str, str], **kwargs):
    if pq is None:
        raise NotImplementedError(
            "Parquet support requires pyarrow. Install with: pip install pyfm[parquet]"
        )

    return pq.read_table(filename, use_threads=True).to_pandas()


def get_hdf5_loader(
    filename: str,
    repl: t.Dict[str, str],
    *,
    is_pytables: bool | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Loads data from an HDF5 file and returns it as a DataFrame.

    Format-aware dispatch (D2): a single first-file root-attr probe determines
    whether the batch is PyTables output (route to ``pd.read_hdf``) or raw Grid
    HDF5 (route to ``h5py.File`` + the index template). The probe is O(1) and is
    performed once per batch by ``_resolve_load_context`` and threaded in via the
    ``is_pytables`` hint; when called directly without a hint, the file is probed
    individually (backward-compatible).

    Args:
        filename: Path to the HDF5 file.
        repl: String replacements applied to the configuration.
        is_pytables: Optional pre-resolved format hint. When ``None`` the file is
            probed individually. When truthy, routes straight to ``pd.read_hdf``.
            When falsy, skips the probe and opens via ``h5py.File``.
        **kwargs: Additional keyword arguments passed to ``LoadH5Config``.

    Returns:
        pd.DataFrame: The loaded data.
    """
    if is_pytables is None:
        is_pytables = _detect_hdf5_format(filename) == "pytables"

    if is_pytables:
        return pd.read_hdf(filename)

    with h5py.File(filename) as file:
        h5_config = LoadH5Config.create(**kwargs).format_data_strings(repl)
        return data_to_frame(file, h5_config)


def get_file_loader(file_path: str):
    ext = os.path.splitext(file_path)[1]

    match ext:
        case ".p" | ".npy":
            return get_pickle_loader
        case ".h5":
            return get_hdf5_loader
        case ".csv":
            return get_csv_loader
        case ".parquet":
            return get_parquet_loader
        case _:
            raise ValueError(
                "File must have extension '.p', '.h5', '.csv', or '.parquet'"
            )


def _resolve_load_context(
    file_repls: t.List[t.Tuple[str, t.Dict]],
    **kwargs,
) -> ResolvedLoadContext:
    """Resolve per-batch load context once: format probe + build-once template.

    For HDF5 batches: probes the first file's format once (D2), and for raw Grid
    builds the index template once (D1) so the legacy and chunked paths both
    reuse a single index/config object across all files. For non-HDF5 formats
    (`h5_context is None`) the legacy partial is returned unchanged.

    `max_workers` is popped by the caller before reaching here (top-level kwarg,
    never a dict entry — baking it into `LoadArrayConfig.create` would TypeError).
    """
    file_loader_fn = get_file_loader(file_repls[0][0])
    h5_context = None
    if file_loader_fn is get_hdf5_loader:
        is_pytables = _detect_hdf5_format(file_repls[0][0]) == "pytables"
        h5_config = None
        template = None
        if not is_pytables:
            h5_config = LoadH5Config.create(**kwargs).format_data_strings(
                dict(file_repls[0][1])
            )
            template = build_hdf5_template(h5_config)
        h5_context = ResolvedHdf5Context(
            is_pytables=is_pytables, h5_config=h5_config, template=template
        )
    if h5_context is not None:
        # Bind the resolved format hint into the partial so every file in a
        # PyTables batch routes straight to pd.read_hdf without re-probing (D2's
        # single O(1) probe per batch). Non-HDF5 loaders don't accept is_pytables.
        file_loader = partial(
            file_loader_fn, is_pytables=h5_context.is_pytables, **kwargs
        )
    else:
        file_loader = partial(file_loader_fn, **kwargs)
    return ResolvedLoadContext(file_loader=file_loader, h5_context=h5_context)


def _load_one_raw_hdf5(
    filename: str,
    repl: t.Dict[str, str],
    template: Hdf5FrameTemplate,
) -> pd.DataFrame:
    """Worker for one raw Grid HDF5 file using the pre-built index template.

    Opens the file, reads arrays (GIL-released), and assembles the frame in O(1)
    by reusing `template`'s cached MultiIndex by reference (no index construction
    on the hot path), then attaches the replacement columns. The template is
    read-only and never mutated, so it is safe to share across threads. Returns
    only the per-file DataFrame; the caller owns GroupTuple construction so the
    legacy yield path and the chunked concat path share one worker body.
    """
    with h5py.File(filename) as file:
        df = _fill_hdf5_frame(file, template)
    if repl:
        df[list(repl.keys())] = tuple(repl.values())
    return df


def load_files(
    filestem: str | t.List[str],
    replacements: t.Dict | None = None,
    regex: t.Dict | None = None,
    wildcard_fill: bool = False,
    aggregate: bool = False,
    skip_file_set: t.List[str] | None = None,
    **kwargs,
) -> WrappedDataPipe | pd.DataFrame:
    def file_factory():
        file_repls = utils.io.process_files(
            filestem, lambda f, r: (f, r), replacements, regex, wildcard_fill
        )

        if skip_file_set:
            file_repls = [f for f in file_repls if f[0] not in skip_file_set]

        if not file_repls:
            file0 = filestem if isinstance(filestem, str) else filestem[0] + ", ..."
            raise ValueError(f"No files found for file search pattern: {file0}")

        max_workers = min(kwargs.pop("max_workers", 1), len(file_repls))

        ctx = _resolve_load_context(file_repls, **kwargs)
        group_cols = list(file_repls[0][1].keys())
        GroupTuple = utils.create_group_tuple(*group_cols)

        def load_one(filename, repl):
            utils.get_logger().debug(f"Loading file: {filename}")
            if ctx.h5_context is not None and ctx.h5_context.template is not None:
                df = _load_one_raw_hdf5(
                    filename, repl, ctx.h5_context.template
                )
            else:
                df = ctx.file_loader(filename, repl)
                if repl:
                    df[list(repl.keys())] = tuple(repl.values())
            return repl, df

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(load_one, fn, r) for fn, r in file_repls]
            for fut in futures:
                repl, df = fut.result()
                yield GroupTuple(**repl), df

    if aggregate:
        return WrappedDataPipe(file_factory).agg()
    else:
        return WrappedDataPipe(file_factory)


def load_files_chunked(
    filestem: str | t.List[str],
    replacements: t.Dict | None = None,
    regex: t.Dict | None = None,
    wildcard_fill: bool = False,
    skip_file_set: t.List[str] | None = None,
    max_workers: int = 1,
    **kwargs,
) -> pd.DataFrame:
    """Load, concat, and return one DataFrame using a chunked thread pool.

    New chunk-aware entry point (D3/D4): resolves the load context once (template
    + format probe), owns a `ThreadPoolExecutor`, submits worker-aligned chunks
    of ``ceil(N/max_workers)`` files, drains and concatenates each chunk, then
    concatenates the chunk results into a single DataFrame. Bounds peak RAM to
    one chunk's worth of frames in flight rather than all N.

    `max_workers` is a top-level kwarg (never a dict entry). Caller is expected
    to set `HDF5_USE_FILE_LOCKING=FALSE` when `max_workers > 1`.
    """
    file_repls = utils.io.process_files(
        filestem, lambda f, r: (f, r), replacements, regex, wildcard_fill
    )

    if skip_file_set:
        file_repls = [f for f in file_repls if f[0] not in skip_file_set]

    if not file_repls:
        file0 = filestem if isinstance(filestem, str) else filestem[0] + ", ..."
        raise ValueError(f"No files found for file search pattern: {file0}")

    n_files = len(file_repls)
    max_workers = min(max_workers, n_files)

    ctx = _resolve_load_context(file_repls, **kwargs)

    def load_one(filename, repl):
        utils.get_logger().debug(f"Loading file: {filename}")
        if ctx.h5_context is not None and ctx.h5_context.template is not None:
            df = _load_one_raw_hdf5(
                filename, repl, ctx.h5_context.template
            )
        else:
            df = ctx.file_loader(filename, repl)
            if repl:
                df[list(repl.keys())] = tuple(repl.values())
        return repl, df

    chunk_size = max(1, math.ceil(n_files / max_workers))
    chunks = [file_repls[i : i + chunk_size] for i in range(0, n_files, chunk_size)]

    chunks_out = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for chunk in chunks:
            futures = [pool.submit(load_one, fn, r) for fn, r in chunk]
            chunk_frames = [fut.result()[1] for fut in futures]
            chunks_out.append(pd.concat(chunk_frames))

    if not chunks_out:
        return pd.DataFrame()
    return pd.concat(chunks_out)
