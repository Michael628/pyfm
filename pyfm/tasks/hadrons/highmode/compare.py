"""Numerical comparison of high-mode correlator outputs between two configs."""

import typing as t

import h5py
import numpy as np
import pandas as pd

from pyfm import utils
from pyfm.tasks.hadrons.highmode.strategy import create_outfile_catalog
from pyfm.tasks.hadrons.types import HighModeConfig

_PAIR_KEYS = ["gamma_label", "mass", "tsource", "dset"]


def _col(row: pd.Series, key: str, default: t.Any = False) -> t.Any:
    """Read a (possibly NaN / absent) catalog column from a merged row."""
    if key not in row:
        return default
    val = row[key]
    return default if pd.isna(val) else val


def _find_op(config: HighModeConfig, gamma_label: str):
    """Return the OpList.Op whose gamma name matches ``gamma_label``, or None."""
    for op in config.op_list:
        if op.gamma.name.lower() == gamma_label:
            return op
    return None


def _dataset_paths(op) -> t.List[str]:
    """HDF5 dataset paths for each gamma component of ``op``."""
    return [f"/meson/meson_{i}/corr" for i in range(len(op.gamma.gamma_list))]


def _compare_h5_file(
    filepath_a: str,
    filepath_b: str,
    dataset_paths: t.List[str],
    *,
    rtol: float,
    atol: float,
) -> t.Tuple[float, float, bool]:
    """Compare matching datasets in two HDF5 files.

    Correlator data is stored as float64 and viewed as ``np.complex128`` on read
    (see ``pyfm/dataio/converter.py::hdf5_to_frame``).

    Returns ``(max_abs_diff, max_rel_diff, within_tolerance)``. A shape mismatch
    between corresponding datasets reports an infinite diff and within=False.
    """
    max_abs = 0.0
    max_rel = 0.0
    within = True
    try:
        with h5py.File(filepath_a, "r") as fa, h5py.File(filepath_b, "r") as fb:
            for ds in dataset_paths:
                if ds not in fa:
                    raise ValueError(f"dataset {ds!r} not found in {filepath_a!r}")
                if ds not in fb:
                    raise ValueError(f"dataset {ds!r} not found in {filepath_b!r}")
                a = fa[ds][:].view(np.complex128)
                b = fb[ds][:].view(np.complex128)
                if a.shape != b.shape:
                    return float("inf"), float("inf"), False
                diff = np.abs(a - b)
                max_abs = max(max_abs, float(np.max(diff)) if diff.size else 0.0)
                denom = np.abs(b)
                nonzero = denom > 0
                if np.any(nonzero):
                    rel = diff[nonzero] / denom[nonzero]
                    max_rel = max(max_rel, float(np.max(rel)))
                within = within and bool(np.all(diff <= atol + rtol * denom))
    except (OSError, KeyError) as e:
        # A file that passed step-1 good_size but is corrupt / missing a dataset
        # should surface as a clean ValueError (caught by the CLI), not a traceback.
        raise ValueError(
            f"Failed to read HDF5 output for comparison "
            f"({filepath_a!r} vs {filepath_b!r}): {e}"
        ) from e
    return max_abs, max_rel, within


def compare_outputs(
    config_a: HighModeConfig,
    config_b: HighModeConfig,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> pd.DataFrame:
    """Compare the high-mode correlator outputs of two configs.

    Builds both outfile catalogs, pairs expected files by
    ``(gamma_label, mass, tsource, dset)``, loads each pair's HDF5 datasets
    (``/meson/meson_{i}/corr``), and reports per-file max abs/rel diff plus
    whether the pair is within tolerance.

    Returns a DataFrame with columns: ``gamma_label, mass, tsource, dset,
    filepath_a, filepath_b, max_abs_diff, max_rel_diff, within_tolerance, status``
    where status is one of ``compared``, ``missing_file``, ``missing_op``.
    """
    logger = utils.get_logger()

    catalog_a = create_outfile_catalog(config_a)
    catalog_b = create_outfile_catalog(config_b)

    paired = catalog_a.merge(
        catalog_b, on=_PAIR_KEYS, how="outer", suffixes=("_a", "_b"), indicator=True
    )

    rows = []
    for _, r in paired.iterrows():
        gamma_label = r["gamma_label"]
        op = _find_op(config_a, gamma_label) or _find_op(config_b, gamma_label)
        filepath_a = _col(r, "filepath_a", None)
        filepath_b = _col(r, "filepath_b", None)
        exists_a = bool(_col(r, "exists_a", False))
        exists_b = bool(_col(r, "exists_b", False))

        base = {
            "gamma_label": gamma_label,
            "mass": r["mass"],
            "tsource": r["tsource"],
            "dset": r["dset"],
            "filepath_a": filepath_a,
            "filepath_b": filepath_b,
        }

        if op is None:
            rows.append(
                base
                | {
                    "max_abs_diff": float("nan"),
                    "max_rel_diff": float("nan"),
                    "within_tolerance": False,
                    "status": "missing_op",
                }
            )
            logger.warning(
                f"No op found for gamma_label={gamma_label!r}; cannot compare."
            )
            continue

        if not (exists_a and exists_b):
            rows.append(
                base
                | {
                    "max_abs_diff": float("nan"),
                    "max_rel_diff": float("nan"),
                    "within_tolerance": False,
                    "status": "missing_file",
                }
            )
            logger.info(
                f"Missing output for gamma_label={gamma_label!r} "
                f"tsource={r['tsource']} dset={r['dset']} "
                f"(exists_a={exists_a}, exists_b={exists_b})."
            )
            continue

        max_abs, max_rel, within = _compare_h5_file(
            filepath_a, filepath_b, _dataset_paths(op), rtol=rtol, atol=atol
        )
        rows.append(
            base
            | {
                "max_abs_diff": max_abs,
                "max_rel_diff": max_rel,
                "within_tolerance": within,
                "status": "compared",
            }
        )

    columns = [
        "gamma_label",
        "mass",
        "tsource",
        "dset",
        "filepath_a",
        "filepath_b",
        "max_abs_diff",
        "max_rel_diff",
        "within_tolerance",
        "status",
    ]
    return pd.DataFrame(rows, columns=columns)
