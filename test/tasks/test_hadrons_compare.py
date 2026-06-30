"""Tests for high-mode output comparison (pyfm/tasks/hadrons/highmode/compare.py)."""

import os

import h5py
import numpy as np
import pytest

from pyfm.domain import Gamma, MassDict, OpList, Outfile
from pyfm.tasks.hadrons.highmode import compare as highmode_compare
from pyfm.tasks.hadrons.highmode.strategy import create_outfile_catalog
from pyfm.tasks.hadrons.types import HighModeConfig


def _make_config(high_modes_stem: str) -> HighModeConfig:
    return HighModeConfig(
        formatting={},
        logging_level="INFO",
        runid="test",
        mass=MassDict.from_dict({"l": 0.01}),
        action_name="action_{mass}",
        solver_name="solver_{solver}_{mass}",
        low_modes_name="low_modes",
        operations=OpList([OpList.Op(gamma=Gamma.VEC_LOCAL, mass=("l",))]),
        high_modes=Outfile(filestem=high_modes_stem, ext=".h5", good_size=1),
        tstart=0,
        tstop=0,
        dt=1,
        noise=1,
        time=4,
        skip_cg=True,
        shift_gauge_name="shift_gauge",
    )


def _write_corr(path: str, values):
    """Write an h5 file with /meson/meson_{i}/corr datasets (one per gamma component)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        for i, val in enumerate(values):
            f.create_dataset(
                f"/meson/meson_{i}/corr", data=np.asarray(val, dtype=np.complex128)
            )


def _expected_filepaths(config):
    return list(create_outfile_catalog(config)["filepath"])


def test_compare_equal_outputs_reports_zero_diff(tmp_path):
    stem_a = str(tmp_path / "a" / "corr_{gamma_label}_{dset}_m{mass}_t{tsource}")
    stem_b = str(tmp_path / "b" / "corr_{gamma_label}_{dset}_m{mass}_t{tsource}")
    cfg_a, cfg_b = _make_config(stem_a), _make_config(stem_b)

    rng = np.random.default_rng(0)
    values = [rng.standard_normal(4) + 1j * rng.standard_normal(4) for _ in range(3)]
    for fp in _expected_filepaths(cfg_a):
        _write_corr(fp, values)
    for fp in _expected_filepaths(cfg_b):
        _write_corr(fp, values)  # identical data

    report = highmode_compare.compare_outputs(cfg_a, cfg_b)
    assert len(report) == 1
    row = report.iloc[0]
    assert row["status"] == "compared"
    assert bool(row["within_tolerance"]) is True
    assert row["max_abs_diff"] == pytest.approx(0.0, abs=1e-15)


def test_compare_perturbed_outputs_reports_mismatch(tmp_path):
    stem_a = str(tmp_path / "a" / "corr_{gamma_label}_{dset}_m{mass}_t{tsource}")
    stem_b = str(tmp_path / "b" / "corr_{gamma_label}_{dset}_m{mass}_t{tsource}")
    cfg_a, cfg_b = _make_config(stem_a), _make_config(stem_b)

    values_a = [np.ones(4, dtype=np.complex128) for _ in range(3)]
    values_b = [np.ones(4, dtype=np.complex128) for _ in range(3)]
    values_b[1] = values_b[1] * 2  # large perturbation in dataset 1

    for fp in _expected_filepaths(cfg_a):
        _write_corr(fp, values_a)
    for fp in _expected_filepaths(cfg_b):
        _write_corr(fp, values_b)

    report = highmode_compare.compare_outputs(cfg_a, cfg_b, rtol=1e-9, atol=1e-12)
    assert len(report) == 1
    row = report.iloc[0]
    assert row["status"] == "compared"
    assert bool(row["within_tolerance"]) is False
    assert row["max_abs_diff"] == pytest.approx(1.0, abs=1e-12)


def test_compare_missing_file_in_b(tmp_path):
    stem_a = str(tmp_path / "a" / "corr_{gamma_label}_{dset}_m{mass}_t{tsource}")
    stem_b = str(tmp_path / "b" / "corr_{gamma_label}_{dset}_m{mass}_t{tsource}")
    cfg_a, cfg_b = _make_config(stem_a), _make_config(stem_b)

    values = [np.ones(4, dtype=np.complex128) for _ in range(3)]
    for fp in _expected_filepaths(cfg_a):
        _write_corr(fp, values)
    # intentionally do NOT write b's file

    report = highmode_compare.compare_outputs(cfg_a, cfg_b)
    assert len(report) == 1
    row = report.iloc[0]
    assert row["status"] == "missing_file"
    assert bool(row["within_tolerance"]) is False


def test_hadrons_lmi_handler_has_compare_outputs():
    import pyfm.tasks.hadrons.lmi  # noqa: F401  (registers hadrons_lmi; idempotent)
    from pyfm.domain.protocols import OutputComparisonProtocol
    from pyfm.tasks.register import get_task_handler

    handler = get_task_handler(job_type="hadrons", task_type="lmi", strict=False)
    assert handler is not None
    assert isinstance(handler, OutputComparisonProtocol)
    assert callable(handler.compare_outputs)
