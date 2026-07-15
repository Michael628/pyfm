"""pytest-benchmark suite for the HDF5 load loop.

Measures load_files_chunked across N-files × max-workers, a template-build
micro-benchmark (D1), and a legacy-vs-chunked comparison. Run with:

    pytest test/dataio/bench_load.py --benchmark-only
"""
import os

import pytest

from pyfm.dataio import load_files, load_files_chunked
from pyfm.dataio.converter import build_hdf5_template
from pyfm.domain.io import LoadH5Config

N_FILES_GRID = [1, 10, 50]
MAX_WORKERS_GRID = [1, 2, 4]


@pytest.fixture
def scaled_raw_grid(tmp_path):
    """Factory: writes N raw Grid HDF5 files into tmp_path; returns (filestem, h5kwargs)."""

    def _make(n_cfg):
        import h5py
        import numpy as np

        h5kwargs = dict(
            name="gamma",
            datasets={
                g: f"/meson/meson_{i}/corr"
                for i, g in enumerate(["g0", "g1"])
            },
            order=["t"],
            labels={"t": "0..23"},
        )
        for cfg in range(n_cfg):
            p = tmp_path / "a" / str(cfg) / "corr.h5"
            p.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(p, "w") as f:
                for i in range(2):
                    g = f.create_group(f"meson/meson_{i}")
                    ds = g.create_dataset(
                        "corr", data=np.arange(24, dtype=np.complex128) + i
                    )
                    ds.attrs["sourceGamma"] = "GX_GX"
        filestem = str(tmp_path / "{series}/{cfg}/corr.h5")
        return filestem, h5kwargs

    return _make


@pytest.mark.parametrize("n_files,max_workers", [
    (n, w) for n in N_FILES_GRID for w in MAX_WORKERS_GRID
])
def test_bench_load_files_chunked(benchmark, scaled_raw_grid, n_files, max_workers):
    """Benchmark load_files_chunked across the N-files × max-workers grid."""
    filestem, h5kwargs = scaled_raw_grid(n_files)
    reps = {"series": ["a"], "cfg": [str(i) for i in range(n_files)]}
    regex = {"series": "[a-z]", "cfg": "[0-9]+"}

    def load():
        return load_files_chunked(
            filestem=filestem,
            replacements=reps,
            regex=regex,
            max_workers=max_workers,
            **h5kwargs,
        )

    df = benchmark.pedantic(load, rounds=3, iterations=1)
    assert len(df) == n_files * 48  # 24 t × 2 gamma per file


def test_bench_template_build(benchmark):
    """Micro-benchmark: build_hdf5_template once (the D1 amortization target)."""
    h5kwargs = dict(
        name="gamma",
        datasets={g: f"/meson/meson_{i}/corr" for i, g in enumerate(["g0", "g1"])},
        order=["t"],
        labels={"t": "0..23"},
    )
    cfg = LoadH5Config.create(**h5kwargs).format_data_strings(
        {"series": "a", "cfg": "0"}
    )
    tmpl = benchmark.pedantic(
        build_hdf5_template, args=(cfg,), rounds=5, iterations=10
    )
    assert tmpl.name == "gamma"


@pytest.mark.parametrize("n_files", [1, 10, 50])
def test_bench_legacy_vs_chunked(benchmark, scaled_raw_grid, n_files):
    """Comparison: legacy load_files().agg() for baseline reference."""
    filestem, h5kwargs = scaled_raw_grid(n_files)
    reps = {"series": ["a"], "cfg": [str(i) for i in range(n_files)]}
    regex = {"series": "[a-z]", "cfg": "[0-9]+"}

    def load():
        return load_files(
            filestem=filestem, replacements=reps, regex=regex, **h5kwargs
        ).agg()

    df = benchmark.pedantic(load, rounds=3, iterations=1)
    assert len(df) == n_files * 48
