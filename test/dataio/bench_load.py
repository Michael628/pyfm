"""pytest-benchmark suite + peak-memory report for the HDF5 load loop.

Measures load_files_chunked across N-files × max-workers, a template-build
micro-benchmark (D1), a legacy-vs-chunked comparison, and a peak-memory report
(tracemalloc + ru_maxrss) that doubles as the Phase B "measured OOM" sensor.

Evidence invocations (results are otherwise discarded):

    # Persist benchmark timings into .benchmarks/ (pytest-benchmark native):
    pytest test/dataio/bench_load.py --benchmark-only --benchmark-save <label>

    # Peak-memory report at scale (HPC / real parallel FS; stdlib sensors):
    PYFM_BENCH_N_FILES=5000 PYFM_BENCH_WORKERS=8 \\
        pytest test/dataio/bench_load.py -k memory_report
    # -> .benchmarks/load_report_n5000_w8_<ts>.json
"""
import json
import os
import resource
import sys
import time
import tracemalloc
from datetime import datetime

import h5py
import pandas as pd
import pytest

from pyfm.dataio import load_files, load_files_chunked
from pyfm.dataio.converter import build_hdf5_template
from pyfm.domain.io import LoadH5Config

N_FILES_GRID = [1, 10, 50]
MAX_WORKERS_GRID = [1, 2, 4]

LOAD_REPORT_SCHEMA_VERSION = 1
PYFM_BENCH_N_FILES = int(os.environ.get("PYFM_BENCH_N_FILES", "50"))
PYFM_BENCH_WORKERS = int(os.environ.get("PYFM_BENCH_WORKERS", "1"))
PYFM_BENCH_REPORT_DIR = os.environ.get("PYFM_BENCH_REPORT_DIR", ".benchmarks")


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


def test_load_memory_report(scaled_raw_grid):
    """Peak-memory + wall-time report for one load_files_chunked run.

    Doubles as the Phase B trigger sensor ("measured OOM under Phase A"):
    tracemalloc peak is the per-run allocation peak (numpy buffers tracked);
    ru_maxrss is the process-wide high-water mark (upper bound when run inside
    the suite). Runnable at production scale on HPC via env knobs:

        PYFM_BENCH_N_FILES=5000 PYFM_BENCH_WORKERS=8 \\
            pytest test/dataio/bench_load.py -k memory_report
    """
    n_files = PYFM_BENCH_N_FILES
    workers = PYFM_BENCH_WORKERS
    filestem, h5kwargs = scaled_raw_grid(n_files)
    reps = {"series": ["a"], "cfg": [str(i) for i in range(n_files)]}
    regex = {"series": "[a-z]", "cfg": "[0-9]+"}

    tracemalloc.start()
    t0 = time.perf_counter()
    df = load_files_chunked(
        filestem=filestem,
        replacements=reps,
        regex=regex,
        max_workers=workers,
        **h5kwargs,
    )
    wall_seconds = time.perf_counter() - t0
    _, tracemalloc_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert len(df) == n_files * 48

    report = {
        "schema_version": LOAD_REPORT_SCHEMA_VERSION,
        "kind": "load_memory",
        "n_files": n_files,
        "max_workers": workers,
        "rows": len(df),
        "wall_seconds": round(wall_seconds, 4),
        "tracemalloc_peak_bytes": tracemalloc_peak_bytes,
        "ru_maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "python_version": sys.version.split()[0],
        "h5py_version": h5py.__version__,
        "pandas_version": pd.__version__,
        "note": (
            "ru_maxrss is a process-wide watermark (upper bound); "
            "tracemalloc_peak_bytes is the per-run allocation peak."
        ),
    }
    os.makedirs(PYFM_BENCH_REPORT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = os.path.join(
        PYFM_BENCH_REPORT_DIR, f"load_report_n{n_files}_w{workers}_{stamp}.json"
    )
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    assert os.path.exists(report_path)
    with open(report_path) as f:
        persisted = json.load(f)
    assert persisted["schema_version"] == LOAD_REPORT_SCHEMA_VERSION
    assert persisted["tracemalloc_peak_bytes"] > 0
