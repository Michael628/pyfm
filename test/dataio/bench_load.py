"""pytest-benchmark suite + peak-memory report for the HDF5 load loop.

Measures load_files_chunked across N-files × max-workers, a template-build
micro-benchmark (D1), a legacy-vs-chunked comparison, and a schema-v2
peak-memory report (tracemalloc + ru_maxrss self/children, cpu-second deltas,
phase timers, environment snapshot) that doubles as the Phase B "measured OOM"
sensor and the process-pool acceptance-gate sensor (D8/D9/D10).

Evidence invocations (results are otherwise discarded):

    # Persist benchmark timings into .benchmarks/ (pytest-benchmark native):
    pytest test/dataio/bench_load.py --benchmark-only --benchmark-save <label>

    # Peak-memory / acceptance report at scale (HPC / real parallel FS):
    PYFM_BENCH_N_FILES=5000 PYFM_BENCH_WORKERS=8 \\
        pytest test/dataio/bench_load.py -k memory_report
    # -> .benchmarks/load_report_n5000_w8_m<start-method>_<ts>.json

    # Dense calibration grid (brackets the fork crossover; slow — not for
    # default collection):
    PYFM_BENCH_N_GRID=1,2,3,5,8,12,20,30,50 PYFM_BENCH_W_GRID=1,2,4,8 \\
        pytest test/dataio/bench_load.py --benchmark-only --benchmark-save calib

    # Start-method axis (explicit; production pins fork per D10):
    PYFM_BENCH_START_METHOD=fork PYFM_BENCH_N_FILES=1000 \\
        PYFM_BENCH_WORKERS=8 pytest test/dataio/bench_load.py -k memory_report

    # Production-scale file sizes (~35 KB/file: n_t=1120; ~1 MB: n_t=32768):
    PYFM_BENCH_N_T=1120 PYFM_BENCH_N_FILES=100 \\
        pytest test/dataio/bench_load.py -k memory_report

    # Parallel-FS placement + threshold override (force the pool below the
    # production small-batch threshold to calibrate the crossover):
    PYFM_BENCH_DATA_DIR=/scratch/pyfm-bench PYFM_BENCH_POOL_THRESHOLD=0 \\
        pytest test/dataio/bench_load.py -k memory_report

    # Regex-glob enumeration arm (exercises the glob.glob path production
    # stems hit; the default fully-enumerated replacements short-circuit it):
    PYFM_BENCH_ENUM=regex pytest test/dataio/bench_load.py -k memory_report

    # Sampling knobs (pedantic rounds/iterations are passed explicitly, so
    # the pytest-benchmark CLI flags --benchmark-rounds/--benchmark-warmup
    # are ignored): more rounds for stabler medians under shared-node noise,
    # warmup rounds to pull the fixture tree into page cache before timing:
    PYFM_BENCH_ROUNDS=10 PYFM_BENCH_WARMUP=1 \\
        pytest test/dataio/bench_load.py --benchmark-only -k "1000-8"
"""
import inspect
import json
import os
import pathlib
import resource
import shutil
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

N_FILES_GRID = [
    int(v) for v in os.environ.get("PYFM_BENCH_N_GRID", "1,10,50").split(",")
]
MAX_WORKERS_GRID = [
    int(v) for v in os.environ.get("PYFM_BENCH_W_GRID", "1,2,4").split(",")
]

LOAD_REPORT_SCHEMA_VERSION = 2
PYFM_BENCH_N_FILES = int(os.environ.get("PYFM_BENCH_N_FILES", "50"))
PYFM_BENCH_WORKERS = int(os.environ.get("PYFM_BENCH_WORKERS", "1"))
PYFM_BENCH_REPORT_DIR = os.environ.get("PYFM_BENCH_REPORT_DIR", ".benchmarks")
PYFM_BENCH_N_T = int(os.environ.get("PYFM_BENCH_N_T", "24"))
PYFM_BENCH_START_METHOD = os.environ.get("PYFM_BENCH_START_METHOD") or None
PYFM_BENCH_DATA_DIR = os.environ.get("PYFM_BENCH_DATA_DIR") or None
PYFM_BENCH_POOL_THRESHOLD = (
    int(os.environ["PYFM_BENCH_POOL_THRESHOLD"])
    if "PYFM_BENCH_POOL_THRESHOLD" in os.environ
    else None
)
PYFM_BENCH_ENUM = os.environ.get("PYFM_BENCH_ENUM", "replacements")
PYFM_BENCH_ROUNDS = int(os.environ.get("PYFM_BENCH_ROUNDS", "3"))
PYFM_BENCH_WARMUP = int(os.environ.get("PYFM_BENCH_WARMUP", "0"))
PYFM_BENCH_ITERATIONS = int(os.environ.get("PYFM_BENCH_ITERATIONS", "1"))


def _loader_supports(kwarg: str) -> bool:
    """Whether the running load_files_chunked accepts ``kwarg``.

    The schema-v2 knobs (``start_method`` / ``pool_threshold`` / ``stats``)
    land with the D8 process-pool swap; the pre-swap baseline must run against
    the threaded loader, which takes none of them.
    """
    return kwarg in inspect.signature(load_files_chunked).parameters


def _loader_kwargs(**extra) -> dict:
    """Forward only kwargs the running loader supports and that are set.

    Pre-swap baseline: ``start_method``/``pool_threshold``/``stats`` are
    dropped (threaded loader). Post-swap: forwarded, so the bench drives the
    calibration axes (start method, threshold override, phase timers) without
    branching on the loader version itself.
    """
    return {k: v for k, v in extra.items() if v is not None and _loader_supports(k)}


def _enum_reps_regex(n_files: int) -> tuple:
    """File-enumeration knobs: (replacements, regex) for the load calls.

    ``replacements`` mode (default): fully-enumerated cfg lists —
    string_replacement_gen resolves every brace, file_regex_gen short-circuits
    glob.glob (fast, but the enumerate phase under-measures the production
    MDS-bound glob). ``regex`` mode: regex-only enumeration, exercising the
    glob.glob + pattern-match path production stems hit (e.g.
    pyfm/tasks/contract/diagram.py stems are regex-only).
    """
    if PYFM_BENCH_ENUM == "regex":
        return None, {"series": "[a-z]", "cfg": "[0-9]+"}
    return {"series": ["a"], "cfg": [str(i) for i in range(n_files)]}, {}


@pytest.fixture
def scaled_raw_grid(tmp_path):
    """Factory: writes N raw Grid HDF5 files; returns (filestem, h5kwargs).

    File size scales with PYFM_BENCH_N_T (complex128 = 16 B/element; rows per
    file = n_t × 2 meson). Default n_t=24 → 768 B/file (the schema-v1 fixture
    size); n_t=1120 ≈ 35 KB/file (production size, 2026-06-22 FRD); n_t=32768
    ≈ 1 MB/file. PYFM_BENCH_DATA_DIR relocates the fixture tree onto a real
    parallel FS (Aurora acceptance runs) — the owned directory is wiped and
    recreated up front and removed on teardown; tmp_path otherwise.
    """
    n_t = PYFM_BENCH_N_T
    n_meson = 2

    base = tmp_path
    if PYFM_BENCH_DATA_DIR:
        base = pathlib.Path(PYFM_BENCH_DATA_DIR)
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True)

    def _make(n_cfg):
        import h5py
        import numpy as np

        h5kwargs = dict(
            name="gamma",
            datasets={
                g: f"/meson/meson_{i}/corr"
                for i, g in enumerate(["g0", "g1"][:n_meson])
            },
            order=["t"],
            labels={"t": f"0..{n_t - 1}"},
        )
        for cfg in range(n_cfg):
            p = base / "a" / str(cfg) / "corr.h5"
            p.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(p, "w") as f:
                for i in range(n_meson):
                    g = f.create_group(f"meson/meson_{i}")
                    ds = g.create_dataset(
                        "corr", data=np.arange(n_t, dtype=np.complex128) + i
                    )
                    ds.attrs["sourceGamma"] = "GX_GX"
        filestem = str(base / "{series}/{cfg}/corr.h5")
        return filestem, h5kwargs

    yield _make

    if PYFM_BENCH_DATA_DIR:
        shutil.rmtree(base, ignore_errors=True)


@pytest.mark.parametrize("n_files,max_workers", [
    (n, w) for n in N_FILES_GRID for w in MAX_WORKERS_GRID
])
def test_bench_load_files_chunked(benchmark, scaled_raw_grid, n_files, max_workers):
    """Benchmark load_files_chunked across the N-files × max-workers grid."""
    filestem, h5kwargs = scaled_raw_grid(n_files)
    reps, regex = _enum_reps_regex(n_files)

    def load():
        return load_files_chunked(
            filestem=filestem,
            replacements=reps,
            regex=regex,
            max_workers=max_workers,
            **_loader_kwargs(
                start_method=PYFM_BENCH_START_METHOD,
                pool_threshold=PYFM_BENCH_POOL_THRESHOLD,
            ),
            **h5kwargs,
        )

    df = benchmark.pedantic(
        load,
        rounds=PYFM_BENCH_ROUNDS,
        warmup_rounds=PYFM_BENCH_WARMUP,
        iterations=PYFM_BENCH_ITERATIONS,
    )
    assert len(df) == n_files * PYFM_BENCH_N_T * 2  # n_t rows × 2 gamma per file


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
    reps, regex = _enum_reps_regex(n_files)

    def load():
        return load_files(
            filestem=filestem, replacements=reps, regex=regex, **h5kwargs
        ).agg()

    df = benchmark.pedantic(
        load,
        rounds=PYFM_BENCH_ROUNDS,
        warmup_rounds=PYFM_BENCH_WARMUP,
        iterations=PYFM_BENCH_ITERATIONS,
    )
    assert len(df) == n_files * PYFM_BENCH_N_T * 2


def _env_snapshot() -> dict:
    """Environment fingerprint: HDF5 build identity inputs + launcher + affinity.

    OMP_NUM_THREADS leakage (systems/aurora/env.sh:38 exports 8) and PBS_*/
    SLURM_* launcher vars explain cross-system variance; affinity bounds the
    parallelism the OS will actually schedule.
    """
    keys = (
        "OMP_NUM_THREADS",
        "PYFM_BENCH_N_T",
        "PYFM_BENCH_START_METHOD",
        "PYFM_BENCH_DATA_DIR",
        "PYFM_BENCH_POOL_THRESHOLD",
        "PYFM_BENCH_ENUM",
        "PYFM_BENCH_ROUNDS",
        "PYFM_BENCH_WARMUP",
    )
    snap = {k: os.environ.get(k) for k in keys}
    snap.update(
        {k: v for k, v in os.environ.items() if k.startswith(("PBS_", "SLURM_"))}
    )
    try:
        snap["cpu_affinity_count"] = len(os.sched_getaffinity(0))
    except AttributeError:  # not every platform exposes affinity
        pass
    return snap


def test_load_memory_report(scaled_raw_grid):
    """Peak-memory + parallelism-signature report for one load_files_chunked run.

    Schema v2 (D8/D9/D10 acceptance-gate sensor): child-side memory via
    RUSAGE_CHILDREN.ru_maxrss (max-over-children — the peak bound the
    acceptance gate needs; per-worker attribution deferred), cpu-second deltas
    (under a process pool cpu_children ≈ wall × effective_workers is the
    healthy-parallelism signature — vs cpu ≈ wall, which diagnosed the thread
    serialization), loader-truth effective_workers + phase timers via the
    ``stats`` kwarg when the running loader supports it (the pre-swap baseline
    computes the clamp locally instead), the HDF5 build identity behind the
    incident, and an environment snapshot. rusage counters are
    session-contaminated high-water marks, so a baseline is read at test start.
    Runnable at production scale on HPC via env knobs (see module docstring).
    """
    n_files = PYFM_BENCH_N_FILES
    workers = PYFM_BENCH_WORKERS
    filestem, h5kwargs = scaled_raw_grid(n_files)
    reps, regex = _enum_reps_regex(n_files)

    stats: dict = {}
    base_self = resource.getrusage(resource.RUSAGE_SELF)
    base_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    base_cpu = (
        base_self.ru_utime + base_self.ru_stime,
        base_children.ru_utime + base_children.ru_stime,
    )

    tracemalloc.start()
    t0 = time.perf_counter()
    df = load_files_chunked(
        filestem=filestem,
        replacements=reps,
        regex=regex,
        max_workers=workers,
        **_loader_kwargs(
            stats=stats,
            start_method=PYFM_BENCH_START_METHOD,
            pool_threshold=PYFM_BENCH_POOL_THRESHOLD,
        ),
        **h5kwargs,
    )
    wall_seconds = time.perf_counter() - t0
    _, tracemalloc_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert len(df) == n_files * PYFM_BENCH_N_T * 2

    final_self = resource.getrusage(resource.RUSAGE_SELF)
    final_children = resource.getrusage(resource.RUSAGE_CHILDREN)

    effective_workers = stats.get("effective_workers", min(workers, n_files))
    report = {
        "schema_version": LOAD_REPORT_SCHEMA_VERSION,
        "kind": "load_memory",
        "n_files": n_files,
        "max_workers": workers,
        "effective_workers": effective_workers,
        "pool_used": stats.get("pool_used", False),
        "start_method": stats.get(
            "start_method", PYFM_BENCH_START_METHOD or "default"
        ),
        "n_t": PYFM_BENCH_N_T,
        "rows": len(df),
        "wall_seconds": round(wall_seconds, 4),
        "tracemalloc_peak_bytes": tracemalloc_peak_bytes,
        "ru_maxrss_kb": final_self.ru_maxrss,
        "ru_maxrss_children_kb": final_children.ru_maxrss,
        "ru_maxrss_children_baseline_kb": base_children.ru_maxrss,
        "cpu_seconds_parent": round(
            (final_self.ru_utime + final_self.ru_stime) - base_cpu[0], 4
        ),
        "cpu_seconds_children": round(
            (final_children.ru_utime + final_children.ru_stime) - base_cpu[1], 4
        ),
        "python_version": sys.version.split()[0],
        "h5py_version": h5py.__version__,
        "hdf5_version": h5py.version.hdf5_version,
        "pandas_version": pd.__version__,
        "environment": _env_snapshot(),
        "stats": stats,
        "note": (
            "ru_maxrss(_children)_kb are process-wide watermarks (upper "
            "bounds); the children baseline is read at test start "
            "(session-contaminated); under a process pool cpu_children ≈ "
            "wall × effective_workers is the expected parallelism signature; "
            "tracemalloc wraps the timed region (its wall is not comparable "
            "to uninstrumented runs); when the loader does not fill stats "
            "(pre-swap baseline), effective_workers reflects the thread "
            "clamp and pool_used is False (the threaded executor is not a "
            "process pool)."
        ),
    }
    os.makedirs(PYFM_BENCH_REPORT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    method_tag = (PYFM_BENCH_START_METHOD or "default").replace("/", "-")
    report_path = os.path.join(
        PYFM_BENCH_REPORT_DIR,
        f"load_report_n{n_files}_w{workers}_m{method_tag}_{stamp}.json",
    )
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    assert os.path.exists(report_path)
    with open(report_path) as f:
        persisted = json.load(f)
    assert persisted["schema_version"] == LOAD_REPORT_SCHEMA_VERSION
    assert persisted["tracemalloc_peak_bytes"] > 0
    assert persisted["effective_workers"] >= 1
