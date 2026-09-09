import pandas as pd
import pytest

from pyfm.dataio import load_files, load_files_chunked
from pyfm.dataio.loader import _detect_hdf5_format


def test_h5py_opens_carry_locking_flag_by_worker_count(raw_grid_h5_factory, monkeypatch):
    """Concurrency guard is observable at the h5py.File / submit boundaries.

    max_workers > 1: the format probe (in-parent) and every worker payload
    (asserted at the ProcessPoolExecutor submit boundary) carry locking=False.
    The submit-boundary capture is required because children re-import h5py
    unpatched (spawn) or append to COW-private copies (fork) — an in-process
    spy alone would pass vacuously for the pool legs. max_workers == 1:
    locking stays None (fail-fast on files held by a writer) and no
    chunk-task is submitted at all (serial fast path).
    """
    import h5py as _h5py

    from pyfm.dataio import loader as loader_mod

    tmpdir, h5kwargs = raw_grid_h5_factory(n_cfg=4)
    fs = f"{tmpdir}/{{series}}/{{cfg}}/corr.h5"
    reps = {"series": ["a"], "cfg": [str(i) for i in range(4)]}
    regex = {"series": "[a-z]", "cfg": "[0-9]+"}

    seen: list = []
    real_file = _h5py.File

    def spy_file(name, *args, **kwargs):
        seen.append(kwargs.get("locking", "__unset__"))
        return real_file(name, *args, **kwargs)

    monkeypatch.setattr(loader_mod.h5py, "File", spy_file)

    real_pool_cls = loader_mod.ProcessPoolExecutor
    submitted: list = []

    class SpyPool(real_pool_cls):
        def submit(self, fn, /, *args, **kwargs):
            submitted.append((fn, args))
            return super().submit(fn, *args, **kwargs)

    monkeypatch.setattr(loader_mod, "ProcessPoolExecutor", SpyPool)

    # W>1 pool leg (threshold forced to 0: 4 files would otherwise clamp).
    df = load_files_chunked(
        filestem=fs, replacements=reps, regex=regex, max_workers=2,
        pool_threshold=0, **h5kwargs,
    )
    assert len(df) == 4 * 48
    assert seen and all(v is False for v in seen)  # parent-side probe: False
    assert submitted, "no chunk-tasks submitted — pool did not engage"
    assert all(fn is loader_mod._load_chunk for fn, _ in submitted)
    assert all(args[1].h5_context.locking is False for _, args in submitted)
    assert [len(args[0]) for _, args in submitted] == [2, 2]

    # Legacy path leg: still threaded — the in-process spy observes every
    # open directly.
    seen.clear()
    load_files(
        filestem=fs, replacements=reps, regex=regex, max_workers=2, **h5kwargs
    ).agg()
    assert seen and all(v is False for v in seen)

    # W=1 leg: serial fast path — locking None, and the pool is bypassed
    # entirely (no additional submissions).
    seen.clear()
    n_submitted = len(submitted)
    load_files_chunked(
        filestem=fs, replacements=reps, regex=regex, max_workers=1, **h5kwargs
    )
    assert seen and all(v is None for v in seen)
    assert len(submitted) == n_submitted


def test_small_batch_threshold_clamps_to_serial(raw_grid_h5_factory, monkeypatch):
    """D9 gate: n_files < pool_threshold (default _POOL_MIN_FILES = 8) clamps
    to the serial path — no executor constructed, locking reverts to None
    (fail-fast preserved), stats reports effective_workers == 1."""
    import h5py as _h5py

    from pyfm.dataio import loader as loader_mod

    tmpdir, h5kwargs = raw_grid_h5_factory(n_cfg=4)  # 4 < 8
    fs = f"{tmpdir}/{{series}}/{{cfg}}/corr.h5"
    reps = {"series": ["a"], "cfg": [str(i) for i in range(4)]}
    regex = {"series": "[a-z]", "cfg": "[0-9]+"}

    def no_pool(*args, **kwargs):
        raise AssertionError("ProcessPoolExecutor constructed below threshold")

    monkeypatch.setattr(loader_mod, "ProcessPoolExecutor", no_pool)

    seen: list = []
    real_file = _h5py.File

    def spy_file(name, *args, **kwargs):
        seen.append(kwargs.get("locking", "__unset__"))
        return real_file(name, *args, **kwargs)

    monkeypatch.setattr(loader_mod.h5py, "File", spy_file)

    stats: dict = {}
    df = load_files_chunked(
        filestem=fs, replacements=reps, regex=regex, max_workers=2,
        stats=stats, **h5kwargs,
    )
    assert len(df) == 4 * 48
    assert stats["effective_workers"] == 1
    assert stats["pool_used"] is False
    assert seen and all(v is None for v in seen)


def test_w1_bypasses_pool_above_threshold(raw_grid_h5_factory, monkeypatch):
    """W=1 is a plain serial loop even above the threshold — no executor."""
    from pyfm.dataio import loader as loader_mod

    tmpdir, h5kwargs = raw_grid_h5_factory(n_cfg=10)  # 10 >= 8
    fs = f"{tmpdir}/{{series}}/{{cfg}}/corr.h5"
    reps = {"series": ["a"], "cfg": [str(i) for i in range(10)]}
    regex = {"series": "[a-z]", "cfg": "[0-9]+"}

    def no_pool(*args, **kwargs):
        raise AssertionError("ProcessPoolExecutor constructed at max_workers=1")

    monkeypatch.setattr(loader_mod, "ProcessPoolExecutor", no_pool)

    stats: dict = {}
    df = load_files_chunked(
        filestem=fs, replacements=reps, regex=regex, max_workers=1,
        stats=stats, **h5kwargs,
    )
    assert len(df) == 10 * 48
    assert stats["effective_workers"] == 1
    assert stats["pool_used"] is False


def test_discriminator_classifies_raw_grid(raw_grid_h5_factory):
    """Raw Grid HDF5 (datasets carry physics attrs, empty root attrs) -> 'raw'."""
    tmpdir, _ = raw_grid_h5_factory(n_cfg=1)
    assert _detect_hdf5_format(f"{tmpdir}/a/0/corr.h5") == "raw"


def test_discriminator_classifies_pytables(pytables_h5_factory):
    """PyTables HDF5 (root attr PYTABLES_FORMAT_VERSION) -> 'pytables'."""
    p = pytables_h5_factory()
    assert _detect_hdf5_format(p) == "pytables"


def test_chunked_equals_legacy_single_worker(raw_grid_h5_factory):
    """load_files_chunked(max_workers=1) == load_files().agg()."""
    tmpdir, h5kwargs = raw_grid_h5_factory(n_cfg=6)
    fs = f"{tmpdir}/{{series}}/{{cfg}}/corr.h5"
    reps = {"series": ["a"], "cfg": [str(i) for i in range(6)]}
    regex = {"series": "[a-z]", "cfg": "[0-9]+"}
    legacy = load_files(filestem=fs, replacements=reps, regex=regex, **h5kwargs).agg()
    chunked = load_files_chunked(filestem=fs, replacements=reps, regex=regex, **h5kwargs)
    pd.testing.assert_frame_equal(legacy.sort_index(), chunked.sort_index())


def test_chunked_equals_legacy_max_workers_sweep(raw_grid_h5_factory):
    """Output is identical across max_workers ∈ {1,2,3,4,6}.

    Multi-worker legs force the process pool (pool_threshold=0) so the sweep
    exercises real chunked-process execution (D8) rather than threshold-
    clamped serial passes (D9) — n_cfg=6 < _POOL_MIN_FILES=8 would otherwise
    clamp every leg to serial and the sweep would cover nothing."""
    tmpdir, h5kwargs = raw_grid_h5_factory(n_cfg=6)
    fs = f"{tmpdir}/{{series}}/{{cfg}}/corr.h5"
    reps = {"series": ["a"], "cfg": [str(i) for i in range(6)]}
    regex = {"series": "[a-z]", "cfg": "[0-9]+"}
    base = load_files_chunked(filestem=fs, replacements=reps, regex=regex, **h5kwargs)
    for mw in [2, 3, 4, 6]:
        df = load_files_chunked(
            filestem=fs, replacements=reps, regex=regex, max_workers=mw,
            pool_threshold=0, **h5kwargs
        )
        pd.testing.assert_frame_equal(base.sort_index(), df.sort_index())


def test_chunked_skip_file_set(raw_grid_h5_factory):
    """skip_file_set removes the specified files from output."""
    tmpdir, h5kwargs = raw_grid_h5_factory(n_cfg=6)
    fs = f"{tmpdir}/{{series}}/{{cfg}}/corr.h5"
    reps = {"series": ["a"], "cfg": [str(i) for i in range(6)]}
    regex = {"series": "[a-z]", "cfg": "[0-9]+"}
    full = load_files_chunked(filestem=fs, replacements=reps, regex=regex, **h5kwargs)
    skip = {f"{tmpdir}/a/0/corr.h5"}
    skipped = load_files_chunked(
        filestem=fs, replacements=reps, regex=regex, skip_file_set=skip, **h5kwargs
    )
    assert len(full) - len(skipped) == 48  # 1 file × 24 t × 2 gamma = 48 rows


def test_chunked_row_order_matches_legacy_unsorted(raw_grid_h5_factory):
    """Row order is identical by construction (submission-order drain in both
    paths); pin it WITHOUT sorting — the sorted guards above would pass an
    order regression. The multi-worker leg forces the pool so the drain-order
    canary covers the process path."""
    tmpdir, h5kwargs = raw_grid_h5_factory(n_cfg=6)
    fs = f"{tmpdir}/{{series}}/{{cfg}}/corr.h5"
    reps = {"series": ["a"], "cfg": [str(i) for i in range(6)]}
    regex = {"series": "[a-z]", "cfg": "[0-9]+"}
    legacy = load_files(filestem=fs, replacements=reps, regex=regex, **h5kwargs).agg()
    chunked = load_files_chunked(filestem=fs, replacements=reps, regex=regex, **h5kwargs)
    pd.testing.assert_frame_equal(legacy, chunked)  # no sort_index()
    multi = load_files_chunked(
        filestem=fs, replacements=reps, regex=regex, max_workers=3,
        pool_threshold=0, **h5kwargs
    )
    pd.testing.assert_frame_equal(legacy, multi)  # pool drain order also pinned


def test_mixed_pytables_raw_batch_raises(raw_grid_h5_factory, tmp_path):
    """Mixed PyTables + raw-Grid batch: first-file-wins dispatch assumes batch
    homogeneity (D2, `_resolve_load_context` first-file probe). A mixed batch
    raises — pinned here as the production assumption rather than restored
    per-file tolerance.

    The filestem LIST fixes dispatch order deterministically: process_files
    iterates stems in list order (glob order within a single stem is not
    guaranteed).

    Pool-forced legs (D8): the same ValueErrors cross the process boundary
    with type + message intact (pickled re-raise; identity does not survive
    and nothing asserts it)."""
    import numpy as np

    tmpdir, h5kwargs = raw_grid_h5_factory(n_cfg=2)
    pt_path = str(tmp_path / "pt.h5")
    pd.DataFrame(
        {"corr": np.arange(48, dtype=float)}, index=pd.Index([0] * 48, name="t")
    ).to_hdf(pt_path, key="corr", mode="w", format="fixed")

    raw_fs = f"{tmpdir}/{{series}}/{{cfg}}/corr.h5"
    reps = {"series": ["a"]}
    regex = {"cfg": "[0-9]+"}

    # First file raw: the raw template path opens the PyTables file and fails
    # on the missing physics dataset.
    with pytest.raises(ValueError, match="not found in file"):
        load_files_chunked(
            filestem=[raw_fs, pt_path], replacements=reps, regex=regex, **h5kwargs
        )

    with pytest.raises(ValueError, match="not found in file"):
        load_files_chunked(
            filestem=[raw_fs, pt_path], replacements=reps, regex=regex,
            max_workers=2, pool_threshold=0, **h5kwargs
        )

    # First file PyTables: every file routes to pd.read_hdf, which rejects the
    # raw Grid file.
    with pytest.raises(ValueError, match="incompatible with Pandas data types"):
        load_files_chunked(
            filestem=[pt_path, raw_fs], replacements=reps, regex=regex, **h5kwargs
        )

    with pytest.raises(ValueError, match="incompatible with Pandas data types"):
        load_files_chunked(
            filestem=[pt_path, raw_fs], replacements=reps, regex=regex,
            max_workers=2, pool_threshold=0, **h5kwargs
        )
