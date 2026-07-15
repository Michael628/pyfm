import pandas as pd

from pyfm.dataio import load_files, load_files_chunked
from pyfm.dataio.loader import _detect_hdf5_format


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
    """Output is identical across max_workers ∈ {1,2,3,4,6}."""
    tmpdir, h5kwargs = raw_grid_h5_factory(n_cfg=6)
    fs = f"{tmpdir}/{{series}}/{{cfg}}/corr.h5"
    reps = {"series": ["a"], "cfg": [str(i) for i in range(6)]}
    regex = {"series": "[a-z]", "cfg": "[0-9]+"}
    base = load_files_chunked(filestem=fs, replacements=reps, regex=regex, **h5kwargs)
    for mw in [2, 3, 4, 6]:
        df = load_files_chunked(
            filestem=fs, replacements=reps, regex=regex, max_workers=mw, **h5kwargs
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
