import itertools

import h5py
import numpy as np
import pandas as pd

from pyfm.dataio.converter import build_hdf5_template, _fill_hdf5_frame
from pyfm.domain.io import LoadH5Config


def _expected_frame(file: h5py.File, cfg: LoadH5Config) -> pd.DataFrame:
    """Build the expected frame via the legacy index algorithm, INDEPENDENTLY of
    `build_hdf5_template`/`_fill_hdf5_frame`.

    Reads config attributes (data) but constructs the MultiIndex + frames inline
    using the same `pd.MultiIndex.from_tuples(itertools.product(...))` algorithm
    legacy `ndarray_to_frame` used, so the D1 split is actually guarded rather
    than compared to itself.
    """
    frames = []
    for key in cfg.datasets:
        dataset_label = cfg.datasets[key][0]
        arr = file[dataset_label][:].view(np.complex128)
        array_config = cfg.array_config[key]
        indices = [array_config.labels[dim] for dim in array_config.order]
        index = pd.MultiIndex.from_tuples(
            itertools.product(*indices), names=array_config.order
        )
        frame = pd.Series(arr.reshape(-1), index=index, name="corr").to_frame()
        frame[cfg.name] = key
        frames.append(frame)
    expected = pd.concat(frames)
    expected.set_index(cfg.name, append=True, inplace=True)
    return expected


def test_template_equivalent_to_legacy(raw_grid_h5_factory):
    """build_hdf5_template + _fill_hdf5_frame matches an independently-built frame."""
    tmpdir, h5kwargs = raw_grid_h5_factory(n_cfg=1)
    cfg = (
        LoadH5Config.create(**h5kwargs)
        .format_data_strings({"series": "a", "cfg": "0"})
    )
    fpath = f"{tmpdir}/a/0/corr.h5"
    with h5py.File(fpath) as f:
        tmpl = build_hdf5_template(cfg)
        candidate = _fill_hdf5_frame(f, tmpl)
        expected = _expected_frame(f, cfg)
    pd.testing.assert_frame_equal(
        candidate.sort_index(), expected.sort_index()
    )


def test_template_reusable_across_files(raw_grid_h5_factory):
    """Build template once; fill many files -> each matches an independent frame."""
    tmpdir, h5kwargs = raw_grid_h5_factory(n_cfg=6)
    cfg = (
        LoadH5Config.create(**h5kwargs)
        .format_data_strings({"series": "a", "cfg": "0"})
    )
    tmpl = build_hdf5_template(cfg)
    for i in range(6):
        fpath = f"{tmpdir}/a/{i}/corr.h5"
        with h5py.File(fpath) as f:
            candidate = _fill_hdf5_frame(f, tmpl)
            expected = _expected_frame(f, cfg)
        pd.testing.assert_frame_equal(
            candidate.sort_index(), expected.sort_index()
        )
