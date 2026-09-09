import os
import tempfile
import shutil

import h5py
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def h5_tmpdir():
    """Ephemeral tmpdir; auto-cleaned."""
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)


@pytest.fixture
def raw_grid_h5_factory(h5_tmpdir):
    """Factory: writes N raw Grid HDF5 files under {series}/{cfg}/corr.h5.

    Returns ``(tmpdir, h5kwargs)`` where ``h5kwargs`` is the LoadH5Config kwargs
    (name, datasets, order, labels) matching the written files.
    """

    def _make(n_cfg=6, n_t=24, n_meson=2, series="a"):
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
            p = os.path.join(h5_tmpdir, series, str(cfg), "corr.h5")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with h5py.File(p, "w") as f:
                for i in range(n_meson):
                    g = f.create_group(f"meson/meson_{i}")
                    ds = g.create_dataset(
                        "corr", data=np.arange(n_t, dtype=np.complex128) + i
                    )
                    ds.attrs["sourceGamma"] = "GX_GX"
        return h5_tmpdir, h5kwargs

    return _make


@pytest.fixture
def pytables_h5_factory(h5_tmpdir):
    """Factory: writes a PyTables-format HDF5 file via pandas to_hdf(format='fixed')."""

    def _make(name="sample", n_rows=48):
        p = os.path.join(h5_tmpdir, f"{name}.h5")
        df = pd.DataFrame(
            {"corr": np.arange(n_rows, dtype=float)},
            index=pd.Index([0] * n_rows, name="t"),
        )
        df.to_hdf(p, key="corr", mode="w", format="fixed")
        return p

    return _make
