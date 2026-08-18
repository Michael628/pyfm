"""Tests for pyfm.dataio.writer dict format support."""
import numpy as np
import pandas as pd

from pyfm.dataio import write_files
from pyfm.dataio.converter import frame_to_dict


def _df():
    # Minimal aggregated-style frame: MultiIndex (gamma, t) + 'corr' column.
    idx = pd.MultiIndex.from_product([["g0", "g1"], [0, 1]], names=["gamma", "t"])
    return pd.DataFrame({"corr": [1.0, 2.0, 3.0, 4.0]}, index=idx)


def test_write_dict_default_dict_depth(tmp_path):
    """format='dict' without an explicit dict_depth defaults to 1 (no KeyError)."""
    df = _df()
    out = tmp_path / "out"
    write_files(df, str(out), format="dict")
    assert (tmp_path / "out.npy").exists()
    expected = frame_to_dict(df, 1)
    loaded = np.load(tmp_path / "out.npy", allow_pickle=True).item()
    # dict-of-ndarray: compare keys, then per-key arrays (== would raise on arrays).
    assert set(loaded) == set(expected)
    for k in expected:
        np.testing.assert_array_equal(loaded[k], expected[k])


def test_write_dict_explicit_dict_depth(tmp_path):
    df = _df()
    out = tmp_path / "out"
    write_files(df, str(out), format="dict", dict_depth=0)
    assert (tmp_path / "out.npy").exists()
    loaded = np.load(tmp_path / "out.npy", allow_pickle=True)
    np.testing.assert_array_equal(loaded, frame_to_dict(df, 0))
