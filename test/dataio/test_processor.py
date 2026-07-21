"""Unit tests for pyfm.dataio.processor action functions.

Focuses on the ``time_average`` action's handling of data that has already been
time-averaged (carries a ``t`` column), which is what the contract
``task aggregate --average`` path must consume without assuming ``t1``/``t2``
columns still exist.
"""
import numpy as np
import pandas as pd

from pyfm.dataio import processor as pc


def _raw_twopoint_df(nt: int = 4) -> pd.DataFrame:
    """Build a raw two-point DataFrame with ``t1``/``t2`` index levels."""
    idx = pd.MultiIndex.from_product(
        [["a"], ["100"], ["G5_G5"], range(nt), range(nt)],
        names=["series", "cfg", "gamma", "t1", "t2"],
    )
    vals = [complex(i, j) for i in range(nt) for j in range(nt)]
    return pd.DataFrame({"corr": vals}, index=idx)


class TestTimeAverageSkip:
    def test_skips_when_t_column_present(self):
        """time_average is a no-op when a regular ``t`` column already exists.

        This mirrors pre-averaged contract data: the ``--average`` path must not
        assume ``t1``/``t2`` columns exist.
        """
        df = pd.DataFrame(
            {
                "series": ["a", "a"],
                "cfg": ["100", "100"],
                "gamma": ["G5_G5", "G5_G5"],
                "t": [0, 1],
                "corr": [1.0 + 1j, 2.0 + 2j],
            }
        )
        out = pc.time_average(df.copy(), "corr", "t1", "t2")

        # Unchanged frame: no reshape, no new index level, no KeyError on t1/t2.
        pd.testing.assert_frame_equal(out, df)

    def test_skips_when_t_index_level_present(self):
        """time_average is a no-op when ``t`` is already an index level."""
        df = pd.DataFrame(
            {"corr": [1.0 + 1j, 2.0 + 2j]}, index=pd.Index([0, 1], name="t")
        )
        out = pc.time_average(df.copy(), "corr", "t1", "t2")
        pd.testing.assert_frame_equal(out, df)

    def test_contract_averaged_actions_run_on_preaveraged_data(self):
        """End-to-end: the contract ``average=True`` action set runs cleanly on
        data that already carries a ``t`` column (time_average skipped, ``real``
        applied, index set to series.cfg/gamma/t)."""
        agg_actions = {
            "drop": "perm",
            "real": True,
            "time_average": ["t1", "t2"],
            "index": ["series.cfg", "gamma", "t"],
        }
        rows = [
            {"series": "a", "cfg": "100", "gamma": "G5_G5", "t": t, "corr": 1.0 + 0.5j}
            for t in range(4)
        ]
        df = pd.DataFrame(rows)

        out = pc.execute(df.copy(), dict(agg_actions))

        assert out.index.names == ["series_cfg", "gamma", "t"]
        # ``real`` action applied: imaginary part removed.
        assert np.allclose(out["corr"].apply(np.imag), 0.0)
        assert len(out) == 4


class TestTimeAverageRaw:
    def test_averages_raw_twopoint_array(self):
        """Regression: raw two-point data (``t1``/``t2`` present, no ``t``) is
        still averaged into a single ``t`` index."""
        nt = 4
        df = _raw_twopoint_df(nt)
        out = pc.time_average(df.copy(), "corr", "t1", "t2")

        assert "t" in out.index.names
        # One averaged value per t per (series, cfg, gamma) group.
        assert len(out) == nt
        # All output rows share the constant trace-average of the input matrix.
        expected = np.mean([complex(i, j) for i in range(nt) for j in range(nt)])
        assert np.allclose(out["corr"].to_numpy(), expected)
