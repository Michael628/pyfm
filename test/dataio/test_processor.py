"""Unit tests for pyfm.dataio.processor action functions.

Focuses on the ``time_average`` action's handling of data that has already been
time-averaged (carries a ``t`` column), which is what the contract
``task aggregate --average`` path must consume without assuming ``t1``/``t2``
columns still exist.
"""
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

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

    def test_skips_when_none_named_level_has_sequential_ints(self):
        """time_average is a no-op when a None-named index level has
        sequential integer values 0..nt-1 (mixed t/dt concatenation).

        When multiple files are concatenated — some with 't' and others with
        'dt' as the time index name — pandas resolves the conflict by
        naming the merged level None.  The guard detects this and renames
        it to 't', so the existing skip check fires.
        """
        nt = 4
        idx = pd.MultiIndex.from_product(
            [["a"], ["100"], ["G5_G5"], range(nt)],
            names=["series", "cfg", "gamma", None],
        )
        rows = [
            {"series": "a", "cfg": "100", "gamma": "G5_G5", None: t, "corr": 1.0 + 0.5j}
            for t in range(nt)
        ]
        df = pd.DataFrame(rows, index=idx)

        out = pc.time_average(df.copy(), "corr", "t1", "t2")

        # The None-named level was renamed to 't', so time_average skipped.
        assert out.index.names == ["series", "cfg", "gamma", "t"]
        assert len(out) == nt
        # Data unchanged (skip path) but index name was fixed.
        assert out["corr"].tolist() == df["corr"].tolist()


class TestTimeAverageRaw:
    def test_averages_raw_twopoint_array(self):
        """Regression: raw two-point data (``t1``/``t2`` present, no ``t``) is
        still averaged into a single ``t`` index."""
        nt = 4
        df = _raw_twopoint_df(nt)
        out = pc.time_average(df.copy(), "corr", "t1", "t2")

        assert out.index.names == ["series", "cfg", "gamma", "t"]
        # One averaged value per t per (series, cfg, gamma) group.
        assert len(out) == nt
        # All output rows share the constant trace-average of the input matrix.
        expected = np.mean([complex(i, j) for i in range(nt) for j in range(nt)])
        assert np.allclose(out["corr"].to_numpy(), expected)


class TestTimeAverageNormalization:
    """Regression tests: time_average always normalizes the innermost index
    name to 't', regardless of what group_apply returns."""

    def test_normalizes_none_index_name(self):
        """Regression: time_average restores 't' when group_apply loses the
        name to None (pandas version-dependent behavior)."""
        nt = 4
        df = _raw_twopoint_df(nt)
        _real_group_apply = pc.group_apply

        def corrupt_none(*args, **kwargs):
            result = _real_group_apply(*args, **kwargs)
            result.index = result.index.set_names(None, level=-1)
            return result

        with patch.object(pc, "group_apply", side_effect=corrupt_none):
            out = pc.time_average(df.copy(), "corr", "t1", "t2")

        assert out.index.names[-1] == "t"

    def test_normalizes_dt_index_name(self):
        """Regression: time_average normalizes legacy 'dt' to 't'."""
        nt = 4
        df = _raw_twopoint_df(nt)
        _real_group_apply = pc.group_apply

        def corrupt_dt(*args, **kwargs):
            result = _real_group_apply(*args, **kwargs)
            result.index = result.index.set_names("dt", level=-1)
            return result

        with patch.object(pc, "group_apply", side_effect=corrupt_dt):
            out = pc.time_average(df.copy(), "corr", "t1", "t2")

        assert out.index.names[-1] == "t"


class TestAverageUnnamedWarning:
    """Tests for average() warning on unnamed index levels."""

    def test_warns_on_unnamed_index_levels(self):
        """average() warns when a None-named index level is excluded from
        the groupby key list."""
        idx = pd.MultiIndex.from_tuples(
            [(0, 0, "a"), (0, 1, "a"), (1, 0, "a"), (1, 1, "a")],
            names=["t1", None, "gamma"],
        )
        df = pd.DataFrame({"corr": [1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j]}, index=idx)

        mock_logger = MagicMock()
        with patch("pyfm.dataio.processor.utils.get_logger", return_value=mock_logger):
            pc.average(df, "corr", "t1")

        mock_logger.warning.assert_called_once()
        msg = mock_logger.warning.call_args[0][0]
        assert "unnamed" in msg.lower()
