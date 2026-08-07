"""Tests for meson field loading in A2A contractions.

Regression coverage for the mass-shift cache bug in ``iter_meson_fields``:
two mesons that read the same original-mass file but apply different mass
shifts (e.g. l->r and l->s) must each be loaded and shifted independently.
Keying the cache on ``(file, time)`` alone collapsed them, silently applying
only the first meson's shift to both endpoints.
"""
import numpy as np
import pytest

from pyfm.a2a import mesonloader
from pyfm.a2a.mesonloader import iter_meson_fields, clear_meson_cache
from pyfm.a2a.types import DiagramConfig, MesonLoaderConfig, ContractType
from pyfm.domain import Outfile
from pyfm.domain.ops import MassDict


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_meson_cache()
    yield
    clear_meson_cache()


def _meson(updated):
    mass = MassDict.from_dict({"l": 0.01, "r": 0.1, "s": 0.4})
    outfile = Outfile(filestem="/tmp/m01/mf", ext=".h5", good_size=1)
    return MesonLoaderConfig(
        formatting={},
        logging_level="INFO",
        runid="t",
        mass=mass,
        file=outfile,
        evalfile=outfile,
        mass_shift=MesonLoaderConfig.MassShift(original="l", updated=updated),
    )


def _diagram(mesons):
    outfile = Outfile(filestem="/tmp/m01/mf", ext=".h5", good_size=1)
    return DiagramConfig(
        formatting={},
        logging_level="INFO",
        runid="t",
        time=4,
        contraction_type=ContractType.TWOPOINT,
        mesons=mesons,
        outfile=outfile,
        gammas=["G5T_G5"],
        eig_range=DiagramConfig.MesonIndex(max=10, min=0),
        symmetric=True,
    )


def _patch_load_meson(monkeypatch):
    """Record the shift each load is asked to apply; return a marker array."""
    calls = []
    marker = {"r": 0.1, "s": 0.4, None: 0.01}

    def fake_load(file, meson_config, vmax_index, wmax_index, time):
        shift = meson_config.mass_shift.updated
        calls.append(shift)
        return np.array([marker[shift]])

    monkeypatch.setattr(mesonloader, "load_meson", fake_load)
    return calls


def test_distinct_mass_shifts_both_applied(monkeypatch):
    """Two mesons sharing an input file but shifting to r vs s must both shift."""
    calls = _patch_load_meson(monkeypatch)
    diagram = _diagram([_meson("r"), _meson("s")])

    # Both endpoints read the same original-mass (l) file.
    mesonfiles = ("/tmp/m01/mf.h5", "/tmp/m01/mf.h5")
    times = ([slice(0, 1)], [slice(0, 1)])
    contraction = ("e", "e", "e", "e")

    yielded = list(iter_meson_fields(diagram, mesonfiles, times, contraction))

    assert calls == ["r", "s"], (
        f"Expected both l->r and l->s shifts; load_meson was called with {calls}"
    )
    [((_, m1), (_, m2))] = yielded
    assert m1.tolist() == [0.1] and m2.tolist() == [0.4]


def test_identical_shifts_still_share_cache(monkeypatch):
    """Identical shifts on the same file/time must still dedupe via the cache."""
    calls = _patch_load_meson(monkeypatch)
    diagram = _diagram([_meson("r"), _meson("r")])  # both l->r

    mesonfiles = ("/tmp/m01/mf.h5", "/tmp/m01/mf.h5")
    times = ([slice(0, 1)], [slice(0, 1)])
    contraction = ("e", "e", "e", "e")

    yielded = list(iter_meson_fields(diagram, mesonfiles, times, contraction))

    # Loaded once, shared for the second endpoint.
    assert calls == ["r"], f"Expected a single load; got {calls}"
    [((_, m1), (_, m2))] = yielded
    assert m1.tolist() == [0.1] and m2.tolist() == [0.1]
