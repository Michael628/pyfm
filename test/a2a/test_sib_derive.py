"""Tests for the SIB pair-basis derivation layer (pyfm/a2a/sib_derive.py).

Synthetic split-noise fixtures: random raw blocks, the writer's ket-side eval
folding applied where the producer applies it, HDF5 files laid out in the
producer's path grammar. The accessor's derived/refolded/pure-high blocks are
checked against hand-computed formulas — the pyfm-side analog of
../HadronsMILC/test/compare_sib_pair_identity.py's checks 2-4 (by
construction: the "stored" h-side blocks are built as derived + delta, so
pure_lh/pure_nh must return exactly delta).
"""
import numpy as np
import pytest
import h5py

from pyfm.a2a.sib_derive import (
    SIBBlockAccessor,
    eval_pairs,
    ket_refold,
    load_evals,
    w_rows,
)
from pyfm.domain import MassDict, Outfile

N_EIG = 2
NT = 4
N_SLICES = 2
T0 = 1
T_STEP = 2
NOISE = 2
MASSES = {"l": 0.01, "u": 0.02}
DEFL = "l"
LAM = np.array([0.7, 1.9])
VECTOR_GAMMA = "GX_GX"
SCALAR_GAMMA = "G1_G1"
GAMMAS = [SCALAR_GAMMA, VECTOR_GAMMA]


def _write_block(path, gamma, arr):
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dt = np.dtype([("re", np.float64), ("im", np.float64)])
    c = np.ascontiguousarray(arr.astype(np.complex128))
    with h5py.File(path, "w") as f:
        grp = f.create_group(f"{gamma}_0_0_0")
        grp.create_dataset(
            "a2aMatrix",
            data=np.stack([c.real, c.imag], axis=-1).view(dt).reshape(c.shape),
        )


def _ref_lp(raw_ll, tab, mass):
    """Reference derived lp: column (s, c) = raw_ll[t] @ (w(m)*tab[t_s])[:, c]."""
    w = w_rows(LAM, mass)
    out = np.zeros((NT, raw_ll.shape[1], N_SLICES * 3), dtype=np.complex128)
    for t in range(NT):
        for s in range(N_SLICES):
            out[t, :, 3 * s : 3 * s + 3] = raw_ll[t] @ (
                tab[T0 + s * T_STEP] * w[:, None]
            )
    return out


def _ref_np(nl_raw, tab_hp, mass):
    """Reference derived np: column (s, c) = nl_raw[t] @ (w(m)*tab_hp[t_s])[:, c]."""
    w = w_rows(LAM, mass)
    out = np.zeros((NT, 3, N_SLICES * 3), dtype=np.complex128)
    for t in range(NT):
        for s in range(N_SLICES):
            out[t, :, 3 * s : 3 * s + 3] = nl_raw[t] @ (
                tab_hp[T0 + s * T_STEP] * w[:, None]
            )
    return out


class _World:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.tab = rng.normal(size=(NT, 2 * N_EIG, 3)) + 1j * rng.normal(
            size=(NT, 2 * N_EIG, 3)
        )
        self.nl_raw = rng.normal(size=(NT, 3, 2 * N_EIG)) + 1j * rng.normal(
            size=(NT, 3, 2 * N_EIG)
        )

    def nl_raw_for(self, gamma):
        if gamma == SCALAR_GAMMA:
            return np.conjugate(self.tab).transpose(0, 2, 1)
        return self.nl_raw


@pytest.fixture()
def surface(tmp_path):
    """Write a complete synthetic split-noise surface.

    Stored blocks follow the producer's conventions: ll/nl carry the ket-side
    eval folding at the deflation mass; tab is unweighted; lh/nh are written
    as derived-plus-delta so the pure-high differences must equal delta.
    """
    rng = np.random.default_rng(42)
    worlds = [_World(seed) for seed in range(NOISE)]
    fold = eval_pairs(LAM, MASSES[DEFL])[None, None, :]
    raw_ll = {
        g: rng.normal(size=(NT, 2 * N_EIG, 2 * N_EIG))
        + 1j * rng.normal(size=(NT, 2 * N_EIG, 2 * N_EIG))
        for g in GAMMAS
    }
    delta_lh = {
        (g, w, m): rng.normal(size=(NT, 2 * N_EIG, N_SLICES * 3))
        + 1j * rng.normal(size=(NT, 2 * N_EIG, N_SLICES * 3))
        for g in GAMMAS
        for w in range(NOISE)
        for m in MASSES
    }
    delta_nh = {
        (g, i, j, m): rng.normal(size=(NT, 3, N_SLICES * 3))
        + 1j * rng.normal(size=(NT, 3, N_SLICES * 3))
        for g in GAMMAS
        for i in range(NOISE)
        for j in range(NOISE)
        for m in MASSES
    }

    stem = str(tmp_path / "sib")
    blocks = Outfile(
        filestem=f"{stem}/{{leg_pair}}{{mass}}_a{{n_index}}{{hp_index}}",
        ext=f".20/{{gamma}}_0_0_0.h5",
        good_size=1,
    )
    tab_out = Outfile(
        filestem=f"{stem}/tab_a{{n_index}}",
        ext=".20/G1_G1_0_0_0.h5",
        good_size=1,
    )
    evalfile = Outfile(filestem=f"{stem}/evals", ext=".20.h5", good_size=1)

    from pathlib import Path

    Path(stem).mkdir(parents=True, exist_ok=True)
    with h5py.File(evalfile.filename, "w") as f:
        f.create_dataset("evals", data=LAM**2)

    for g in GAMMAS:
        _write_block(
            blocks.filename.format(
                leg_pair="ll", mass="", gamma=g, n_index="", hp_index=""
            ),
            g,
            raw_ll[g] / fold,
        )
        for w, world in enumerate(worlds):
            if g != SCALAR_GAMMA:
                _write_block(
                    blocks.filename.format(
                        leg_pair="nl",
                        mass="",
                        gamma=g,
                        n_index=f"_n{w}",
                        hp_index="",
                    ),
                    g,
                    world.nl_raw / fold,
                )
            for m in MASSES:
                lp = _ref_lp(raw_ll[g], world.tab, MASSES[m])
                _write_block(
                    blocks.filename.format(
                        leg_pair="lh",
                        mass=f"_m{m}",
                        gamma=g,
                        n_index="",
                        hp_index=f"_n{w}",
                    ),
                    g,
                    lp + delta_lh[(g, w, m)],
                )
                for w_hp, world_hp in enumerate(worlds):
                    np_ = _ref_np(
                        world.nl_raw_for(g), world_hp.tab, MASSES[m]
                    )
                    _write_block(
                        blocks.filename.format(
                            leg_pair="nh",
                            mass=f"_m{m}",
                            gamma=g,
                            n_index=f"_n{w}",
                            hp_index=f"_n{w_hp}",
                        ),
                        g,
                        np_ + delta_nh[(g, w, w_hp, m)],
                    )

    for w, world in enumerate(worlds):
        _write_block(
            tab_out.filename.format(gamma=SCALAR_GAMMA, n_index=f"_n{w}"),
            SCALAR_GAMMA,
            world.tab,
        )

    accessor = SIBBlockAccessor(
        blocks=blocks,
        tab=tab_out,
        evalfile=evalfile,
        mass=MassDict.from_dict(MASSES),
        defl_mass=DEFL,
        noise=NOISE,
        t0=T0,
        t_step=T_STEP,
        n_slices=N_SLICES,
    )
    return accessor, worlds, raw_ll, delta_lh, delta_nh


class TestWeights:
    def test_w_rows_diagonal_form(self):
        w = w_rows(LAM, 0.01)
        m = 0.02
        invmag = 1.0 / (m * m + LAM**2)
        assert np.allclose(w[0::2], invmag * (m - 1j * LAM))
        assert np.allclose(w[1::2], invmag * (m + 1j * LAM))

    def test_eval_pairs_parity(self):
        ev = eval_pairs(LAM, 0.01)
        assert np.allclose(ev[0::2], 0.02 + 1j * LAM)
        assert np.allclose(ev[1::2], 0.02 - 1j * LAM)

    def test_ket_refold_round_trip_and_ratio(self):
        assert np.allclose(ket_refold(LAM, 0.01, 0.01), 1.0)
        ratio = ket_refold(LAM, 0.01, 0.02)
        expected = eval_pairs(LAM, 0.01) / eval_pairs(LAM, 0.02)
        assert np.allclose(ratio, expected)

    def test_load_evals_takes_sqrt(self, tmp_path):
        path = str(tmp_path / "evals.20.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("evals", data=LAM**2)
        assert np.allclose(np.asarray(load_evals(path)), LAM)

    def test_load_evals_loud_shape_guard(self, tmp_path):
        path = str(tmp_path / "evals.20.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("evals", data=(LAM**2).reshape(1, -1))
        with pytest.raises(ValueError, match="1-d"):
            load_evals(path)


class TestGeometry:
    def test_slice_of_window(self, surface):
        accessor, *_ = surface
        # window: t = 1 (s=0), t = 3 (s=1); everything else outside.
        assert accessor.slice_of(1) == 0
        assert accessor.slice_of(3) == 1
        assert accessor.slice_of(0) is None
        assert accessor.slice_of(2) is None
        assert accessor.slice_of(4) is None
        assert accessor.window_times() == [1, 3]

    def test_cols_selection_and_none(self, surface):
        accessor, *_ = surface
        block = np.arange(2 * N_SLICES * 3, dtype=np.complex128).reshape(
            1, 2, N_SLICES * 3
        )
        sel = accessor.cols(block, T0)
        assert np.allclose(sel, block[..., 0:3])
        sel1 = accessor.cols(block, T0 + T_STEP)
        assert np.allclose(sel1, block[..., 3:6])
        assert accessor.cols(block, 0) is None

    def test_cols_loud_shape_guard(self, surface):
        accessor, *_ = surface
        bad = np.zeros((1, 2, 7), dtype=np.complex128)
        with pytest.raises(ValueError, match="batch window"):
            accessor.cols(bad, T0)


class TestRefold:
    def test_ll_identity_at_defl_and_refolded(self, surface):
        accessor, _, raw_ll, _, _ = surface
        for g in GAMMAS:
            stored = raw_ll[g] / eval_pairs(LAM, MASSES[DEFL])[None, None, :]
            assert np.allclose(np.asarray(accessor.ll(g, DEFL)), stored)
            ratio = ket_refold(LAM, MASSES[DEFL], MASSES["u"])
            assert np.allclose(
                np.asarray(accessor.ll(g, "u")), stored * ratio[None, None, :]
            )

    def test_vector_nl_refolded(self, surface):
        accessor, worlds, _, _, _ = surface
        g = VECTOR_GAMMA
        w = worlds[1]
        stored = w.nl_raw / eval_pairs(LAM, MASSES[DEFL])[None, None, :]
        assert np.allclose(np.asarray(accessor.nl(g, 1, DEFL)), stored)
        ratio = ket_refold(LAM, MASSES[DEFL], MASSES["u"])
        assert np.allclose(
            np.asarray(accessor.nl(g, 1, "u")), stored * ratio[None, None, :]
        )

    def test_scalar_nl_derived(self, surface):
        accessor, worlds, _, _, _ = surface
        ev_l = eval_pairs(LAM, MASSES[DEFL])
        ev_u = eval_pairs(LAM, MASSES["u"])
        for w_i, world in enumerate(worlds):
            expected_l = np.conjugate(world.tab).transpose(0, 2, 1) / ev_l[
                None, None, :
            ]
            expected_u = np.conjugate(world.tab).transpose(0, 2, 1) / ev_u[
                None, None, :
            ]
            assert np.allclose(
                np.asarray(accessor.nl(SCALAR_GAMMA, w_i, DEFL)), expected_l
            )
            assert np.allclose(
                np.asarray(accessor.nl(SCALAR_GAMMA, w_i, "u")), expected_u
            )

    def test_mass_cache_is_distinct(self, surface):
        accessor, *_ = surface
        a = accessor.ll(SCALAR_GAMMA, DEFL)
        b = accessor.ll(SCALAR_GAMMA, "u")
        assert a is not b


class TestDerivedBlocks:
    def test_pure_lh_returns_delta(self, surface):
        """Harness check 2 analog: stored lh = derived lp + delta."""
        accessor, worlds, raw_ll, delta_lh, _ = surface
        for g in GAMMAS:
            for w in range(NOISE):
                for m in MASSES:
                    assert np.allclose(
                        np.asarray(accessor.pure_lh(g, w, m)),
                        delta_lh[(g, w, m)],
                        atol=1e-10,
                    )

    def test_pure_nh_returns_delta(self, surface):
        """Harness check 3 analog: stored nh = derived np + delta."""
        accessor, worlds, _, _, delta_nh = surface
        for g in GAMMAS:
            for i in range(NOISE):
                for j in range(NOISE):
                    for m in MASSES:
                        assert np.allclose(
                            np.asarray(
                                accessor.pure_nh(g, i, j, m)
                            ),
                            delta_nh[(g, i, j, m)],
                            atol=1e-10,
                        )

    def test_derived_lp_reference_formula(self, surface):
        accessor, worlds, raw_ll, _, _ = surface
        for g in GAMMAS:
            for w, world in enumerate(worlds):
                for m in MASSES:
                    lp = accessor._derived_lp(g, w, m)
                    assert np.allclose(
                        np.asarray(lp), _ref_lp(raw_ll[g], world.tab, MASSES[m])
                    )

    def test_derived_np_reference_formula(self, surface):
        accessor, worlds, _, _, _ = surface
        for g in GAMMAS:
            for i in range(NOISE):
                for j in range(NOISE):
                    for m in MASSES:
                        np_ = accessor._derived_np(g, i, j, m)
                        assert np.allclose(
                            np.asarray(np_),
                            _ref_np(
                                worlds[i].nl_raw_for(g),
                                worlds[j].tab,
                                MASSES[m],
                            ),
                        )


class TestPathGuards:
    def test_unresolved_token_raises(self, tmp_path):
        blocks = Outfile(
            filestem=str(tmp_path / "{leg_pair}{series}"),
            ext=".20/{gamma}_0_0_0.h5",
            good_size=1,
        )
        accessor = SIBBlockAccessor(
            blocks=blocks,
            tab=blocks,
            evalfile=blocks,
            mass=MassDict.from_dict(MASSES),
            defl_mass=DEFL,
            noise=NOISE,
            t0=T0,
            t_step=T_STEP,
            n_slices=N_SLICES,
        )
        with pytest.raises(ValueError, match="unresolved"):
            accessor.block_path(
                leg_pair="ll", mass_token="", gamma=SCALAR_GAMMA
            )

    def test_unknown_mass_label_raises(self, surface):
        accessor, *_ = surface
        with pytest.raises(ValueError, match="mass label"):
            accessor.ll(SCALAR_GAMMA, "nope")
