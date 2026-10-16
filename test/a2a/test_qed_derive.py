"""Tests for the QED block accessor (derive layer).

A synthetic hadrons_qed_mf surface (random blocks in the producer's
grammar, all fermion worlds inside the matrix axes, photon worlds as
files) is read back through QEDBlockAccessor and checked against
explicit reference constructions: per-world axis slicing, the ported
pair-basis derivations, ket refolds, j1l component sums, loud shape
guards, and cache-key completeness across content-affecting axes (the
d38ef6c lesson).
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from pyfm.a2a.qed_derive import QEDBlockAccessor, load_block
from pyfm.a2a.sib_derive import SCALAR_GAMMA, eval_pairs, w_rows
from pyfm.domain import MassDict, Outfile
from pyfm.domain.blockkeys import (
    EM_FAMILY_JA,
    EM_FAMILY_J1L,
    em_file_key,
    mass_token,
    vec_file_key,
)

N_EIG = 2
NT = 4
NOISE = 2
MASSES = {"l": 0.01, "u": 0.02}
DEFL = "l"
LAM = np.array([0.7, 1.9])
VEC_GAMMA = "GX_GX"
MOM = "1 0 0"
MOM2 = "0 2 0"
SEEDS = ("qed_a_20_n0", "qed_a_20_n1")
VEC_FAMILY = "vl"


def _write_block(path, dataset, arr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dt = np.dtype([("re", np.float64), ("im", np.float64)])
    c = np.ascontiguousarray(arr.astype(np.complex128))
    with h5py.File(path, "w") as f:
        grp = f.create_group(dataset)
        grp.create_dataset(
            "a2aMatrix",
            data=np.stack([c.real, c.imag], axis=-1).view(dt).reshape(c.shape),
        )


def _ref_lp(ll_raw, tab, world, mass_label):
    w = w_rows(LAM, MASSES[mass_label])
    tab_w = tab[:, :, 3 * world : 3 * world + 3]
    out = np.zeros((NT, ll_raw.shape[1], NT * 3), dtype=np.complex128)
    for t in range(NT):
        for s in range(NT):
            out[t, :, 3 * s : 3 * s + 3] = ll_raw[t] @ (
                tab_w[s] * w[:, None]
            )
    return out


def _ref_np(nl_raw, tab, world_n, world_hp, mass_label):
    w = w_rows(LAM, MASSES[mass_label])
    tab_w = tab[:, :, 3 * world_hp : 3 * world_hp + 3]
    nl_w = nl_raw[:, 3 * world_n : 3 * world_n + 3, :]
    out = np.zeros((NT, 3, NT * 3), dtype=np.complex128)
    for t in range(NT):
        for s in range(NT):
            out[t, :, 3 * s : 3 * s + 3] = nl_w[t] @ (
                tab_w[s] * w[:, None]
            )
    return out


class _Surface:
    """Synthetic qed_mf output in the producer's grammar."""

    def __init__(self, root, seed=11):
        self.root = Path(root)

        rng = np.random.default_rng(seed)

        def c(*shape):
            return rng.normal(size=shape) + 1j * rng.normal(size=shape)

        self.ll_raw = c(NT, 2 * N_EIG, 2 * N_EIG)
        self.nl_raw = c(NT, 3 * NOISE, 2 * N_EIG)
        self.tab = c(NT, 2 * N_EIG, 3 * NOISE)
        self.pure_lh = {
            (w, m): c(NT, 2 * N_EIG, NT * 3)
            for w in range(NOISE)
            for m in MASSES
        }
        self.pure_nh = {
            (a, b, m): c(NT, 3, NT * 3)
            for a in range(NOISE)
            for b in range(NOISE)
            for m in MASSES
        }
        self.evals_path = self.root / "evals.20.h5"
        with h5py.File(self.evals_path, "w") as f:
            f.create_dataset("evals", data=(LAM**2).astype(np.float64))
        # Deterministic per-content-key scale so different momenta /
        # em-worlds hold genuinely different data (cache-key tests).
        self.scales: dict = {}

    def _scale(self, key):
        if key not in self.scales:
            n = len(self.scales)
            self.scales[key] = (1.0 + 0.3j) ** n
        return self.scales[key]

    def block_path(self, family, leg_pair, mass_label, em_world, file_key):
        stem = self.root / (
            f"qed/{family}_{leg_pair}{mass_token(mass_label)}"
            f"_a{'' if em_world is None else f'_em{em_world}'}"
        )
        return f"{stem}.20/{file_key}.h5"

    def write_tab(self):
        _write_block(
            self.root / f"qed/tab_a.20/{SCALAR_GAMMA}_0_0_0.h5",
            f"{SCALAR_GAMMA}_0_0_0",
            self.tab,
        )

    def write_vec(self, family, gamma, mom):
        ev = eval_pairs(LAM, MASSES[DEFL])
        fk = vec_file_key(gamma, mom)
        sc = self._scale(("vec", gamma, mom))
        _write_block(
            self.block_path(family, "ll", None, None, fk), fk,
            sc * self.ll_raw / ev[None, None, :],
        )
        _write_block(
            self.block_path(family, "nl", None, None, fk), fk,
            sc * self.nl_raw / ev[None, None, :],
        )
        for m in MASSES:
            lh = np.concatenate(
                [
                    _ref_lp(self.ll_raw, self.tab, w, m)
                    + self.pure_lh[(w, m)]
                    for w in range(NOISE)
                ],
                axis=-1,
            )
            _write_block(
                self.block_path(family, "lh", m, None, fk), fk, sc * lh
            )
            nh = np.concatenate(
                [
                    np.concatenate(
                        [
                            _ref_np(self.nl_raw, self.tab, a, b, m)
                            + self.pure_nh[(a, b, m)]
                            for b in range(NOISE)
                        ],
                        axis=-1,
                    )
                    for a in range(NOISE)
                ],
                axis=1,
            )
            _write_block(
                self.block_path(family, "nh", m, None, fk), fk, sc * nh
            )

    def write_em(self, family, em_world):
        """EM blocks for one photon world (j1l writes 4 summed channels)."""
        ev = eval_pairs(LAM, MASSES[DEFL])
        seed = SEEDS[em_world]
        sc = self._scale(("em", family, em_world))
        blocks = {
            ("ll", None): sc * self.ll_raw / ev[None, None, :],
            ("nl", None): sc * self.nl_raw / ev[None, None, :],
        }
        for m in MASSES:
            blocks[("lh", m)] = sc * np.concatenate(
                [
                    _ref_lp(self.ll_raw, self.tab, w, m)
                    + self.pure_lh[(w, m)]
                    for w in range(NOISE)
                ],
                axis=-1,
            )
            blocks[("nh", m)] = sc * np.concatenate(
                [
                    np.concatenate(
                        [
                            _ref_np(self.nl_raw, self.tab, a, b, m)
                            + self.pure_nh[(a, b, m)]
                            for b in range(NOISE)
                        ],
                        axis=-1,
                    )
                    for a in range(NOISE)
                ],
                axis=1,
            )

        def write(leg_pair, mass_label, arr):
            if family == EM_FAMILY_J1L:
                # four component files, each arr/4 — the inline sum
                # must restore arr
                for k in range(4):
                    fk = em_file_key(family, seed, k)
                    _write_block(
                        self.block_path(
                            family, leg_pair, mass_label, em_world, fk
                        ),
                        fk,
                        arr / 4.0,
                    )
            else:
                fk = em_file_key(family, seed, None)
                _write_block(
                    self.block_path(family, leg_pair, mass_label, em_world, fk),
                    fk,
                    arr,
                )

        for (leg_pair, mass_label), arr in blocks.items():
            write(leg_pair, mass_label, arr)


def _accessor(root, seeds=SEEDS):
    return QEDBlockAccessor(
        blocks=Outfile(
            str(Path(root) / "qed/{family}_{leg_pair}{mass}_a{em_index}"),
            ".20/{file_key}.h5",
            1,
        ),
        tab=Outfile(str(Path(root) / "qed/tab_a"), ".20/{gamma}_0_0_0.h5", 1),
        evalfile=Outfile(str(Path(root) / "evals"), ".20.h5", 1),
        mass=MassDict.from_dict(MASSES),
        defl_mass=DEFL,
        noise=NOISE,
        em_seeds=seeds,
    )


@pytest.fixture()
def surface(tmp_path):
    s = _Surface(tmp_path)
    s.write_tab()
    s.write_vec(VEC_FAMILY, VEC_GAMMA, MOM)
    return s


class TestLoadBlock:
    def test_round_trip(self, tmp_path):
        arr = np.arange(2 * 3 * 4).reshape(2, 3, 4) * (1 + 1j)
        _write_block(tmp_path / "b.h5", "some_key", arr)
        out = load_block(str(tmp_path / "b.h5"), "some_key")
        np.testing.assert_array_equal(out, arr)

    def test_wrong_dataset_raises(self, tmp_path):
        _write_block(tmp_path / "b.h5", "some_key", np.zeros((2, 3, 4)))
        with pytest.raises(ValueError, match="a2aMatrix"):
            load_block(str(tmp_path / "b.h5"), "other_key")


class TestGeometry:
    def test_nt_from_tab(self, surface, tmp_path):
        acc = _accessor(tmp_path)
        assert acc.nt() == NT

    def test_cols_pinning(self, surface, tmp_path):
        acc = _accessor(tmp_path)
        acc._nt = NT  # geometry-only test: bypass the tab read
        block = np.arange(2 * 3 * (NT * 3)).reshape(2, 3, NT * 3) * 1.0
        out = acc.cols(block, 2)
        assert out.shape == (2, 3, 3)
        np.testing.assert_array_equal(
            out, block.reshape(2, 3, NT, 3)[..., 2, :]
        )

    def test_cols_guards(self, tmp_path):
        acc = _accessor(tmp_path)
        acc._nt = NT
        with pytest.raises(ValueError, match="per-world width"):
            acc.cols(np.zeros((2, 3, NT * 3 + 1)), 0)
        with pytest.raises(ValueError, match="outside"):
            acc.cols(np.zeros((2, 3, NT * 3)), NT)

    def test_eta_world_slicing_and_guards(self, tmp_path):
        acc = _accessor(tmp_path)
        acc._nt = NT
        arr = np.arange(NT * 3 * NOISE * 5).reshape(NT, 3 * NOISE, 5) * 1.0
        np.testing.assert_array_equal(acc._eta_world(arr, 1), arr[:, 3:6, :])
        with pytest.raises(ValueError, match="3\\*noise"):
            acc._eta_world(np.zeros((NT, 3 * NOISE + 1, 5)), 0)
        with pytest.raises(ValueError, match="outside"):
            acc._eta_world(arr, NOISE)

    def test_h_world_slicing_and_guards(self, tmp_path):
        acc = _accessor(tmp_path)
        acc._nt = NT
        w3 = 3 * NT
        arr = np.arange(2 * 3 * w3 * NOISE).reshape(2, 3, w3 * NOISE) * 1.0
        np.testing.assert_array_equal(
            acc._h_world(arr, 1), arr[..., w3 : 2 * w3]
        )
        with pytest.raises(ValueError, match="3\\*nt\\*noise"):
            acc._h_world(np.zeros((2, 3, w3 * NOISE + 1)), 0)
        with pytest.raises(ValueError, match="outside"):
            acc._h_world(arr, NOISE)


class TestRefoldsAndDerivations:
    def test_ll_refold(self, surface, tmp_path):
        acc = _accessor(tmp_path)
        sc = surface.scales[("vec", VEC_GAMMA, MOM)]
        for m in MASSES:
            np.testing.assert_allclose(
                acc.ll(VEC_FAMILY, VEC_GAMMA, m, mom=MOM),
                sc * surface.ll_raw
                / eval_pairs(LAM, MASSES[m])[None, None, :],
                rtol=1e-12,
            )

    def test_nl_per_world(self, surface, tmp_path):
        acc = _accessor(tmp_path)
        sc = surface.scales[("vec", VEC_GAMMA, MOM)]
        for w in range(NOISE):
            for m in MASSES:
                np.testing.assert_allclose(
                    acc.nl(VEC_FAMILY, VEC_GAMMA, w, m, mom=MOM),
                    sc * surface.nl_raw[:, 3 * w : 3 * w + 3, :]
                    / eval_pairs(LAM, MASSES[m])[None, None, :],
                    rtol=1e-12,
                )

    def test_pure_lh_per_world(self, surface, tmp_path):
        acc = _accessor(tmp_path)
        sc = surface.scales[("vec", VEC_GAMMA, MOM)]
        for w in range(NOISE):
            for m in MASSES:
                np.testing.assert_allclose(
                    acc.pure_lh(VEC_FAMILY, VEC_GAMMA, w, m, mom=MOM),
                    sc * surface.pure_lh[(w, m)],
                    rtol=1e-10,
                )

    def test_pure_nh_per_world_pair(self, surface, tmp_path):
        acc = _accessor(tmp_path)
        sc = surface.scales[("vec", VEC_GAMMA, MOM)]
        for a in range(NOISE):
            for b in range(NOISE):
                for m in MASSES:
                    np.testing.assert_allclose(
                        acc.pure_nh(VEC_FAMILY, VEC_GAMMA, a, b, m, mom=MOM),
                        sc * surface.pure_nh[(a, b, m)],
                        rtol=1e-10,
                    )


class TestEMFamilies:
    def test_ja_blocks(self, surface, tmp_path):
        surface.write_em(EM_FAMILY_JA, 0)
        acc = _accessor(tmp_path)
        sc = surface.scales[("em", EM_FAMILY_JA, 0)]
        for w in range(NOISE):
            np.testing.assert_allclose(
                acc.nl(EM_FAMILY_JA, None, w, "l", em_world=0),
                sc * surface.nl_raw[:, 3 * w : 3 * w + 3, :]
                / eval_pairs(LAM, MASSES["l"])[None, None, :],
                rtol=1e-12,
            )
        np.testing.assert_allclose(
            acc.pure_lh(EM_FAMILY_JA, None, 0, "u", em_world=0),
            sc * surface.pure_lh[(0, "u")],
            rtol=1e-10,
        )

    def test_j1l_component_sum(self, surface, tmp_path):
        surface.write_em(EM_FAMILY_J1L, 1)
        acc = _accessor(tmp_path)
        sc = surface.scales[("em", EM_FAMILY_J1L, 1)]
        # four channel files were written as arr/4 — the inline sum
        # restores arr
        np.testing.assert_allclose(
            acc.ll(EM_FAMILY_J1L, None, "l", em_world=1),
            sc * surface.ll_raw / eval_pairs(LAM, MASSES["l"])[None, None, :],
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            acc.pure_nh(EM_FAMILY_J1L, None, 1, 0, "l", em_world=1),
            sc * surface.pure_nh[(1, 0, "l")],
            rtol=1e-10,
        )

    def test_em_requires_em_world(self, surface, tmp_path):
        acc = _accessor(tmp_path)
        with pytest.raises(ValueError, match="em_world"):
            acc.ll(EM_FAMILY_JA, None, "l")

    def test_vec_requires_mom(self, surface, tmp_path):
        acc = _accessor(tmp_path)
        with pytest.raises(ValueError, match="momentum"):
            acc.ll(VEC_FAMILY, VEC_GAMMA, "l")


class TestCacheAxes:
    def test_momentum_distinguishes_content(self, surface, tmp_path):
        surface.write_vec(VEC_FAMILY, VEC_GAMMA, MOM2)
        acc = _accessor(tmp_path)
        a = acc.ll(VEC_FAMILY, VEC_GAMMA, "l", mom=MOM)
        b = acc.ll(VEC_FAMILY, VEC_GAMMA, "l", mom=MOM2)
        assert a is not b
        assert not np.allclose(a, b)

    def test_mass_distinguishes_content(self, surface, tmp_path):
        acc = _accessor(tmp_path)
        a = acc.pure_lh(VEC_FAMILY, VEC_GAMMA, 0, "l", mom=MOM)
        b = acc.pure_lh(VEC_FAMILY, VEC_GAMMA, 0, "u", mom=MOM)
        assert not np.allclose(a, b)

    def test_repeat_call_is_cached(self, surface, tmp_path):
        acc = _accessor(tmp_path)
        assert (
            acc.pure_lh(VEC_FAMILY, VEC_GAMMA, 0, "l", mom=MOM)
            is acc.pure_lh(VEC_FAMILY, VEC_GAMMA, 0, "l", mom=MOM)
        )

    def test_em_world_distinguishes_content(self, surface, tmp_path):
        surface.write_em(EM_FAMILY_JA, 0)
        surface.write_em(EM_FAMILY_JA, 1)
        acc = _accessor(tmp_path)
        a = acc.ll(EM_FAMILY_JA, None, "l", em_world=0)
        b = acc.ll(EM_FAMILY_JA, None, "l", em_world=1)
        assert not np.allclose(a, b)


class TestPathTokens:
    def test_unresolved_stem_raises(self, tmp_path):
        with h5py.File(Path(tmp_path) / "evals.20.h5", "w") as f:
            f.create_dataset("evals", data=(LAM**2).astype(np.float64))
        acc = QEDBlockAccessor(
            blocks=Outfile(
                str(Path(tmp_path) / "qed/{family}_{leg_pair}{mass}_{series}"),
                ".20/{file_key}.h5",
                1,
            ),
            tab=Outfile(
                str(Path(tmp_path) / "qed/tab_a"), ".20/{gamma}_0_0_0.h5", 1
            ),
            evalfile=Outfile(str(Path(tmp_path) / "evals"), ".20.h5", 1),
            mass=MassDict.from_dict(MASSES),
            defl_mass=DEFL,
            noise=NOISE,
            em_seeds=SEEDS,
        )
        with pytest.raises(ValueError, match="unresolved"):
            acc.ll(VEC_FAMILY, VEC_GAMMA, "l", mom=MOM)
