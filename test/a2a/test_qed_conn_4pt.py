"""Tests for the QED four-point contraction kernel (qed_conn_4pt).

A synthetic hadrons_qed_mf surface (two momentum elems with distinct
content, photon worlds as files, all fermion worlds in the matrix axes)
is contracted by the kernel — through the accessor, so the reader,
world slicing, derivations, and refolds are exercised end-to-end — and
compared term-by-term against a naive reference implementation of the
sixteen-term decomposition: explicit time/junction/world loops over the
ground-truth arrays. Terms whose world selection is empty at the test's
noise count (k > noise) must be identically zero. Both vertex orderings
(photex and selfen) are covered, plus the execute() dispatch entry.
"""

import itertools
from pathlib import Path

import h5py
import numpy as np

from pyfm.a2a.contractions import qed_conn_4pt
from pyfm.a2a.execution import execute
from pyfm.a2a.sib_derive import SCALAR_GAMMA, eval_pairs, w_rows
from pyfm.a2a.types import (
    ContractType,
    QEDContractConfig,
    QEDDiagramConfig,
    QED_TERM_LABELS,
)
from pyfm.domain import MassDict, OpList, Outfile
from pyfm.domain.blockkeys import (
    EM_FAMILY_JA,
    em_file_key,
    mass_token,
    vec_file_key,
)

N_EIG = 2
NT = 3
NOISE = 2
N_EM = 2
MASSES = {"l": 0.01, "u": 0.02}
DEFL = "l"
LAM = np.array([0.7, 1.9])
VEC_GAMMA = "GX_GX"
MOM = "1 0 0"
CONJ_MOM = "-1 0 0"
SEEDS = ("qed_a_20_n0", "qed_a_20_n1")
VEC_FAMILY = "vl"
# MassDict.to_string("l", True) — the sib convention (prefix stripped;
# the "m" rides the outfile stem text).
MASS_TOKEN = "01"


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


def _raw_set(rng):
    def c(*shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    return {
        "ll_raw": c(NT, 2 * N_EIG, 2 * N_EIG),
        "nl_raw": c(NT, 3 * NOISE, 2 * N_EIG),
        "pure_lh": {
            (w, m): c(NT, 2 * N_EIG, NT * 3)
            for w in range(NOISE)
            for m in MASSES
        },
        "pure_nh": {
            (a, b, m): c(NT, 3, NT * 3)
            for a in range(NOISE)
            for b in range(NOISE)
            for m in MASSES
        },
    }


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


class _KernelSurface:
    """Two-momentum + per-photon-world synthetic surface."""

    def __init__(self, root, seed=23):
        self.root = Path(root)
        self.vec_raw = {
            mom: _raw_set(np.random.default_rng(seed + k))
            for k, mom in enumerate((MOM, CONJ_MOM))
        }
        self.em_raw = {
            w: _raw_set(np.random.default_rng(seed + 100 + w))
            for w in range(N_EM)
        }
        rng = np.random.default_rng(seed + 7)
        self.tab = rng.normal(size=(NT, 2 * N_EIG, 3 * NOISE)) + 1j * rng.normal(
            size=(NT, 2 * N_EIG, 3 * NOISE)
        )
        with h5py.File(self.root / "evals.20.h5", "w") as f:
            f.create_dataset("evals", data=(LAM**2).astype(np.float64))

    def block_path(self, family, leg_pair, mass_label, em_world, file_key):
        suffix = "" if em_world is None else f"_em{em_world}"
        stem = self.root / (
            f"qed/{family}_{leg_pair}{mass_token(mass_label)}"
            f"_a{suffix}.20"
        )
        return stem / f"{file_key}.h5"

    def _write_legs(self, raw, family, em_world, gamma, mom):
        ev = eval_pairs(LAM, MASSES[DEFL])
        if mom is None:
            keys = [em_file_key(family, SEEDS[em_world], None)]
        else:
            keys = [vec_file_key(gamma, mom)]
        for fk in keys:
            _write_block(
                self.block_path(family, "ll", None, em_world, fk), fk,
                raw["ll_raw"] / ev[None, None, :],
            )
            _write_block(
                self.block_path(family, "nl", None, em_world, fk), fk,
                raw["nl_raw"] / ev[None, None, :],
            )
            for m in MASSES:
                lh = np.concatenate(
                    [
                        _ref_lp(raw["ll_raw"], self.tab, w, m)
                        + raw["pure_lh"][(w, m)]
                        for w in range(NOISE)
                    ],
                    axis=-1,
                )
                _write_block(
                    self.block_path(family, "lh", m, em_world, fk), fk, lh
                )
                nh = np.concatenate(
                    [
                        np.concatenate(
                            [
                                _ref_np(raw["nl_raw"], self.tab, a, b, m)
                                + raw["pure_nh"][(a, b, m)]
                                for b in range(NOISE)
                            ],
                            axis=-1,
                        )
                        for a in range(NOISE)
                    ],
                    axis=1,
                )
                _write_block(
                    self.block_path(family, "nh", m, em_world, fk), fk, nh
                )

    def write(self, gamma):
        _write_block(
            self.root / f"qed/tab_a.20/{SCALAR_GAMMA}_0_0_0.h5",
            f"{SCALAR_GAMMA}_0_0_0",
            self.tab,
        )
        for mom in (MOM, CONJ_MOM):
            self._write_legs(self.vec_raw[mom], VEC_FAMILY, None, gamma, mom)
        for w in range(N_EM):
            self._write_legs(self.em_raw[w], EM_FAMILY_JA, w, None, None)


# ----------------------------------------------------------------------
# naive reference
# ----------------------------------------------------------------------
def _matrix_at(raw, kind, world_n, world_hp, mass_label, t, t_next):
    """Ground-truth matrix for one field at (t, t_next)."""
    ev_m = eval_pairs(LAM, MASSES[mass_label])
    if kind == "ll":
        return raw["ll_raw"][t] / ev_m[None, :]
    if kind == "nl":
        return raw["nl_raw"][t][3 * world_n : 3 * world_n + 3, :] / ev_m[None, :]
    if kind == "lh":
        m = raw["pure_lh"][(world_hp, mass_label)][t]
        return m.reshape(m.shape[:-1] + (NT, 3))[..., t_next, :]
    m = raw["pure_nh"][(world_n, world_hp, mass_label)][t]
    return m.reshape(m.shape[:-1] + (NT, 3))[..., t_next, :]


def _ref_term(fields_raw, term, mass_label):
    """Naive C4 for one term, summed over world selections."""
    out = np.zeros((NT,) * 4, dtype=np.complex128)
    dims = [(2 * N_EIG if ch == "l" else 3) for ch in term]
    n_junctions = [j for j, ch in enumerate(term) if ch == "n"]
    for ws in itertools.permutations(range(NOISE), len(n_junctions)):
        world_of = dict(zip(n_junctions, ws))
        for tv in itertools.product(range(NT), repeat=4):
            for jidx in itertools.product(*(range(d) for d in dims)):
                prod = 1 + 0j
                for i in range(4):
                    left_j, right_j = (i - 1) % 4, i
                    if term[left_j] == "l" and term[right_j] == "l":
                        kind, wn, wh = "ll", None, None
                    elif term[left_j] == "n" and term[right_j] == "l":
                        kind, wn, wh = "nl", world_of[left_j], None
                    elif term[left_j] == "l" and term[right_j] == "n":
                        kind, wn, wh = "lh", None, world_of[right_j]
                    else:
                        kind, wn, wh = "nh", world_of[left_j], world_of[right_j]
                    m = _matrix_at(
                        fields_raw[i], kind, wn, wh, mass_label,
                        tv[i], tv[(i + 1) % 4],
                    )
                    prod *= m[jidx[left_j], jidx[right_j]]
                out[tv] += prod
    return out


def _config(tmp_path, ordering):
    def o(stem, ext):
        return Outfile(str(tmp_path / stem), ext, 1)

    diagram = QEDDiagramConfig(
        formatting={},
        logging_level="INFO",
        runid="test",
        contraction_type=ContractType.QED,
        ordering=ordering,
        em=(EM_FAMILY_JA, EM_FAMILY_JA),
        operations=OpList.from_dict({"vec_local": {"mass": ["l"]}}),
        mass=MassDict.from_dict(MASSES),
        blocks=o(
            "qed/{family}_{leg_pair}{mass}_a{em_index}", ".20/{file_key}.h5"
        ),
        tab=o("qed/tab_a", ".20/{gamma}_0_0_0.h5"),
        evalfile=o("evals", ".20.h5"),
        outfile=o("c4_{gamma}_m{mass}_P{momentum}_em{em}", ".20.h5"),
        noise=NOISE,
        n_em=N_EM,
        em_seeds=SEEDS,
        momentum=((1, 0, 0),),
    )
    return QEDContractConfig(
        formatting={},
        logging_level="INFO",
        runid="test",
        diagrams={"d": diagram},
        time=NT,
    )


def _surface(tmp_path):
    s = _KernelSurface(tmp_path)
    for gamma in ("GX_GX", "GY_GY", "GZ_GZ"):  # VEC_LOCAL gamma_list
        s.write(gamma)
    return s


class TestPhotexOrdering:
    def test_keys(self, tmp_path):
        _surface(tmp_path)
        config = _config(tmp_path, ("vec", "em", "vec", "em"))
        corr = qed_conn_4pt(("qed",), config.diagrams["d"], config)
        assert set(corr.keys()) == {
            (g, MASS_TOKEN, "1_0_0", str(w))
            for g in ("GX_GX", "GY_GY", "GZ_GZ")
            for w in range(N_EM)
        }
        for terms in corr.values():
            assert set(terms.keys()) == set(QED_TERM_LABELS)
            for term in QED_TERM_LABELS:
                assert terms[term].shape == (NT, NT, NT, NT)

    def test_all_terms_vs_reference_em0(self, tmp_path):
        s = _surface(tmp_path)
        config = _config(tmp_path, ("vec", "em", "vec", "em"))
        corr = qed_conn_4pt(("qed",), config.diagrams["d"], config)
        fields_raw = [
            s.vec_raw[MOM], s.em_raw[0], s.vec_raw[CONJ_MOM], s.em_raw[0],
        ]
        got = corr[(VEC_GAMMA, MASS_TOKEN, "1_0_0", "0")]
        for term in QED_TERM_LABELS:
            np.testing.assert_allclose(
                got[term],
                _ref_term(fields_raw, term, "l"),
                rtol=1e-9,
                atol=1e-12,
            )

    def test_terms_vs_reference_em1_subset(self, tmp_path):
        s = _surface(tmp_path)
        config = _config(tmp_path, ("vec", "em", "vec", "em"))
        corr = qed_conn_4pt(("qed",), config.diagrams["d"], config)
        fields_raw = [
            s.vec_raw[MOM], s.em_raw[1], s.vec_raw[CONJ_MOM], s.em_raw[1],
        ]
        got = corr[(VEC_GAMMA, MASS_TOKEN, "1_0_0", "1")]
        for term in ("llll", "nlll", "lnnn", "nnnn"):
            np.testing.assert_allclose(
                got[term],
                _ref_term(fields_raw, term, "l"),
                rtol=1e-9,
                atol=1e-12,
            )


class TestSelfenOrdering:
    def test_subset_vs_reference(self, tmp_path):
        s = _surface(tmp_path)
        config = _config(tmp_path, ("vec", "em", "em", "vec"))
        corr = qed_conn_4pt(("qed",), config.diagrams["d"], config)
        assert set(corr.keys()) == {
            (g, MASS_TOKEN, "1_0_0", str(w))
            for g in ("GX_GX", "GY_GY", "GZ_GZ")
            for w in range(N_EM)
        }
        fields_raw = [
            s.vec_raw[MOM], s.em_raw[0], s.em_raw[0], s.vec_raw[CONJ_MOM],
        ]
        got = corr[(VEC_GAMMA, MASS_TOKEN, "1_0_0", "0")]
        for term in ("llll", "nlll", "nnll", "llnn", "nnnn"):
            np.testing.assert_allclose(
                got[term],
                _ref_term(fields_raw, term, "l"),
                rtol=1e-9,
                atol=1e-12,
            )


class TestWorldSelection:
    def test_empty_selection_terms_are_zero(self, tmp_path):
        _surface(tmp_path)
        config = _config(tmp_path, ("vec", "em", "vec", "em"))
        corr = qed_conn_4pt(("qed",), config.diagrams["d"], config)
        for terms in corr.values():
            for term in QED_TERM_LABELS:
                if term.count("n") > NOISE:
                    np.testing.assert_array_equal(terms[term], 0)


class TestDispatch:
    def test_execute_routes_qed(self, tmp_path):
        _surface(tmp_path)
        config = _config(tmp_path, ("vec", "em", "vec", "em"))
        corr = execute(("qed",), config.diagrams["d"], config)
        assert set(corr.keys()) == {
            (g, MASS_TOKEN, "1_0_0", str(w))
            for g in ("GX_GX", "GY_GY", "GZ_GZ")
            for w in range(N_EM)
        }
