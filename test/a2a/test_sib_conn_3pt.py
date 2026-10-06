"""Tests for the SIB three-point contraction kernel (sib_conn_3pt).

A synthetic split-noise surface (random blocks in the producer's grammar)
is contracted by the kernel and compared term-by-term against a naive
reference implementation of the eight-term decomposition (D5/D6): plain
triple loops with explicit world sums and explicit slice-pinned column
selection. Equality of the two proves the einsum/world/pinning bookkeeping
independently of the optimized structure.
"""
import itertools

import numpy as np
import pytest
import h5py

from pyfm.a2a.contractions import sib_conn_3pt
from pyfm.a2a.sib_derive import SIBBlockAccessor, eval_pairs, w_rows
from pyfm.a2a.types import (
    ContractType,
    SIBContractConfig,
    SIBDiagramConfig,
    SIB_TERM_LABELS,
)
from pyfm.domain import MassDict, OpList, Outfile

N_EIG = 2
NT = 4
NOISE = 2
MASSES = {"l": 0.01, "u": 0.02}
DEFL = "l"
LAM = np.array([0.7, 1.9])
SCALAR_GAMMA = "G1_G1"
VECTOR_GAMMAS = ["GX_GX", "GY_GY", "GZ_GZ"]
GAMMAS = [SCALAR_GAMMA, *VECTOR_GAMMAS]


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


def _ref_lp(raw_ll, tab, mass):
    w = w_rows(LAM, mass)
    out = np.zeros((NT, raw_ll.shape[1], NT * 3), dtype=np.complex128)
    for t in range(NT):
        for tt in range(NT):
            out[t, :, 3 * tt : 3 * tt + 3] = raw_ll[t] @ (
                tab[tt] * w[:, None]
            )
    return out


def _ref_np(nl_raw, tab_hp, mass):
    w = w_rows(LAM, mass)
    out = np.zeros((NT, 3, NT * 3), dtype=np.complex128)
    for t in range(NT):
        for tt in range(NT):
            out[t, :, 3 * tt : 3 * tt + 3] = nl_raw[t] @ (
                tab_hp[tt] * w[:, None]
            )
    return out


@pytest.fixture()
def surface(tmp_path):
    return _make_surface(tmp_path, NOISE)


@pytest.fixture()
def surface3(tmp_path):
    return _make_surface(tmp_path, 3)


def _make_surface(tmp_path, noise):
    rng = np.random.default_rng(123)
    worlds = [_World(seed) for seed in range(noise)]
    fold = eval_pairs(LAM, MASSES[DEFL])[None, None, :]
    raw_ll = {
        g: rng.normal(size=(NT, 2 * N_EIG, 2 * N_EIG))
        + 1j * rng.normal(size=(NT, 2 * N_EIG, 2 * N_EIG))
        for g in GAMMAS
    }
    delta_lh = {
        (g, w, m): rng.normal(size=(NT, 2 * N_EIG, NT * 3))
        + 1j * rng.normal(size=(NT, 2 * N_EIG, NT * 3))
        for g in GAMMAS
        for w in range(noise)
        for m in MASSES
    }
    delta_nh = {
        (g, i, j, m): rng.normal(size=(NT, 3, NT * 3))
        + 1j * rng.normal(size=(NT, 3, NT * 3))
        for g in GAMMAS
        for i in range(noise)
        for j in range(noise)
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
    outfile = Outfile(
        filestem=f"{stem}/corr/m{{mass}}/{{gamma}}/c3", ext=".20.h5", good_size=1
    )

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
                        leg_pair="nl", mass="", gamma=g,
                        n_index=f"_n{w}", hp_index="",
                    ),
                    g,
                    world.nl_raw / fold,
                )
            for m in MASSES:
                lp = _ref_lp(raw_ll[g], world.tab, MASSES[m])
                _write_block(
                    blocks.filename.format(
                        leg_pair="lh", mass=f"_m{m}", gamma=g,
                        n_index="", hp_index=f"_n{w}",
                    ),
                    g,
                    lp + delta_lh[(g, w, m)],
                )
                for w_hp, world_hp in enumerate(worlds):
                    np_ = _ref_np(world.nl_raw_for(g), world_hp.tab, MASSES[m])
                    _write_block(
                        blocks.filename.format(
                            leg_pair="nh", mass=f"_m{m}", gamma=g,
                            n_index=f"_n{w}", hp_index=f"_n{w_hp}",
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

    operations = OpList.from_dict({"gamma": ["vec_local"], "mass": ["l", "u"]})
    diagram = SIBDiagramConfig(
        formatting={},
        logging_level="INFO",
        runid="t",
        contraction_type=ContractType.SIB,
        operations=operations,
        mass=MassDict.from_dict(MASSES),
        blocks=blocks,
        tab=tab_out,
        evalfile=evalfile,
        outfile=outfile,
        noise=noise,
    )
    contract = SIBContractConfig(
        formatting={},
        logging_level="INFO",
        runid="t",
        diagrams={"hvp": diagram},
        time=NT,
        overwrite=True,
        hardware="cpu",
    )
    accessor = SIBBlockAccessor(
        blocks=blocks,
        tab=tab_out,
        evalfile=evalfile,
        mass=MassDict.from_dict(MASSES),
        defl_mass=DEFL,
        noise=noise,
    )
    return accessor, contract, worlds, raw_ll


def _colsel(row, t_pin):
    """Reference time-pinned column selection (explicit index arithmetic):
    slice-major ``3*t + c`` — the block for time ``t`` is ``3*t..3*t+3``."""
    return row[..., 3 * t_pin : 3 * t_pin + 3]


def _ref_term(accessor, gamma, mass_label, term):
    """Naive reference: plain triple loops implementing D5/D6 literally."""
    worlds = list(range(accessor.noise))
    ll_o = np.asarray(accessor.ll(gamma, mass_label))
    ll_s = np.asarray(accessor.ll(SCALAR_GAMMA, mass_label))
    nl_o = {w: np.asarray(accessor.nl(gamma, w, mass_label)) for w in worlds}
    nl_s = {
        w: np.asarray(accessor.nl(SCALAR_GAMMA, w, mass_label)) for w in worlds
    }
    pl_o = {
        w: np.asarray(accessor.pure_lh(gamma, w, mass_label)) for w in worlds
    }
    pl_s = {
        w: np.asarray(
            accessor.pure_lh(SCALAR_GAMMA, w, mass_label)
        )
        for w in worlds
    }
    pn_o = {
        (i, j): np.asarray(accessor.pure_nh(gamma, i, j, mass_label))
        for i in worlds
        for j in worlds
    }
    pn_s = {
        (i, j): np.asarray(
            accessor.pure_nh(SCALAR_GAMMA, i, j, mass_label)
        )
        for i in worlds
        for j in worlds
    }

    C = np.zeros((NT, NT, NT), dtype=np.complex128)
    for t1, t2, t3 in itertools.product(range(NT), repeat=3):
        val = 0.0 + 0.0j
        if term == "lll":
            val = np.einsum("ab,bc,ca->", ll_o[t1], ll_s[t2], ll_o[t3])
        elif term == "nll":
            for i in worlds:
                x = _colsel(pl_o[i][t1], t2)
                val += np.einsum(
                    "ac,cb,ba->", x, nl_s[i][t2], ll_o[t3]
                )
        elif term == "lnl":
            for i in worlds:
                x = _colsel(pl_s[i][t2], t3)
                val += np.einsum(
                    "ab,bc,ca->", ll_o[t1], x, nl_o[i][t3]
                )
        elif term == "lln":
            for i in worlds:
                x = _colsel(pl_o[i][t3], t1)
                val += np.einsum(
                    "ca,ab,bc->", nl_o[i][t1], ll_s[t2], x
                )
        elif term == "nnl":
            for a in worlds:
                for b in worlds:
                    if a == b:
                        continue
                    x = _colsel(pl_o[a][t1], t2)
                    y = _colsel(pn_s[(a, b)][t2], t3)
                    val += np.einsum(
                        "ac,cd,da->", x, y, nl_o[b][t3]
                    )
        elif term == "nln":
            for a in worlds:
                for c in worlds:
                    if a == c:
                        continue
                    x = _colsel(pn_o[(c, a)][t1], t2)
                    z = _colsel(pl_o[c][t3], t1)
                    val += np.einsum(
                        "ed,dk,ke->", x, nl_s[a][t2], z
                    )
        elif term == "lnn":
            for a in worlds:
                for b in worlds:
                    if a == b:
                        continue
                    u = _colsel(pl_s[b][t2], t3)
                    v = _colsel(pn_o[(b, a)][t3], t1)
                    val += np.einsum(
                        "ck,kw,wc->", nl_o[a][t1], u, v
                    )
        elif term == "nnn":
            for a in worlds:
                for b in worlds:
                    for c in worlds:
                        if a == b or b == c or c == a:
                            continue
                        x1 = _colsel(pn_o[(c, a)][t1], t2)
                        x2 = _colsel(pn_s[(a, b)][t2], t3)
                        x3 = _colsel(pn_o[(b, c)][t3], t1)
                        val += np.einsum("ed,df,fe->", x1, x2, x3)
        C[t1, t2, t3] = val
    return C


class TestKernel:
    @pytest.mark.parametrize("term", SIB_TERM_LABELS)
    def test_terms_match_reference(self, surface, term):
        """Every term of the kernel equals the naive reference."""
        accessor, contract, worlds, _ = surface
        diagram = contract.diagrams["hvp"]
        corr = sib_conn_3pt(("sib",), diagram, contract)
        for op in diagram.op_list:
            for gamma in op.gamma.gamma_list:
                for m in op.mass:
                    token = diagram.mass.to_string(m, True)
                    got = corr[(gamma, token)][term]
                    ref = _ref_term(accessor, gamma, m, term)
                    assert np.allclose(got, ref, atol=1e-10), (
                        f"{gamma}/{m}/{term}: max dev "
                        f"{np.abs(got - ref).max()}"
                    )

    def test_all_scalar_outer_family(self, surface):
        """The all-scalar three-point works through the derived nl_s."""
        accessor, contract, worlds, _ = surface
        diagram = contract.diagrams["hvp"]
        from pyfm.domain import OpList as _OL

        scalar_diagram = SIBDiagramConfig(
            **{
                **{
                    f.name: getattr(diagram, f.name)
                    for f in diagram.__dataclass_fields__.values()
                },
                "operations": _OL.from_dict(
                    {"gamma": ["scalar_local"], "mass": ["l"]}
                ),
            }
        )
        scalar_contract = SIBContractConfig(
            formatting={},
            logging_level="INFO",
            runid="t",
            diagrams={"hvp_s": scalar_diagram},
            time=NT,
        )
        corr = sib_conn_3pt(("sib",), scalar_diagram, scalar_contract)
        got = corr[("G1_G1", diagram.mass.to_string("l", True))]["lnl"]
        ref = _ref_term(accessor, "G1_G1", "l", "lnl")
        assert np.allclose(got, ref, atol=1e-10)

    def test_full_volume_no_zero_time_slabs(self, surface):
        """Full-volume coverage: every t2 slab of nll receives a
        stochastic contribution (no window — nothing is skipped)."""
        accessor, contract, *_ = surface
        diagram = contract.diagrams["hvp"]
        corr = sib_conn_3pt(("sib",), diagram, contract)
        token = diagram.mass.to_string("l", True)
        nll = corr[(VECTOR_GAMMAS[0], token)]["nll"]
        for t2 in range(NT):
            assert not np.allclose(nll[:, t2, :], 0.0)

    def test_nnn_empty_selection_at_noise_two(self, surface):
        """noise=2 admits no pairwise-distinct triple: nnn is identically
        zero (raw world sum over an empty selection)."""
        accessor, contract, *_ = surface
        diagram = contract.diagrams["hvp"]
        corr = sib_conn_3pt(("sib",), diagram, contract)
        token = diagram.mass.to_string("l", True)
        assert np.allclose(corr[(VECTOR_GAMMAS[0], token)]["nnn"], 0.0)

    @pytest.mark.parametrize("term", ["nnl", "nln", "lnn", "nnn"])
    def test_world_selection_noise_three(self, surface3, term):
        """noise=3: three-world off-diagonal pairs and the non-empty nnn
        triple selection match the naive reference."""
        accessor, contract, *_ = surface3
        diagram = contract.diagrams["hvp"]
        corr = sib_conn_3pt(("sib",), diagram, contract)
        token = diagram.mass.to_string("l", True)
        gamma = VECTOR_GAMMAS[0]
        got = corr[(gamma, token)][term]
        ref = _ref_term(accessor, gamma, "l", term)
        assert np.allclose(got, ref, atol=1e-10)
        if term == "nnn":
            assert not np.allclose(got, 0.0)

    def test_returns_all_terms_and_keys(self, surface):
        accessor, contract, *_ = surface
        diagram = contract.diagrams["hvp"]
        corr = sib_conn_3pt(("sib",), diagram, contract)
        # vec_local: 3 gamma names x 2 masses
        assert len(corr) == 6
        for terms in corr.values():
            assert set(terms) == set(SIB_TERM_LABELS)
            for arr in terms.values():
                assert arr.shape == (NT, NT, NT)
                assert arr.dtype == np.complex128

    def test_symmetric_rejected(self, surface):
        import dataclasses

        accessor, contract, *_ = surface
        diagram = dataclasses.replace(
            contract.diagrams["hvp"], symmetric=True
        )
        with pytest.raises(ValueError, match="Symmetric"):
            sib_conn_3pt(("sib",), diagram, contract)
