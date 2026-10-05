"""Core contraction operations for A2A calculations."""

import itertools
import typing as t

import opt_einsum as oe
import pandas as pd

try:
    import cupy as xp
except ImportError:
    import numpy as xp

from pyfm import utils
from pyfm.a2a.types import (
    ContractConfig,
    DiagramConfig,
    SIBContractConfig,
    SIBDiagramConfig,
    SIB_TERM_LABELS,
)
from pyfm.a2a.mesonloader import iter_meson_fields, clear_meson_cache
from pyfm.a2a.time_operations import convert_to_numpy


def _reduce_sum_to_root(sendbuf, recvbuf):
    """Sum ``sendbuf`` across all ranks onto ``recvbuf`` at rank 0.

    Imports mpi4py lazily so this module can be imported without running
    MPI_Init; callers only reach here when ``comm_size > 1``, i.e. under an
    active MPI launch.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    comm.Barrier()
    comm.Reduce(sendbuf, recvbuf, op=MPI.SUM, root=0)


def contract(
    m1: xp.ndarray,
    m2: xp.ndarray,
    m3: xp.ndarray = None,
    m4: xp.ndarray = None,
    open_indices: t.Tuple = (0, -1),
):
    """Performs contraction of up to 4 3-dim arrays down to one 2-dim array

    Parameters
    ----------
    m1 : ndarray
    m2 : ndarray
    m3 : ndarray, optional
    m4 : ndarray, optional
    open_indices : tuple, default=(0,-1)
        A two-dimensional tuple containing the time indices that will
        not be contracted in the full product. The default=(0,-1) leaves
        indices of the first and last matrix open, summing over all others.

    Returns
    -------
    ndarray
        The resultant 2-dim array from the contraction
    """
    npoint = sum([1 for m in [m1, m2, m3, m4] if m is not None])

    if len(open_indices) > npoint:
        raise ValueError(
            (f"Length of open_indices must be <= diagram degree (maximum: 4)")
        )

    index_list = ["i", "j", "k", "l"][:npoint]
    out_indices = "".join(index_list[i] for i in open_indices)

    if npoint == 2:  # two-point contractions
        cij = oe.contract(f"imn,jnm->{out_indices}", m1, m2)

    elif npoint == 3:  # three-point contractions
        cij = oe.contract(f"imn,jno,kom->{out_indices}", m1, m2, m3)

    else:  # four-point contractions
        cij = oe.contract(f"imn,jno,kop,lpm->{out_indices}", m1, m2, m3, m4)

    return cij


def generate_time_sets(diagram_config: DiagramConfig, contract_config: ContractConfig):
    """Breaks meson field time extent into `comm_size` blocks and
    returns unique list of blocks for each `rank`.

    Returns
    -------
    tuple
        A tuple of length `diagram_config.npoint` containing lists of slices.
        One list for each meson field
    """

    workers = contract_config.comm_size

    slice_indices = list(
        itertools.product(range(workers), repeat=diagram_config.npoint)
    )

    if diagram_config.symmetric:  # filter for only upper-triangular slices
        slice_indices = list(filter(lambda x: list(x) == sorted(x), slice_indices))
        workers = int((len(slice_indices) + workers - 1) / workers)

    offset = int(contract_config.rank * workers)

    slice_indices = list(zip(*slice_indices[offset : offset + workers]))

    tspacing = int(contract_config.time / contract_config.comm_size)

    return tuple(
        [slice(int(ti * tspacing), int((ti + 1) * tspacing)) for ti in times]
        for times in slice_indices
    )


def conn_2pt(
    contraction: t.Tuple[str],
    diagram_config: DiagramConfig,
    contract_config: ContractConfig,
):
    """Execute 2-point contraction."""
    corr = {}

    times = generate_time_sets(diagram_config, contract_config)

    logger = utils.get_logger()

    for gamma in diagram_config.gammas:
        clear_meson_cache()
        mesonfiles = tuple(
            meson_outfile.file.filename.format(
                w_index=contraction[i],
                v_index=contraction[i + 1],
                gamma=gamma,
            )
            for i, meson_outfile in zip([0, 2], diagram_config.mesons)
        )

        cij = xp.zeros(
            (contract_config.time, contract_config.time), dtype=xp.complex128
        )

        for (t1, m1), (t2, m2) in iter_meson_fields(
            diagram_config, mesonfiles, times, contraction
        ):
            logger.info(f"Contracting {gamma}: {t1},{t2}")

            cij[t1, t2] = contract(m1, m2)
            if diagram_config.symmetric and t1 != t2:
                cij[t2, t1] = cij[t1, t2].T

        logger.debug("Contraction completed")

        if contract_config.comm_size > 1:
            temp = None
            if contract_config.rank == 0:
                temp = xp.empty_like(cij)
            _reduce_sum_to_root(cij, temp)

            if contract_config.rank == 0:
                corr[gamma] = convert_to_numpy(temp)
        else:
            corr[gamma] = convert_to_numpy(cij)

        del m1, m2
    return corr


def sib_conn_3pt(
    contraction: t.Tuple[str],
    diagram_config: SIBDiagramConfig,
    contract_config: SIBContractConfig,
):
    """Execute the SIB three-point contraction (connected HVP, V·S·V).

    Computes all eight L/H junction terms (``SIB_TERM_LABELS``) per (outer
    GammaName, correlator mass) from the ``hadrons_sib_mf`` split-noise
    block surface, deriving the removed p-side blocks via the pair-basis
    identity (``pyfm.a2a.sib_derive.SIBBlockAccessor``). World selection:
    single noise junctions sum diagonal world pairs; two-junction terms sum
    ordered off-diagonal pairs; ``nnn`` sums pairwise-distinct triples. At
    each noise junction the h/p column slice is pinned to the eta-side
    field's time row (the masked noise sees operators only at its source
    slice; pinned times outside the batch window contribute nothing).

    Outputs are RAW per-term (T,T,T) world sums — no normalization factors
    (D7: aggregation applies per-term world-count factors then sums the
    terms). Returns ``{(gamma_name, mass_token): {term: ndarray}}`` with
    numpy arrays on rank 0 (entries absent on other ranks).

    ``contraction`` is accepted for signature parity with the other
    kernels; the SIB scheme carries its noise structure in-matrix, so the
    legacy (w, v) seed tuple is unused.
    """
    from pyfm.a2a.sib_derive import SIBBlockAccessor

    if diagram_config.symmetric:
        raise ValueError(
            "Symmetric optimization is not implemented for SIB 3-point "
            "contractions."
        )

    logger = utils.get_logger()

    accessor = SIBBlockAccessor(
        blocks=diagram_config.blocks,
        tab=diagram_config.tab,
        evalfile=diagram_config.evalfile,
        mass=diagram_config.mass,
        defl_mass=diagram_config.defl_mass,
        noise=diagram_config.noise,
        t0=diagram_config.t0,
        t_step=diagram_config.t_step,
        n_slices=diagram_config.n_slices,
    )

    times = generate_time_sets(diagram_config, contract_config)
    block_triples = list(zip(*[list(t) for t in times])) or [
        (slice(0, contract_config.time),) * 3
    ]

    corr: t.Dict[t.Tuple[str, str], t.Dict[str, xp.ndarray]] = {}
    for op in diagram_config.op_list:
        for gamma in op.gamma.gamma_list:
            for mass_label in op.mass:
                mass_token = diagram_config.mass.to_string(mass_label, True)
                key = (gamma, mass_token)
                logger.info(f"Contracting SIB 3pt: {gamma} at mass {mass_label}")

                cij = {
                    term: xp.zeros(
                        (contract_config.time,) * 3, dtype=xp.complex128
                    )
                    for term in SIB_TERM_LABELS
                }
                for s1, s2, s3 in block_triples:
                    _sib_terms(
                        cij,
                        accessor,
                        gamma,
                        mass_label,
                        s1,
                        s2,
                        s3,
                    )

                for term in SIB_TERM_LABELS:
                    if contract_config.comm_size > 1:
                        temp = None
                        if contract_config.rank == 0:
                            temp = xp.empty_like(cij[term])
                        _reduce_sum_to_root(cij[term], temp)
                        if contract_config.rank == 0:
                            corr.setdefault(key, {})[term] = convert_to_numpy(
                                temp
                            )
                    else:
                        corr.setdefault(key, {})[term] = convert_to_numpy(
                            cij[term]
                        )
                del cij
    return corr


def _sib_worlds(noise: int) -> t.List[int]:
    return list(range(noise))


def _sib_offdiag(noise: int) -> t.List[t.Tuple[int, int]]:
    return [(i, j) for i in range(noise) for j in range(noise) if i != j]


def _sib_distinct(noise: int) -> t.List[t.Tuple[int, int, int]]:
    return [
        (a, b, c)
        for a in range(noise)
        for b in range(noise)
        for c in range(noise)
        if a != b and b != c and c != a
    ]


def _sib_terms(
    cij: t.Dict[str, xp.ndarray],
    accessor,
    gamma: str,
    mass_label: str,
    s1: slice,
    s2: slice,
    s3: slice,
) -> None:
    """Accumulate one rank's time-block contribution to the eight terms.

    Field positions: M1@t1 (outer, ``gamma``), M2@t2 (scalar middle,
    G1_G1), M3@t3 (outer, ``gamma``). Junctions: J12 (M1.cols x M2.rows),
    J23 (M2.cols x M3.rows), J31 (M3.cols x M1.rows); term ids follow
    (J12, J23, J31) letters, 'l' = eig, 'n' = noise. At each noise
    junction the h/p column slice is pinned to the eta-side field's time
    row (D5/D6); ``cols`` returns None outside the window and the
    contribution is skipped.
    """
    from pyfm.a2a.sib_derive import SCALAR_GAMMA

    cols = accessor.cols
    worlds = _sib_worlds(accessor.noise)

    # ---- lll: plain cyclic einsum --------------------------------------
    ll_o = accessor.ll(gamma, mass_label)
    ll_s = accessor.ll(SCALAR_GAMMA, mass_label)
    cij["lll"][s1, s2, s3] += oe.contract(
        "uab,vbc,wca->uvw", ll_o[s1], ll_s[s2], ll_o[s3]
    )

    # shared per-mass pieces for the noise terms
    nl_o = {w: accessor.nl(gamma, w, mass_label) for w in worlds}
    nl_s = {w: accessor.nl(SCALAR_GAMMA, w, mass_label) for w in worlds}
    pl_o = {w: accessor.pure_lh(gamma, w, mass_label) for w in worlds}
    pl_s = {w: accessor.pure_lh(SCALAR_GAMMA, w, mass_label) for w in worlds}

    # ---- nll: J12 noise, world i on M1 (pure_lh) and M2 (nl) -----------
    for i in worlds:
        for t2 in range(s2.start, s2.stop):
            x = cols(pl_o[i][s1], t2)
            if x is None:
                continue
            y = oe.contract("uac,cb->uab", x, nl_s[i][t2])
            cij["nll"][s1, t2, s3] += oe.contract("uab,vba->uv", y, ll_o[s3])

    # ---- lnl: J23 noise, world i on M2 (pure_lh) and M3 (nl) -----------
    for i in worlds:
        for t3 in range(s3.start, s3.stop):
            x = cols(pl_s[i][s2], t3)
            if x is None:
                continue
            cij["lnl"][s1, s2, t3] += oe.contract(
                "uab,vbc,ca->uv", ll_o[s1], x, nl_o[i][t3]
            )

    # ---- lln: J31 noise, world i on M1 (nl) and M3 (pure_lh) -----------
    for i in worlds:
        for t1 in range(s1.start, s1.stop):
            x = cols(pl_o[i][s3], t1)
            if x is None:
                continue
            cij["lln"][t1, s2, s3] += oe.contract(
                "ca,uab,vbc->uv", nl_o[i][t1], ll_s[s2], x
            )

    # ---- two-junction terms: ordered off-diagonal world pairs ----------
    pn_o = {}
    pn_s = {}
    for i in worlds:
        for j in worlds:
            pn_o[(i, j)] = accessor.pure_nh(gamma, i, j, mass_label)
            pn_s[(i, j)] = accessor.pure_nh(SCALAR_GAMMA, i, j, mass_label)

    # nnl: M1 = pure_lh^a (outer), M2 = pure_nh^(a,b) (middle),
    #      M3 = nl^b (outer); J12 pins M1's cols to t2, J23 pins M2's
    #      cols to t3.
    for a, b in _sib_offdiag(accessor.noise):
        for t2 in range(s2.start, s2.stop):
            x = cols(pl_o[a][s1], t2)
            if x is None:
                continue
            for t3 in range(s3.start, s3.stop):
                y = cols(pn_s[(a, b)][t2], t3)
                if y is None:
                    continue
                cij["nnl"][s1, t2, t3] += oe.contract(
                    "ukc,cd,dk->u", x, y, nl_o[b][t3]
                )

    # nln: M1 = pure_nh^(c,a) (outer), M2 = nl^a (middle), M3 =
    #      pure_lh^c (outer); J12 pins M1's cols to t2, J31 pins M3's
    #      cols to t1.
    for a, c in _sib_offdiag(accessor.noise):
        for t2 in range(s2.start, s2.stop):
            nl2 = nl_s[a][t2]
            for t1 in range(s1.start, s1.stop):
                x = cols(pn_o[(c, a)][t1], t2)
                if x is None:
                    continue
                z = cols(pl_o[c][s3], t1)
                if z is None:
                    continue
                cij["nln"][t1, t2, s3] += oe.contract(
                    "ed,dk,vke->v", x, nl2, z
                )

    # lnn: M1 = nl^a (outer), M2 = pure_lh^b (middle), M3 = pure_nh^(b,a)
    #      (outer); J23 pins M2's cols to t3, J31 pins M3's cols to t1.
    for a, b in _sib_offdiag(accessor.noise):
        for t3 in range(s3.start, s3.stop):
            u = cols(pl_s[b][s2], t3)
            if u is None:
                continue
            for t1 in range(s1.start, s1.stop):
                v = cols(pn_o[(b, a)][t3], t1)
                if v is None:
                    continue
                cij["lnn"][t1, s2, t3] += oe.contract(
                    "ck,vkw,wc->v", nl_o[a][t1], u, v
                )

    # ---- nnn: pairwise-distinct world triples ---------------------------
    for a, b, c in _sib_distinct(accessor.noise):
        for t2 in range(s2.start, s2.stop):
            x1_all = cols(pn_o[(c, a)][s1], t2)
            if x1_all is None:
                continue
            for t3 in range(s3.start, s3.stop):
                x2 = cols(pn_s[(a, b)][t2], t3)
                if x2 is None:
                    continue
                for t1 in range(s1.start, s1.stop):
                    x3 = cols(pn_o[(b, c)][t3], t1)
                    if x3 is None:
                        continue
                    cij["nnn"][t1, t2, t3] += oe.contract(
                        "ed,df,fe->", x1_all[t1 - s1.start], x2, x3
                    )


def qed_conn_4pt(
    contraction: t.Tuple[str],
    diagram_config: DiagramConfig,
    contract_config: ContractConfig,
) -> pd.DataFrame:
    """Execute 4-point QED contraction."""
    corr = pd.DataFrame()

    times = generate_time_sets(diagram_config, contract_config)

    logger = utils.get_logger()
    for gamma in diagram_config.gammas:
        for i in range(diagram_config.n_em):
            emlabel = f"{diagram_config.emseedstring}_{i}"
            if subdiagram == diagram_config.contraction_type.PHOTEX:
                ops = [gamma, emlabel, gamma, emlabel]
            elif subdiagram == diagram_config.contraction_type.SELFEN:
                ops = [gamma, emlabel, emlabel, gamma]
            else:
                raise ValueError("Invalid qed diagram.")
            mesonfiles = tuple(
                m_path.format(
                    w_index=contraction[i],
                    v_index=contraction[i + 1],
                    gamma=g,
                )
                for i, g, m_path in zip([0, 2, 4, 6], ops, diagram_config.mesonfiles)
            )

            cij = xp.zeros((contract_config.time,) * 4, dtype=xp.complex128)

            for (t1, m1), (t2, m2), (t3, m3), (t4, m4) in iter_meson_fields(
                diagram_config, mesonfiles, times, contraction
            ):
                logger.info(f"Contracting ({gamma},{emlabel}): {t1}, {t2}, {t3}, {t4}")
                cij[t1, t2, t3, t4] = contract(
                    m1, m2, m3, m4, open_indices=[0, 1, 2, 3]
                )

                if diagram_config.symmetric:
                    raise Exception("Symmetric 4dim optimization not implemented.")

            logger.debug("Contraction completed.")

            if contract_config.comm_size > 1:
                temp = None
                if contract_config.rank == 0:
                    temp = xp.empty_like(cij)
                _reduce_sum_to_root(cij, temp)

                if contract_config.rank == 0:
                    corr[gamma] = convert_to_numpy(temp)
            else:
                corr[gamma] = convert_to_numpy(cij)

    return corr
