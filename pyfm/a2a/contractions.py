"""Core contraction operations for A2A calculations."""
import itertools
import logging
import typing as t

import opt_einsum as oe
import pandas as pd

try:
    import cupy as xp
except ImportError:
    import numpy as xp

try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
except ImportError:
    pass

from pyfm.a2a import config
from pyfm.a2a.meson_loader import MesonLoader
from pyfm.a2a.time_operations import convert_to_numpy


def make_contraction_key(
    contraction: t.Tuple[str], diagram_config: config.DiagramConfig
):
    con_key = "_".join(contraction)
    return con_key


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


def generate_time_sets(diagram_config: config.DiagramConfig, run_config: config.RunContractConfig):
    """Breaks meson field time extent into `comm_size` blocks and
    returns unique list of blocks for each `rank`.

    Returns
    -------
    tuple
        A tuple of length `diagram_config.npoint` containing lists of slices.
        One list for each meson field
    """

    workers = run_config.comm_size

    slice_indices = list(
        itertools.product(range(workers), repeat=diagram_config.npoint)
    )

    if diagram_config.symmetric:  # filter for only upper-triangular slices
        slice_indices = list(filter(lambda x: list(x) == sorted(x), slice_indices))
        workers = int((len(slice_indices) + workers - 1) / workers)

    offset = int(run_config.rank * workers)

    slice_indices = list(zip(*slice_indices[offset : offset + workers]))

    tspacing = int(run_config.time / run_config.comm_size)

    return tuple(
        [slice(int(ti * tspacing), int((ti + 1) * tspacing)) for ti in times]
        for times in slice_indices
    )


def conn_2pt(
    contraction: t.Tuple[str],
    diagram_config: config.DiagramConfig,
    run_config: config.RunContractConfig,
):
    """Execute 2-point contraction."""
    corr = {}

    times = generate_time_sets(diagram_config, run_config)
    run_config_replacements = run_config.string_dict()
    diagram_config_replacements = diagram_config.string_dict()

    for gamma in diagram_config.gammas:
        mesonfiles = tuple(
            m_path.format(
                w_index=contraction[i],
                v_index=contraction[i + 1],
                gamma=gamma,
                **run_config_replacements,
                **diagram_config_replacements,
            )
            for i, m_path in zip([0, 2], diagram_config.mesonfiles)
        )

        mat_gen = MesonLoader(
            mesonfiles=mesonfiles, times=times, **diagram_config.meson_params
        )

        cij = xp.zeros((run_config.time, run_config.time), dtype=xp.complex128)

        for (t1, m1), (t2, m2) in mat_gen:
            logging.info(f"Contracting {gamma}: {t1},{t2}")

            cij[t1, t2] = contract(m1, m2)
            if diagram_config.symmetric and t1 != t2:
                cij[t2, t1] = cij[t1, t2].T

        logging.debug("Contraction completed")

        if run_config.comm_size > 1:
            temp = None
            if run_config.rank == 0:
                temp = xp.empty_like(cij)
            COMM.Barrier()
            COMM.Reduce(cij, temp, op=MPI.SUM, root=0)

            if run_config.rank == 0:
                corr[gamma] = convert_to_numpy(temp)
        else:
            corr[gamma] = convert_to_numpy(cij)

        del m1, m2
    return corr


def sib_conn_3pt(
    contraction: t.Tuple[str],
    diagram_config: config.DiagramConfig,
    run_config: config.RunContractConfig,
):
    """Execute 3-point contraction."""
    corr = {}

    times = generate_time_sets(diagram_config, run_config)

    run_config_replacements = run_config.string_dict()
    diagram_config_replacements = diagram_config.string_dict()

    for gamma in diagram_config.gammas:
        mesonfiles = tuple(
            m_path.format(
                w_index=contraction[i],
                v_index=contraction[i + 1],
                gamma=g,
                **run_config_replacements,
                **diagram_config_replacements,
            )
            for i, g, m_path in zip(
                [0, 2, 4], [gamma, "G1_G1", gamma], diagram_config.mesonfiles
            )
        )

        mat_gen = MesonLoader(
            mesonfiles=mesonfiles, times=times, **diagram_config.meson_params
        )

        cij = xp.zeros(
            (run_config.time, run_config.time, run_config.time), dtype=xp.complex128
        )

        for (t1, m1), (t2, m2), (t3, m3) in mat_gen:
            logging.info(f"Contracting {gamma}: {t1},{t2},{t3}")
            cij[t1, t2, t3] = contract(m1, m2, m3, open_indices=[0, 1, 2])

            if diagram_config.symmetric:
                raise Exception("Symmetric 3dim optimization not implemented.")

        logging.debug("Contraction completed.")

        if run_config.comm_size > 1:
            temp = None
            if run_config.rank == 0:
                temp = xp.empty_like(cij)
            COMM.Barrier()
            COMM.Reduce(cij, temp, op=MPI.SUM, root=0)

            if run_config.rank == 0:
                corr[gamma] = convert_to_numpy(temp)
        else:
            corr[gamma] = convert_to_numpy(cij)

    return corr


def qed_conn_4pt(
    contraction: t.Tuple[str],
    diagram_config: config.DiagramConfig,
    run_config: config.RunContractConfig,
    subdiagram: config.Diagrams,
) -> pd.DataFrame:
    """Execute 4-point QED contraction."""
    corr = pd.DataFrame()

    times = generate_time_sets(diagram_config, run_config)
    run_config_replacements = run_config.string_dict()
    diagram_config_replacements = diagram_config.string_dict()

    for gamma in diagram_config.gammas:
        for i in range(diagram_config.n_em):
            emlabel = f"{diagram_config.emseedstring}_{i}"
            if subdiagram == config.Diagrams.photex:
                ops = [gamma, emlabel, gamma, emlabel]
            elif subdiagram == config.Diagrams.selfen:
                ops = [gamma, emlabel, emlabel, gamma]
            else:
                raise ValueError("Invalid qed diagram.")
            mesonfiles = tuple(
                m_path.format(
                    w_index=contraction[i],
                    v_index=contraction[i + 1],
                    gamma=g,
                    **run_config_replacements,
                    **diagram_config_replacements,
                )
                for i, g, m_path in zip(
                    [0, 2, 4, 6], ops, diagram_config.mesonfiles
                )
            )

            mat_gen = MesonLoader(
                mesonfiles=mesonfiles, times=times, **diagram_config.meson_params
            )

            cij = xp.zeros((run_config.time,) * 4, dtype=xp.complex128)

            for (t1, m1), (t2, m2), (t3, m3), (t4, m4) in mat_gen:
                logging.info(
                    f"Contracting ({gamma},{emlabel}): {t1}, {t2}, {t3}, {t4}"
                )
                cij[t1, t2, t3, t4] = contract(
                    m1, m2, m3, m4, open_indices=[0, 1, 2, 3]
                )

                if diagram_config.symmetric:
                    raise Exception("Symmetric 4dim optimization not implemented.")

            logging.debug("Contraction completed.")

            if run_config.comm_size > 1:
                temp = None
                if run_config.rank == 0:
                    temp = xp.empty_like(cij)
                COMM.Barrier()
                COMM.Reduce(cij, temp, op=MPI.SUM, root=0)

                if run_config.rank == 0:
                    corr[gamma] = convert_to_numpy(temp)
            else:
                corr[gamma] = convert_to_numpy(cij)

    return corr