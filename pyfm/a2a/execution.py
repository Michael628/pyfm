"""Main execution logic for A2A contractions."""

import itertools
import logging
import os
import pickle
import typing as t
from time import perf_counter

from sympy.utilities.iterables import multiset_permutations

try:
    import cupy as xp
except ImportError:
    import numpy as xp

from pyfm import utils
from pyfm.domain import DiagramConfig, RunContractConfig, Diagrams
from .contractions import (
    conn_2pt,
    make_contraction_key,
    qed_conn_4pt,
    sib_conn_3pt,
)


def execute(
    contraction: t.Tuple[str],
    diagram_config: DiagramConfig,
    run_config: RunContractConfig,
):
    """Execute the appropriate contraction based on diagram configuration."""
    if hasattr(xp, "cuda"):
        my_device = run_config.rank % xp.cuda.runtime.getDeviceCount()
        logging.debug(f"Rank {run_config.rank} is using gpu device {my_device}")
        xp.cuda.Device(my_device).use()

    logging.info(f"Processing mode: {', '.join(contraction)}")

    contraction_types = {
        "conn_2pt": lambda: conn_2pt(contraction, diagram_config, run_config),
        "sib_conn_3pt": lambda: sib_conn_3pt(contraction, diagram_config, run_config),
        "qed_conn_photex_4pt": lambda: qed_conn_4pt(
            contraction, diagram_config, run_config, Diagrams.photex
        ),
        "qed_conn_selfen_4pt": lambda: qed_conn_4pt(
            contraction, diagram_config, run_config, Diagrams.selfen
        ),
    }

    if diagram_config.contraction_type in contraction_types:
        run = contraction_types[diagram_config.contraction_type]
    else:
        raise ValueError(
            f"No contraction implementation for `{diagram_config.contraction_type}`."
        )

    return run()


def main(param_file: str):
    """Main execution function for A2A contractions."""
    params = utils.io.load_param(param_file)

    run_config = get_contract_config(params)

    logging_level = getattr(run_config, "logging_level", "INFO")
    utils.set_logging_level(logging_level)

    if run_config.hardware == "cpu":
        import numpy as xp

        globals()["xp"] = xp

    overwrite = run_config.overwrite_correlators

    diagrams = run_config.diagrams
    for diagram_config in diagrams:
        if diagram_config.evalfile:
            diagram_config.format_evalfile()

        nmesons = diagram_config.npoint

        low_min = 0 if diagram_config.has_high else nmesons
        low_max = nmesons + 1 if diagram_config.has_low else 1

        perms = sum(
            [
                list(multiset_permutations(["L"] * nlow + ["H"] * (nmesons - nlow)))
                for nlow in range(low_min, low_max)
            ],
            [],
        )
        perms = list(map("".join, perms))
        # Overwrite chosen permutations with user input, if provided
        if diagram_config.perms:
            perms = diagram_config.perms

        logging.debug(f"Computing permutations: {perms}")

        for perm in perms:
            nlow = perm.count("L")

            permkey = "".join(
                sum(((perm[i], perm[(i + 1) % nmesons]) for i in range(nmesons)), ())
            )

            if diagram_config.has_high:
                # Build list of high source indices,
                # e.g. [[0,1], [0,2], ...]
                seeds = list(
                    map(
                        list,
                        itertools.combinations(
                            list(range(diagram_config.high_count)), nmesons - nlow
                        ),
                    )
                )
            else:
                seeds = [[]]

            # Fill low-mode indices with None
            # e.g. [[None,0,1], [None,0,2], ...]
            _ = [
                seed.insert(i, None)
                for i in range(len(perm))
                if perm[i] == "L"
                for seed in seeds
            ]

            # Double indices for <bra | ket> and cycle
            # e.g. [[None,0,0,1,1,None], [None,0,0,2,2,None], ...]
            seeds = [list(sum(zip(seed, seed), ())) for seed in seeds]
            seeds = [seed[1:] + seed[:1] for seed in seeds]

            outfile = diagram_config.outfile.format(permkey=permkey)

            if overwrite or not os.path.exists(outfile):
                logging.info(
                    f"Contracting diagram: {diagram_config.gamma_label} ({permkey})"
                )
            else:
                logging.info(f"Skipping write. File exists: {outfile}")
                continue

            contraction_list = [
                ["e" if seed[i] is None else s for i, s in enumerate(map(str, seed))]
                for seed in seeds
            ]

            start_time = perf_counter()

            corr = dict(
                zip(
                    map(
                        lambda x: make_contraction_key(x, diagram_config),
                        contraction_list,
                    ),
                    map(
                        lambda x: execute(x, diagram_config, run_config),
                        contraction_list,
                    ),
                )
            )

            stop_time = perf_counter()

            logging.debug("")
            logging.debug(
                "    Total elapsed time for %s = %g seconds."
                % (permkey, stop_time - start_time)
            )
            logging.debug("")

            if run_config.rank < 1:
                os.makedirs(os.path.dirname(outfile), exist_ok=True)
                pickle.dump(corr, open(outfile, "wb"))
