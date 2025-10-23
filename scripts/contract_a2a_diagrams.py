import typing as t
import os
import pickle
import itertools
import argparse

from sympy.utilities.iterables import multiset_permutations

from pyfm.domain import ContractConfig, LoadDictConfig
from pyfm.builder import build_config
from pyfm.dataio import data_to_frame, write_files

from pyfm.a2a import get_a2a_handler, execute
from pyfm import utils
from time import perf_counter


def make_contraction_key(contraction: t.Tuple[str]):
    con_key = "_".join(contraction)
    return con_key


def main():
    """Main execution function for A2A contractions."""

    parser = argparse.ArgumentParser(
        description="Generate a2a (2pt, 3pt, 4pt, QED, SIB, etc.) contractions as described from input parameters."
    )
    parser.add_argument(
        "-p",
        "--param-file",
        type=str,
        required=True,
        help="Contraction input parameter file location",
    )
    args = parser.parse_args()
    params = utils.io.load_param(args.param_file)

    config: ContractConfig = build_config(
        ContractConfig, params, get_handler=get_a2a_handler
    )

    logging_level = getattr(config, "logging_level", "INFO")
    utils.set_logging_level(logging_level)

    logger = utils.get_logger()
    if config.hardware == "cpu":
        import numpy as xp

        globals()["xp"] = xp

    overwrite = config.overwrite

    for diagram_label, diagram_config in config.diagrams.items():
        nmesons = diagram_config.npoint

        has_high = diagram_config.stoch_range is not None
        has_low = diagram_config.eig_range is not None

        low_min = 0 if has_high else nmesons
        low_max = nmesons + 1 if has_low else 1

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

        logger.debug(f"Computing permutations: {perms}")

        for perm in perms:
            nlow = perm.count("L")

            permkey = "".join(
                sum(((perm[i], perm[(i + 1) % nmesons]) for i in range(nmesons)), ())
            )

            if has_high:
                # Build list of high source indices,
                # e.g. [[0,1], [0,2], ...]
                seeds = list(
                    map(
                        list,
                        itertools.combinations(
                            diagram_config.stoch_seed_indices, nmesons - nlow
                        ),
                    )
                )
            else:
                seeds = [[]]

            # Fill low-mode indices with None
            # e.g. Low-High-High -> [[None,0,1], [None,0,2], ...]
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

            outfile = diagram_config.outfile.filename.format(permkey=permkey)

            if overwrite or not os.path.exists(outfile):
                logger.info(f"Contracting diagram: {diagram_label} ({permkey})")
            else:
                logger.info(f"Skipping write. File exists: {outfile}")
                continue

            contraction_list = [
                ["e" if seed[i] is None else s for i, s in enumerate(map(str, seed))]
                for seed in seeds
            ]

            start_time = perf_counter()

            corr = dict(
                zip(
                    map(
                        make_contraction_key,
                        contraction_list,
                    ),
                    map(
                        lambda x: execute(x, diagram_config, config),
                        contraction_list,
                    ),
                )
            )

            stop_time = perf_counter()

            logger.debug("")
            logger.debug(
                "    Total elapsed time for %s = %g seconds."
                % (permkey, stop_time - start_time)
            )
            logger.debug("")

            if config.rank < 1:
                os.makedirs(os.path.dirname(outfile), exist_ok=True)

                array_order = [f"t{i+1}" for i in range(diagram_config.npoint)]
                data_config = LoadDictConfig.create(
                    dict_labels=["perm", "gamma"],
                    array_order=array_order,
                    array_labels={o: f"0..{config.time-1}" for o in array_order},
                )
                df = data_to_frame(corr, data_config)
                write_files(df, "hdf5", outfile)
                # pickle.dump(corr, open(outfile, "wb"))


if __name__ == "__main__":
    main()
