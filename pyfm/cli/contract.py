import os
import itertools
import typing as t
from time import perf_counter

import click
from sympy.utilities.iterables import multiset_permutations

from pyfm.a2a.types import ContractConfig
from pyfm.domain import LoadDictConfig
from pyfm.core.builder import build_config
from pyfm.dataio import data_to_frame, write_files
from pyfm.a2a import execute, time_average
from pyfm import utils


def _make_contraction_key(contraction: t.Tuple[str]):
    return "_".join(contraction)


@click.group()
def contract():
    """Run A2A all-to-all contraction calculations."""
    pass


@contract.command()
@click.argument(
    "param-file", type=click.Path(dir_okay=False), required=False, default=None
)
@click.option(
    "-p",
    "--param-file-opt",
    type=click.Path(dir_okay=False),
    default=None,
    help="Path to YAML parameter file.",
)
@click.option(
    "--time-average",
    "do_time_average",
    is_flag=True,
    default=False,
    help="Apply time averaging to all contractions before writing.",
)
def run(param_file, param_file_opt, do_time_average):
    """Execute A2A contractions for all diagrams defined in the parameter file."""
    param_file = param_file or param_file_opt
    if param_file is None:
        raise click.UsageError(
            "A parameter file is required (pass as argument or with -p)."
        )
    params = utils.io.load_param(param_file)

    config: ContractConfig = build_config(ContractConfig, params, normalized=True)

    logging_level = getattr(config, "logging_level", "INFO")
    logger = utils.set_logging_level(logging_level)

    logger.info(
        f"Starting A2A contractions with {config.comm_size} MPI rank(s) "
        f"(current rank: {config.rank}, hardware: {config.hardware})"
    )

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
        if diagram_config.perms:
            perms = diagram_config.perms

        logger.debug(f"Computing permutations: {perms}")

        for perm in perms:
            nlow = perm.count("L")

            permkey = "".join(
                sum(((perm[i], perm[(i + 1) % nmesons]) for i in range(nmesons)), ())
            )

            if has_high:
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

            _ = [
                seed.insert(i, None)
                for i in range(len(perm))
                if perm[i] == "L"
                for seed in seeds
            ]

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
                    map(_make_contraction_key, contraction_list),
                    map(lambda x: execute(x, diagram_config, config), contraction_list),
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

                if do_time_average:
                    corr = {
                        k: {g: time_average(v) for g, v in gamma_dict.items()}
                        for k, gamma_dict in corr.items()
                    }
                    array_order = [f"t{i+1}" for i in range(1, nmesons - 1)] + ["dt"]
                else:
                    array_order = [f"t{i+1}" for i in range(nmesons)]
                data_config = LoadDictConfig.create(
                    dict_labels=["perm", "gamma"],
                    array_order=array_order,
                    array_labels={o: f"0..{config.time-1}" for o in array_order},
                )
                df = data_to_frame(corr, data_config)
                write_files(df, outfile, format="hdf5")
