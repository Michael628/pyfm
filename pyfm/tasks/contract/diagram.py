import typing as t
from pyrsistent import freeze, thaw

from pyfm.a2a.types import DiagramConfig
from pyfm.utils.string import PartialFormatter

from pyfm import utils

from pyfm.tasks.register import register_task
from pyfm.tasks.contract import mesonloader

import pandas as pd


def build_input_params(config: DiagramConfig) -> t.Dict[str, t.Any]:
    mass_map = PartialFormatter(mass=config.mass_label)
    yaml_params = {
        "contraction_type": config.contraction_type.name,
        "gammas": config.gammas,
        "mesons": [mesonloader.build_input_params(m) for m in config.mesons],
        "outfile": config.outfile.format_map(mass_map),
        "symmetric": config.symmetric,
    }

    if config.perms is not None:
        yaml_params["perms"] = config.perms

    for key in ["eig_range", "stoch_range", "efield_indices", "stoch_seed_indices"]:
        if getattr(config, key) is not None:
            yaml_params[key] = getattr(config, key)._asdict()

    return yaml_params


def preprocess_params(params: t.Dict) -> t.Dict:
    """Top-level preprocessing for DiagramConfig.

    Flattens _tasks into the top-level params.
    """
    task_data = params.get("_tasks", {})
    return params | {"_tasks": {}} | task_data


def preprocess_subconfig(params: t.Dict, subconfig_label: str) -> t.Dict:
    """Per-subconfig preprocessing for DiagramConfig.

    For the ``mesons`` LIST container, renames ``mass`` → ``mass_original``
    and ``new_mass`` → ``mass_updated`` in each meson entry so that
    MesonLoaderConfig can construct its ``mass_shift`` field correctly.
    """
    if subconfig_label != "mesons":
        return params

    mesons = params.get("mesons", [])
    if not isinstance(mesons, list):
        mesons = [mesons]
    new_mesons = [
        thaw(
            freeze(m)
            .discard("mass")
            .discard("new_mass")
            .update({"mass_original": m["mass"]} if "mass" in m else {})
            .update({"mass_updated": m["new_mass"]} if "new_mass" in m else {})
        )
        for m in mesons
    ]
    return params | {"mesons": new_mesons}


def build_aggregator_params(config: DiagramConfig, average: bool) -> t.Dict:
    agg_params = {"run": ["diagram"], "diagram": {}}

    mass_map = PartialFormatter(mass=config.mass_label)
    infile = config.outfile.format_map(mass_map)

    suffix = "_avg" if average else ""
    outfile_stem = utils.io.get_processed_filename(
        infile.filestem, remove=["series"], suffix=suffix
    )

    agg_params["diagram"] = {
        "logging_level": config.logging_level,
        "load_files": {
            "filestem": infile.filename,
            "regex": {"series": "[a-z]", "cfg": "[0-9]+"},
            "dict_labels": ["perm", "gamma"],
        },
        "out_files": {"filestem": outfile_stem},
    }

    actions = {}

    index = ["series.cfg", "gamma"]
    if config.has_high:
        index.append("perm")
    else:
        actions["drop"] = "perm"

    if average:
        actions["real"] = True

    t_order = [f"t{i}" for i in range(1, config.npoint + 1)]
    array_params = {
        "array_order": t_order,
        "array_labels": {},
    }

    t_labels = f"0..{config.time - 1}"
    if average:
        actions["time_average"] = [
            t_order[0],
            t_order[-1],
        ]  # Assumes averaging first and last time index
        index += t_order[1:-1] + ["t"]
    else:
        index += array_params["array_order"]

    actions["index"] = index

    for t_index in array_params["array_order"]:
        array_params["array_labels"][t_index] = t_labels

    if actions:
        agg_params["diagram"]["actions"] = actions

    agg_params["diagram"]["load_files"] |= array_params

    return agg_params


def create_outfile_catalog(config: DiagramConfig) -> pd.DataFrame:
    def generate_outfile_formatting():
        """Generator for meson field file formatting parameters."""
        res = {"mass": config.mass_label}
        yield res, config.outfile

    outfile_generator = generate_outfile_formatting()

    return utils.io.catalog_files(outfile_generator)


register_task(
    "contract_diagram",
    DiagramConfig,
    build_input_params=build_input_params,
    create_outfile_catalog=create_outfile_catalog,
    build_aggregator_params=build_aggregator_params,
    preprocess=preprocess_params,
    preprocess_subconfig=preprocess_subconfig,
)
