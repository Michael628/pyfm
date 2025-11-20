import typing as t
from pydantic.dataclasses import dataclass
from dataclasses import field
from pyrsistent import freeze, thaw

from pyfm.domain import (
    Outfile,
    MesonLoaderConfig,
    DiagramConfig,
    ContractConfig,
    SimpleConfig,
    CompositeConfig,
    ContractType,
    MassDict,
    PartialFormatter,
)

from pyfm import utils

from pyfm.tasks.register import register_task

import pandas as pd


def meson_loader_preprocess_params(
    params: t.Dict, subconfig: str | None = None
) -> t.Dict:
    mass_shift = {}

    for key in ["mass_original", "mass_updated", "milc_mass"]:
        if key in params:
            mass_shift[key.removeprefix("mass_")] = params[key]
    return params | {"mass_shift": mass_shift}


def meson_loader_build_input_params(config: MesonLoaderConfig) -> t.Dict[str, t.Any]:
    mass_map = PartialFormatter(mass=config.get_mass_label(include_shift=False))
    yaml_params = {
        "mass": config.mass._asdict(),
        "file": config.file.format_map(mass_map),
        "mass_shift": config.mass_shift._asdict(),
    }
    if config.evalfile is not None:
        yaml_params["evalfile"] = config.evalfile.format_map(mass_map)

    return yaml_params


def diagram_build_input_params(config: DiagramConfig) -> t.Dict[str, t.Any]:
    mass_map = PartialFormatter(mass=config.mass_label)
    yaml_params = {
        "contraction_type": config.contraction_type.name,
        "gammas": config.gammas,
        "mesons": [meson_loader_build_input_params(m) for m in config.mesons],
        "outfile": config.outfile.format_map(mass_map),
        "symmetric": config.symmetric,
    }

    if config.perms is not None:
        yaml_params["perms"] = config.perms

    for key in ["eig_range", "stoch_range", "efield_indices", "stoch_seed_indices"]:
        if getattr(config, key) is not None:
            yaml_params[key] = getattr(config, key)._asdict()

    return yaml_params


def diagram_preprocess_params(params: t.Dict, subconfig: str | None = None) -> t.Dict:
    if subconfig is None:
        return params

    new_mesons = []
    mesons = params.get("mesons", [])
    if not isinstance(mesons, list):
        mesons = [mesons]
    for m in mesons:
        new_meson = freeze(m)
        if "mass" in m:
            new_meson = new_meson.remove("mass").update({"mass_original": m["mass"]})
        if "new_mass" in m:
            new_meson = new_meson.remove("new_mass").update(
                {"mass_updated": m["new_mass"]}
            )

        new_mesons.append(thaw(new_meson))

    return params | {"mesons": new_mesons}


def contract_preprocess_params(params: t.Dict, subconfig: str | None = None) -> t.Dict:
    diagrams = params.get("diagrams", [])
    diagram_params = params.get("diagram_params", [])

    if isinstance(diagrams, list) and len(diagrams) == 0:
        raise ValueError("No diagrams provided in config parameters")
    if len(diagram_params) == 0:
        raise ValueError("No diagram_params provided in config parameters")
    for d in diagrams:
        if d not in diagram_params:
            raise ValueError(f"Diagram {d} not found in diagram_params")

    return params | {
        "diagrams": {k: v for k, v in diagram_params.items() if k in diagrams}
    }


def contract_build_input_params(
    config: ContractConfig,
) -> t.Dict[str, t.Any]:
    input_yaml = {
        "diagrams": {},
        "logging_level": config.logging_level,
        "runid": config.runid,
        "time": config.time,
    }
    for dlabel, diagram in config.diagrams.items():
        input_yaml["diagrams"][dlabel] = diagram_build_input_params(diagram)

    return input_yaml


def diagram_build_aggregator_params(config: DiagramConfig) -> t.Dict:
    agg_params = {"run": ["diagram"], "diagram": {}}

    mass_map = PartialFormatter(mass=config.mass_label)
    infile_stem = config.outfile.filename.format_map(mass_map)

    outfile = (
        config.outfile.filestem.format_map(mass_map)
        .replace("correlators", "processed/{format}")
        .replace("_{series}", "")
    ) + ".h5"

    agg_params["diagram"] = {
        "logging_level": config.logging_level,
        "load_files": {
            "filestem": infile_stem,
            "regex": {"series": "[a-z]", "cfg": "[0-9]+"},
            "dict_labels": ["perm", "gamma"],
        },
        "out_files": {"filestem": outfile},
    }

    actions = {}

    index = ["series.cfg", "gamma"]
    if config.has_high:
        index.append("perm")
    else:
        actions["drop"] = "perm"

    t_order = [f"t{i}" for i in range(1, config.npoint + 1)]
    array_params = {
        "array_order": t_order,
        "array_labels": {},
    }

    t_labels = f"0..{config.time - 1}"
    # TODO: Make time_average bool determined by something else.
    time_average = True
    if time_average:
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


def contract_build_aggregator_params(config: ContractConfig) -> t.Dict:
    agg_params = {"run": []}

    for dlabel, diagram in config.diagrams.items():

        agg_params[dlabel] = diagram_build_aggregator_params(diagram)["diagram"]
        agg_params["run"].append(dlabel)

    return agg_params


def diagram_create_outfile_catalog(config: DiagramConfig) -> pd.DataFrame:
    def generate_outfile_formatting():
        """Generator for meson field file formatting parameters."""
        res = {"mass": config.mass_label}
        yield res, config.outfile

    outfile_generator = generate_outfile_formatting()

    return utils.io.catalog_files(outfile_generator)


def contract_create_outfile_catalog(config: ContractConfig) -> pd.DataFrame:
    df = [diagram_create_outfile_catalog(d) for d in config.diagrams.values()]
    return pd.concat(df)


# Register ContractConfig as the config for 'contract' task type
register_task(
    ContractConfig,
    build_input_params=contract_build_input_params,
    build_aggregator_params=contract_build_aggregator_params,
    create_outfile_catalog=contract_create_outfile_catalog,
    preprocess_params=contract_preprocess_params,
)
register_task(
    DiagramConfig,
    build_input_params=diagram_build_input_params,
    create_outfile_catalog=diagram_create_outfile_catalog,
    build_aggregator_params=diagram_build_aggregator_params,
    preprocess_params=diagram_preprocess_params,
)
register_task(
    MesonLoaderConfig,
    build_input_params=meson_loader_build_input_params,
    preprocess_params=meson_loader_preprocess_params,
)
