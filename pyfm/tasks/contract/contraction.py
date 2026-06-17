import typing as t

from pyfm.a2a.types import ContractConfig

from pyfm.tasks.register import register_task

import pandas as pd

from pyfm.tasks.contract import diagram as dmod


def normalize_params(params: t.Dict) -> t.Dict:
    """Normalize ContractConfig input: select the requested diagrams.

    Broad input supplies the full ``diagram_params`` catalog plus a ``diagrams``
    list naming the subset to run. Canonical output replaces both with a single
    ``diagrams`` map of the selected diagrams.
    """
    combined_params = params | params.pop("_preprocessor", {})

    diagrams = combined_params.pop("diagrams", [])
    diagram_params = combined_params.pop("diagram_params", {})

    if isinstance(diagrams, list) and len(diagrams) == 0:
        raise ValueError("No diagrams provided in config parameters")
    if len(diagram_params) == 0:
        raise ValueError("No diagram_params provided in config parameters")
    for d in diagrams:
        if d not in diagram_params:
            raise ValueError(f"Diagram {d} not found in diagram_params")

    filtered_diagrams = {k: v for k, v in diagram_params.items() if k in diagrams}
    return combined_params | dict(diagrams=filtered_diagrams)


def route_params(params: t.Dict) -> t.Dict:
    """Route the canonical ``diagrams`` map to per-diagram subconfigs."""
    combined_params = params | params.pop("_preprocessor", {})

    diagrams = combined_params.pop("diagrams", {})
    return combined_params | dict(
        diagrams={k: {} for k in diagrams},
        _preprocessor=dict(diagrams=diagrams),
    )


def build_input_params(
    config: ContractConfig,
) -> t.Dict[str, t.Any]:
    input_yaml = {
        "diagrams": {},
        "logging_level": config.logging_level,
        "runid": config.runid,
        "time": config.time,
    }
    for dlabel, diagram in config.diagrams.items():
        input_yaml["diagrams"][dlabel] = dmod.build_input_params(diagram)

    return input_yaml


def build_aggregator_params(config: ContractConfig, average: bool) -> t.Dict:
    agg_params = {"run": []}

    for dlabel, diagram in config.diagrams.items():

        agg_params[dlabel] = dmod.build_aggregator_params(diagram, average)["diagram"]
        agg_params["run"].append(dlabel)

    return agg_params


def create_outfile_catalog(config: ContractConfig) -> pd.DataFrame:
    df = [dmod.create_outfile_catalog(d) for d in config.diagrams.values()]
    return pd.concat(df)


def validate_config(config: ContractConfig) -> None:
    if len(config.diagrams) == 0:
        raise ValueError("ContractConfig.diagrams must not be empty")


# Register ContractConfig as the config for 'contract' task type
register_task(
    "contract",
    ContractConfig,
    build_input_params,
    build_aggregator_params,
    create_outfile_catalog,
    normalize_params,
    route_params,
    validate=validate_config,
)
