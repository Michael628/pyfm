import typing as t

from pyfm.a2a.types import ContractConfig

from pyfm.tasks.register import register_task

import pandas as pd

from pyfm.tasks.contract import diagram as dmod


def preprocess_params(params: t.Dict) -> t.Dict:
    """Top-level preprocessing for ContractConfig.

    Extracts non-diagram entries from _tasks into top-level params and sets
    _tasks to the list of diagram names to include.
    """
    task_data = params.get("_tasks", {})
    return (
        params
        | {k: v for k, v in task_data.items() if k != "diagrams"}
        | {"_tasks": task_data.get("diagrams", {})}
    )


def preprocess_subconfig(params: t.Dict, subconfig_label: str) -> t.Dict:
    """Per-subconfig preprocessing for ContractConfig.

    For the ``diagrams`` DICT container, constructs the diagrams dict by
    filtering ``diagram_params`` down to the requested diagram names.
    """
    if subconfig_label != "diagrams":
        return params

    diagrams = params.get("_tasks", {})
    diagram_params = params.get("diagram_params", {})

    if isinstance(diagrams, list) and len(diagrams) == 0:
        raise ValueError("No diagrams provided in config parameters")
    if len(diagram_params) == 0:
        raise ValueError("No diagram_params provided in config parameters")
    for d in diagrams:
        if d not in diagram_params:
            raise ValueError(f"Diagram {d} not found in diagram_params")

    return params | {
        "_tasks": {},
        "diagrams": {k: v for k, v in diagram_params.items() if k in diagrams},
    }


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


# Register ContractConfig as the config for 'contract' task type
register_task(
    "contract",
    ContractConfig,
    build_input_params=build_input_params,
    create_outfile_catalog=create_outfile_catalog,
    build_aggregator_params=build_aggregator_params,
    preprocess=preprocess_params,
    preprocess_subconfig=preprocess_subconfig,
)
