import logging
import os
import typing as t
from dataclasses import field

import pandas as pd
from pydantic.dataclasses import dataclass

from pyfm import config as c
from pyfm import utils
from pyfm.a2a.config import DiagramConfig
from pyfm.nanny import SubmitConfig, TaskBase


@c.dataclass_with_getters
class SubmitContractConfig(SubmitConfig):
    _diagram_params: t.Dict[str, DiagramConfig] = field(default_factory=dict)
    series: t.Optional[str] = None
    cfg: t.Optional[str] = None
    hardware: t.Optional[str] = None
    logging_level: t.Optional[str] = None

    def __init__(self, **kwargs):
        params = kwargs.copy()
        self.hardware = params.pop("hardware", None)
        self.logging_level = params.pop("logging_level", None)
        self.series = params.pop("series", None)
        self.cfg = params.pop("cfg", None)
        self._diagram_params = {}
        for k, v in params.pop("_diagram_params", {}).items():
            self._diagram_params[k] = DiagramConfig.create(**v)
        super().__init__(**params)


@dataclass
class ContractTask(TaskBase):
    diagrams: t.List[str]


def input_params(
    task_config: ContractTask,
    submit_config: SubmitContractConfig,
) -> t.Tuple[t.List[str], None]:
    input_yaml = submit_config.public_dict
    input_yaml["diagrams"] = {}
    for diagram in task_config.diagrams:
        d_params = submit_config.diagram_params[diagram]
        d_params.set_filenames(submit_config.files)
        input_yaml["diagrams"][diagram] = d_params.string_dict()

    return input_yaml, None


def catalog_files(
    task_config: ContractTask,
    submit_config: SubmitContractConfig,
) -> t.List[str]:
    def generate_outfile_formatting():
        outfile_config = submit_config.files["contract"]
        for diagram in task_config.diagrams:
            diagram_replacements: t.Dict = submit_config.diagram_params[
                diagram
            ].string_dict()
            yield diagram_replacements, outfile_config

    outfile_generator = generate_outfile_formatting()

    replacements = submit_config.string_dict()

    df = utils.catalog_files(outfile_generator, replacements)

    return df


def bad_files(
    task_config: ContractTask,
    submit_config: SubmitContractConfig,
) -> t.List[str]:
    df = catalog_files(task_config, submit_config)

    return list(df[(df["file_size"] >= df["good_size"]) != True]["filepath"])


def processing_params(
    task_config: ContractTask,
    submit_config: SubmitContractConfig,
) -> t.Dict:
    outfile_dict = submit_config.files
    infile_stem = outfile_dict["contract"].filename
    outfile = outfile_dict["contract"].filestem
    filekeys = utils.format_keys(infile_stem)
    proc_params: t.Dict[str, t.Any] = {"run": task_config.diagrams}
    outfile = outfile.replace("correlators", "dataframes")
    outfile = outfile.replace("_{series}", "")
    outfile += ".h5"
    replacements = {
        k: v for k, v in submit_config.string_dict().items() if k in filekeys
    }

    for diagram_key in task_config.diagrams:
        diagram = submit_config.diagram_params[diagram_key]

        # TODO: Make processing_params general to 3, and 4pt functions
        assert diagram.npoint == 2, "Only 2-point diagrams are currently supported"

        diagram_dict = diagram.string_dict()
        logging_level = (
            submit_config.logging_level if submit_config.logging_level else "INFO"
        )
        replacements.update({k: v for k, v in diagram_dict.items() if k in filekeys})
        proc_params[diagram_key] = {
            "logging_level": logging_level,
            "load_files": {
                "filestem": infile_stem,
                "regex": {"series": "[a-z]", "cfg": "[0-9]+"},
                "dict_labels": ["seedkey", "gamma"],
            },
            "out_files": {"filestem": outfile, "type": "dataframe"},
        }
        index = ["series.cfg", "gamma"]
        actions = {}
        if diagram.has_high:
            actions["drop"] = "seedkey"
        else:
            index.append("seedkey")

        t_order = [f"t{i}" for i in range(1, diagram.npoint + 1)]
        array_params = {
            "array_order": t_order,
            "array_labels": {},
        }

        t_labels = f"0..{submit_config.time - 1}"
        # TODO: Make time_average bool determined by something else.
        time_average = True
        if time_average:
            actions["time_average"] = [t_order[0], t_order[-1]]
            index += t_order[1:-1] + ["t"]
        else:
            index += array_params["order"]

        actions["index"] = index

        for t_index in array_params["order"]:
            array_params["labels"][t_index] = t_labels

        if actions:
            proc_params[diagram_key]["actions"] = actions

        proc_params[diagram_key]["load_files"]["replacements"] = replacements.copy()
        proc_params[diagram_key]["load_files"] |= array_params

    return proc_params


def get_task_factory():
    return ContractTask.from_dict


def get_submit_factory() -> t.Callable[..., SubmitContractConfig]:
    return SubmitContractConfig.create
