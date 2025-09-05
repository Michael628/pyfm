import typing as t
from typing import List, Dict, Tuple

import pandas as pd

from pyfm import utils
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class ContractTaskConfig(CompositeConfig):
    diagram_params: t.Dict[str, DiagramConfig]
    diagrams: t.List[str]
    hardware: str | None = None
    logging_level: str | None = None


def build_input_params(
    config: ContractConfig,
) -> t.List[str]:
    input_yaml = config.__dict__
    input_yaml["diagrams"] = {}
    for dlabel, diagram in config.diagrams.items():
        input_yaml["diagrams"][dlabel] = diagram.string_dict()

    return input_yaml


# def catalog_files(
#     config: ContractTask,
# ) -> pd.DataFrame:
#     def generate_outfile_formatting():
#         outfile_config = config.files["contract"]
#         for diagram in task_config.diagrams:
#             diagram_replacements: t.Dict = config.diagram_params[diagram].string_dict()
#             yield diagram_replacements, outfile_config
#
#     outfile_generator = generate_outfile_formatting()
#
#     replacements = config.string_dict()
#
#     df = utils.io.catalog_files(outfile_generator, replacements)
#
#     return df
#
#
# def bad_files(
#     config: ContractTask,
# ) -> t.List[str]:
#     df = catalog_files(task_config, config)
#
#     return list(df[(df["file_size"] >= df["good_size"]) != True]["filepath"])
#
#
# def processing_params(
#     config: ContractTask,
# ) -> t.Dict:
#     outfile_dict = config.files
#     infile_stem = outfile_dict["contract"].filename
#     outfile = outfile_dict["contract"].filestem
#     filekeys = utils.format_keys(infile_stem)
#     proc_params: t.Dict[str, t.Any] = {"run": task_config.diagrams}
#     outfile = outfile.replace("correlators", "processed/{format}")
#     outfile = outfile.replace("_{series}", "")
#     outfile += ".h5"
#     replacements = {k: v for k, v in config.string_dict().items() if k in filekeys}
#
#     for diagram_key in task_config.diagrams:
#         diagram = config.diagram_params[diagram_key]
#
#         # TODO: Make processing_params general to 3, and 4pt functions
#         assert diagram.npoint == 2, "Only 2-point diagrams are currently supported"
#
#         diagram_dict = diagram.string_dict()
#         logging_level = config.logging_level if config.logging_level else "INFO"
#         replacements.update({k: v for k, v in diagram_dict.items() if k in filekeys})
#         proc_params[diagram_key] = {
#             "logging_level": logging_level,
#             "load_files": {
#                 "filestem": infile_stem,
#                 "regex": {"series": "[a-z]", "cfg": "[0-9]+"},
#                 "dict_labels": ["seedkey", "gamma"],
#             },
#             "out_files": {"filestem": outfile},
#         }
#         index = ["series.cfg", "gamma"]
#         actions = {}
#         if diagram.has_high:
#             actions["drop"] = "seedkey"
#         else:
#             index.append("seedkey")
#
#         t_order = [f"t{i}" for i in range(1, diagram.npoint + 1)]
#         array_params = {
#             "array_order": t_order,
#             "array_labels": {},
#         }
#
#         t_labels = f"0..{config.time - 1}"
#         # TODO: Make time_average bool determined by something else.
#         time_average = True
#         if time_average:
#             actions["time_average"] = [t_order[0], t_order[-1]]
#             index += t_order[1:-1] + ["t"]
#         else:
#             index += array_params["array_order"]
#
#         actions["index"] = index
#
#         for t_index in array_params["array_order"]:
#             array_params["array_labels"][t_index] = t_labels
#
#         if actions:
#             proc_params[diagram_key]["actions"] = actions
#
#         proc_params[diagram_key]["load_files"]["replacements"] = replacements.copy()
#         proc_params[diagram_key]["load_files"] |= array_params
#
#     return proc_params
