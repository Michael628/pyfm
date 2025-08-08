import typing as t
from pydantic.dataclasses import dataclass

from pyfm.nanny import TaskBase
from pyfm.nanny.tasks.hadrons import SubmitHadronsConfig


@dataclass
class RawTask(TaskBase):
    """Configuration for raw hadrons tasks that load XML files.

    Attributes
    ----------
    xml_file : str
        Path to the XML file to load
    """

    xml_file: str = ""


def input_params(task: RawTask, _: SubmitHadronsConfig) -> t.Tuple[t.Dict, None]:
    return {"xml_file": task.xml_file}, None


def processing_params(task: RawTask, config: SubmitHadronsConfig) -> t.Dict:
    return {}


def catalog_files(task: RawTask, config: SubmitHadronsConfig) -> t.List[str]:
    return []


def bad_files(task: RawTask, config: SubmitHadronsConfig) -> t.List[str]:
    return []


def get_task_factory():
    return RawTask.from_dict
