import typing as t
from dataclasses import dataclass

from .. import common
from pyfm.utils import get_logger
from pyfm.core.nanny.registry import register_task, register_builder
from pyfm.domain.common import SimpleConfigBuilder


@dataclass
class RawConfig:
    """Configuration for raw hadrons tasks that load XML files.

    Attributes
    ----------
    formatting : Dict[str, str]
        Dictionary of formatting keys and values for the XML file
        (potentially populated by parameter yaml file).

    xml_file : str
        Path to the XML file to load
    """

    formatting: t.Dict[str, str]
    xml_file: str


@register_task("hadrons", "raw")
@dataclass
class RawTask:
    config: RawConfig

    def write_input_file(self, input_file: str):
        with open(self.config.xml_file, "r") as f:
            contents = f.read().format_map(self.formatting)
        common.write_plain_text(input_file, contents, ext="xml")

    def processing_params(self) -> t.Dict:
        raise NotImplementedError("RawTask does not implement processing_params")

    def catalog_files(self) -> t.List[str]:
        get_logger().warning("RawTask does not implement catalog_files.")
        return []

    def bad_files(self) -> t.List[str]:
        get_logger().warning("RawTask cannot check output files.")
        return []
