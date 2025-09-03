import logging
import sys
import typing as t
from dataclasses import fields
from enum import Enum, auto


class Gamma(Enum):
    ONELINK = auto()
    LOCAL = auto()
    VEC_ONELINK = auto()
    VEC_LOCAL = auto()
    PION_LOCAL = auto()
    G1_G1 = auto()
    GX_GX = auto()
    GY_GY = auto()
    GZ_GZ = auto()
    G5_G5 = auto()
    GX_G1 = auto()
    GY_G1 = auto()
    GZ_G1 = auto()

    @property
    def gamma_list(self) -> t.List[str]:
        if self in [Gamma.ONELINK, Gamma.VEC_ONELINK]:
            return ["GX_G1", "GY_G1", "GZ_G1"]
        if self == Gamma.LOCAL:
            return ["G5_G5", "GX_GX", "GY_GY", "GZ_GZ"]
        if self == Gamma.VEC_LOCAL:
            return ["GX_GX", "GY_GY", "GZ_GZ"]
        if self == Gamma.PION_LOCAL:
            return ["G5_G5"]
        else:
            return [self.name]
            # raise ValueError(f"Unexpected Gamma value: {self}")

    @property
    def gamma_string(self) -> str:
        gammas = self.gamma_list
        gammas = [f"({gamma})" for gamma in gammas]
        gammas = " ".join(gammas)
        gammas = gammas.replace("_", " ")
        return gammas

    @property
    def _local_gammas(self) -> t.List:
        return [
            Gamma.LOCAL,
            Gamma.PION_LOCAL,
            Gamma.VEC_LOCAL,
            Gamma.G1_G1,
            Gamma.GX_GX,
            Gamma.GY_GY,
            Gamma.GZ_GZ,
            Gamma.G5_G5,
        ]

    @property
    def local(self) -> bool:
        if self in self._local_gammas:
            return True
        else:
            return False


def setup_logging(logging_level: str):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)-5s - %(message)s",
        style="%",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger().setLevel(logging_level)


BuilderT = t.TypeVar("BuilderT")


class ConfigBuilder(t.Generic[BuilderT]):
    """Base builder for all configuration types."""

    def __init__(self):
        self._config_data: t.Dict[str, t.Any] = {}
        self._format_data: t.Dict[str, t.Callable] = {}
        self._task_fields: t.List[str] = [field.name for field in fields(BuilderT)]

    def with_formatter(self, key: str, value: t.Any):
        if key in self._format_data:
            logging.debug(
                f"Formatting {key} already in format_data with {self._format_data[key]}"
            )
            logging.debug(f"Attempting to replace with {value}")

        if isinstance(value, list):
            self._format_data[key] = [str(item) for item in value]
        elif isinstance(value, bool):
            self._format_data[key] = str(value).lower()
        elif value is not None and not isinstance(value, dict):
            self._format_data[key] = str(value)

        return self

    def with_field(self, key: str, value: t.Any):
        if key in _task_fields:
            self._config_data[key] = value
        else:
            self.with_formatter(key, value)
        return self

    def with_run_params(self, series: str, cfg: str):
        return self.with_field("series", series).with_field("cfg", cfg)

    def get_formatter(self) -> t.Dict[str, str]:
        """Generate string dictionary for formatting."""
        return self._format_data

    def build(self) -> BuilderT:
        """Build the final configuration object. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement build()")
