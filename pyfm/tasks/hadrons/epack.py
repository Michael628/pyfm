import typing as t
import pandas as pd

from pydantic.dataclasses import dataclass
from pydantic import Field

from pyfm import utils
from pyfm.tasks.hadrons.types import HadronsInput
import pyfm.tasks.hadrons.modules as hadmods
from pyfm.domain import (
    SimpleConfig,
    Outfile,
    MassDict,
)
from pyfm.tasks.register import register_task
from pyfm.tasks.hadrons.types import LanczosParams


@dataclass(frozen=True)
class EpackConfig(SimpleConfig):
    mass: MassDict
    eigs: int
    eig: Outfile
    eigdir: Outfile
    eval: Outfile
    load: bool = True
    lanczos: LanczosParams | None = None
    action_name: str | None = None
    low_modes_name: str | None = None
    multifile: bool = False
    save_eigs: bool = False
    save_evals: bool = True
    mass_shifts: t.List[str] = Field(default_factory=list)

    key: t.ClassVar[str] = "hadrons_epack"

    @property
    def masses(self) -> t.List[str]:
        return [] if self.load == True else ["zero"]

    def __post_init__(self):
        if not self.load:
            if self.lanczos is None:
                raise ValueError(
                    "No Lanczos parameters provided. Required for using IRL solver."
                )
            if self.action_name is None:
                raise ValueError(
                    "No action_name parameter provided. Required for using IRL solver."
                )
        if len(self.mass_shifts) != 0:
            if self.low_modes_name is None:
                raise ValueError(
                    "No low_modes_name parameter provided. Required for shifting mass of eigenvalues."
                )


def build_input_params(config: EpackConfig) -> HadronsInput:
    modules = {}
    schedule = []
    multifile = str(config.multifile).lower()
    epack_path = ""
    if config.load or config.save_eigs:
        epack_path = config.eig.filestem

    # Load or generate eigenvectors
    if config.load:
        modules["epack"] = hadmods.epack_load(
            name="epack",
            filestem=epack_path,
            size=str(config.eigs),
            multifile=multifile,
        )
    else:
        modules["stag_op"] = hadmods.op(
            "stag_op", config.action_name.format(mass="zero")
        )
        schedule.append("stag_op")
        modules["epack"] = hadmods.irl(
            name="epack",
            op="stag_op_schur",
            multifile=multifile,
            output=epack_path,
            **config.lanczos.to_string(),
        )
    schedule.append("epack")

    # Shift mass of eigenvalues
    for mass_label in config.mass_shifts:
        if mass_label == "zero":
            continue
        mass = config.mass.to_string(mass_label)
        name = config.low_modes_name.format(mass=mass_label)
        modules[name] = hadmods.epack_modify(name=name, eigen_pack="epack", mass=mass)
        schedule.append(name)

    if config.save_evals:
        eval_path = config.eval.filestem
        modules["eval_save"] = hadmods.eval_save(
            name="eval_save", eigen_pack="epack", output=eval_path
        )
        schedule.append("eval_save")
    return HadronsInput(modules=modules, schedule=schedule)


def create_outfile_catalog(config: EpackConfig) -> pd.DataFrame:
    def generate_outfile_formatting():
        """
        A generator function that yields file formatting details for different
        components of the task configuration.

        Yields:
            Tuple[Dict[str, Any], str]: A dictionary of replacements and the
            corresponding output file path for each component.
        """

        if config.save_eigs:
            if config.multifile:
                yield ({"eig_index": list(range(int(config.eigs)))}, config.eigdir)
            else:
                yield {}, config.eig
        if config.save_eigs:
            yield {}, config.eval

    outfile_generator = generate_outfile_formatting()

    df = utils.io.catalog_files(outfile_generator)

    return df


# Register GaugeConfig as the config for 'hadrons_gauge' task type
register_task(EpackConfig, build_input_params, create_outfile_catalog)
