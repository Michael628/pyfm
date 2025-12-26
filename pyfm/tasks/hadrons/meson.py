import typing as t
from pyfm import utils
import pandas as pd
from pydantic.dataclasses import dataclass
from dataclasses import fields

from pyfm.tasks.hadrons.types import HadronsInput
import pyfm.tasks.hadrons.modules as hadmods
from pyfm.domain import (
    SimpleConfig,
    Gamma,
    OpList,
    Outfile,
    MassDict,
)
from pyfm.tasks.register import register_task


@dataclass(frozen=True)
class MesonConfig(SimpleConfig):
    action_name: str
    low_modes_name: str
    mass: MassDict
    blocksize: int
    operations: OpList
    meson: Outfile
    overwrite: bool = False
    apply_g5: bool = False

    key: t.ClassVar[str] = "hadrons_meson"

    @property
    def op_list(self) -> t.List[OpList.Op]:
        """Get list of gamma operations."""
        return self.operations.op_list

    @property
    def masses(self) -> t.List[str]:
        """Get list of unique mass labels required by all operations."""
        return self.operations.mass


def create_outfile_catalog(config: MesonConfig) -> pd.DataFrame:
    def generate_outfile_formatting():
        """Generator for meson field file formatting parameters."""
        for op in config.op_list:
            res = {
                "gamma": op.gamma.gamma_list,
                "mass": [config.mass.to_string(m, remove_prefix=True) for m in op.mass],
            }
            yield res, config.meson

    outfile_generator = generate_outfile_formatting()

    return utils.io.catalog_files(outfile_generator)


def check_files_complete(
    config: MesonConfig, gammas: t.List[str], mass_label: str, bad_files: t.List[str]
) -> bool:
    meson_files = [
        config.meson.filename.format(
            mass=config.mass.to_string(mass_label, remove_prefix=True),
            gamma=g,
        )
        for g in gammas
    ]

    if not any([mf in bad_files for mf in meson_files]):
        return True
    return False


def build_input_params(config: MesonConfig) -> HadronsInput:
    modules = {}
    schedule = []

    meson_template = config.meson.filestem

    bad_files = None
    if not config.overwrite:
        bad_files = utils.io.get_bad_files(create_outfile_catalog(config))

    for op in config.op_list:
        op_type = op.gamma.name.lower()
        gauge = "" if op.gamma.local else "gauge"

        for mass_label in op.mass:
            if not config.overwrite and check_files_complete(
                config, op.gamma.gamma_list, mass_label, bad_files
            ):
                continue

            output = meson_template.format(
                mass=config.mass.to_string(mass_label, remove_prefix=True)
            )

            module_name = f"mf_{op_type}_mass_{mass_label}"

            schedule.append(module_name)
            modules[module_name] = hadmods.meson_field(
                name=module_name,
                action=config.action_name.format(mass=mass_label),
                block=config.blocksize,
                gammas=op.gamma.gamma_string,
                apply_g5=str(config.apply_g5).lower(),
                gauge=gauge,
                low_modes=config.low_modes_name.format(mass=mass_label),
                left="",
                right="",
                output=output,
            )

    return HadronsInput(modules=modules, schedule=schedule)


def create_outfile_catalog(config: MesonConfig) -> pd.DataFrame:
    def generate_outfile_formatting():
        """
        A generator function that yields file formatting details for different
        components of the task configuration.

        Yields:
            Tuple[Dict[str, Any], str]: A dictionary of replacements and the
            corresponding output file path for each component.
        """

        for op in config.operations.op_list:
            res = {
                "gamma": op.gamma.gamma_list,
                "mass": [config.mass.to_string(m, True) for m in op.mass],
            }
            yield res, config.meson

    outfile_generator = generate_outfile_formatting()

    df = utils.io.catalog_files(outfile_generator)

    return df


def preprocess_params(params: t.Dict, subconfig: str | None = None) -> t.Dict:
    """Preprocessing for MesonConfig.

    Handles routing of task data to 'operations' field to avoid collision
    between MassDict (from params['mass']) and OpList mass labels (from params['_tasks']['mass']).
    """
    # Extract task configs (contains gamma, mass lists for OpList)
    task_data = params.get("_tasks", {})

    # Get field names from MesonConfig, excluding 'mass'
    # - 'mass' comes from top-level params (MassDict)
    config_fields = {f.name for f in fields(MesonConfig) if f.name != "mass"}

    return (
        params
        | {
            "operations": {
                k: v for k, v in task_data.items() if k not in config_fields
            },
            "_tasks": {},
        }
        | {k: v for k, v in task_data.items() if k in config_fields}
    )


# Register GaugeConfig as the config for 'hadrons_gauge' task type
register_task(
    MesonConfig, build_input_params, create_outfile_catalog, preprocess_params
)
