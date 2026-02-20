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


def get_incomplete_gammas(
    config: MesonConfig, gammas: t.List[Gamma], mass_label: str, bad_files: t.List[str]
) -> bool:
    meson_files = [
        [
            config.meson.filename.format(
                mass=config.mass.to_string(mass_label, remove_prefix=True),
                gamma=g_str,
            )
            for g_str in gamma.gamma_list
        ]
        for gamma in gammas
    ]

    return [
        g for i, g in enumerate(gammas) if any(mf in bad_files for mf in meson_files[i])
    ]


def build_input_params(config: MesonConfig) -> HadronsInput:
    modules = {}
    schedule = []

    meson_template = config.meson.filestem

    bad_files = None
    if not config.overwrite:
        bad_files = utils.io.get_bad_files(create_outfile_catalog(config))

    for op_type, gammas in config.operations.group_by_mass_and_shift():
        assert len(op_type.mass) == 1, "Grouped operations should each have only 1 mass"
        op_label = op_type.gamma.name.lower()
        mass_label = op_type.mass[0]
        gauge = "" if op_type.gamma.local else "gauge"

        if not config.overwrite:
            gammas = get_incomplete_gammas(config, gammas, mass_label, bad_files)

        gamma_string = " ".join([x.gamma_string for x in gammas])

        output = meson_template.format(
            mass=config.mass.to_string(mass_label, remove_prefix=True)
        )

        module_name = f"mf_{op_label}_mass_{mass_label}"

        schedule.append(module_name)
        modules[module_name] = hadmods.meson_field(
            name=module_name,
            action=config.action_name.format(mass=mass_label),
            block=config.blocksize,
            gammas=gamma_string,
            apply_g5=str(config.apply_g5).lower(),
            gauge=gauge,
            low_modes=config.low_modes_name.format(mass=mass_label),
            left="",
            right="",
            output=output,
        )

    return HadronsInput(modules=modules, schedule=schedule)


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


def postprocess_config(config: MesonConfig) -> MesonConfig:
    # Backward compatibility. Convert local operation into pion_local and vec_local
    try:
        local_index = config.op_list.index(Gamma.LOCAL)
    except ValueError:
        return config

    local_op = config.op_list.pop(local_index)
    config.op_list.insert(
        local_index, OpList.Op(gamma=Gamma.VEC_LOCAL, mass=local_op.mass)
    )
    config.op_list.insert(
        local_index, OpList.Op(gamma=Gamma.PION_LOCAL, mass=local_op.mass)
    )
    return config


# Register GaugeConfig as the config for 'hadrons_gauge' task type
register_task(
    MesonConfig,
    build_input_params,
    create_outfile_catalog,
    preprocess_params,
    postprocess_config,
)
