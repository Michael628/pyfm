import typing as t
import pandas as pd

from pydantic.dataclasses import dataclass
from dataclasses import replace

from pyfm import utils
from pyfm.tasks.hadrons.types import HadronsInput
from pyfm.domain import CompositeConfig
from pyfm.tasks.register import register_task

from . import gauge, meson, epack, highmode
from .types import HighModeConfig


@dataclass(frozen=True)
class LMIConfig(CompositeConfig):
    gauge_config: gauge.GaugeConfig
    epack_config: epack.EpackConfig
    meson_config: meson.MesonConfig
    high_modes_config: HighModeConfig
    skip_epack: bool = False
    skip_meson: bool = False
    skip_high_modes: bool = False

    key: t.ClassVar[str] = "hadrons_lmi"

    def __post_init__(self):
        for k, skip in [
            (k, getattr(self, f"skip_{k}")) for k in ["meson", "high_modes", "epack"]
        ]:
            if skip:
                utils.get_logger().debug(f"Skipping {k} step")

        if self.skip_epack and not self.skip_meson:
            raise ValueError(
                "Epack parameters must be set to perform meson calculation"
            )


def preprocess_params(params: t.Dict) -> t.Dict:
    """Perform any necessary modifications to task input parameters before they
    are passed to the subtask constructor.
    """
    preprocessor = params.get("_preprocessor", {})

    # Skip configs where user provides no _preprocessor entry
    optional_configs = ["meson", "high_modes", "epack"]
    skip_optional = {
        f"skip_{k}": True for k in optional_configs if k not in preprocessor
    }

    action_name = "stag_mass_{mass}"
    solver_name = "stag_{solver}_mass_{mass}"
    low_modes_name = "evecs_mass_{mass}"
    shift_gauge_name = "gauge"

    return params | skip_optional | {
        "action_name": action_name,
        "solver_name": solver_name,
        "low_modes_name": low_modes_name,
        "shift_gauge_name": shift_gauge_name,
        "skip_low_modes": "epack" not in preprocessor,
    }


def postprocess_config(config: LMIConfig) -> LMIConfig:
    """Update the subtask config properties to reflect the needs of the other subtasks.

    In this case,
     - the epack_config gets updated with any masses used in meson or high_modes.
     - the high gets updated with any masses used in meson or high_modes.

    """

    def update_single_precision(config: LMIConfig) -> LMIConfig:
        new_config = config
        if not config.skip_high_modes and config.high_modes_config.solver == "mpcg":
            new_gauge_config = replace(
                config.gauge_config, sp_masses=config.high_modes_config.masses
            )
            new_config = replace(new_config, gauge_config=new_gauge_config)

        return new_config

    def update_masses(config: LMIConfig) -> LMIConfig:
        masses = set()
        if not config.skip_meson:
            masses |= set(config.meson_config.masses)
        if not config.skip_high_modes:
            masses |= set(config.high_modes_config.masses)
        if not config.skip_epack:
            masses |= set(config.epack_config.masses)
        masses = list(masses)

        gauge_with_masses = replace(config.gauge_config, action_masses=masses)
        new_config = replace(config, gauge_config=gauge_with_masses)

        if not config.skip_epack:
            epack_with_masses = replace(new_config.epack_config, mass_shifts=masses)
            new_config = replace(new_config, epack_config=epack_with_masses)

        return new_config

    new_config = update_masses(config)
    new_config = update_single_precision(new_config)

    return new_config


def build_input_params(config: LMIConfig) -> HadronsInput:
    modules, schedule = gauge.build_input_params(config.gauge_config)
    if not config.skip_epack:
        m, s = epack.build_input_params(config.epack_config)
        modules |= m
        schedule += s
    if not config.skip_meson:
        m, s = meson.build_input_params(config.meson_config)
        modules |= m
        schedule += s
    if not config.skip_high_modes:
        m, s = highmode.build_input_params(config.high_modes_config)
        modules |= m
        schedule += s

    return HadronsInput(modules=modules, schedule=schedule)


def create_outfile_catalog(config: LMIConfig) -> pd.DataFrame:
    df = [
        m.create_outfile_catalog(c)
        for m, c in zip(
            [epack, meson, highmode],
            [config.epack_config, config.meson_config, config.high_modes_config],
        )
        if c is not None
    ]
    return pd.concat(df)


def build_aggregator_params(config: LMIConfig, average: bool) -> t.Dict:
    return (
        highmode.build_aggregator_params(config.high_modes_config, average)
        if not config.skip_high_modes
        else {}
    )


# Register SmearConfig as the config for 'smear' task type
register_task(
    LMIConfig,
    create_outfile_catalog,
    build_input_params,
    build_aggregator_params,
    preprocess_params,
    postprocess_config,
)
