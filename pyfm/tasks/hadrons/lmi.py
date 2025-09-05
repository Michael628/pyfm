import typing as t
import pandas as pd

from pydantic.dataclasses import dataclass
from dataclasses import replace

from pyfm.domain import TaskRegistry, CompositeConfig, HadronsInput
from . import gauge, meson, epack, highmode


# ============LMI Task Configuration===========
@dataclass(frozen=True)
class LMIConfig(CompositeConfig):
    gauge_config: gauge.GaugeConfig
    epack_config: epack.EpackConfig | None = None
    meson_config: meson.MesonConfig | None = None
    high_modes_config: highmode.HighModeConfig | None = None


def preprocess_params(params: t.Dict, subconfig: str | None = None) -> t.Dict:
    """Perform any necessary modifications to task input parameters before they
    are passed to the subtask constructor.
    """

    if subconfig is None:
        return params

    key = subconfig.removesuffix("_config")
    sub_params = params.get(key, {})

    if key == "high_modes" or key == "meson" and "operations" not in sub_params:
        sub_params = {"operations": sub_params}

    action_name = "stag_mass_{mass}"
    solver_name = "stag_{solver}_mass_{mass}"
    low_modes_name = "evecs_mass_{mass}"
    shift_gauge_name = "gauge"
    if key == "meson":
        return {
            "action_name": action_name,
            "low_modes_name": low_modes_name,
        } | sub_params
    elif key == "gauge":
        return {
            "action_name": action_name,
        } | sub_params
    elif key == "epack":
        return {
            "action_name": action_name,
            "low_modes_name": low_modes_name,
        } | sub_params
    elif key == "high_modes":
        return {
            "shift_gauge_name": shift_gauge_name,
            "action_name": action_name,
            "solver_name": solver_name,
            "low_modes_name": low_modes_name,
            "skip_low_modes": "epack_config" in params,
        } | sub_params
    else:
        raise ValueError(f"Unknown subconfig: {subconfig}")


def postprocess_config(config: LMIConfig) -> LMIConfig:
    """Update the subtask config properties to reflect the needs of the other subtasks.

    In this case,
     - the epack_config gets updated with any masses used in meson or high_modes.
     - the high gets updated with any masses used in meson or high_modes.

    """

    def update_single_precision(config: LMIConfig) -> LMIConfig:
        new_config = config
        if config.high_modes_config and config.high_modes_config.solver == "mpcg":
            new_gauge_config = replace(
                config.gauge_config, sp_masses=config.high_modes_config.masses
            )
            new_config = replace(new_config, gauge_config=new_gauge_config)

        return new_config

    def update_masses(config: LMIConfig) -> LMIConfig:
        masses = set()
        if config.meson_config:
            masses |= set(config.meson_config.masses)
        if config.high_modes_config:
            masses |= set(config.high_modes_config.masses)
        if config.epack_config:
            masses |= set(config.epack_config.masses)
        masses = list(masses)

        gauge_with_masses = replace(config.gauge_config, action_masses=masses)
        new_config = replace(config, gauge_config=gauge_with_masses)

        if config.epack_config:
            epack_with_masses = replace(new_config.epack_config, mass_shifts=masses)
            new_config = replace(new_config, epack_config=epack_with_masses)

        return new_config

    new_config = update_masses(config)
    new_config = update_single_precision(new_config)

    return new_config


def build_input_params(config: LMIConfig) -> HadronsInput:
    modules, schedule = gauge.build_input_params(config.gauge_config)
    if config.epack_config:
        m, s = epack.build_input_params(config.epack_config)
        modules |= m
        schedule += s
    if config.meson_config:
        m, s = meson.build_input_params(config.meson_config)
        modules |= m
        schedule += s
    if config.high_modes_config:
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


def build_aggregator_params(
    config: LMIConfig,
) -> t.Dict:
    return (
        {}
        if not config.high_modes_config
        else highmode.build_aggregator_params(config.high_modes_config)
    )


# Register SmearConfig as the config for 'smear' task type
TaskRegistry.register_config("hadrons_lmi", LMIConfig)

# Register all functions for the 'smear' task type
TaskRegistry.register_functions(
    "hadrons_lmi",
    create_outfile_catalog,
    build_input_params,
    build_aggregator_params,
    preprocess_params,
    postprocess_config,
)
