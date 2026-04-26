import typing as t
import pandas as pd

from pydantic.dataclasses import dataclass

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



def preprocess_params(params: t.Dict) -> t.Dict:
    """Perform any necessary modifications to task input parameters before they
    are passed to the subtask constructor.
    """

    ACTION_NAME = "stag_mass_{mass}"
    SOLVER_NAME = "stag_{solver}_mass_{mass}"
    LOW_MODES_NAME = "evecs_mass_{mass}"
    SHIFT_GAUGE_NAME = "gauge"

    # Extract task configs (may not exist for all callers)
    preprocessor_params = params.pop("_preprocessor", {})

    # Skip configs where user provides no input
    optional_configs = ["meson", "high_modes", "epack"]
    skip_flags = {
        f"skip_{k}": True for k in optional_configs if k not in preprocessor_params
    }

    # Set defaults for child configs
    child_preprocessor = dict(
        gauge_config=dict(action_name=ACTION_NAME),
        epack_config=dict(
            action_name=ACTION_NAME,
            low_modes_name=LOW_MODES_NAME,
        ),
        meson_config=dict(
            action_name=ACTION_NAME,
            shift_gauge_name=SHIFT_GAUGE_NAME,
            low_modes_name=LOW_MODES_NAME,
        ),
        high_modes_config=dict(
            action_name=ACTION_NAME,
            low_modes_name=LOW_MODES_NAME,
            solver_name=SOLVER_NAME,
            shift_gauge_name=SHIFT_GAUGE_NAME,
            skip_low_modes="epack" not in preprocessor_params,
        ),
    )

    # Update child processor with corresponding params passed to parent
    for k, v in preprocessor_params.items():
        child_preprocessor[f"{k}_config"] |= v

    return params | skip_flags | dict(_preprocessor=child_preprocessor)


def validate_config(config: LMIConfig) -> None:
    """Validate LMIConfig after construction and postprocessing.

    Validates that if epack is skipped, meson must also be skipped.
    """

    for k, skip in [
        (k, getattr(config, f"skip_{k}")) for k in ["meson", "high_modes", "epack"]
    ]:
        if skip:
            utils.get_logger().debug(f"Skipping {k} step")

    if config.skip_epack and not config.skip_meson:
        raise ValueError("Epack parameters must be set to perform meson calculation")


def build_input_params(config: LMIConfig) -> HadronsInput:
    """Generate input parameters for the full LMI task.

    Orchestrates gauge module generation with submodule computation, ensuring that
    gauge action modules are generated only when needed by the submodules that use them.
    """
    modules = {}
    schedule = []

    # 1. Always start with base gauge
    base_gauge = gauge.build_base_gauge(config.gauge_config)
    modules |= base_gauge.modules
    schedule += base_gauge.schedule

    # 2. EPACK section: generate actions then compute
    if not config.skip_epack:
        epack_masses = config.epack_config.masses
        actions = gauge.build_action_modules(
            config.gauge_config, dp_masses=epack_masses
        )
        modules |= actions.modules
        schedule += actions.schedule

        epack_input = epack.build_input_params(config.epack_config)
        modules |= epack_input.modules
        schedule += epack_input.schedule

        # Handle epack mass shifts for meson and highmode
        epack_mass_shifts = []
        if not config.skip_meson:
            epack_mass_shifts.extend(config.meson_config.masses)
        if not config.skip_high_modes:
            epack_mass_shifts.extend(config.high_modes_config.masses)

        if epack_mass_shifts:
            mass_shifts_input = epack.build_epack_mass_shifts(
                config.epack_config, epack_mass_shifts
            )
            modules |= mass_shifts_input.modules
            schedule += mass_shifts_input.schedule

    # 3. MESON section: generate actions then compute
    if not config.skip_meson:
        meson_masses = config.meson_config.masses
        actions = gauge.build_action_modules(
            config.gauge_config, dp_masses=meson_masses
        )
        modules |= actions.modules
        schedule += actions.schedule

        meson_input = meson.build_input_params(config.meson_config)
        modules |= meson_input.modules
        schedule += meson_input.schedule

    # 4. HIGHMODE section: generate actions then compute
    if not config.skip_high_modes:
        # Compute sp_masses if highmode uses mixed precision
        highmode_sp_masses = []
        if config.high_modes_config.solver == "mpcg":
            highmode_sp_masses = config.high_modes_config.masses
            sp_gauge = gauge.build_sp_gauge(config.gauge_config)
            modules |= sp_gauge.modules
            schedule += sp_gauge.schedule

        highmode_masses = config.high_modes_config.masses
        actions = gauge.build_action_modules(
            config.gauge_config, dp_masses=highmode_masses, sp_masses=highmode_sp_masses
        )
        modules |= actions.modules
        schedule += actions.schedule

        highmode_input = highmode.build_input_params(config.high_modes_config)
        modules |= highmode_input.modules
        schedule += highmode_input.schedule

    # Deduplicate schedule: keep first occurrence of each module name
    deduplicated_schedule = list(dict.fromkeys(schedule))

    return HadronsInput(modules=modules, schedule=deduplicated_schedule)


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


# Register LMIConfig with all handlers
register_task(
    "hadrons_lmi",
    LMIConfig,
    create_outfile_catalog,
    build_input_params,
    build_aggregator_params,
    preprocess_params,
    validate=validate_config,
)
