from pyfm.tasks.hadrons import gauge, modules, meson, epack, highmode, lmi
from pyfm.tasks.hadrons import meson_v2, highmode_v2, lma_new, sib_mf

from pyfm.tasks.hadrons.lmi import LMIConfig
from pyfm.tasks.hadrons.lma_new import LMANewConfig
from pyfm.tasks.hadrons.sib_mf import SIBMFConfig
from pyfm.tasks.hadrons.types import HighModeConfig

from pyfm.tasks.register import register_task

from pyfm.tasks.hadrons.highmode import (
    build_input_params,
    create_outfile_catalog,
    build_aggregator_params,
    normalize_params,
    route_params,
    validate_config as validate_high_mode_config,
)

hadmods = modules

__all__ = [
    "HighModeConfig",
    "LMIConfig",
    "LMANewConfig",
    "SIBMFConfig",
    "hadmods",
    "gauge",
    "meson",
    "epack",
    "highmode",
    "lmi",
    "meson_v2",
    "highmode_v2",
    "lma_new",
    "sib_mf",
]

# Register HighModeConfig as the config for 'hadrons_high_modes' task type
register_task(
    "hadrons_high_modes",
    HighModeConfig,
    build_input_params,
    create_outfile_catalog,
    build_aggregator_params,
    route_params,
    normalize_params=normalize_params,
    validate=validate_high_mode_config,
)
