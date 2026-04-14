from pyfm.tasks.hadrons import gauge, modules, meson, epack, highmode, lmi

from pyfm.tasks.hadrons.lmi import LMIConfig
from pyfm.tasks.hadrons.types import HighModeConfig

from pyfm.tasks.register import register_task

from pyfm.tasks.hadrons.highmode import (
    build_input_params,
    create_outfile_catalog,
    build_aggregator_params,
    preprocess_params,
    validate_config as validate_high_mode_config,
)

hadmods = modules

__all__ = [
    "HighModeConfig",
    "LMIConfig",
    "hadmods",
    "gauge",
    "meson",
    "epack",
    "highmode",
    "lmi",
]

# Register HighModeConfig as the config for 'hadrons_high_modes' task type
register_task(
    "hadrons_high_modes",
    HighModeConfig,
    build_input_params=build_input_params,
    create_outfile_catalog=create_outfile_catalog,
    build_aggregator_params=build_aggregator_params,
    preprocess=preprocess_params,
    validate=validate_high_mode_config,
)
