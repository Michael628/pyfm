from pyfm.tasks.hadrons import gauge, modules, meson, epack, highmode, lmi

from pyfm.tasks.hadrons.types import HighModeConfig

from pyfm.tasks.register import register_task

from pyfm.tasks.hadrons.highmode import (
    build_input_params,
    create_outfile_catalog,
    build_aggregator_params,
)

hadmods = modules

__all__ = ["HighModeConfig", "hadmods", "gauge", "meson", "epack", "highmode", "lmi"]

# Register HighModeConfig as the config for 'hadrons_high_modes' task type
register_task(
    HighModeConfig, build_input_params, create_outfile_catalog, build_aggregator_params
)
