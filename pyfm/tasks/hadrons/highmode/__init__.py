from pyfm.tasks.hadrons.highmode.strategy import (
    build_input_params,
    create_outfile_catalog,
    build_aggregator_params,
)

from pyfm.tasks.register import register_task

from pyfm.tasks.hadrons.highmode.domain import HighModeConfig

__all__ = [
    "HighModeConfig",
    "build_input_params",
    "create_outfile_catalog",
    "build_aggregator_params",
]

# Register HighModeConfig as the config for 'hadrons_high_modes' task type
register_task(
    HighModeConfig, build_input_params, create_outfile_catalog, build_aggregator_params
)
