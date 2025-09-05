from .strategy import (
    build_input_params,
    create_outfile_catalog,
    build_aggregator_params,
)

from pyfm.domain import TaskRegistry

from .domain import HighModeConfig

__all__ = [
    "HighModeConfig",
    "build_input_params",
    "create_outfile_catalog",
    "build_aggregator_params",
]

# Register HighModeConfig as the config for 'hadrons_high_modes' task type
TaskRegistry.register_config("hadrons_high_modes", HighModeConfig)

# Register all functions for the 'high_modes' task type
TaskRegistry.register_functions(
    "hadrons_high_modes",
    build_input_params,
    create_outfile_catalog,
    build_aggregator_params,
)
