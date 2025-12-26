import typing as t

from pyfm.tasks.hadrons.highmode.strategy import (
    build_input_params,
    create_outfile_catalog,
    build_aggregator_params,
    preprocess_params,
)

from pyfm.tasks.register import register_task

from pyfm.tasks.hadrons.highmode.domain import HighModeConfig


def preprocess_params(params: t.Dict) -> t.Dict:
    """Merge the _preprocessor slice into params, wrapping in 'operations' if needed."""
    sub = params.get("_preprocessor", {})
    if "operations" not in sub:
        sub = {"operations": sub}
    return params | sub


__all__ = [
    "build_input_params",
    "create_outfile_catalog",
    "build_aggregator_params",
    "preprocess_params",
]

# Register HighModeConfig as the config for 'hadrons_high_modes' task type
register_task(
    HighModeConfig,
    build_input_params,
    create_outfile_catalog,
    build_aggregator_params,
    preprocess_params=preprocess_params,
)
