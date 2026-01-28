import typing as t

from pyfm.utils.string import PartialFormatter
from pyfm.a2a.types import MesonLoaderConfig

from pyfm.tasks.register import register_task


def preprocess_params(params: t.Dict, subconfig: str | None = None) -> t.Dict:
    """Preprocessing for MesonLoaderConfig."""
    task_data = params.get("_tasks", {})

    # Flatten params and task_data
    result = params | task_data

    # Build mass_shift from combined data
    mass_shift = {
        key.removeprefix("mass_"): result[key]
        for key in ["mass_original", "mass_updated", "milc_mass"]
        if key in result
    }

    return params | task_data | {"_tasks": {}, "mass_shift": mass_shift}


def build_input_params(config: MesonLoaderConfig) -> t.Dict[str, t.Any]:
    mass_map = PartialFormatter(mass=config.get_mass_label(include_shift=False))
    yaml_params = {
        "mass": config.mass._asdict(),
        "file": config.file.format_map(mass_map),
        "mass_shift": config.mass_shift._asdict(),
    }
    if config.evalfile is not None:
        yaml_params["evalfile"] = config.evalfile.format_map(mass_map)

    return yaml_params


register_task(
    MesonLoaderConfig,
    build_input_params=build_input_params,
    preprocess_params=preprocess_params,
)
