import typing as t

from pyfm.utils.string import PartialFormatter
from pyfm.a2a.types import MesonLoaderConfig

from pyfm.tasks.register import register_task


def normalize_params(params: t.Dict) -> t.Dict:
    """Normalize MesonLoaderConfig input: assemble the ``mass_shift`` field.

    Collapses the loose ``mass_original``/``mass_updated``/``milc_mass`` keys into
    the single ``mass_shift`` mapping the config expects. (Routing is the default
    ``_preprocessor`` absorb — no custom ``route`` hook needed for this leaf.)
    """
    combined_params = params | params.pop("_preprocessor", {})

    mass_shift = {
        key.removeprefix("mass_"): combined_params[key]
        for key in ["mass_original", "mass_updated", "milc_mass"]
        if key in combined_params
    }

    return combined_params | dict(mass_shift=mass_shift)


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
    "contract_mesonloader",
    MesonLoaderConfig,
    build_input_params=build_input_params,
    normalize_params=normalize_params,
)
