import typing as t
from pydantic.dataclasses import dataclass

from pyfm.tasks.hadrons.types import HadronsInput
import pyfm.tasks.hadrons.modules as hadmods
from pyfm.domain import (
    Outfile,
    SimpleConfig,
    MassDict,
)

from pyfm.tasks.register import register_task


@dataclass(frozen=True)
class GaugeConfig(SimpleConfig):
    mass: MassDict
    gauge_links: Outfile
    long_links: Outfile
    fat_links: Outfile
    free: bool = False
    action_name: str | None = None


def build_base_gauge(config: GaugeConfig) -> HadronsInput:
    """Create base gauge modules, including the APBC shift gauge.

    These are always generated as they are the foundation for all computations.
    """
    modules = {}
    schedule = ["gauge", "gauge_fat", "gauge_long"]
    for name in schedule:
        if config.free:
            modules[name] = hadmods.unit_gauge(name)
        else:
            ofile_label = f"{name.split('_')[-1]}_links"
            modules[name] = hadmods.load_gauge(
                name, getattr(config, ofile_label).filestem
            )

    modules["gauge_apbc"] = hadmods.apbc_gauge("gauge_apbc", "gauge")
    schedule.append("gauge_apbc")

    return HadronsInput(modules=modules, schedule=schedule)


def build_sp_gauge(config: GaugeConfig) -> HadronsInput:
    """Create single-precision gauge modules: gauge_fatf, gauge_longf.

    These are generated when needed (e.g., when highmode uses mixed precision).
    """
    modules = {}
    schedule = ["gauge_fatf", "gauge_longf"]
    modules["gauge_fatf"] = hadmods.cast_gauge("gauge_fatf", "gauge_fat")
    modules["gauge_longf"] = hadmods.cast_gauge("gauge_longf", "gauge_long")
    return HadronsInput(modules=modules, schedule=schedule)


def build_action_modules(
    config: GaugeConfig,
    dp_masses: t.List[str] | None = None,
    sp_masses: t.List[str] | None = None,
) -> HadronsInput:
    """Create action modules for double and single precision.

    Args:
        config: GaugeConfig instance
        dp_masses: List of masses requiring double-precision actions
        sp_masses: List of masses requiring single-precision actions

    Returns:
        HadronsInput with action modules and their schedule entries.
    """
    if dp_masses is None:
        dp_masses = []
    if sp_masses is None:
        sp_masses = []

    modules = {}
    schedule = []

    # Double-precision actions
    for mass_label in dp_masses:
        mass = config.mass.to_string(mass_label)
        name = config.action_name.format(mass=mass_label)
        modules[name] = hadmods.action(
            name=name, mass=mass, gauge_fat="gauge_fat", gauge_long="gauge_long"
        )
        schedule.append(name)

    # Single-precision actions
    for mass_label in sp_masses:
        mass = config.mass.to_string(mass_label)
        iname = f"i{config.action_name.format(mass=mass_label)}"
        modules[iname] = hadmods.action_float(
            name=iname,
            mass=mass,
            gauge_fat="gauge_fatf",
            gauge_long="gauge_longf",
        )
        schedule.append(iname)

    return HadronsInput(modules=modules, schedule=schedule)


# Register GaugeConfig (not as a complete handler task, just for infrastructure)
register_task("hadrons_gauge", GaugeConfig)
