import typing as t
from pydantic.dataclasses import dataclass
from pydantic import Field

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
    action_masses: t.List[str] = Field(default_factory=list)
    sp_masses: t.List[str] = Field(default_factory=list)

    key: t.ClassVar[str] = "hadrons_gauge"

    def __post_init__(self):
        if len(self.action_masses) != 0:
            if self.action_name is None:
                raise ValueError(
                    "No action_name parameter provided. Required when `action_masses` are provided."
                )
        else:
            if len(self.sp_masses) != 0:
                raise ValueError(
                    f"sp_masses ({self.sp_masses}) should be a subset of action_masses ({self.action_masses})"
                )


def build_input_params(
    config: GaugeConfig,
) -> HadronsInput:
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
    has_sp_masses = len(config.sp_masses) != 0
    if has_sp_masses:
        schedule += ["gauge_fatf", "gauge_longf"]
        modules["gauge_fatf"] = hadmods.cast_gauge("gauge_fatf", "gauge_fat")
        modules["gauge_longf"] = hadmods.cast_gauge("gauge_longf", "gauge_long")

    for mass_label in config.action_masses:
        mass = config.mass.to_string(mass_label)
        name = config.action_name.format(mass=mass_label)

        modules[name] = hadmods.action(
            name=name, mass=mass, gauge_fat="gauge_fat", gauge_long="gauge_long"
        )
        schedule.append(name)

        if mass_label in config.sp_masses:
            iname = f"i{name}"
            modules[iname] = hadmods.action_float(
                name=iname,
                mass=mass,
                gauge_fat="gauge_fatf",
                gauge_long="gauge_longf",
            )
            schedule.append(iname)

    return HadronsInput(modules=modules, schedule=schedule)


# Register GaugeConfig as the config for 'hadrons_gauge' task type
register_task(GaugeConfig, build_input_params)
