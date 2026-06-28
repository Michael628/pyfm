import typing as t
from pydantic.dataclasses import dataclass

from pyfm.tasks.hadrons.types import HadronsInput
import pyfm.tasks.hadrons.modules as hadmods
from pyfm.domain import (
    Outfile,
    SimpleConfig,
    MassDict,
    SerializableEnum,
)

from pyfm.tasks.register import register_task


class ActionType(SerializableEnum):
    FREE = 0
    IMPROVED = 1
    HISQ = 2


@dataclass(frozen=True)
class GaugeConfig(SimpleConfig):
    mass: MassDict
    gauge_links: Outfile
    long_links: Outfile
    fat_links: Outfile
    action_type: ActionType = ActionType.IMPROVED
    action_name: str | None = None


def build_base_gauge(config: GaugeConfig) -> HadronsInput:
    """Create base gauge modules, including the APBC shift gauge.

    These are always generated as they are the foundation for all computations.
    ``action_type`` controls construction: ``FREE`` uses unit gauge; ``IMPROVED``
    and ``HISQ`` load gauge fields from disk. For ``HISQ`` the
    ``gauge_fat``/``gauge_long`` modules are omitted (the HISQ action smears the
    thin gauge internally), while ``FREE`` and ``IMPROVED`` still create them.
    """
    modules = {}
    schedule = []

    # Thin gauge: always present. FREE uses unit gauge; IMPROVED/HISQ load it.
    if config.action_type == ActionType.FREE:
        modules["gauge"] = hadmods.unit_gauge("gauge")
    else:
        modules["gauge"] = hadmods.load_gauge("gauge", config.gauge_links.filestem)
    schedule.append("gauge")

    # Fat/long links: required by FREE (unit) and IMPROVED (loaded) actions.
    # Unused by HISQ, which smears the thin gauge internally.
    if config.action_type != ActionType.HISQ:
        for name in ("gauge_fat", "gauge_long"):
            schedule.append(name)
            if config.action_type == ActionType.FREE:
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
    """Create single-precision gauge modules for mixed-precision solvers.

    For ``IMPROVED``/``FREE`` actions these are the single-precision fat/long
    links (``gauge_fatf``, ``gauge_longf``). For ``HISQ`` the action smears the
    thin gauge internally, so only the single-precision thin gauge (``gauge_f``)
    is produced.
    """
    modules = {}
    schedule = []
    if config.action_type == ActionType.HISQ:
        modules["gauge_f"] = hadmods.cast_gauge("gauge_f", "gauge")
        schedule.append("gauge_f")
    else:
        modules["gauge_fatf"] = hadmods.cast_gauge("gauge_fatf", "gauge_fat")
        modules["gauge_longf"] = hadmods.cast_gauge("gauge_longf", "gauge_long")
        schedule += ["gauge_fatf", "gauge_longf"]
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

    ``action_type`` selects the module family: ``IMPROVED``/``FREE`` use the
    fat/long-link ``ImprovedStaggeredMILC`` action; ``HISQ`` uses the
    thin-gauge ``HighlyImprovedStaggeredMILC`` action (which smears internally).
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
        if config.action_type == ActionType.HISQ:
            modules[name] = hadmods.hisq_action(name=name, mass=mass, gauge="gauge")
        else:
            modules[name] = hadmods.action(
                name=name, mass=mass, gauge_fat="gauge_fat", gauge_long="gauge_long"
            )
        schedule.append(name)

    # Single-precision actions
    for mass_label in sp_masses:
        mass = config.mass.to_string(mass_label)
        iname = f"i{config.action_name.format(mass=mass_label)}"
        if config.action_type == ActionType.HISQ:
            modules[iname] = hadmods.hisq_action_float(
                name=iname, mass=mass, gauge="gauge_f"
            )
        else:
            modules[iname] = hadmods.action_float(
                name=iname,
                mass=mass,
                gauge_fat="gauge_fatf",
                gauge_long="gauge_longf",
            )
        schedule.append(iname)

    return HadronsInput(modules=modules, schedule=schedule)


def normalize_params(params: t.Dict) -> t.Dict:
    """Normalize GaugeConfig input: translate the legacy ``free`` flag.

    Legacy configs may pass ``free: true/false`` instead of the canonical
    ``action_type``. This hook runs before routing (and is skipped for
    already-canonical generated inputs) and maps the legacy flag onto
    ``action_type``. An explicit ``action_type`` always wins; absent both, the
    ``IMPROVED`` default applies at construction.
    """
    combined = params | params.pop("_preprocessor", {})
    if "free" in combined:
        raw = combined.pop("free")
        is_free = raw is True or (
            isinstance(raw, str) and raw.strip().lower() == "true"
        )
        combined.setdefault("action_type", "free" if is_free else "improved")
    return combined


# Register GaugeConfig (not as a complete handler task, just for infrastructure)
register_task("hadrons_gauge", GaugeConfig, normalize_params=normalize_params)
