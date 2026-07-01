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
    LOAD = 1
    SMEAR = 2


class GaugeFileFormat(SerializableEnum):
    """On-disk format of a gauge configuration, selecting its reader module.

    Each member's value is the Hadrons module that loads that format, so
    :func:`pyfm.tasks.hadrons.modules.load_gauge` can pick the reader from
    ``format.value`` without importing this module (avoiding a circular import).
    """

    ILDG = "MIO::LoadIldg"
    MILCV5 = "MIO::LoadMilc"


# Fat/long link field names are action-type independent. For SMEAR they are the
# outputs of the MGauge::HISQSmear module (named "<module>_fat"/"<module>_long");
# for FREE/LOAD they are directly created fields. One name pair for every action
# type keeps the single-precision cast and the ImprovedStaggered actions free of
# any action_type branching.
_FAT = "gauge_smear_fat"
_LONG = "gauge_smear_long"
# The HISQSmear module name; its outputs are _FAT/_LONG.
_SMEAR_MODULE = "gauge_smear"


@dataclass(frozen=True)
class GaugeConfig(SimpleConfig):
    mass: MassDict
    gauge_links: Outfile
    long_links: Outfile
    fat_links: Outfile
    action_type: ActionType = ActionType.LOAD
    action_name: str | None = None
    save_smear: bool = False
    format: GaugeFileFormat = GaugeFileFormat.ILDG


def build_base_gauge(config: GaugeConfig) -> HadronsInput:
    """Create base gauge modules, including the APBC shift gauge.

    ``action_type`` only controls how the thin gauge and the fat/long links are
    *produced* — the field names they are published under are identical for every
    action type (``gauge``, ``gauge_smear_fat``, ``gauge_smear_long``):

    * ``FREE`` — unit thin gauge, smeared on the fly via ``MGauge::HISQSmear``.
    * ``LOAD`` — thin gauge and pre-smeared fat/long links loaded from disk.
    * ``SMEAR`` — thin gauge loaded, fat/long links derived on the fly via
      ``MGauge::HISQSmear``.

    ``FREE`` and ``SMEAR`` both route the thin gauge through ``HISQSmear`` so
    that the KS phases and boundary conditions are baked into the fat/long links
    (via rephase), matching the convention of the disk-resident links used by
    ``LOAD``. When ``config.save_smear`` is set, those on-the-fly links are also
    written to the ``fat_links``/``long_links`` outfile paths via
    ``MIO::SaveIldg``.
    """
    modules = {}
    schedule = []

    # Thin gauge: FREE uses unit gauge; LOAD/SMEAR load it.
    if config.action_type == ActionType.FREE:
        modules["gauge"] = hadmods.unit_gauge("gauge")
    else:
        modules["gauge"] = hadmods.load_gauge(
            "gauge", config.gauge_links.filestem, config.format.value
        )
    schedule.append("gauge")

    # Fat/long links. Field names are action-type independent; only the producer
    # differs. LOAD reads them from disk (already smeared, KS phases carried);
    # FREE/SMEAR derive them on the fly via HISQSmear, which bakes the KS phases
    # and boundary into the links via rephase.
    if config.action_type == ActionType.LOAD:
        for field, ofile in ((_FAT, config.fat_links), (_LONG, config.long_links)):
            modules[field] = hadmods.load_gauge(
                field, ofile.filestem, config.format.value
            )
            schedule.append(field)
    else:
        modules[_SMEAR_MODULE] = hadmods.hisq_smear(_SMEAR_MODULE, gauge="gauge")
        schedule.append(_SMEAR_MODULE)

        if config.save_smear:
            modules["save_fat"] = hadmods.save_ildg(
                "save_fat", gauge=_FAT, filestem=config.fat_links.filestem
            )
            modules["save_long"] = hadmods.save_ildg(
                "save_long", gauge=_LONG, filestem=config.long_links.filestem
            )
            schedule += ["save_fat", "save_long"]

    modules["gauge_apbc"] = hadmods.apbc_gauge("gauge_apbc", "gauge")
    schedule.append("gauge_apbc")

    return HadronsInput(modules=modules, schedule=schedule)


def build_sp_gauge(config: GaugeConfig) -> HadronsInput:
    """Create single-precision fat/long link modules for mixed-precision solvers.

    Casts the double-precision fat/long links (loaded, unit, or smeared) down to
    single precision. The source field names are action-type independent, so no
    branching on ``action_type`` is needed here.
    """
    modules = {
        f"{smear}f": hadmods.cast_gauge(f"{smear}f", smear) for smear in [_FAT, _LONG]
    }
    return HadronsInput(modules=modules, schedule=list(modules.keys()))


def build_action_modules(
    config: GaugeConfig,
    dp_masses: t.List[str] | None = None,
    sp_masses: t.List[str] | None = None,
) -> HadronsInput:
    """Create ``ImprovedStaggeredMILC`` action modules for double/single precision.

    Args:
        config: GaugeConfig instance
        dp_masses: List of masses requiring double-precision actions
        sp_masses: List of masses requiring single-precision actions

    Returns:
        HadronsInput with action modules and their schedule entries.

    Every action type feeds the fat/long links into the fat/long-link
    ``ImprovedStaggeredMILC`` action. The link field names are action-type
    independent (``gauge_smear_fat``/``gauge_smear_long`` for dp,
    ``gauge_smear_fatf``/``gauge_smear_longf`` for sp).
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
            name=name, mass=mass, gauge_fat=_FAT, gauge_long=_LONG
        )
        schedule.append(name)

    # Single-precision actions
    for mass_label in sp_masses:
        mass = config.mass.to_string(mass_label)
        iname = f"i{config.action_name.format(mass=mass_label)}"
        modules[iname] = hadmods.action_float(
            name=iname, mass=mass, gauge_fat=f"{_FAT}f", gauge_long=f"{_LONG}f"
        )
        schedule.append(iname)

    return HadronsInput(modules=modules, schedule=schedule)


def normalize_params(params: t.Dict) -> t.Dict:
    """Normalize GaugeConfig input: translate the legacy ``free`` flag.

    Legacy configs may pass ``free: true/false`` instead of the canonical
    ``action_type``. This hook runs before routing (and is skipped for
    already-canonical generated inputs) and maps the legacy flag onto
    ``action_type``. An explicit ``action_type`` always wins; absent both, the
    ``LOAD`` default applies at construction.
    """
    combined = params | params.pop("_preprocessor", {})
    if "free" in combined:
        raw = combined.pop("free")
        is_free = raw is True or (
            isinstance(raw, str) and raw.strip().lower() == "true"
        )
        combined.setdefault("action_type", "free" if is_free else "load")
    return combined


# Register GaugeConfig (not as a complete handler task, just for infrastructure)
register_task("hadrons_gauge", GaugeConfig, normalize_params=normalize_params)
