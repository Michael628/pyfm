import typing as t
from dataclasses import replace

from pydantic.dataclasses import dataclass

from pyfm.tasks.hadrons.types import HadronsInput
from pyfm.domain import CompositeConfig
from pyfm.tasks.register import register_task

from . import gauge, meson, epack, meson_v2, highmode_v2, lmi
from .types import HighModeConfig


@dataclass(frozen=True)
class LMANewConfig(CompositeConfig):
    """HadronsMILC-SpinTaste-API-aware sibling of ``LMIConfig``.

    Reuses ``gauge.GaugeConfig``/``epack.EpackConfig``/``meson.MesonConfig``/
    ``HighModeConfig`` as field types unchanged (mirrors
    ``grid/lma.py``'s ``GridLMAConfig`` precedent) — a genuinely new class,
    not a subclass or reuse of ``LMIConfig`` (``register.py``'s
    ``_config_to_task_key`` reverse-lookup keys by class object; reusing
    ``LMIConfig`` verbatim would silently steal ``hadrons_lmi``'s
    registration).
    """

    gauge_config: gauge.GaugeConfig
    epack_config: epack.EpackConfig
    meson_config: t.List[meson.MesonConfig]
    high_modes_config: t.List[HighModeConfig]
    skip_epack: bool = False
    skip_meson: bool = False
    skip_high_modes: bool = False
    build_lh_cache: bool = False

    @property
    def split_mpi_layout(self) -> str | None:
        """Re-expose the split-grid MPI layout from the high-modes subconfigs.

        Identical semantics to ``LMIConfig.split_mpi_layout`` (duplicated,
        not inherited — see the class docstring on why ``LMANewConfig``
        isn't a subclass): the first entry that sets a layout wins;
        conflicting layouts are a hard misconfiguration (one ``<split>`` per
        XML job). ``None`` when no entry opts in or the list is empty.
        """
        layouts = [
            hm.split_mpi_layout
            for hm in self.high_modes_config
            if hm.split_mpi_layout is not None
        ]
        if not layouts:
            return None
        if any(layout != layouts[0] for layout in layouts[1:]):
            raise ValueError(
                "Conflicting split_mpi_layout values across high_modes_config "
                f"entries: {layouts!r}; a job admits exactly one <split>."
            )
        return layouts[0]


def build_input_params(config: LMANewConfig) -> HadronsInput:
    """Generate input parameters for the full LMA-new task.

    Structurally identical to ``lmi.build_input_params`` — same
    gauge/epack orchestration (spin-taste-free, reused verbatim) — only the
    meson and high-modes sections differ: ``meson_v2.build_input_params``
    and ``highmode_v2.build_input_params`` target the canonical
    (SpinTaste-module-based) HadronsMILC API instead of ``lmi.py``'s
    Legacy modules.
    """
    modules = {}
    schedule = []

    # 1. Always start with base gauge
    base_gauge = gauge.build_base_gauge(config.gauge_config)
    modules |= base_gauge.modules
    schedule += base_gauge.schedule

    # 2. EPACK section: generate actions then compute
    if not config.skip_epack:
        epack_masses = config.epack_config.masses
        actions = gauge.build_action_modules(
            config.gauge_config, dp_masses=epack_masses
        )
        modules |= actions.modules
        schedule += actions.schedule

        epack_input = epack.build_input_params(config.epack_config)
        modules |= epack_input.modules
        schedule += epack_input.schedule

        # Handle epack mass shifts for meson and every high-mode entry
        epack_mass_shifts = []
        if not config.skip_meson:
            for mc in config.meson_config:
                epack_mass_shifts.extend(mc.masses)
        for hm in config.high_modes_config:
            epack_mass_shifts.extend(hm.masses)

        if epack_mass_shifts:
            mass_shifts_input = epack.build_epack_mass_shifts(
                config.epack_config, epack_mass_shifts
            )
            modules |= mass_shifts_input.modules
            schedule += mass_shifts_input.schedule

    # 3. MESON section: generate actions then compute, once per entry
    # (canonical schema).
    if not config.skip_meson:
        meson_masses = []
        for mc in config.meson_config:
            meson_masses.extend(mc.masses)
        actions = gauge.build_action_modules(
            config.gauge_config, dp_masses=meson_masses
        )
        modules |= actions.modules
        schedule += actions.schedule

        for mc in config.meson_config:
            meson_input = meson_v2.build_input_params(mc)
            modules |= meson_input.modules
            schedule += meson_input.schedule

    # 4. HIGHMODE section: generate actions then compute, once per entry
    # (canonical schema).
    if not config.skip_high_modes:
        entry_sp_masses = [
            hm.masses if hm.solver == "mpcg" else []
            for hm in config.high_modes_config
        ]
        if any(entry_sp_masses):
            sp_gauge = gauge.build_sp_gauge(config.gauge_config)
            modules |= sp_gauge.modules
            schedule += sp_gauge.schedule

        for hm, sp_masses in zip(config.high_modes_config, entry_sp_masses):
            actions = gauge.build_action_modules(
                config.gauge_config, dp_masses=hm.masses, sp_masses=sp_masses
            )
            modules |= actions.modules
            schedule += actions.schedule

            hm_input = highmode_v2.build_input_params(hm)
            modules |= hm_input.modules
            schedule += hm_input.schedule

    # Deduplicate schedule: keep first occurrence of each module name
    deduplicated_schedule = list(dict.fromkeys(schedule))

    return HadronsInput(modules=modules, schedule=deduplicated_schedule)


def postprocess_config(config: LMANewConfig) -> LMANewConfig:
    """Synthesize a load-cache MesonConfig entry per applicable high_modes entry.

    When ``build_lh_cache`` is set, every ``high_modes_config`` entry with
    ``low_mode_method == 'load'`` and ``not skip_low_modes`` gets a matching
    ``MesonConfig`` appended to ``meson_config`` — its ``operations``
    copied verbatim from the high-modes entry's own. ``apply_g5=True``
    reproduces the old writer's G5-folding through the existing,
    unmodified per-shift-group loop in ``meson_v2.build_input_params``
    (see the design's Decisions for why a verbatim copy suffices — no
    new branch, no cross-section SpinTaste-name coupling needed).
    ``skip_meson`` is flipped to ``False`` whenever at least one entry is
    synthesized, so ``validate_config``'s guards stay consistent.
    """
    if not config.build_lh_cache:
        return config

    for hm in config.high_modes_config:
        if (
            hm.low_mode_method == "load"
            and not hm.skip_low_modes
            and hm.meson_stoch_proj is None
        ):
            raise ValueError(
                "high_modes entry has low_mode_method='load' and "
                "not skip_low_modes, but meson_stoch_proj is unset — "
                "build_lh_cache synthesis needs a files entry to write the "
                "load-cache meson fields to. Set high_modes.meson_stoch_proj "
                "to a files entry."
            )

    synthesized = [
        meson.MesonConfig(
            formatting=hm.formatting,
            logging_level=hm.logging_level,
            runid=hm.runid,
            action_name=hm.action_name,
            low_modes_name=hm.low_modes_name,
            mass=hm.mass,
            blocksize=hm.blocksize,
            operations=hm.operations,
            meson=hm.meson_stoch_proj,
            overwrite=hm.overwrite,
            apply_g5=True,
            shift_gauge_name=hm.shift_gauge_name,
            high_left_name="",
            high_right_name=f"{hm.noise_name}_vec",
        )
        for hm in config.high_modes_config
        if hm.low_mode_method == "load" and not hm.skip_low_modes
    ]

    if not synthesized:
        return config

    return replace(
        config,
        meson_config=list(config.meson_config) + synthesized,
        skip_meson=False,
    )


def validate_config(config: LMANewConfig) -> None:
    """Validate LMANewConfig after construction and postprocessing.

    Reuses the schema-agnostic checks (epack/meson consistency, filestem
    collisions) from ``lmi.validate_shared_config`` unchanged, then adds
    one ``lma_new``-specific guard: ``postprocess_config`` guarantees
    ``skip_meson=False`` whenever it actually synthesizes an entry, so a
    ``high_modes`` entry needing load-mode caching with ``skip_meson``
    still true means neither ``build_lh_cache`` synthesis nor a
    hand-authored ``tasks.meson`` entry covers it — the loader/producer
    chain would reference a writer output nothing produces.
    """
    lmi.validate_shared_config(config)

    if config.skip_meson:
        needs_cache = any(
            hm.low_mode_method == "load" and not hm.skip_low_modes
            for hm in config.high_modes_config
        )
        if needs_cache:
            raise ValueError(
                "A high_modes entry has low_mode_method='load' and "
                "not skip_low_modes, but skip_meson is true (no meson "
                "entries exist) and build_lh_cache is "
                f"{config.build_lh_cache!r} — the load-mode loader/producer "
                "chain would reference a writer output nothing produces. "
                "Set build_lh_cache=True, or hand-author a tasks.meson "
                "entry producing the required load-cache files."
            )


# Register LMANewConfig, reusing lmi.py's schema-agnostic hooks by direct
# reference (grid/lma.py's GridLMAConfig precedent) — only build_input_params,
# postprocess_config, and validate_config are lma_new-specific.
register_task(
    "hadrons_lma_new",
    LMANewConfig,
    lmi.create_outfile_catalog,
    build_input_params,
    lmi.build_aggregator_params,
    lmi.compare_outputs,
    lmi.normalize_params,
    lmi.route_params,
    postprocess_config,
    validate=validate_config,
)
