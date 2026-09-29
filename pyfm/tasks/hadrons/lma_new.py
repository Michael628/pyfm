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
    # skip_high_modes normally skips this whole section; build_lh_cache's
    # cache-only entries (postprocess_config-flagged hm.cache_only) are the
    # one exception — they still need highmode_v2.build_input_params to
    # emit the writer's noise dependency, just without the per-entry
    # action-module build the mass-loop/solver/quark/contract machinery
    # needs (the noise module has no action/gauge reference of its own).
    active_entries = [
        hm
        for hm in config.high_modes_config
        if not config.skip_high_modes or hm.cache_only
    ]
    if active_entries:
        entry_sp_masses = [
            hm.masses if hm.solver == "mpcg" and not hm.cache_only else []
            for hm in active_entries
        ]
        if any(entry_sp_masses):
            sp_gauge = gauge.build_sp_gauge(config.gauge_config)
            modules |= sp_gauge.modules
            schedule += sp_gauge.schedule

        for hm, sp_masses in zip(active_entries, entry_sp_masses):
            if not hm.cache_only:
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


def _needs_cache(hm: HighModeConfig) -> bool:
    """Whether ``hm`` is a candidate for lh-cache synthesis/cache_only routing.

    Single predicate shared by ``postprocess_config`` (which entries get a
    synthesized writer / ``cache_only=True``) and ``validate_config`` (which
    entries a direct, hand-set ``cache_only=True`` is valid on) — keeps both
    checks from silently drifting apart.
    """
    return hm.low_mode_method == "load" and not hm.skip_low_modes


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

    When ``skip_high_modes`` is also set, the same entries additionally get
    ``cache_only=True`` — ``build_input_params``'s HIGHMODE section then
    still calls ``highmode_v2.build_input_params`` for them (emitting just
    the writer's noise dependency) instead of skipping them outright.
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
        if _needs_cache(hm)
    ]

    if not synthesized:
        return config

    high_modes_config = config.high_modes_config
    if config.skip_high_modes:
        high_modes_config = [
            replace(hm, cache_only=True) if _needs_cache(hm) else hm
            for hm in config.high_modes_config
        ]

    return replace(
        config,
        high_modes_config=high_modes_config,
        meson_config=list(config.meson_config) + synthesized,
        skip_meson=False,
    )


def validate_config(config: LMANewConfig) -> None:
    """Validate LMANewConfig after construction and postprocessing.

    Reuses the schema-agnostic checks (epack/meson consistency, filestem
    collisions) from ``lmi.validate_shared_config`` unchanged, then adds two
    ``lma_new``-specific guards: (1) every load-mode, non-``skip_low_modes``
    ``high_modes`` entry must set ``meson_stoch_proj`` — the loader/producer
    chain (``highmode_v2.build_lma_meson_field_chain``) reads load-cache
    files straight off disk by that entry's filestem, independent of
    whether *this* job's ``meson_config``/``build_lh_cache`` wrote them or a
    separate prior job did (the two-stage build-then-load workflow this
    feature exists for); a missing ``meson_stoch_proj`` is the one thing
    that's wrong in every case, so it's checked unconditionally rather than
    only when ``skip_meson`` is true; (2) ``cache_only`` is only ever set by
    ``postprocess_config`` on entries matching ``_needs_cache``, but
    ``HighModeConfig`` fields route straight from YAML (``lmi.route_params``'s
    ``high_modes_defaults | entry`` layering), so a hand-set
    ``cache_only=True`` on a non-matching entry must be rejected rather than
    silently under-building.
    """
    lmi.validate_shared_config(config)

    for hm in config.high_modes_config:
        if _needs_cache(hm) and hm.meson_stoch_proj is None:
            raise ValueError(
                "A high_modes entry has low_mode_method='load' and "
                "not skip_low_modes, but meson_stoch_proj is unset — the "
                "load-mode loader/producer chain needs a files entry to "
                "read the load-cache meson fields from (whether written by "
                "this job's build_lh_cache synthesis or a prior job's). "
                "Set high_modes.meson_stoch_proj to a files entry."
            )

    for hm in config.high_modes_config:
        if hm.cache_only and not _needs_cache(hm):
            raise ValueError(
                "A high_modes entry has cache_only=True but "
                f"low_mode_method={hm.low_mode_method!r} / "
                f"skip_low_modes={hm.skip_low_modes!r} — cache_only is only "
                "meaningful for low_mode_method='load' entries with low "
                "modes enabled (the same predicate postprocess_config uses "
                "to synthesize the lh-cache writer). Unset cache_only, or "
                "fix low_mode_method/skip_low_modes."
            )


def normalize_params(params: t.Dict) -> t.Dict:
    """lma_new-specific wrapper around ``lmi.normalize_params``.

    ``lmi.normalize_params`` always derives ``skip_high_modes`` from
    ``tasks.high_modes``'s presence (``not entries``) — correct for
    ``hadrons_lmi``, which has no ``build_lh_cache`` concept, but it means a
    cache-only job (real ``tasks.high_modes`` entries, ``skip_high_modes``
    forced ``True``) can never reach ``postprocess_config`` with both set:
    the derived value always wins. ``build_lh_cache`` lives only on
    ``LMANewConfig`` (not ``LMIConfig``), so this override lives here, not
    in the shared ``lmi.py`` hook. When the raw params request both
    ``build_lh_cache`` and ``skip_high_modes``, that explicit intent is
    restored after ``lmi.normalize_params`` runs; otherwise this is a
    pass-through.
    """
    build_lh_cache = params.get("build_lh_cache", False)
    wants_skip_high_modes = params.get("skip_high_modes", False)
    result = lmi.normalize_params(params)
    if build_lh_cache and wants_skip_high_modes:
        result = result | {"skip_high_modes": True}
    return result


# Register LMANewConfig, reusing lmi.py's schema-agnostic hooks by direct
# reference (grid/lma.py's GridLMAConfig precedent) — build_input_params,
# postprocess_config, validate_config, and normalize_params are lma_new-specific.
register_task(
    "hadrons_lma_new",
    LMANewConfig,
    lmi.create_outfile_catalog,
    build_input_params,
    lmi.build_aggregator_params,
    lmi.compare_outputs,
    normalize_params,
    lmi.route_params,
    postprocess_config,
    validate=validate_config,
)
