"""``hadrons_lma_tiered``: the TIERED channel of ``hadrons_lma_new``.

A thin task surface over ``lma_new`` — same config tree (one added
program-level knob), same routing/validation/hooks wrapped around
lma_new's, same registration hooks reused verbatim. ``lma_new`` itself
is untouched; the gather emission deltas land in Phase 3.
"""

import typing as t

import pandas as pd
from pydantic import Field
from pydantic.dataclasses import dataclass

from pyfm.tasks.hadrons.types import HadronsInput, SolveCrossTerms
from pyfm.tasks.register import register_task

from . import epack, gauge, highmode_v2, lma_new, meson, meson_v2
from .lma_new import LMANewConfig, _merge_modules
from .lma_new import (
    build_aggregator_params,
    compare_outputs,
    create_outfile_catalog,
    normalize_params,
)


# --- Slice 1: LMATieredConfig (new module; imports beside lma_new's) ---
@dataclass(frozen=True)
class LMATieredConfig(LMANewConfig):
    """Config for ``hadrons_lma_tiered``: the TIERED channel of
    ``hadrons_lma_new``, owning the program-level ``subgrid_gather`` knob.
    A subclass, not verbatim reuse (registration keys by class object; the
    parent's entry survives — verified 2026-11-10) and not a duplicate
    (fields, phasing knobs, split_mpi_layout/subgrid_ranks inherited)."""

    subgrid_gather: bool = False


# --- Slice 1: route_params — a thin wrapper around lma_new's table ---
def route_params(params: t.Dict) -> t.Dict:
    """lma_tiered routing: lma_new's routing table plus the gather knob.

    The knob is popped BEFORE delegating — lma_new's generic loop folds
    leftover slice keys into ``{k}_config`` children, which for a bool
    crashes (``dict |= bool``) — then re-merged as a plain top-level param,
    the same shape the ``_PROGRAM_KNOBS`` pop produces.
    """
    incoming = params.get("_preprocessor", {})
    subgrid_gather = incoming.pop("subgrid_gather", False)
    if not isinstance(subgrid_gather, bool):
        raise TypeError(
            "tasks.subgrid_gather must be a boolean; got "
            f"{type(subgrid_gather).__name__}."
        )
    routed = lma_new.route_params(params)
    return routed | {"subgrid_gather": subgrid_gather}


# --- Slice 1: validate_config — lma_new's checks verbatim, then the
# --- split-only gate and the TIERED admission rule (D8) ---
def validate_config(config: LMATieredConfig) -> None:
    lma_new.validate_config(config)
    if config.subgrid_gather and config.split_mpi_layout is None:
        raise ValueError(
            "subgrid_gather requires split-grid: set split_mpi_layout and "
            "subgrid_ranks on the high_modes entries — the gather moves "
            "world-grid ranLL producers around the split phases."
        )
    non_tiered = [
        label
        for label, hm in config.high_modes_config.items()
        if hm.effective_solve_cross_terms is not SolveCrossTerms.TIERED
        and not highmode_v2.is_cache_build_only(hm)
    ]
    if non_tiered:
        raise ValueError(
            "hadrons_lma_tiered solves TIERED programs only; entries "
            f"{sorted(non_tiered)} do not run solve_cross_terms tiered — "
            "use hadrons_lma_new for full/diagonal programs (the gather "
            "qualifies nothing there)."
        )


def _option_leaves(options: t.Dict) -> t.Iterator[str]:
    """Every whitespace-separated string token in a module's options (the
    module-reference convention: names appear as bare tokens, mirroring
    the schedule-dependency walkers)."""
    for value in options.values():
        if isinstance(value, dict):
            yield from _option_leaves(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    yield from item.split()
        elif isinstance(value, str):
            yield from value.split()


def _validate_split_world_reads(modules: t.Dict[str, t.Dict]) -> None:
    """D4: in a split program every world-grid read of a tagged module's
    output must target a ``MUtilities::GammaMapElement``. Hadrons
    (bf1c6bbc) gathers elements at the producing phase's exit and aborts
    on any world read of an ungatherable ``TGammaMap`` — so any other
    world read of a tagged output is a build-time bug, not a runtime
    surprise."""
    tagged = {
        name for name, module in modules.items() if module.get("subgrid") is not None
    }
    if not tagged:
        return
    for name, module in modules.items():
        if module.get("subgrid") is not None:
            continue
        for leaf in _option_leaves(module["options"]):
            if leaf in tagged:
                src = modules[leaf]
                if src["id"]["type"] != "MUtilities::GammaMapElement":
                    raise ValueError(
                        f"Module {name!r} reads the tagged output {leaf!r} of a "
                        f"{src['id']['type']} module from the world grid: only "
                        "MUtilities::GammaMapElement outputs gather (Hadrons "
                        "bf1c6bbc); a TGammaMap world read aborts at the "
                        "producing phase's exit."
                    )


def build_input_params(config: LMATieredConfig) -> HadronsInput:
    """Generate input parameters for the tiered LMA task.

    Verbatim mirror of ``lma_new.build_input_params``: base gauge → epack
    actions → epack → epack mass shifts (meson + every entry) → user meson
    entries → high-mode entries (per-entry actions, then the entry's own
    emission) → pooled split program → dedup. No gather deltas yet — the
    three gather deltas (design D5) land in Phase 3; with the flag False
    this is byte-identical to lma_new's emission.
    """
    modules = {}
    schedule = []

    # 1. Always start with base gauge
    base_gauge = gauge.build_base_gauge(config.gauge_config)
    _merge_modules(modules, base_gauge.modules)
    schedule += base_gauge.schedule

    # 2. EPACK section (required)
    epack_masses = config.epack_config.masses
    actions = gauge.build_action_modules(config.gauge_config, dp_masses=epack_masses)
    _merge_modules(modules, actions.modules)
    schedule += actions.schedule

    epack_input = epack.build_input_params(config.epack_config)
    _merge_modules(modules, epack_input.modules)
    schedule += epack_input.schedule

    # Handle epack mass shifts for meson and every high-mode entry.
    epack_mass_shifts = [m for mc in config.meson_config for m in mc.masses]
    for hm in config.high_modes_config.values():
        epack_mass_shifts.extend(hm.masses)
    if epack_mass_shifts:
        mass_shifts_input = epack.build_epack_mass_shifts(
            config.epack_config, epack_mass_shifts
        )
        _merge_modules(modules, mass_shifts_input.modules)
        schedule += mass_shifts_input.schedule

    # 3. MESON section (presence-driven)
    if config.meson_config:
        meson_masses = [m for mc in config.meson_config for m in mc.masses]
        actions = gauge.build_action_modules(config.gauge_config, dp_masses=meson_masses)
        _merge_modules(modules, actions.modules)
        schedule += actions.schedule

        for mc in config.meson_config:
            meson_input = meson_v2.build_input_params(mc)
            _merge_modules(modules, meson_input.modules)
            schedule += meson_input.schedule

    # 4. HIGHMODE section, once per keyed entry. Per-entry sp masses stay
    # solver-scoped (an entry's masses join the sp set only when that
    # entry itself solves mixed-precision); build_only entries contribute
    # none (they have no cg block).
    hms = list(config.high_modes_config.values())
    entry_sp_masses = [
        hm.masses
        if hm.cg_config is not None and hm.cg_config.solver == "mpcg"
        else []
        for hm in hms
    ]
    if any(entry_sp_masses):
        sp_gauge = gauge.build_sp_gauge(config.gauge_config)
        _merge_modules(modules, sp_gauge.modules)
        schedule += sp_gauge.schedule

    # Split entries with refs to run are emitted TOGETHER, after the
    # per-entry sections, as one pooled split program (phases outer,
    # entries inner); build_only split entries emit no tagged modules
    # and stay per-entry.
    emitting_split = [
        hm
        for hm in hms
        if hm.subgrid_ranks is not None
        and not highmode_v2.is_cache_build_only(hm)
    ]
    pooled_ids = {id(hm) for hm in emitting_split}

    for hm, sp_masses in zip(hms, entry_sp_masses):
        # Unconditional — build_only entries need their dp actions too:
        # the writer's EigenPackCBPairs modules reference stag_mass_<mass>
        # (sp_masses is [] for them: no cg block, nothing solver-scoped).
        actions = gauge.build_action_modules(
            config.gauge_config, dp_masses=hm.masses, sp_masses=sp_masses
        )
        _merge_modules(modules, actions.modules)
        schedule += actions.schedule

        if id(hm) in pooled_ids:
            continue  # emitted below as part of the pooled split program
        # delta 1 — the per-entry call passes the flag too (it serves only
        # non-split/build_only entries post-split-phases; split entries skip
        # it via `pooled_ids`; qualifying is empty unsplit, so harmless):
        hm_input = highmode_v2.build_input_params(
            hm, subgrid_gather=config.subgrid_gather
        )
        _merge_modules(modules, hm_input.modules)
        schedule += hm_input.schedule

    if emitting_split:
        _, placements, n_phases = highmode_v2.pooled_placement(
            emitting_split,
            config.subgrid_ranks,
            n_phases=config.subgrid_phases,
            phase_size=config.subgrid_phase_size,
            last_size=config.subgrid_last_phase_size,
        )
        split_input = highmode_v2.build_split_program(
            emitting_split, placements, n_phases,
            subgrid_gather=config.subgrid_gather,
        )
        _merge_modules(modules, split_input.modules)
        schedule += split_input.schedule

    # delta 3 — after the pooled split emission, before the schedule dedup
    # so the guard sees the merged program — D4: world reads of tagged
    # outputs must target gatherable elements only — every lma_tiered
    # split program, gather or not.
    if config.split_mpi_layout is not None:
        _validate_split_world_reads(modules)

    # Deduplicate schedule: keep first occurrence of each module name
    deduplicated_schedule = list(dict.fromkeys(schedule))

    return HadronsInput(modules=modules, schedule=deduplicated_schedule)


# --- Slice 1: registration — lma_new's catalog/aggregator/compare/
# --- normalize hooks are reused verbatim (an LMATieredConfig instance
# --- satisfies their LMANewConfig annotations) ---
register_task(
    "hadrons_lma_tiered",
    LMATieredConfig,
    create_outfile_catalog,
    build_input_params,
    build_aggregator_params,
    compare_outputs,
    normalize_params,
    route_params,
    validate=validate_config,
)
