"""Per-entry module emission for ``hadrons_lma_new`` high-mode entries.

Fully separate from the legacy ``highmode/`` package (D4): owns the entry
catalog (:func:`create_outfile_catalog`, serving the resume gate here and
aggregation/comparison from the task level), the ranLL gamma demand set
(:func:`needed_ranll_gammas`), the schedule sort, the loader/producer
meson-field chain (one producer per source slice, scheduled with that
slice's consumers), and the
per-entry emission order from the design doc — SpinTaste + sink →
full-volume noise (meson_field) → cache writer (``build_and_load`` /
``build_only``) → stop for ``build_only`` → per-source noise → per-mass
low-mode solvers / meson-field chains / CG solvers → quarks +
contractions, sorted.
"""

import itertools
import re
import typing as t

import h5py
import numpy as np
import pandas as pd
from pyrsistent import freeze, thaw

from pyfm import utils
from pyfm.domain import Gamma
from pyfm.tasks.hadrons.types import HadronsInput, SourceRef, SolveCrossTerms
import pyfm.tasks.hadrons.modules as hadmods
from pyfm.tasks.hadrons import meson, meson_v2
from pyfm.tasks.hadrons.highmode_v2 import twopoint
from pyfm.tasks.hadrons.highmode_v2.phases import (
    PooledRef,
    RefPlacement,
    plan_program,
)
from pyfm.tasks.hadrons.highmode_v2.config import (
    CacheMode,
    LMAHighModeConfig,
    LowModeMethod,
    PreconMode,
)


def create_outfile_catalog(config: LMAHighModeConfig) -> pd.DataFrame:
    if config.output_config is None:
        # Cache 'build_only' entries have no correlator outputs to catalog;
        # the task-level catalog (lma_new) iterates every entry, so this
        # must stay total.
        return pd.DataFrame()

    def generate_outfile_formatting():
        solver_labels = config.get_solver_labels()
        res = {"tsource": config.sources_config.source_axis, "dset": solver_labels}

        for op in config.op_list:
            res["gamma_label"] = op.gamma.name.lower()
            res["mass"] = config.get_mass_labels(op)
            yield res, config.output_config.file

    outfile_generator = generate_outfile_formatting()

    return utils.io.catalog_files(outfile_generator)


def needed_ranll_gammas(config: LMAHighModeConfig) -> t.Dict[str, t.List[Gamma]]:
    """Per-mass ranLL gamma demand for the meson-field chain.

    Derived from ``twopoint.quark_gen`` — the same demand-driven set the
    quark emission uses — so the producer set can never drift from the
    consumer references: each (gamma, mass) entry here corresponds exactly
    to the per-slice ``quark_ranLL_{glabel}_mass_{m}_{label}[_t{t0}]``
    producers, each of whose module name is the propagator contractions
    and the ama guess chain reference.
    """
    needed: t.Dict[str, t.List[Gamma]] = {}
    for op in twopoint.quark_gen(config):
        if op.solver == "ranLL":
            needed.setdefault(op.mass, []).append(op.gamma)
    return needed


def sort_schedule(config: LMAHighModeConfig, module_names: t.List[str]) -> t.List[str]:
    gammas = ["pion_local", "scalar_local", "vec_local", "vec_onelink"]

    def gamma_order(name):
        for i, gamma in enumerate(gammas):
            if gamma in name:
                return i
        return -1

    def mass_order(name):
        for i, mass in enumerate(config.mass.keys()):
            if f"mass_{mass}" in name:
                return i
        return -1

    def mixed_solvers_last(name):
        # Two-list ranking: modules matching a cross label rank strictly
        # above every base rank; everything else ranks against the base
        # list. Under TIERED the `ama` dset leaves the dset list while ama
        # quark modules persist; ranking those against the dset list would
        # invert the ranLL->ama precon order (quark_gen's guess chain).
        base_labels = config.get_solver_labels(skip_cross=True)
        cross_labels = [
            l for l in config.get_solver_labels() if l not in base_labels
        ]
        for i, label in reversed(list(enumerate(cross_labels))):
            if label in name:
                return len(base_labels) + i
        for i, label in reversed(list(enumerate(base_labels))):
            if label in name:
                return i
        return -1

    def mixed_mass_last(name):
        return len(re.findall(r"_mass", name))

    # Biased labels are n{i} and a biased producer is ..._n{i}_t{t0}: key
    # on the block index so it sorts with its ..._n{i} consumers. Grid
    # labels (and producers) carry t{t0} directly.
    slice_pattern = (
        r"_n(\d+)" if config.sources_config.biased_config is not None else r"_t(\d+)"
    )

    def tslice_order(name):
        time = re.findall(slice_pattern, name)
        if len(time):
            return int(time[0])
        else:
            return -1

    sorted_modules = sorted(module_names, key=gamma_order)
    sorted_modules = sorted(sorted_modules, key=mass_order)
    sorted_modules = sorted(sorted_modules, key=mixed_mass_last)
    sorted_modules = sorted(sorted_modules, key=mixed_solvers_last)
    sorted_modules = sorted(sorted_modules, key=tslice_order)

    return sorted_modules


def reorder_cross_contractions(
    schedule: t.List[str],
    modules: t.Dict[str, t.Dict],
    config: LMAHighModeConfig,
) -> t.List[str]:
    """Anchor each cross contraction just after its first diagonal reader.

    ``sort_schedule`` ranks a source's cross contractions after its whole
    ladder, keeping every ama quark of the source live on its subgrid at
    once. Each cross moves to just after the first module in the block
    that reads its ama-side field — the ``source`` option for
    ``ama_ranLL``-shaped labels (the ama quark), the ``sink`` option (the
    antiquark elem) for ``ranLL_ama``-shaped ones — which is the first
    diagonal contraction reading that field (O's ``reorder`` with
    ``sink_first=True``); with no diagonal reader (TIERED drops the HH
    diagonal), just after the module named by that field (its producer:
    the ama quark or the elem bridge). Crosses on one anchor keep their
    prior order; everything else is untouched; tags never change. Always
    on — split or not: the order changes no result, only how many ama
    fields are live at once.
    """
    base_labels = config.get_solver_labels(skip_cross=True)
    cross_labels = [l for l in config.get_solver_labels() if l not in base_labels]
    if not cross_labels:
        return list(schedule)

    def is_cross(name: str) -> bool:
        return any(label in name for label in cross_labels)

    crosses = {
        name
        for name in schedule
        if is_cross(name)
        and modules[name]["id"]["type"] == "MContraction::StagMeson"
    }
    if not crosses:
        return list(schedule)

    pure = [name for name in schedule if name not in crosses]

    def ama_field(module: t.Dict) -> str:
        # solver labels are quark-first: an ama quark side puts the ama
        # field on `source`; otherwise (ranLL_ama) it is the antiquark
        # elem on `sink`
        source = module["options"]["source"]
        return source if "quark_ama" in source else module["options"]["sink"]

    def reads(module: t.Dict, field: str) -> bool:
        options = module["options"]
        return field in (options.get("source"), options.get("sink"))

    anchored: t.Dict[str, t.List[str]] = {}
    for name in schedule:                     # prior order within an anchor
        if name not in crosses:
            continue
        module = modules[name]
        field = ama_field(module)
        anchor = next(
            (
                other
                for other in pure
                if modules[other]["id"]["type"] == "MContraction::StagMeson"
                and not is_cross(other)
                and reads(modules[other], field)
            ),
            None,
        )
        if anchor is None:
            # TIERED: no diagonal reads the field — anchor at its own
            # producer (the ama quark / the elem bridge)
            anchor = field if field in pure else None
        if anchor is not None:
            anchored.setdefault(anchor, []).append(name)
        # anchor None (field not in the block): the cross keeps its
        # sorted position (emitted in place below)
    placed: t.Set[str] = set()
    for anchored_crosses in anchored.values():
        placed.update(anchored_crosses)
    out: t.List[str] = []
    for name in schedule:                     # original order throughout
        if name in placed:
            continue                          # emitted after its anchor
        out.append(name)
        for cross in anchored.pop(name, []):
            cx, ax = modules[cross], modules[name]
            assert (cx.get("subgrid"), cx.get("splitPhase")) == (
                ax.get("subgrid"),
                ax.get("splitPhase"),
            ), cross
            out.append(cross)
    return out


def gather_qualifying_gammas(config: LMAHighModeConfig) -> t.FrozenSet[Gamma]:
    """RanLL gammas whose per-slice maps the subgrid gather may move (D3).

    The TIERED special case: under ``solve_cross_terms`` TIERED the entry's
    only ama solves are the contract gamma's (the LH contractions' antiquark
    side), and those solves guess their own maps — so every OTHER
    operator's map has no tagged reader besides the LH contractions that
    move with it, and it never needs to reach a subgrid. Under any other
    effective mode nothing qualifies ("in a full program no map qualifies
    and nothing moves"): a full entry's ama solves guess their own
    operator's maps.
    """
    if config.effective_solve_cross_terms is not SolveCrossTerms.TIERED:
        return frozenset()
    contract = {con.antiquark.gamma for _, con in twopoint.contraction_gen(config)}
    ranll = {op.gamma for op in twopoint.quark_gen(config) if op.solver == "ranLL"}
    return frozenset(ranll - contract)


def _gather_post_schedule(
    config: LMAHighModeConfig,
    chunk: t.List[SourceRef],
    qualifying: t.FrozenSet[Gamma],
    emitted_names: t.AbstractSet[str],
    contract_inputs: HadronsInput,
) -> t.List[str]:
    """The gather's post-phase schedule for one phase (D6 order): per
    slice (t0 ascending), per mass, per gamma (``sort_schedule``'s gamma
    precedence) — the ranLL map, its LL contraction, then each rung's LH
    contraction in ``get_solver_labels`` order. Membership is decided
    against the emitted module set, so the list can never name a module
    the phase did not emit; contractions are identified by their
    ``source`` (the op-gamma ranLL map name — LL and LH alike carry it on
    the quark side).
    """
    if not qualifying:
        return []
    solver_labels = config.get_solver_labels()
    precedence = ["pion_local", "scalar_local", "vec_local", "vec_onelink"]

    qual_maps = {
        twopoint.quark_name(config, "ranLL", g.name.lower(), mass_label, ref)
        for mass_label in config.masses
        for g in qualifying
        for ref in chunk
    }
    moved: t.Dict[str, t.List[str]] = {}
    for name in contract_inputs.schedule:
        module = contract_inputs.modules[name]
        if (
            module["id"]["type"] == "MContraction::StagMeson"
            and module["options"]["source"] in qual_maps
        ):
            moved.setdefault(module["options"]["source"], []).append(name)

    def _solver_key(name: str) -> int:
        # `corr_{solver_label}_{glabel}_mass...` (label-prefixed names:
        # split after "corr_"): recover the solver label as the LONGEST
        # get_solver_labels() entry matching on a "_" boundary, so
        # ranLL_ama_0.0001 outranks bare ranLL.
        rest = name.split("corr_", 1)[1]
        best = len(solver_labels)
        for i, label in enumerate(solver_labels):
            if (rest == label or rest.startswith(f"{label}_")) and (
                best == len(solver_labels) or len(label) > len(solver_labels[best])
            ):
                best = i
        return best

    for names in moved.values():
        names.sort(key=_solver_key)

    def _gamma_key(g: Gamma):
        glabel = g.name.lower()
        return (0, precedence.index(glabel), "") if glabel in precedence else (1, 0, glabel)

    post: t.List[str] = []
    for ref in sorted(chunk, key=lambda r: r.t0):
        for mass_label in config.masses:
            for g in sorted(qualifying, key=_gamma_key):
                map_name = twopoint.quark_name(
                    config, "ranLL", g.name.lower(), mass_label, ref
                )
                if map_name not in emitted_names:
                    continue  # gamma not demanded for this mass
                post.append(map_name)
                post.extend(moved.get(map_name, []))
    return post


def _cache_writer_config(config: LMAHighModeConfig) -> meson.MesonConfig:
    """Synthesize the per-entry LH-cache writer (D9; never appended to
    ``meson_config``). ``apply_g5=True`` reproduces the writer's G5-folding
    through ``meson_v2``'s per-shift-group loop; the entry prefix (via
    ``meson_v2.build_input_params``) keeps keyed entries' writers disjoint
    while an unkeyed entry's writer stays byte-identical to the old
    synthesized one."""
    mf = config.low_modes_config.meson_field_config
    return meson.MesonConfig(
        formatting=config.formatting,
        logging_level=config.logging_level,
        runid=config.runid,
        action_name=config.action_name,
        low_modes_name=config.low_modes_name,
        mass=config.mass,
        blocksize=mf.blocksize,
        operations=config.operations,
        meson=mf.file,
        overwrite=(
            config.output_config.overwrite
            if config.output_config is not None
            else False
        ),
        apply_g5=True,
        shift_gauge_name=config.shift_gauge_name,
        high_left_name="",
        high_right_name=f"{config.noise_name}_vec",
        # StagLMAMesonFieldProp requires the 2*nEvec |e+o>/|e-o> row layout
        cb_pairs=True,
    )


def build_lma_meson_field_chain(
    config: LMAHighModeConfig,
    mass_label: str,
    action: str,
    low_modes: str,
    gammas: t.List[Gamma],
    spintaste_names: t.Dict[Gamma, str],
    refs: t.List[SourceRef],
) -> t.Tuple[HadronsInput, HadronsInput]:
    """Loader + producer halves of the meson-field chain (per entry).

    Loaders: one per (mass, conjugated gamma), label-prefixed, reading the
    entry's cache files by the writer's filestem grammar, shared by every
    slice. Producers: one per (gamma, mass, ref) in both source modes, a
    single-time window ``tA=tB=ref.t0`` (``tStep=1``). A single-slice
    ``StagLMAMesonFieldProp`` publishes under its own module name, so
    ``twopoint.meson_field_producer_name`` is both the module and the
    object contractions and the ama guess chain reference.

    Returned as ``(loaders, producers)``: loaders join the entry's main
    schedule; producers join the per-source quark schedule so
    ``sort_schedule`` interleaves each slice's producer with its
    consumers and Hadrons frees it before the next slice is built.

    In
    split-grid mode (``subgrid_ranks`` set) :func:`build_input_params`
    hoists the producers ahead of the first tagged module instead, so
    their outputs exist when Hadrons runs its single pre-scatter pass
    (see the split-mode hoist there).

    subgrid_gather (split mode): qualifying ranLL maps move to
    just after their phase's last tagged step with their LL and LH
    contractions, untagged (design D2/D3).
    """
    loader_modules = {}
    loader_schedule = []
    mf = config.low_modes_config.meson_field_config
    # {mass} is filled with the prefix-removed mass VALUE (the writer's
    # output grammar, meson_v2), never the raw massdict key.
    stem = mf.file.filestem.format(
        mass=config.mass.to_string(mass_label, remove_prefix=True)
    )
    noise_vec = f"{config.noise_name}_vec"

    for g in gammas:
        for conj in g.conjugate_gamma_list:
            loader = config.module_name(f"mfload_mass_{mass_label}_{conj}")
            if loader not in loader_modules:
                loader_modules[loader] = hadmods.load_meson_field(
                    name=loader,
                    file=f"{stem}.@traj@/{conj}_0_0_0.h5",
                    dataset=f"{conj}_0_0_0",
                )
                loader_schedule.append(loader)

    producer_modules = {}
    producer_schedule = []
    for g in gammas:
        glabel = g.name.lower()
        meson_field_refs = " ".join(
            config.module_name(f"mfload_mass_{mass_label}_{conj}")
            for conj in g.conjugate_gamma_list
        )
        for ref in refs:
            producer = twopoint.meson_field_producer_name(config, glabel, mass_label, ref)
            producer_modules[producer] = hadmods.lma_meson_field_prop_v2(
                name=producer,
                action=action,
                low_modes=low_modes,
                meson_field=meson_field_refs,
                gammas=spintaste_names[g],
                labels=" ".join(g.gamma_list),
                ta=str(ref.t0),
                tb=str(ref.t0),
                tstep="1",
                noise=noise_vec,
                n_noise=str(config.sources_config.noise),
            )
            producer_schedule.append(producer)

    return (
        HadronsInput(modules=loader_modules, schedule=loader_schedule),
        HadronsInput(modules=producer_modules, schedule=producer_schedule),
    )


def is_cache_build_only(config: LMAHighModeConfig) -> bool:
    """Whether the entry is a cache ``build_only`` writer (no refs run)."""
    mf = config.low_modes_config.meson_field_config
    return mf is not None and mf.cache is CacheMode.BUILD_ONLY


def entry_run_refs(config: LMAHighModeConfig) -> t.List[SourceRef]:
    """The entry's run refs after the resume gate.

    With ``output.overwrite`` off, only sources whose correlator files
    are missing run (catalog-driven); otherwise every source ref. Total:
    entries without an output block return every ref (their emission
    never consumes them — the build_only stop comes first).
    """
    if config.output_config is not None and not config.output_config.overwrite:
        df = create_outfile_catalog(config)
        if df.empty:
            return []
        missing_files = df[df["exists"] == False]
        return [
            ref
            for ref in config.sources_config.source_refs
            if any(missing_files["tsource"] == ref.axis)
        ]
    return config.sources_config.source_refs


def pooled_placement(
    entries: t.Sequence[LMAHighModeConfig],
    subgrid_ranks: int,
    n_phases: int | None = None,
    phase_size: int | None = None,
    last_size: int | None = None,
) -> t.Tuple[
    t.List[t.List[SourceRef]], t.List[t.Mapping[SourceRef, RefPlacement]], int
]:
    """Pool every entry's run refs in entry order and phase them once.

    A ref counts its entry's ``noise`` sources and weighs ``noise ×
    rungs`` (the CHAIN ladder's length, 1 otherwise). Returns
    ``(refs_per_entry, placements_per_entry, n_phases)``: each entry's
    post-gate refs, its ref -> placement slice of the pooled plan, and
    the phase count. No knob set -> one implicit phase (a single split
    range).
    """
    if n_phases is None and phase_size is None:
        n_phases = 1
    refs_per_entry = [entry_run_refs(hm) for hm in entries]
    pool: t.List[PooledRef] = []
    for i, (hm, refs) in enumerate(zip(entries, refs_per_entry)):
        noise = hm.sources_config.noise
        rungs = (
            len(hm.cg_config.residual)
            if hm.cg_config is not None and hm.cg_config.precon is PreconMode.CHAIN
            else 1
        )
        pool.extend(PooledRef(i, ref, noise, noise * rungs) for ref in refs)
    plan = plan_program(
        pool,
        subgrid_ranks=subgrid_ranks,
        n_phases=n_phases,
        phase_size=phase_size,
        last_size=last_size,
    )
    placements = [
        {ref: plan.placement[(i, ref)] for ref in refs}
        for i, refs in enumerate(refs_per_entry)
    ]
    return refs_per_entry, placements, len(plan.phases)


def build_split_program(
    entries: t.Sequence[LMAHighModeConfig],
    placements: t.Sequence[t.Mapping[SourceRef, RefPlacement]],
    n_phases: int,
    *,
    subgrid_gather: bool = False,
) -> HadronsInput:
    """Pooled split-grid emission: phases outer, entries inner.

    ``entries`` are the program's split entries in order (never cache
    ``build_only`` ones — they emit no tagged modules); ``placements[i]``
    maps entry i's run refs to their ``(phase, subgrid)`` (the pooled
    planner's slices). Per entry the global, ref-free modules are emitted
    once, in program order (SpinTaste + sink, full-volume noise, cache
    writer, meson-field loaders, ranLL solver, CG solvers); then per
    phase: EVERY entry's hoist for its refs in that phase — walls,
    meson-field guess producers, bridges on hoisted maps, and the
    untagged world-grid contractions whose named inputs are hoisted or
    global (``corr_ranLL`` — fault 2) — followed by each entry's sorted
    tagged block. Nothing untagged sits between a phase's first and last
    tagged module, so no world-collective step serializes the subgrids
    mid-phase; the eigenpack's last reader is the last phase's hoist, so
    Hadrons frees the pack before that phase's solves.

    subgrid_gather (split mode): qualifying ranLL maps move to
    just after their phase's last tagged step with their LL and LH
    contractions, untagged (design D2/D3).
    """
    modules: t.Dict[str, t.Dict] = {}
    schedule: t.List[str] = []

    context = []
    for config, placement in zip(entries, placements):
        needed: t.Dict[str, t.List[Gamma]] = {}
        if config.use_meson_field:
            needed = needed_ranll_gammas(config)
        spintaste_modules, spintaste_names = twopoint.build_spintaste_modules(config)
        modules |= spintaste_modules
        schedule += list(spintaste_modules.keys())

        sink_module = twopoint.sink_name(config)
        modules[sink_module] = hadmods.sink(name=sink_module, mom="0 0 0")
        schedule.append(sink_module)

        mf = config.low_modes_config.meson_field_config
        if config.use_meson_field and config.masses:
            modules[config.noise_name] = hadmods.full_volume_noise(
                name=config.noise_name, nsrc=str(config.sources_config.noise)
            )
            schedule.append(config.noise_name)

        if mf is not None and mf.cache is not CacheMode.LOAD:
            writer_input = meson_v2.build_input_params(
                _cache_writer_config(config), prefix=config.module_name("")
            )
            modules |= writer_input.modules
            schedule += writer_input.schedule

        for mass_label in config.masses:
            action = config.action_name.format(mass=mass_label)
            if config.low_modes_config.method is not LowModeMethod.NONE:
                low_modes = config.low_modes_name.format(mass=mass_label)
                if config.use_meson_field:
                    loaders, _ = build_lma_meson_field_chain(
                        config,
                        mass_label=mass_label,
                        action=action,
                        low_modes=low_modes,
                        gammas=needed.get(mass_label, []),
                        spintaste_names=spintaste_names,
                        refs=[],
                    )
                    modules |= loaders.modules       # producers empty (refs=[])
                    schedule += loaders.schedule
                else:
                    name = twopoint.solver_module_name(config, "ranLL", mass_label)
                    modules[name] = hadmods.lma_solver(
                        name=name, action=action, low_modes=low_modes,
                    )
                    schedule.append(name)
            if config.cg_config is not None:
                cg_solver_labels: t.List[str] = [
                    s for s in config.get_solver_labels(skip_cross=True) if "ama" in s
                ]
                for resid, sl in zip(map(str, config.cg_config.residual), cg_solver_labels):
                    name = twopoint.solver_module_name(config, sl, mass_label)
                    match config.cg_config.solver:
                        case "rb":
                            modules[name] = hadmods.rb_cg(name=name, action=action, residual=resid)
                        case "cg":
                            modules[name] = hadmods.cg(name=name, action=action, residual=resid)
                        case "mpcg":
                            inner_action = f"i{action}"
                            modules[name] = hadmods.mixed_precision_cg(
                                name=name, outer_action=action,
                                inner_action=inner_action, residual=resid,
                            )
                        case _:
                            raise ValueError(
                                f"Unknown high-mode CG solver: {config.cg_config.solver}"
                            )
                    schedule.append(name)

        context.append(
            (
                config,
                placement,
                spintaste_names,
                needed,
                gather_qualifying_gammas(config) if subgrid_gather else frozenset(),
            )
        )

    for phase in range(n_phases):
        hoist: t.List[str] = []
        tagged: t.List[str] = []
        post: t.List[str] = []
        for config, placement, spintaste_names, needed, qualifying in context:
            chunk = [ref for ref in placement if placement[ref].phase == phase]
            if not chunk:
                continue
            for ref in chunk:                     # walls (hoisted, untagged)
                name = twopoint.noise_rw_name(config, ref)
                modules[name] = hadmods.noise_rw(
                    name=name,
                    nsrc=str(config.sources_config.noise),
                    t0=str(ref.t0),
                    tstep=str(config.sources_config.time),
                    noise=config.noise_name if config.use_meson_field else "",
                )
                hoist.append(name)
            if config.use_meson_field:            # guess producers (hoisted)
                for mass_label in config.masses:
                    _, producers = build_lma_meson_field_chain(
                        config,
                        mass_label=mass_label,
                        action=config.action_name.format(mass=mass_label),
                        low_modes=config.low_modes_name.format(mass=mass_label),
                        gammas=needed.get(mass_label, []),
                        spintaste_names=spintaste_names,
                        refs=chunk,
                    )
                    modules |= producers.modules
                    hoist += producers.schedule
            entry_placement = {ref: placement[ref] for ref in chunk}
            quark_inputs = twopoint.build_quarks(
                config, chunk, spintaste_names, entry_placement
            )
            modules |= quark_inputs.modules
            contract_inputs = twopoint.build_contractions(
                config, chunk, spintaste_names, entry_placement
            )
            modules |= contract_inputs.modules
            # re-bucket: bridges on hoisted maps join the hoist (after the
            # producers) AND register in hoist_set so the contractions
            # sinking on them can see them; untagged world contractions
            # whose named inputs are all hoisted join too (corr_ranLL,
            # fault 2 — under SOLVE low modes the ranLL quarks stay in
            # the sorted block, so their contractions do too); everything
            # else is this entry's tagged block.
            hoist_set = set(hoist)
            quark_schedule_k = list(quark_inputs.schedule)
            deferred: t.List[str] = []
            for name in contract_inputs.schedule:
                module = contract_inputs.modules[name]
                mtype = module["id"]["type"]
                if (
                    mtype == "MUtilities::GammaMapElement"
                    and module["options"]["map"] in hoist_set
                ):
                    hoist.append(name)
                    hoist_set.add(name)
                else:
                    deferred.append(name)
            # second pass: untagged world contractions whose named inputs
            # are all hoisted join too (corr_ranLL, fault 2 — under SOLVE
            # low modes the ranLL quarks stay in the sorted block, so
            # their contractions do too); everything else is this entry's
            # tagged block. Deferring keeps the hoist order walls →
            # producers → bridges → corr.
            for name in deferred:
                module = contract_inputs.modules[name]
                if (
                    module["id"]["type"] == "MContraction::StagMeson"
                    and module.get("subgrid") is None
                    and module["options"]["source"] in hoist_set
                    and module["options"]["sink"] in hoist_set
                ):
                    hoist.append(name)
                else:
                    quark_schedule_k.append(name)
            post_k: t.List[str] = []
            if qualifying:
                post_k = _gather_post_schedule(
                    config, chunk, qualifying, set(modules), contract_inputs
                )
                if post_k:
                    post_set = set(post_k)
                    for name in post_set:
                        # moved LH contractions were tagged (placement-
                        # based, post-split-phases); producers and LL
                        # contractions never were. Strip any tag so the
                        # moved modules are pure world-grid steps.
                        modules[name].pop("subgrid", None)
                        modules[name].pop("splitPhase", None)
                    hoist[:] = [n for n in hoist if n not in post_set]
                    quark_schedule_k[:] = [
                        n for n in quark_schedule_k if n not in post_set
                    ]
                    post += post_k        # phase-level: entries in order (D6)
            tagged += reorder_cross_contractions(
                sort_schedule(config, quark_schedule_k), modules, config
            )
        schedule += hoist
        schedule += tagged
        schedule += post                  # after the phase's last tagged step
    return HadronsInput(modules=modules, schedule=schedule)


def build_input_params(
    config: LMAHighModeConfig, *, subgrid_gather: bool = False
) -> HadronsInput:
    """Per-entry emission (design doc order): SpinTaste + sink →
    full-volume noise → cache writer → (stop for build_only) → per-source
    noise → mass loop (meson-field chains / ranLL solver / CG solvers) →
    quarks + contractions, sorted.

    Split-grid mode (``subgrid_ranks`` set, refs to run, not a cache
    ``build_only`` entry) delegates to :func:`build_split_program` as a
    pooled program of one — a single phase, greedy placement over the
    entry's refs, per-phase hoist (walls, producers, bridges, the
    untagged world contractions) ahead of the sorted tagged block, so
    Hadrons' single pre-scatter pass per phase sees every input (the Q7
    contract). The eigenpack stays resident through every phase but the
    last, whose hoist carries its last reader; Hadrons' GC frees it
    there.

    The resume gate keys on the entry's correlator catalog (output files);
    ``build_only`` entries have no output block and skip the gate (all
    source refs, unused — emission stops after the cache writer).
    """
    run_refs = entry_run_refs(config)

    if (
        config.subgrid_ranks is not None
        and run_refs
        and not is_cache_build_only(config)
    ):
        noise = config.sources_config.noise
        rungs = (
            len(config.cg_config.residual)
            if config.cg_config is not None
            and config.cg_config.precon is PreconMode.CHAIN
            else 1
        )
        pool = [PooledRef(0, ref, noise, noise * rungs) for ref in run_refs]
        plan = plan_program(pool, subgrid_ranks=config.subgrid_ranks, n_phases=1)
        placement = {ref: plan.placement[(0, ref)] for ref in run_refs}
        return build_split_program(
            [config], [placement], len(plan.phases),
            subgrid_gather=subgrid_gather,
        )

    modules = {}
    schedule = []

    mf = config.low_modes_config.meson_field_config
    cache_build_only = is_cache_build_only(config)

    needed: t.Dict[str, t.List[Gamma]] = {}
    if config.use_meson_field:
        needed = needed_ranll_gammas(config)

    # 1. SpinTaste modules, sink.
    spintaste_modules, spintaste_names = twopoint.build_spintaste_modules(config)
    modules |= spintaste_modules
    schedule += list(spintaste_modules.keys())

    sink_module = twopoint.sink_name(config)
    modules[sink_module] = hadmods.sink(name=sink_module, mom="0 0 0")
    schedule.append(sink_module)

    # 2. Full-volume noise (meson_field only) — the writer's and the
    # producers' noise dependency.
    if config.use_meson_field and config.masses:
        modules[config.noise_name] = hadmods.full_volume_noise(
            name=config.noise_name, nsrc=str(config.sources_config.noise)
        )
        schedule.append(config.noise_name)

    # 3. Cache writer (build_and_load / build_only), after the noise it reads.
    if mf is not None and mf.cache is not CacheMode.LOAD:
        writer_input = meson_v2.build_input_params(
            _cache_writer_config(config), prefix=config.module_name("")
        )
        modules |= writer_input.modules
        schedule += writer_input.schedule

    # 4. Stop here for build_only: no quarks, solvers or correlators.
    if cache_build_only:
        return HadronsInput(modules=modules, schedule=schedule)

    # 5. Per-source noise (interleaved — split mode never reaches here).
    quark_schedule = []
    for ref in run_refs:
        name = twopoint.noise_rw_name(config, ref)
        modules[name] = hadmods.noise_rw(
            name=name,
            nsrc=str(config.sources_config.noise),
            t0=str(ref.t0),
            tstep=str(config.sources_config.time),
            noise=config.noise_name if config.use_meson_field else "",
        )
        quark_schedule.append(name)

    # 6. Per mass: low-mode solvers / meson-field chains, then CG solvers.
    # The meson-field PRODUCERS join quark_schedule — the interleaved
    # non-split behavior (split mode delegates above and never sorts).
    for mass_label in config.masses:
        action = config.action_name.format(mass=mass_label)  # parent-owned: unprefixed
        if config.low_modes_config.method is not LowModeMethod.NONE:
            low_modes = config.low_modes_name.format(mass=mass_label)
            if config.use_meson_field:
                loaders, producers = build_lma_meson_field_chain(
                    config,
                    mass_label=mass_label,
                    action=action,
                    low_modes=low_modes,
                    gammas=needed.get(mass_label, []),
                    spintaste_names=spintaste_names,
                    refs=run_refs,
                )
                modules |= loaders.modules
                schedule += loaders.schedule
                # per-slice producers sort with their slice's consumers
                modules |= producers.modules
                quark_schedule += producers.schedule
            else:
                name = twopoint.solver_module_name(config, "ranLL", mass_label)
                modules[name] = hadmods.lma_solver(
                    name=name,
                    action=action,
                    low_modes=low_modes,
                )
                schedule.append(name)

        if config.cg_config is not None:
            cg_solver_labels: t.List[str] = [
                s for s in config.get_solver_labels(skip_cross=True) if "ama" in s
            ]
            for resid, sl in zip(map(str, config.cg_config.residual), cg_solver_labels):
                name = twopoint.solver_module_name(config, sl, mass_label)

                match config.cg_config.solver:
                    case "rb":
                        modules[name] = hadmods.rb_cg(
                            name=name,
                            action=action,
                            residual=resid,
                        )
                    case "cg":
                        modules[name] = hadmods.cg(
                            name=name,
                            action=action,
                            residual=resid,
                        )
                    case "mpcg":
                        inner_action = f"i{action}"
                        modules[name] = hadmods.mixed_precision_cg(
                            name=name,
                            outer_action=action,
                            inner_action=inner_action,
                            residual=resid,
                        )
                    case _:
                        raise ValueError(
                            f"Unknown high-mode CG solver: {config.cg_config.solver}"
                        )
                schedule.append(name)

    # 7. Quarks, contractions (sorted, untagged — split mode delegates above).
    quark_inputs = twopoint.build_quarks(config, run_refs, spintaste_names)
    modules |= quark_inputs.modules
    quark_schedule += quark_inputs.schedule

    contract_inputs = twopoint.build_contractions(config, run_refs, spintaste_names)
    modules |= contract_inputs.modules
    quark_schedule += contract_inputs.schedule

    schedule += reorder_cross_contractions(
        sort_schedule(config, quark_schedule), modules, config
    )

    return HadronsInput(modules=modules, schedule=schedule)


def build_aggregator_params(config: LMAHighModeConfig, average: bool) -> t.Dict:
    """Aggregation parameters for one entry's correlator files.

    Ported from ``highmode.strategy.build_aggregator_params`` with the
    field mapping onto the sub-block tree; the old positional
    ``run_prefix`` is gone — run keys are label-prefixed via
    ``config.module_name("")`` (D6), so an unkeyed entry keeps today's
    unprefixed keys byte-for-byte while keyed entries never clobber each
    other. Cache ``build_only`` entries have no correlator files to
    aggregate and aggregate nothing.
    """
    if config.output_config is None:
        return {}

    agg_params = freeze({})

    suffix = "_avg" if average else ""
    outfile = utils.io.get_processed_filename(
        config.output_config.file.filestem,
        remove=["series", "tsource"],
        suffix=suffix,
    )

    infile = config.output_config.file.filename

    e_rep = freeze({"tsource": config.sources_config.source_axis}).evolver()

    solver_labels = config.get_solver_labels()

    run_list = []

    actions: t.Dict[str, t.Any] = {"index": ["series_cfg", "gamma", "t"]}

    if average:
        actions["average"] = ["tsource"]
        actions["real"] = True

    label_prefix = config.module_name("")
    for op in config.op_list:
        gamma_label = op.gamma.name.lower()
        e_rep["gamma_label"] = gamma_label
        # Mass axis from get_mass_labels so cross-mass dsets aggregate like
        # diagonal ones (the catalog/resume gate has always used this axis).
        for mass_label, dset in itertools.product(
            config.get_mass_labels(op), solver_labels
        ):
            file_label = f"{label_prefix}{gamma_label}_{mass_label}_{dset}"
            run_list.append(file_label)
            e_rep["mass"] = mass_label
            e_rep["dset"] = dset
            replacements = e_rep.persistent()

            h5_datasets = {
                g: f"/meson/meson_{i}/corr" for i, g in enumerate(op.gamma.gamma_list)
            }

            array_params = {
                "order": ["t"],
                "labels": {"t": f"0..{config.sources_config.time - 1}"},
            }

            agg_params = agg_params.set(
                file_label,
                {
                    "actions": actions,
                    "logging_level": config.logging_level,
                    "load_files": {
                        "filestem": infile,
                        "regex": {"series": "[a-z]", "cfg": "[0-9]+"},
                        "replacements": thaw(replacements),
                        "name": "gamma",
                        "datasets": h5_datasets,
                        **array_params,
                    },
                    "out_files": {"filestem": outfile},
                },
            )
            e_rep = replacements.evolver()
    agg_params = agg_params.set("run", run_list)

    return dict(thaw(agg_params))


_PAIR_KEYS = ["gamma_label", "mass", "tsource", "dset"]


def _col(row: pd.Series, key: str, default: t.Any = False) -> t.Any:
    """Read a (possibly NaN / absent) catalog column from a merged row."""
    if key not in row:
        return default
    val = row[key]
    return default if pd.isna(val) else val


def _find_op(config: LMAHighModeConfig, gamma_label: str):
    """Return the OpList.Op whose gamma name matches ``gamma_label``, or None."""
    for op in config.op_list:
        if op.gamma.name.lower() == gamma_label:
            return op
    return None


def _dataset_paths(op) -> t.List[str]:
    """HDF5 dataset paths for each gamma component of ``op``."""
    return [f"/meson/meson_{i}/corr" for i in range(len(op.gamma.gamma_list))]


def _compare_h5_file(
    filepath_a: str,
    filepath_b: str,
    dataset_paths: t.List[str],
    *,
    rtol: float,
    atol: float,
) -> t.Tuple[float, float, bool]:
    """Compare matching datasets in two HDF5 files.

    Correlator data is stored as float64 and viewed as ``np.complex128`` on read
    (see ``pyfm/dataio/converter.py::hdf5_to_frame``).

    Returns ``(max_abs_diff, max_rel_diff, within_tolerance)``. A shape mismatch
    between corresponding datasets reports an infinite diff and within=False.
    """
    max_abs = 0.0
    max_rel = 0.0
    within = True
    try:
        with h5py.File(filepath_a, "r") as fa, h5py.File(filepath_b, "r") as fb:
            for ds in dataset_paths:
                if ds not in fa:
                    raise ValueError(f"dataset {ds!r} not found in {filepath_a!r}")
                if ds not in fb:
                    raise ValueError(f"dataset {ds!r} not found in {filepath_b!r}")
                a = fa[ds][:].view(np.complex128)
                b = fb[ds][:].view(np.complex128)
                if a.shape != b.shape:
                    return float("inf"), float("inf"), False
                diff = np.abs(a - b)
                max_abs = max(max_abs, float(np.max(diff)) if diff.size else 0.0)
                denom = np.abs(b)
                nonzero = denom > 0
                if np.any(nonzero):
                    rel = diff[nonzero] / denom[nonzero]
                    max_rel = max(max_rel, float(np.max(rel)))
                within = within and bool(np.all(diff <= atol + rtol * denom))
    except (OSError, KeyError) as e:
        # A file that passed step-1 good_size but is corrupt / missing a dataset
        # should surface as a clean ValueError (caught by the CLI), not a traceback.
        raise ValueError(
            f"Failed to read HDF5 output for comparison "
            f"({filepath_a!r} vs {filepath_b!r}): {e}"
        ) from e
    return max_abs, max_rel, within


def compare_outputs(
    config_a: LMAHighModeConfig,
    config_b: LMAHighModeConfig,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> pd.DataFrame:
    """Compare the entry's correlator outputs between two configs.

    Ported from ``highmode.compare.compare_outputs`` (module-local
    catalog, LMAHighModeConfig typing): builds both outfile catalogs,
    pairs expected files by ``(gamma_label, mass, tsource, dset)``, loads
    each pair's HDF5 datasets (``/meson/meson_{i}/corr``), and reports
    per-file max abs/rel diff plus whether the pair is within tolerance.

    Returns a DataFrame with columns: ``gamma_label, mass, tsource, dset,
    filepath_a, filepath_b, max_abs_diff, max_rel_diff, within_tolerance, status``
    where status is one of ``compared``, ``missing_file``, ``missing_op``.
    """
    logger = utils.get_logger()

    catalog_a = create_outfile_catalog(config_a)
    catalog_b = create_outfile_catalog(config_b)

    paired = catalog_a.merge(
        catalog_b, on=_PAIR_KEYS, how="outer", suffixes=("_a", "_b"), indicator=True
    )

    rows = []
    for _, r in paired.iterrows():
        gamma_label = r["gamma_label"]
        op = _find_op(config_a, gamma_label) or _find_op(config_b, gamma_label)
        filepath_a = _col(r, "filepath_a", None)
        filepath_b = _col(r, "filepath_b", None)
        exists_a = bool(_col(r, "exists_a", False))
        exists_b = bool(_col(r, "exists_b", False))

        base = {
            "gamma_label": gamma_label,
            "mass": r["mass"],
            "tsource": r["tsource"],
            "dset": r["dset"],
            "filepath_a": filepath_a,
            "filepath_b": filepath_b,
        }

        if op is None:
            rows.append(
                base
                | {
                    "max_abs_diff": float("nan"),
                    "max_rel_diff": float("nan"),
                    "within_tolerance": False,
                    "status": "missing_op",
                }
            )
            logger.warning(
                f"No op found for gamma_label={gamma_label!r}; cannot compare."
            )
            continue

        if not (exists_a and exists_b):
            rows.append(
                base
                | {
                    "max_abs_diff": float("nan"),
                    "max_rel_diff": float("nan"),
                    "within_tolerance": False,
                    "status": "missing_file",
                }
            )
            logger.info(
                f"Missing output for gamma_label={gamma_label!r} "
                f"tsource={r['tsource']} dset={r['dset']} "
                f"(exists_a={exists_a}, exists_b={exists_b})."
            )
            continue

        max_abs, max_rel, within = _compare_h5_file(
            filepath_a, filepath_b, _dataset_paths(op), rtol=rtol, atol=atol
        )
        rows.append(
            base
            | {
                "max_abs_diff": max_abs,
                "max_rel_diff": max_rel,
                "within_tolerance": within,
                "status": "compared",
            }
        )

    columns = [
        "gamma_label",
        "mass",
        "tsource",
        "dset",
        "filepath_a",
        "filepath_b",
        "max_abs_diff",
        "max_rel_diff",
        "within_tolerance",
        "status",
    ]
    return pd.DataFrame(rows, columns=columns)
