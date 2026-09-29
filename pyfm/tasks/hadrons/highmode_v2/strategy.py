import typing as t

from pyfm.tasks.hadrons.types import (
    HadronsInput,
    HighModeConfig,
    CorrelatorStrategy,
    SourceRef,
)
import pyfm.tasks.hadrons.modules as hadmods
from pyfm.domain import Gamma
from pyfm.tasks.hadrons.highmode_v2 import twopoint
from pyfm.tasks.hadrons.highmode.strategy import (
    create_outfile_catalog,
    needed_ranll_gammas,
    sort_schedule,
)


def build_lma_meson_field_chain(
    config: HighModeConfig,
    mass_label: str,
    action: str,
    low_modes: str,
    gammas: t.List[Gamma],
    spintaste_names: t.Dict[Gamma, str],
) -> HadronsInput:
    """Loader + producer half of the load-mode meson-field chain.

    The writer (cbpairs + SpinTaste + ``meson_field_v2``) that used to
    live here has moved to the meson section — an ordinary ``MesonConfig``
    entry, synthesized by ``LMANewConfig.postprocess_config`` when
    ``build_lh_cache`` is set. This function only loads the resulting
    files and builds the eager ``StagLMAMesonFieldProp`` producers;
    ``gammas`` is this mass's demand set (:func:`needed_ranll_gammas`).
    """
    modules = {}
    schedule = []
    stem = config.meson_stoch_proj.filestem.format(mass=mass_label)
    noise_vec = f"{config.noise_name}_vec"

    for g in gammas:
        for conj in g.conjugate_gamma_list:
            loader = f"mfload_mass_{mass_label}_{conj}"
            if loader not in modules:
                modules[loader] = hadmods.load_meson_field(
                    name=loader,
                    file=f"{stem}.@traj@/{conj}_0_0_0.h5",
                    dataset=f"{conj}_0_0_0",
                )
                schedule.append(loader)

    for g in gammas:
        glabel = g.name.lower()
        producer = f"quark_ranLL_{glabel}_mass_{mass_label}"
        modules[producer] = hadmods.lma_meson_field_prop_v2(
            name=producer,
            action=action,
            low_modes=low_modes,
            meson_field=" ".join(
                f"mfload_mass_{mass_label}_{conj}" for conj in g.conjugate_gamma_list
            ),
            gammas=spintaste_names[g],
            labels=" ".join(g.gamma_list),
            ta=str(config.tstart),
            tb=str(config.tstop),
            tstep=str(config.dt),
            noise=noise_vec,
            n_noise=str(config.noise),
        )
        schedule.append(producer)

    return HadronsInput(modules=modules, schedule=schedule)


def build_quark_strategy(
    config: HighModeConfig,
    run_refs: t.List[SourceRef],
    spintaste_names: t.Dict[Gamma, str],
) -> HadronsInput:
    match config.correlator_strategy:
        case CorrelatorStrategy.TWOPOINT:
            return twopoint.build_quarks(config, run_refs, spintaste_names)
        case _:
            raise ValueError(
                "hadrons_lma_new only supports correlator_strategy=TWOPOINT "
                "(SIB is dead/unreachable code upstream too — arity-broken "
                f"callee); got {config.correlator_strategy}"
            )


def build_contract_strategy(
    config: HighModeConfig,
    run_refs: t.List[SourceRef],
    spintaste_names: t.Dict[Gamma, str],
) -> HadronsInput:
    match config.correlator_strategy:
        case CorrelatorStrategy.TWOPOINT:
            return twopoint.build_contractions(config, run_refs, spintaste_names)
        case _:
            raise ValueError(
                "hadrons_lma_new only supports correlator_strategy=TWOPOINT "
                "(SIB is dead/unreachable code upstream too — arity-broken "
                f"callee); got {config.correlator_strategy}"
            )


def build_input_params(config: HighModeConfig) -> HadronsInput:
    """Canonical-schema counterpart of ``highmode.strategy.build_input_params``.

    Same overall shape (resume gate, meson-field-chain gating, per-mass CG
    solvers, schedule sort) — one addition: the config-scoped SpinTaste
    module set is built ONCE via ``twopoint.build_spintaste_modules`` and
    threaded through every consumer (the meson-field chain's producers, the
    quark/contract dispatch), instead of each consumer building its own
    inline gamma options.
    """
    modules = {}
    schedule = []

    if config.cache_only:
        # Cache-only mode (lma_new's build_lh_cache + skip_high_modes path):
        # the lh-cache writer (a synthesized MesonConfig entry, built in the
        # meson section) references this entry's noise module by name
        # (meson_v2.py's high_right_name) — full_volume_noise is its sole
        # dependency here. Everything else below (resume gate, SpinTaste,
        # sink, per-source noise, mass-loop solvers, quark/contraction
        # dispatch) exists only to build quarks/correlators, which this
        # pathway explicitly skips.
        if config.masses:
            modules[config.noise_name] = hadmods.full_volume_noise(
                name=config.noise_name, nsrc=str(config.noise)
            )
            schedule.append(config.noise_name)
        return HadronsInput(modules=modules, schedule=schedule)

    if not config.overwrite:
        df = create_outfile_catalog(config)
        if df.empty:
            run_refs = []
        else:
            missing_files = df[df["exists"] == False]
            run_refs = [
                ref
                for ref in config.source_refs
                if any(missing_files["tsource"] == ref.axis)
            ]
    else:
        run_refs = config.source_refs

    use_meson_field = config.low_mode_method == "load" and not config.skip_low_modes
    needed: t.Dict[str, t.List[Gamma]] = {}
    if use_meson_field:
        needed = needed_ranll_gammas(config)

    spintaste_modules, spintaste_names = twopoint.build_spintaste_modules(config)
    modules |= spintaste_modules
    schedule += list(spintaste_modules.keys())

    modules["sink"] = hadmods.sink(name="sink", mom="0 0 0")
    schedule.append("sink")

    if use_meson_field and config.masses:
        modules[config.noise_name] = hadmods.full_volume_noise(
            name=config.noise_name, nsrc=str(config.noise)
        )
        schedule.append(config.noise_name)

    quark_schedule = []
    for ref in run_refs:
        name = f"noise_{ref.label}"
        modules[name] = hadmods.noise_rw(
            name=name,
            nsrc=str(config.noise),
            t0=str(ref.t0),
            tstep=str(config.time),
            noise=config.noise_name if use_meson_field else "",
        )
        quark_schedule.append(name)

    for mass_label in config.masses:
        action = config.action_name.format(mass=mass_label)
        if not config.skip_low_modes:
            low_modes = config.low_modes_name.format(mass=mass_label)
            if use_meson_field:
                chain = build_lma_meson_field_chain(
                    config,
                    mass_label=mass_label,
                    action=action,
                    low_modes=low_modes,
                    gammas=needed.get(mass_label, []),
                    spintaste_names=spintaste_names,
                )
                modules |= chain.modules
                schedule += chain.schedule
            else:
                name = config.solver_name.format(solver="ranLL", mass=mass_label)
                modules[name] = hadmods.lma_solver(
                    name=name,
                    action=action,
                    low_modes=low_modes,
                )
                schedule.append(name)

        cg_solver_labels: t.List = [
            s for s in config.get_solver_labels(skip_cross=True) if "ama" in s
        ]
        for resid, sl in zip(map(str, config.residual), cg_solver_labels):
            name = config.solver_name.format(solver=sl, mass=mass_label)

            match config.solver:
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
                    raise ValueError(f"Unknown high-mode CG solver: {config.solver}")
            schedule.append(name)

    quark_inputs = build_quark_strategy(config, run_refs, spintaste_names)
    modules |= quark_inputs.modules
    quark_schedule += quark_inputs.schedule

    contract_inputs = build_contract_strategy(config, run_refs, spintaste_names)
    modules |= contract_inputs.modules
    quark_schedule += contract_inputs.schedule

    schedule += sort_schedule(config, quark_schedule)

    return HadronsInput(modules=modules, schedule=schedule)
