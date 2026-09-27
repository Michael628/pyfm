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
    create_meson_field_catalog,
    sort_schedule,
)

from pyfm import utils


def build_lma_meson_field_chain(
    config: HighModeConfig,
    mass_label: str,
    action: str,
    low_modes: str,
    gammas: t.List[Gamma],
    write: bool,
    spintaste_names: t.Dict[Gamma, str],
) -> HadronsInput:
    """Canonical-schema counterpart of
    ``highmode.strategy.build_lma_meson_field_chain``.

    Same writer/loader/producer shape and file-naming convention — the
    module calls differ: the writer (``meson_field_v2``) has no ``action``
    field at all (dropped from the canonical Par struct) and its gamma set
    is published through a dedicated per-mass union ``SpinTaste`` module
    (covering every gamma this mass needs, since one writer call spans all
    of them); each producer (``lma_meson_field_prop_v2``) references the
    SAME per-gamma SpinTaste module ``build_quarks``/``build_contractions``
    already use (``spintaste_names[g]``, built once by
    ``twopoint.build_spintaste_modules`` and threaded through) with a
    REQUIRED ``labels`` list — ``g``'s own raw ``gamma_list``, positionally
    parallel to the conjugated-gamma loader list (unchanged from today's
    file-naming/loader convention).
    """
    modules = {}
    schedule = []
    stem = config.meson_stoch_proj.filestem.format(mass=mass_label)

    if write:
        cbpairs_l = f"cbpairs_l_mass_{mass_label}"
        cbpairs_r = f"cbpairs_r_mass_{mass_label}"
        modules[cbpairs_l] = hadmods.eigen_pack_cb_pairs(
            name=cbpairs_l, eigen_pack=low_modes, action=action
        )
        modules[cbpairs_r] = hadmods.eigen_pack_cb_pairs(
            name=cbpairs_r, eigen_pack=low_modes, action=action
        )
        schedule += [cbpairs_l, cbpairs_r]

        writer_spintaste = f"spintaste_mfwrite_mass_{mass_label}"
        modules[writer_spintaste] = hadmods.spin_taste(
            name=writer_spintaste,
            gammas=" ".join(dict.fromkeys(g.gamma_string for g in gammas)),
            gauge=(
                "gauge" if all(g.local for g in gammas) else config.shift_gauge_name
            ),
            apply_g5="true",
        )
        schedule.append(writer_spintaste)

        writer = f"mfwrite_mass_{mass_label}"
        modules[writer] = hadmods.meson_field_v2(
            name=writer,
            block=str(config.blocksize),
            gammas=writer_spintaste,
            low_modes=low_modes,
            left="",
            right="noise_fv_vec",
            output=stem,
            cb_pairs_left=cbpairs_l,
            cb_pairs_right=cbpairs_r,
        )
        schedule.append(writer)

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
            noise="noise_fv_vec",
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
    incomplete_masses: t.Set[str] = set()
    if use_meson_field and not config.overwrite:
        mf_catalog = create_meson_field_catalog(config)
        if not mf_catalog.empty:
            bad = utils.io.get_bad_files(mf_catalog)
            incomplete_masses = set(
                mf_catalog[mf_catalog["filepath"].isin(bad)]["mass"]
            )

    spintaste_modules, spintaste_names = twopoint.build_spintaste_modules(config)
    modules |= spintaste_modules
    schedule += list(spintaste_modules.keys())

    modules["sink"] = hadmods.sink(name="sink", mom="0 0 0")
    schedule.append("sink")

    if use_meson_field and config.masses:
        modules["noise_fv"] = hadmods.full_volume_noise(
            name="noise_fv", nsrc=str(config.noise)
        )
        schedule.append("noise_fv")

    quark_schedule = []
    for ref in run_refs:
        name = f"noise_{ref.label}"
        modules[name] = hadmods.noise_rw(
            name=name,
            nsrc=str(config.noise),
            t0=str(ref.t0),
            tstep=str(config.time),
            noise="noise_fv" if use_meson_field else "",
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
                    write=config.overwrite or mass_label in incomplete_masses,
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
