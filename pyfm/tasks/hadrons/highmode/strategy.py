import typing as t
import re
import pandas as pd
import itertools
from pyrsistent import freeze, thaw
from dataclasses import fields

from pyfm.tasks.hadrons.types import (
    HadronsInput,
    HighModeConfig,
    CorrelatorStrategy,
    SolveCrossTerms,
    SourceRef,
)
import pyfm.tasks.hadrons.modules as hadmods
from pyfm.domain import Gamma, OpList
from pyfm.tasks.hadrons.highmode import sib, twopoint

from pyfm import utils


_LEGACY_CROSS_TERMS: t.Dict[str, t.Tuple[bool, SolveCrossTerms]] = {
    "none": (False, SolveCrossTerms.DIAGONAL),
    "mass": (True, SolveCrossTerms.DIAGONAL),
    "solve": (False, SolveCrossTerms.ALL),
    "all": (True, SolveCrossTerms.ALL),
}


def _translate_legacy_cross_terms(site: t.Dict) -> t.Dict:
    """Return ``site`` unchanged (same object) unless it carries a legacy
    ``cross_terms`` key, in which case return a new dict with the key
    translated onto ``mass_cross_terms``/``solve_cross_terms`` — explicit
    keys win. Raises ``ValueError`` on an unrecognized value."""
    legacy = site.get("cross_terms")
    if legacy is None:
        return site
    try:
        mass_cross, solve_cross = _LEGACY_CROSS_TERMS[str(legacy).lower()]
    except KeyError:
        raise ValueError(
            f"Invalid cross_terms value ({legacy!r}). "
            "options are: none, mass, solve, all"
        ) from None
    utils.get_logger().debug(
        f"Translating legacy cross_terms={legacy!r} -> "
        f"mass_cross_terms={mass_cross}, solve_cross_terms={solve_cross.name}"
    )
    return {k: v for k, v in site.items() if k != "cross_terms"} | {
        "mass_cross_terms": site.get("mass_cross_terms", mass_cross),
        "solve_cross_terms": site.get("solve_cross_terms", solve_cross),
    }


def normalize_params(params: t.Dict) -> t.Dict:
    """Translate the legacy ``cross_terms`` enum onto the split fields.

    Silent by design (gauge ``action_type`` precedent, ``gauge.py``):
    ``none`` -> defaults, ``mass`` -> mass toggle, ``solve`` -> ALL,
    ``all`` -> mass toggle + ALL. Explicit ``mass_cross_terms`` /
    ``solve_cross_terms`` keys always win. Runs before ``route_params``'
    reflection split so the legacy key never leaks into ``operations``.

    Non-mutating (``lmi.normalize_params`` style): the caller's dicts are
    never modified, and the input object itself is returned (identity, not
    a copy) when no legacy key is present at either site.
    """
    result = _translate_legacy_cross_terms(params)
    slice_ = params.get("_preprocessor")
    if slice_ is not None:
        translated = _translate_legacy_cross_terms(slice_)
        if translated is not slice_:
            result = result | {"_preprocessor": translated}
    return result


def route_params(params: t.Dict) -> t.Dict:
    """Route task data to the 'operations' field of HighModeConfig.

    Avoids a collision between MassDict (from params['mass']) and OpList mass
    labels (from params['_preprocessor']['mass']). This is pure ``_preprocessor``
    routing — there is nothing to normalize.
    """
    # Extract task configs (contains gamma, mass lists for OpList)
    preprocessor_params = params.pop("_preprocessor", {})

    # Split-grid is opt-in and both-or-neither: a partial config (exactly one of
    # `split_mpi_layout`/`subgrid_ranks` set) is meaningless -- Hadrons needs the
    # global <split> to define the subgrids that <subgrid> tags reference. Strip
    # both and warn so the job falls back to non-split behavior rather than
    # emitting orphan <subgrid> tags or a stray <split>.
    split_mpi_layout = preprocessor_params.get("split_mpi_layout")
    subgrid_ranks = preprocessor_params.get("subgrid_ranks")
    if (split_mpi_layout is None) != (subgrid_ranks is None):
        utils.get_logger().warning(
            "Split-grid requires both `split_mpi_layout` and `subgrid_ranks`; "
            "only one was provided. Stripping both and falling back to "
            "non-split behavior."
        )
        preprocessor_params.pop("split_mpi_layout", None)
        preprocessor_params.pop("subgrid_ranks", None)

    # Bias sampling seed: compose the user's base with series/cfg here — the
    # route hook is the only place both are visible (post-construction hooks
    # cannot see them, and the aggregation path builds without series/cfg).
    # Leave the field untouched when the base is absent so validate_config
    # can raise a clear "bias_seed is required" error.
    if preprocessor_params.get("nbias") is not None:
        seed_base = preprocessor_params.get("bias_seed")
        if seed_base is not None:
            preprocessor_params["bias_seed"] = (
                f"{seed_base}_{params.get('series', '')}_{params.get('cfg', '')}"
            )

    # Get field names from HighModeConfig, excluding 'mass'
    # - 'mass' comes from top-level params (MassDict)
    # !NOTE: Don't squash params['mass']
    config_fields = {f.name for f in fields(HighModeConfig) if f.name != "mass"}

    return (
        params
        | {
            "operations": {
                k: v for k, v in preprocessor_params.items() if k not in config_fields
            },
        }
        | {k: v for k, v in preprocessor_params.items() if k in config_fields}
    )


def create_outfile_catalog(config: HighModeConfig) -> pd.DataFrame:
    def generate_outfile_formatting():
        solver_labels = config.get_solver_labels()
        res = {"tsource": config.source_axis, "dset": solver_labels}

        for op in config.op_list:
            res["gamma_label"] = op.gamma.name.lower()
            res["mass"] = config.get_mass_labels(op)
            yield res, config.high_modes

    outfile_generator = generate_outfile_formatting()

    df = utils.io.catalog_files(outfile_generator)

    return df


def needed_ranll_gammas(config: HighModeConfig) -> t.Dict[str, t.List[Gamma]]:
    """Per-mass ranLL gamma demand for the load-mode chain.

    Derived from ``twopoint.quark_gen`` — the same demand-driven set
    ``build_quarks`` emits GaugeProp names for — so the producer set can
    never drift from the consumer references: each (gamma, mass) entry
    here corresponds exactly to a ``quark_ranLL_{glabel}_mass_{m}``
    producer whose outputs are the propagators contractions and the ama
    guess chain reference. Mass-major, gamma-name order within mass
    (quark_gen's own ordering); one entry per pair, no duplicates.
    """
    needed: t.Dict[str, t.List[Gamma]] = {}
    for op in twopoint.quark_gen(config):
        if op.solver == "ranLL":
            needed.setdefault(op.mass, []).append(op.gamma)
    return needed


def create_meson_field_catalog(config: HighModeConfig) -> pd.DataFrame:
    """Catalog the per-(mass, gamma) meson-field intermediate files (load mode).

    Deliberately NOT concatenated into :func:`create_outfile_catalog`:
    intermediates must not join the correlator catalog that drives the
    resume gate, ``compare_outputs`` pairing, or the nanny compare gate.
    ``{cfg}`` is pre-formatted by the config builder in real builds; the
    catalog is only consumed by ``build_input_params``' skip-if-complete
    check. The gamma axis carries the G5-CONJUGATED pair names — the
    writer names files by the applied gamma — so ``{gamma}_0_0_0.h5``
    resolves per conjugated entry of each mass's needed-gamma set
    (:func:`needed_ranll_gammas`).
    """
    if config.meson_stoch_proj is None:
        raise ValueError(
            "low_mode_method='load' requires a meson_stoch_proj files entry "
            "(filestem + good_size) for the meson-field intermediates."
        )
    if not config.masses:
        return pd.DataFrame()

    needed = needed_ranll_gammas(config)

    def generate_outfile_formatting():
        for mass_label, gammas in needed.items():
            res = {
                "mass": [mass_label],
                "gamma": [
                    conj for g in gammas for conj in g.conjugate_gamma_list
                ],
            }
            yield res, config.meson_stoch_proj

    return utils.io.catalog_files(generate_outfile_formatting())


def build_lma_meson_field_chain(
    config: HighModeConfig,
    mass_label: str,
    action: str,
    low_modes: str,
    gammas: t.List[Gamma],
    write: bool,
) -> HadronsInput:
    """Per-mass file-driven LMA producer chain (``low_mode_method='load'``).

    ``cbpairs_l/r`` → multi-gamma meson-field writer → one loader per
    (mass, conjugated gamma) → one eager ``StagLMAMesonFieldProp``
    producer per (gamma, mass). The writer chain is emitted only when
    ``write`` (any of the mass's gamma files missing or undersized, or
    ``overwrite``); the loaders and producers always run — loaders read
    disk, so complete files make the writer redundant. Declaration order
    carries the writer→loader file edge, which the Hadrons scheduler
    graph cannot see; the producer additionally depends on the loaders
    through its ``mesonField`` input edges.

    Producers are named ``quark_ranLL_{glabel}_mass_{m}`` so their
    outputs (``..._t{t}`` for a single gamma, ``..._t{t}_<spin>_<taste>``
    for multiple) collide exactly with the propagator names
    ``build_contractions`` and the ama guess chain already reference —
    the per-source timeslice binding survives as the output-name suffix.
    ``gammas`` is this mass's demand set (:func:`needed_ranll_gammas`).
    The writer folds the union of REQUESTED gamma strings with
    ``applyG5="true"`` (files are named by the conjugated gammas
    internally — pion files stay ``G1_G1_0_0_0.h5``); loaders are keyed
    by the conjugated names and referenced in raw order (the producer's
    positional parallel list). The time window is source-matched:
    ``tA/tB/tStep`` from the configured source range so every sampled
    ``t0`` lands on an output — never the resume-gated subset.
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

        writer = f"mfwrite_mass_{mass_label}"
        modules[writer] = hadmods.meson_field(
            name=writer,
            action="",
            block=str(config.blocksize),
            gammas=" ".join(dict.fromkeys(g.gamma_string for g in gammas)),
            gauge=(
                "gauge"
                if all(g.local for g in gammas)
                else config.shift_gauge_name
            ),
            low_modes=low_modes,
            left="",
            right="noise_fv_vec",
            output=stem,
            apply_g5="true",
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
        modules[producer] = hadmods.lma_meson_field_prop(
            name=producer,
            action=action,
            low_modes=low_modes,
            meson_field=" ".join(
                f"mfload_mass_{mass_label}_{conj}"
                for conj in g.conjugate_gamma_list
            ),
            ta=str(config.tstart),
            tb=str(config.tstop),
            tstep=str(config.dt),
            gammas=g.gamma_string,
            apply_g5="true",
            noise="noise_fv_vec",
        )
        schedule.append(producer)

    return HadronsInput(modules=modules, schedule=schedule)


def build_input_params(config: HighModeConfig) -> HadronsInput:
    modules = {}
    schedule = []

    # Sources needing (re)generation, keyed by the unique {tsource} axis
    # values (bare times in dt mode, block labels in bias mode) so duplicate
    # sampled times stay distinct sources.
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

    modules["sink"] = hadmods.sink(name="sink", mom="0 0 0")
    schedule.append("sink")

    if use_meson_field and config.masses:
        # Shared full-volume noise: every RandomWall references it by bare
        # name, and the meson-field writer (`right`) and the producers'
        # pairing/normalization self-check (`noise`) consume `noise_fv_vec`.
        # Emitted whenever the chain is (masses non-empty) — the loaders and
        # producers run even with no pending sources, so gating on run_refs
        # would dangle their noise_fv_vec references. The module name is
        # part of the RNG stream — do not rename casually.
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

    quark_inputs = build_quark_strategy(config, run_refs)
    modules |= quark_inputs.modules
    quark_schedule += quark_inputs.schedule

    contract_inputs = build_contract_strategy(config, run_refs)
    modules |= contract_inputs.modules
    quark_schedule += contract_inputs.schedule

    schedule += sort_schedule(config, quark_schedule)

    return HadronsInput(modules=modules, schedule=schedule)


def sort_schedule(config: HighModeConfig, module_names: t.List[str]) -> t.List[str]:
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
        # above every base rank (len(base)+i — byte-identical indices to
        # the legacy appended-at-end list for DIAGONAL/ALL); everything
        # else ranks against the base list. Under TIERED the `ama` dset
        # leaves the dset list while ama quark modules persist; ranking
        # those against the dset list would invert the ranLL->ama precon
        # order (quark_gen's guess chain).
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

    def tslice_order(name):
        time = re.findall(r"_t(\d+)", name)
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


def build_quark_strategy(
    config: HighModeConfig, run_refs: t.List[SourceRef]
) -> HadronsInput:
    match config.correlator_strategy:
        case CorrelatorStrategy.TWOPOINT:
            return twopoint.build_quarks(config, run_refs)
        case CorrelatorStrategy.SIB:
            # Dead path (arity-broken callee); kept for parity, callee untouched.
            return sib.build_quarks(config, run_refs)
        case _:
            raise ValueError(
                f"Unknown correlator_strategy: {config.correlator_strategy}"
            )


def build_contract_strategy(
    config: HighModeConfig, run_refs: t.List[SourceRef]
) -> HadronsInput:
    match config.correlator_strategy:
        case CorrelatorStrategy.SIB:
            return sib.build_contractions(config, run_refs)
        case CorrelatorStrategy.TWOPOINT:
            return twopoint.build_contractions(config, run_refs)
        case _:
            raise ValueError(
                f"Unknown correlator_strategy: {config.correlator_strategy}"
            )


def build_aggregator_params(
    config: HighModeConfig,
    average: bool,
    run_prefix: str = "",
) -> t.Dict:
    agg_params = freeze({})

    suffix = "_avg" if average else ""
    outfile = utils.io.get_processed_filename(
        config.high_modes.filestem, remove=["series", "tsource"], suffix=suffix
    )

    infile = config.high_modes.filename

    e_rep = freeze({"tsource": config.source_axis}).evolver()

    solver_labels = config.get_solver_labels()

    run_list = []

    actions: t.Dict[str, t.Any] = {"index": ["series_cfg", "gamma", "t"]}

    if average:
        actions["average"] = ["tsource"]
        actions["real"] = True

    for op in config.op_list:
        gamma_label = op.gamma.name.lower()
        e_rep["gamma_label"] = gamma_label
        # Mass axis from get_mass_labels so cross-mass dsets aggregate like
        # diagonal ones (the catalog/resume gate has always used this axis).
        for mass_label, dset in itertools.product(
            config.get_mass_labels(op), solver_labels
        ):
            file_label = f"{run_prefix}{gamma_label}_{mass_label}_{dset}"
            run_list.append(file_label)
            e_rep["mass"] = mass_label
            e_rep["dset"] = dset
            replacements = e_rep.persistent()

            h5_datasets = {
                g: f"/meson/meson_{i}/corr" for i, g in enumerate(op.gamma.gamma_list)
            }

            array_params = {
                "order": ["t"],
                "labels": {"t": f"0..{config.time - 1}"},
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


def validate_config(config: HighModeConfig) -> None:
    """Validate HighModeConfig after construction and postprocessing.

    Validates that if non-local operators are used, shift_gauge_name must be set.
    """
    if config.solver not in {"mpcg", "rb", "cg"}:
        raise ValueError(
            "High-mode solver must be one of 'mpcg', 'rb', or 'cg'; "
            f"got {config.solver!r}."
        )

    if config.low_mode_method not in {"compute", "load"}:
        raise ValueError(
            "low_mode_method must be 'compute' or 'load'; got "
            f"{config.low_mode_method!r}."
        )
    if config.low_mode_method == "load" and not config.skip_low_modes:
        if config.noise != 1:
            raise ValueError(
                "low_mode_method='load' requires noise == 1 (the "
                "StagLMAMesonFieldProp producer reconstructs each color "
                "from a window of 3 adjacent columns of a single "
                "color-diluted source, noiseIndex=0); got "
                f"noise={config.noise}."
            )
        if config.nbias is not None:
            raise ValueError(
                "low_mode_method='load' is incompatible with nbias (bias "
                "source labels n{i} cannot bind to the producer's fixed "
                "_t{t} output-name grammar); use dt-mode sources or "
                "low_mode_method='compute'."
            )
        if config.meson_stoch_proj is None:
            raise ValueError(
                "low_mode_method='load' requires a meson_stoch_proj files entry "
                "(filestem + good_size) for the meson-field intermediates; add "
                "one under files: in the job YAML."
            )
        if "{mass}" not in config.meson_stoch_proj.filestem:
            raise ValueError(
                "low_mode_method='load' requires the meson_stoch_proj "
                "filestem to carry the {mass} token (one meson field per "
                f"mass); got {config.meson_stoch_proj.filestem!r}."
            )
    if config.low_mode_method == "load" and config.skip_low_modes:
        utils.get_logger().warning(
            "low_mode_method='load' is inert for this entry: skip_low_modes "
            "is set, so no ranLL solver (and no meson-field chain) is "
            "emitted."
        )

    has_nonlocal_ops = any([not op.gamma.local for op in config.operations.op_list])
    if has_nonlocal_ops and config.shift_gauge_name is None:
        raise ValueError(
            "Non-local operators detected, but shift_gauge_name is not set."
        )

    if config.subgrid_ranks is not None and config.subgrid_ranks <= 0:
        raise ValueError(
            f"subgrid_ranks must be a positive integer; got {config.subgrid_ranks}."
        )

    if config.nbias is not None:
        if config.nbias < 1:
            raise ValueError(f"nbias must be a positive integer; got {config.nbias}.")
        if not config.bias_seed:
            raise ValueError(
                "bias_seed is required when nbias is set (it seeds the "
                "deterministic time-slice draws; set it in the tasks.high_modes "
                "list entry)."
            )
        if not config.bias_replace and config.nbias > config.time:
            raise ValueError(
                f"nbias ({config.nbias}) exceeds the time extent ({config.time}); "
                "without-replacement sampling (bias_replace=False) requires "
                "nbias <= time."
            )

    effective = config.effective_solve_cross_terms
    if effective != config.solve_cross_terms:
        triggers = ", ".join(
            flag
            for flag, enabled in (
                ("skip_low_modes", config.skip_low_modes),
                ("skip_cg", config.skip_cg),
            )
            if enabled
        )
        utils.get_logger().warning(
            f"solve_cross_terms={config.solve_cross_terms.name} downgraded to "
            f"{effective.name}: {triggers} "
            f"{'are' if ', ' in triggers else 'is'} set — cross modes require "
            "both solver classes (low modes and CG); only same-solver "
            "(diagonal) pairs are admitted."
        )
