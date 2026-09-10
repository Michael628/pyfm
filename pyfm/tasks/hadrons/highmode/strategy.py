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
from pyfm.domain import OpList
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

    modules["sink"] = hadmods.sink(name="sink", mom="0 0 0")
    schedule.append("sink")

    quark_schedule = []
    for ref in run_refs:
        name = f"noise_{ref.label}"
        modules[name] = hadmods.noise_rw(
            name=name,
            nsrc=str(config.noise),
            t0=str(ref.t0),
            tstep=str(config.time),
        )
        quark_schedule.append(name)

    for mass_label in config.masses:
        action = config.action_name.format(mass=mass_label)
        if not config.skip_low_modes:
            name = config.solver_name.format(solver="ranLL", mass=mass_label)
            low_modes = config.low_modes_name.format(mass=mass_label)
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
