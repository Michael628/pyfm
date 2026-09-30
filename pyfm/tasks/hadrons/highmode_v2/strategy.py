"""Per-entry module emission for ``hadrons_lma_new`` high-mode entries.

Fully separate from the legacy ``highmode/`` package (D4): owns the entry
catalog (:func:`create_outfile_catalog`, serving the resume gate here and
aggregation/comparison from the task level), the ranLL gamma demand set
(:func:`needed_ranll_gammas`), the schedule sort, the loader/producer
meson-field chain (biased-aware: one producer per drawn slice), and the
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
from pyfm.tasks.hadrons.types import HadronsInput, SourceRef
import pyfm.tasks.hadrons.modules as hadmods
from pyfm.tasks.hadrons import meson, meson_v2
from pyfm.tasks.hadrons.highmode_v2 import twopoint
from pyfm.tasks.hadrons.highmode_v2.config import (
    CacheMode,
    LMAHighModeConfig,
    LowModeMethod,
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
    to a ``quark_ranLL_{glabel}_mass_{m}`` producer whose outputs are the
    propagators contractions and the ama guess chain reference.
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
) -> HadronsInput:
    """Loader + producer half of the meson-field chain (per entry).

    Loaders: one per (mass, conjugated gamma), label-prefixed, reading the
    entry's cache files by the writer's filestem grammar. Producers: one
    per (gamma, mass) in grid mode (a single ``tstart..tstop`` stride
    ``dt`` window); biased sources emit one producer per drawn slice
    (``tA=tB=t0``, sharing the loaders) — a documented ADR consequence
    (one producer per slice until ``StagLMAMesonFieldProp`` accepts a
    time-slice list). Producer/output naming goes through
    ``twopoint.meson_field_producer_name`` so contractions and the ama
    guess chain resolve the published objects (``_t{t0}``) regardless of
    source mode.
    """
    modules = {}
    schedule = []
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
            if loader not in modules:
                modules[loader] = hadmods.load_meson_field(
                    name=loader,
                    file=f"{stem}.@traj@/{conj}_0_0_0.h5",
                    dataset=f"{conj}_0_0_0",
                )
                schedule.append(loader)

    is_biased = config.sources_config.biased_config is not None
    # One (ref, tA, tB, tStep) window per producer to emit: grid shares a
    # single producer over the whole configured range (no ref needed);
    # biased draws one dedicated single-time producer per slice.
    windows = (
        [
            (ref, str(ref.t0), str(ref.t0), "1")
            for ref in config.sources_config.source_refs
        ]
        if is_biased
        else [
            (
                None,
                str(config.sources_config.grid_config.tstart),
                str(config.sources_config.grid_config.tstop),
                str(config.sources_config.grid_config.dt),
            )
        ]
    )

    for g in gammas:
        glabel = g.name.lower()
        meson_field_refs = " ".join(
            config.module_name(f"mfload_mass_{mass_label}_{conj}")
            for conj in g.conjugate_gamma_list
        )
        for ref, ta, tb, tstep in windows:
            producer = twopoint.meson_field_producer_name(config, glabel, mass_label, ref)
            modules[producer] = hadmods.lma_meson_field_prop_v2(
                name=producer,
                action=action,
                low_modes=low_modes,
                meson_field=meson_field_refs,
                gammas=spintaste_names[g],
                labels=" ".join(g.gamma_list),
                ta=ta,
                tb=tb,
                tstep=tstep,
                noise=noise_vec,
                n_noise=str(config.sources_config.noise),
            )
            schedule.append(producer)

    return HadronsInput(modules=modules, schedule=schedule)


def build_input_params(config: LMAHighModeConfig) -> HadronsInput:
    """Per-entry emission (design doc order): SpinTaste + sink →
    full-volume noise → cache writer → (stop for build_only) → per-source
    noise → mass loop (meson-field chains / ranLL solver / CG solvers) →
    quarks + contractions, sorted.

    The resume gate keys on the entry's correlator catalog (output files);
    ``build_only`` entries have no output block and skip the gate (all
    source refs, unused — emission stops after the cache writer).
    """
    modules = {}
    schedule = []

    mf = config.low_modes_config.meson_field_config
    cache_build_only = mf is not None and mf.cache is CacheMode.BUILD_ONLY

    if config.output_config is not None and not config.output_config.overwrite:
        df = create_outfile_catalog(config)
        if df.empty:
            run_refs = []
        else:
            missing_files = df[df["exists"] == False]
            run_refs = [
                ref
                for ref in config.sources_config.source_refs
                if any(missing_files["tsource"] == ref.axis)
            ]
    else:
        run_refs = config.sources_config.source_refs

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

    # 5. Per-source noise.
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
    for mass_label in config.masses:
        action = config.action_name.format(mass=mass_label)  # parent-owned: unprefixed
        if config.low_modes_config.method is not LowModeMethod.NONE:
            low_modes = config.low_modes_name.format(mass=mass_label)
            if config.use_meson_field:
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

    # 7. Quarks, contractions (sorted as before).
    quark_inputs = twopoint.build_quarks(config, run_refs, spintaste_names)
    modules |= quark_inputs.modules
    quark_schedule += quark_inputs.schedule

    contract_inputs = twopoint.build_contractions(config, run_refs, spintaste_names)
    modules |= contract_inputs.modules
    quark_schedule += contract_inputs.schedule

    schedule += sort_schedule(config, quark_schedule)

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
