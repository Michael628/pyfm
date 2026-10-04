"""Connected-SIB meson-field producer task (``hadrons_sib_mf``).

Sibling composite of ``hadrons_lma_new`` driving the HadronsMILC SIB HVP
A2A-batch module surface (``../HadronsMILC/test/params/sib-hvp-mesonfields.xml``):
full-volume noise → ``noise_t{t0}_vec`` batch source → ``tab`` (⟨ℓ|η⟩
overlap table) → ``p`` (the ``StagLMAMesonFieldProp`` ``a2a_batch`` guess)
→ per-mass ``h`` solves → six leg-pair ``StagA2AMesonField`` blocks per
Γ family. The task produces meson-field artifacts only — aggregation has
no meaning here, so ``build_aggregator_params`` returns ``{}``.

Flat single-module layout (``lma_new.py`` style): config tree, hooks,
emission, catalog, and registration in one file. Batch data contract
pinned by construction (upstream deliberately does not cross-check it):
``tA=t0``, ``tB=t0+(n_slices-1)·t_step``, ``tStep=t_step``,
``nNoise=noise`` (the RandomWall ``nSrc``), single label ``G1_G1``.
Emission is demand-driven: the resume gate catalogs the block/tab
outputs at (family, leg_pair, mass, gamma) granularity and narrows the
chain to the missing ones (the meson_v2 skip-if-complete precedent).
"""

import copy
import typing as t

import h5py
import numpy as np
import pandas as pd
from pydantic.dataclasses import dataclass

from pyfm import utils
from pyfm.domain import CompositeConfig, Gamma, MassDict, OpList, Outfile, SimpleConfig
from pyfm.tasks.hadrons.types import HadronsInput
from pyfm.tasks.register import register_task

from . import gauge, epack
from .lma_new import _merge_modules
import pyfm.tasks.hadrons.modules as hadmods

# Name-template defaults layered by route_params (lma_new conventions).
_ACTION_NAME = "stag_mass_{mass}"
_LOW_MODES_NAME = "evecs_mass_{mass}"
_SHIFT_GAUGE_NAME = "gauge_apbc"

# The six leg-pair block kinds per Gamma family (pyfm-side letters, from
# the research naming decisions): l = low modes (eig), n = full-volume
# noise, p = LMA batch precon guess, h = unprojected high solve.
# Reference blocks (ll/lp/nl/np) are emitted once at defl_mass; h-legged
# blocks (lh/nh) are emitted per operations mass. Upstream writes the
# noise leg as `e` (eta) and the guess leg as `g`.
_REFERENCE_LEG_PAIRS = ("ll", "lp", "nl", "np")
_H_LEG_PAIRS = ("lh", "nh")
# Pairs with no eigenvector leg on either side: empty lowModes, no
# CB pairs (upstream sib-hvp-mesonfields.xml eh/eg blocks).
_PURE_HIGH_LEG_PAIRS = frozenset({"np", "nh"})

# The three Γ families the block table covers (spin-taste structure lives
# in the meson-field kernel, not the solves — all four field sets are
# scalar-spin-taste).
_SIB_FAMILIES = frozenset({Gamma.SCALAR_LOCAL, Gamma.VEC_LOCAL, Gamma.VEC_ONELINK})

# Module-name prefix per family — the upstream s/vl/vo nicknames. Module
# names need the family axis for uniqueness (mf_s_ll vs mf_vl_ll; the
# _merge_modules one-module-per-name rule); output stems do NOT (flattened
# stems: every family's blocks for one leg pair share <stem>.<traj>/,
# one file per GammaName).
_FAMILY_NICKNAMES = {
    Gamma.SCALAR_LOCAL: "s",
    Gamma.VEC_LOCAL: "vl",
    Gamma.VEC_ONELINK: "vo",
}


@dataclass(frozen=True)
class SIBBatchConfig(SimpleConfig):
    """``sib.batch`` block: the A2A batch source window.

    One ``MSource::StagRandomWall`` publishes ``noise_t{t0}_vec`` with
    ``3 * noise * n_slices`` columns (slice-major, column
    ``3*(noise*n_slices + slice) + color``); the batched
    ``StagLMAMesonFieldProp`` guess and the ``h`` solve consume the whole
    window at once. ``noise`` and ``time`` absorb from shared/job params
    when omitted from the block (the builder's shared-param layering), so
    a job-level ``params: {noise: 2}`` keeps ``nSrc``, ``nNoise``, the
    runid, and the ``{noise}`` filestem token consistent.
    """

    t0: int
    n_slices: int
    noise: int
    time: int
    t_step: int = 1

    @property
    def tb(self) -> int:
        """Last source slice of the batch window (the producer's ``tB``)."""
        return self.t0 + (self.n_slices - 1) * self.t_step


def validate_batch(config: SIBBatchConfig) -> None:
    """Validate SIBBatchConfig after construction: window inside the extent."""
    if config.n_slices < 1:
        raise ValueError(f"sib.batch.n_slices must be >= 1; got {config.n_slices}.")
    if config.noise < 1:
        raise ValueError(f"sib.batch.noise must be >= 1; got {config.noise}.")
    if config.t_step < 1:
        raise ValueError(f"sib.batch.t_step must be >= 1; got {config.t_step}.")
    if not (0 <= config.t0 <= config.tb < config.time):
        raise ValueError(
            f"sib.batch window [t0={config.t0}, tB={config.tb}] stride "
            f"{config.t_step} exceeds the lattice time extent "
            f"{config.time}."
        )


@dataclass(frozen=True)
class SIBSolverConfig(SimpleConfig):
    """``sib.solver`` block: the per-mass ``h`` solve.

    Single residual only: the block topology has exactly one ``h`` leg
    per mass — a residual ladder would need one ``h`` per rung and
    per-rung blocks, for which there is no upstream precedent.
    """

    solver: str = "mpcg"
    residual: float = 1e-8


def validate_solver(config: SIBSolverConfig) -> None:
    """Validate SIBSolverConfig after construction."""
    if config.solver not in {"mpcg", "rb", "cg"}:
        raise ValueError(
            "sib.solver.solver must be one of 'mpcg', 'rb', or 'cg'; got "
            f"{config.solver!r}."
        )


@dataclass(frozen=True)
class SIBOutputConfig(SimpleConfig):
    """``sib.output`` block: meson-field block and tab targets.

    ``file`` is the block files label — its filestem carries ``{leg_pair}``
    and ``{mass}``, where ``{mass}`` formats to ``''`` for the mass-free
    reference blocks (ll/lp/nl/np) and ``'_m<label>'`` for the per-mass
    h-legged blocks (lh/nh). ``tab`` is the scalar-only ⟨ℓ|η⟩ overlap
    table label. Both route to the ``cfg_gamma_h5`` ext
    (``.{cfg}/{gamma}_0_0_0.h5``), so their labels must contain "meson"
    (the ``Outfile.from_param`` name-key routing,
    ``pyfm/domain/outfiles.py``).
    """

    file: Outfile
    tab: Outfile
    overwrite: bool = False


@dataclass(frozen=True)
class SIBMFConfig(CompositeConfig):
    """Own config tree for ``hadrons_sib_mf`` (sibling of ``hadrons_lma_new``).

    A genuinely new class, not a subclass or reuse of a sibling task's
    config class (``register.py``'s ``_config_to_task_key`` reverse-lookup
    keys by class object; reusing a sibling task's config class verbatim
    would silently steal its registration).

    ``operations`` selects the Γ families (a subset of scalar_local /
    vec_local / vec_onelink) and the ``h``-solve masses; ``defl_mass``
    (default ``"l"``, the most commonly calculated mass) names the single
    ``ModifyEigenPackMILC`` shift every low-side artifact (tab, p, the
    ll/lp/nl/np blocks, the CB pairs) builds from — all contributions at
    that mass then contract without reweighting, and every other mass
    reweights downstream from the stored evals.
    """

    gauge_config: gauge.GaugeConfig
    epack_config: epack.EpackConfig
    batch_config: SIBBatchConfig
    solver_config: SIBSolverConfig
    output_config: SIBOutputConfig
    mass: MassDict
    operations: OpList
    defl_mass: str = "l"
    blocksize: int = 12
    shift_gauge_name: str | None = None

    @property
    def op_list(self) -> t.List[OpList.Op]:
        """Get list of gamma operations."""
        return self.operations.op_list

    @property
    def masses(self) -> t.List[str]:
        """Mass labels the ``h`` solves (and lh/nh blocks) run at."""
        return self.operations.mass


def route_params(params: t.Dict) -> t.Dict:
    """Route the ``sib:`` section onto sub-blocks and plain fields.

    The ``sib`` mapping lifts: ``batch``/``solver``/``output`` onto their
    subconfig slices, ``operations`` onto the OpList field, and the
    remaining keys (``defl_mass``, ``blocksize``, ``shift_gauge_name``)
    onto plain composite fields. ``tasks.gauge``/``tasks.epack`` ride onto
    their child slices with the shared name-template defaults (lma_new
    convention). Every other ``tasks:`` key and every unknown ``sib`` key
    is rejected loudly.
    """
    # Deep-copy: this hook pops keys out of the nested sib block (and the
    # gauge/epack slices below), and create_task is called once per cfgno
    # on the same YAML tree (lma_new.normalize_params' rationale).
    preprocessor_params = copy.deepcopy(params.pop("_preprocessor", {}))

    sib = preprocessor_params.pop("sib", {})
    if not isinstance(sib, dict):
        raise TypeError(
            f"tasks.sib must be a mapping; got {type(sib).__name__}."
        )

    def _block(value) -> t.Dict:
        """A sub-block mapping; ``true`` counts as an empty block (defaults)."""
        return value if isinstance(value, dict) else {}

    child_preprocessor = dict(
        gauge_config=dict(action_name=_ACTION_NAME),
        epack_config=dict(
            action_name=_ACTION_NAME,
            low_modes_name=_LOW_MODES_NAME,
        ),
        batch_config=_block(sib.pop("batch", {})),
        solver_config=_block(sib.pop("solver", {})),
        output_config=_block(sib.pop("output", {})),
    )

    operations = sib.pop("operations", {})
    if not isinstance(operations, dict):
        raise TypeError(
            f"sib.operations must be a mapping; got "
            f"{type(operations).__name__}."
        )

    unknown_tasks = sorted(k for k in preprocessor_params if k not in ("gauge", "epack"))
    if unknown_tasks:
        raise ValueError(
            f"Unknown tasks keys for hadrons_sib_mf: {unknown_tasks}; "
            "allowed task blocks are gauge, epack, sib."
        )
    for k, v in preprocessor_params.items():
        child_preprocessor[f"{k}_config"] |= v

    sib_fields = {"defl_mass", "blocksize", "shift_gauge_name"}
    unknown_sib = sorted(k for k in sib if k not in sib_fields)
    if unknown_sib:
        raise ValueError(
            f"Unknown sib section keys: {unknown_sib}. Allowed keys are "
            "operations, batch, solver, output, defl_mass, blocksize, "
            "shift_gauge_name."
        )

    params = dict(shift_gauge_name=_SHIFT_GAUGE_NAME) | params | sib
    return params | {"operations": operations, "_preprocessor": child_preprocessor}


def validate_config(config: SIBMFConfig) -> None:
    """Validate SIBMFConfig after construction (cross-section checks).

    Per-block state was validated when each leaf was built. Here:
    ``defl_mass`` and every operations mass must exist in the MassDict;
    op gammas must be SIB families (the block table covers exactly those);
    non-local families (vec_onelink) require ``shift_gauge_name``;
    ``operations`` must be non-empty.
    """
    if not config.op_list:
        raise ValueError(
            "sib.operations has no operations — provide an operations: "
            "block with at least one of scalar_local / vec_local / "
            "vec_onelink and one mass (e.g. operations: {gamma: "
            "[scalar_local], mass: [l]})."
        )
    for op in config.op_list:
        if not op.mass:
            raise ValueError(
                f"sib.operations gamma {op.gamma.name.lower()!r} has no mass — "
                "every operation needs at least one mass (the h solves and "
                "lh/nh blocks run per operations mass; e.g. operations: "
                "{gamma: [scalar_local], mass: [l]})."
            )
    if config.defl_mass not in config.mass:
        raise ValueError(
            f"sib.defl_mass {config.defl_mass!r} is not present in the "
            f"mass parameters ({sorted(config.mass.keys())}); it names the "
            "single ModifyEigenPackMILC shift every low-side artifact "
            "builds from."
        )
    for m in config.masses:
        if m not in config.mass:
            raise ValueError(
                f"sib.operations mass {m!r} is not present in the mass "
                f"parameters ({sorted(config.mass.keys())})."
            )
    bad = [op.gamma.name.lower() for op in config.op_list if op.gamma not in _SIB_FAMILIES]
    if bad:
        raise ValueError(
            f"sib.operations gammas {bad} are not SIB families; allowed: "
            "scalar_local, vec_local, vec_onelink."
        )
    has_nonlocal_ops = any(not op.gamma.local for op in config.op_list)
    if has_nonlocal_ops and config.shift_gauge_name is None:
        raise ValueError(
            "Non-local operators detected (vec_onelink), but "
            "shift_gauge_name is not set."
        )


def _sib_outfile_catalog(config: SIBMFConfig) -> pd.DataFrame:
    """Catalog the SIB outputs (blocks + tab) at (leg_pair, mass, gamma)
    granularity — the axes the resume gate narrows on and compare_outputs
    pairs on. One generator yield per (family, leg-pair group): the
    reference pairs carry the mass-free stem token (""), the h pairs one
    ``_m<label>`` token per operations mass; the tab row is scalar-only.
    """

    def generate_outfile_formatting():
        for op in config.op_list:
            gammas = op.gamma.gamma_list
            yield (
                {
                    "gamma": gammas,
                    "leg_pair": list(_REFERENCE_LEG_PAIRS),
                    "mass": [""],
                },
                config.output_config.file,
            )
            yield (
                {
                    "gamma": gammas,
                    "leg_pair": list(_H_LEG_PAIRS),
                    "mass": [f"_m{m}" for m in config.masses],
                },
                config.output_config.file,
            )
        yield (
            {"gamma": Gamma.SCALAR_LOCAL.gamma_list},
            config.output_config.tab,
        )

    return utils.io.catalog_files(generate_outfile_formatting())


def _block_filepath(
    config: SIBMFConfig, leg_pair: str, mass: str, gamma_name: str
) -> str:
    """Formatted output path of one block file.

    ``mass`` is the stem token ("" for reference pairs, ``_m<label>``
    for h pairs) — exactly the value the emission formats into
    ``{mass}``.
    """
    return config.output_config.file.filename.format(
        leg_pair=leg_pair, mass=mass, gamma=gamma_name
    )


def _tab_filepath(config: SIBMFConfig) -> str:
    """Formatted output path of the scalar ⟨ℓ|η⟩ overlap table."""
    return config.output_config.tab.filename.format(
        gamma=Gamma.SCALAR_LOCAL.gamma_list[0]
    )


def _needed_blocks(
    config: SIBMFConfig, bad_files: t.Set[str]
) -> t.Tuple[t.Dict[Gamma, t.Set[str]], t.Set[t.Tuple[Gamma, str, str]]]:
    """Demand sets from the resume gate's bad-file list.

    Returns ``(needed_ref, needed_h)``: reference leg pairs per family
    (e.g. ``{Gamma.VEC_LOCAL: {"ll", "np"}}``) and
    ``(family, leg_pair, mass_label)`` triples for h-legged pairs with at
    least one bad file. A bad file anywhere in a leg pair's gamma set
    re-runs that whole leg pair (one module writes all its family's
    GammaNames — per-gamma narrowing is impossible at module granularity).
    """
    needed_ref: t.Dict[Gamma, t.Set[str]] = {}
    needed_h: t.Set[t.Tuple[Gamma, str, str]] = set()
    for op in config.op_list:
        for leg_pair in _REFERENCE_LEG_PAIRS:
            if any(
                _block_filepath(config, leg_pair, "", gamma_name) in bad_files
                for gamma_name in op.gamma.gamma_list
            ):
                needed_ref.setdefault(op.gamma, set()).add(leg_pair)
        for mass_label in config.masses:
            for leg_pair in _H_LEG_PAIRS:
                if any(
                    _block_filepath(config, leg_pair, f"_m{mass_label}", gamma_name)
                    in bad_files
                    for gamma_name in op.gamma.gamma_list
                ):
                    needed_h.add((op.gamma, leg_pair, mass_label))
    return needed_ref, needed_h


def build_input_params(config: SIBMFConfig) -> HadronsInput:
    """Generate input parameters for the connected-SIB meson-field task.

    Emission order IS the execution contract under the naive scheduler
    (the tab writer → loader file dependency has no environment edge), so
    sections append in dependency order: base gauge → epack actions →
    epack → defl-mass shift → sp gauge/actions (mpcg) → SIB actions →
    [resume gate] → CB pairs → h-solve solvers → SpinTaste → noise → tab
    writer/loader → precon → h solves → blocks. The schedule is
    deduplicated first-occurrence (lma_new convention).

    The SIB chain (sections 6-13) is demand-driven: the gate catalogs the
    block/tab outputs and emits only the missing ones plus the chain
    modules they transitively reference. ``output.overwrite`` skips the
    gate (everything runs). Infra sections (gauge/epack/actions/shifts)
    always emit — the lma_new presence-driven convention.
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

    # 3. Single mass shift at defl_mass: every low-side artifact (tab, p,
    #    the ll/lp/nl/np blocks, the CB pairs) builds from this one
    #    shifted pack (D9).
    mass_shifts_input = epack.build_epack_mass_shifts(
        config.epack_config, [config.defl_mass]
    )
    _merge_modules(modules, mass_shifts_input.modules)
    schedule += mass_shifts_input.schedule

    # 4. SIB actions. dp masses = operations masses ∪ {defl_mass} (the p
    #    producer and CB pairs bind stag_mass_{defl_mass} — Meooe-only,
    #    mass-independent, but the action module must exist); sp masses +
    #    the sp gauge only when the h solver is mixed-precision.
    sp_masses = config.masses if config.solver_config.solver == "mpcg" else []
    if sp_masses:
        sp_gauge = gauge.build_sp_gauge(config.gauge_config)
        _merge_modules(modules, sp_gauge.modules)
        schedule += sp_gauge.schedule

    dp_masses = list(dict.fromkeys([*config.masses, config.defl_mass]))
    actions = gauge.build_action_modules(
        config.gauge_config, dp_masses=dp_masses, sp_masses=sp_masses
    )
    _merge_modules(modules, actions.modules)
    schedule += actions.schedule

    # 5. Resume gate: demand sets for the SIB chain. With overwrite, or
    #    when every output file is present and good-sized, nothing below
    #    changes; with partial outputs the chain narrows to the missing
    #    blocks plus the modules they reference.
    batch = config.batch_config
    if config.output_config.overwrite:
        needed_ref = {op.gamma: set(_REFERENCE_LEG_PAIRS) for op in config.op_list}
        needed_h = {
            (op.gamma, leg_pair, mass_label)
            for op in config.op_list
            for mass_label in config.masses
            for leg_pair in _H_LEG_PAIRS
        }
        tab_needed = True
    else:
        bad_files = set(utils.io.get_bad_files(_sib_outfile_catalog(config)))
        needed_ref, needed_h = _needed_blocks(config, bad_files)
        tab_needed = _tab_filepath(config) in bad_files

    ref_pairs = set().union(*needed_ref.values()) if needed_ref else set()
    # h solves run per mass, in operations order, only when an lh/nh
    # block of some family needs them.
    h_masses = [m for m in config.masses if any(mh[2] == m for mh in needed_h)]
    need_precon = "lp" in ref_pairs or "np" in ref_pairs or bool(needed_h)
    need_noise_t0 = bool(needed_h)
    need_noise_fv = (
        tab_needed
        or need_precon
        or need_noise_t0
        or "nl" in ref_pairs
        or "np" in ref_pairs
    )
    need_cbpairs = (
        tab_needed
        or "ll" in ref_pairs
        or "lp" in ref_pairs
        or "nl" in ref_pairs
        or any(leg_pair == "lh" for (_, leg_pair, _) in needed_h)
    )

    action_defl = config.gauge_config.action_name.format(mass=config.defl_mass)
    low_modes_defl = config.epack_config.low_modes_name.format(mass=config.defl_mass)
    cbpairs_l = f"cbpairs_l_mass_{config.defl_mass}"
    cbpairs_r = f"cbpairs_r_mass_{config.defl_mass}"
    precon = f"precon_t{batch.t0}"
    noise_batch = f"noise_t{batch.t0}"
    noise_fv_vec = "noise_fv_vec"

    # 6. CB pairs at defl_mass: two DISTINCT MUtilities::EigenPackCBPairs
    #    instances (the C++ setup requires cbPairsLeft != cbPairsRight),
    #    shared by the tab writer and every low-sided block. Emitted only
    #    when a low-side consumer will run.
    if need_cbpairs:
        for name in (cbpairs_l, cbpairs_r):
            modules[name] = hadmods.eigen_pack_cb_pairs(
                name=name, eigen_pack=low_modes_defl, action=action_defl
            )
            schedule.append(name)

    # 7. h-solve solver modules, one per mass whose h solve will run.
    #    Names carry no solver-label substrings and no slice token
    #    (nothing sorts this schedule — D10).
    for mass_label in h_masses:
        action = config.gauge_config.action_name.format(mass=mass_label)
        name = f"sib_solver_mass_{mass_label}"
        resid = str(config.solver_config.residual)
        match config.solver_config.solver:
            case "rb":
                modules[name] = hadmods.rb_cg(
                    name=name, action=action, residual=resid
                )
            case "cg":
                modules[name] = hadmods.cg(
                    name=name, action=action, residual=resid
                )
            case "mpcg":
                modules[name] = hadmods.mixed_precision_cg(
                    name=name,
                    outer_action=action,
                    inner_action=f"i{action}",
                    residual=resid,
                )
            case _:
                raise ValueError(
                    f"Unknown SIB solver: {config.solver_config.solver}"
                )
        schedule.append(name)

    # 8. SpinTaste modules, one per Γ family that will emit blocks, plus
    #    the scalar family whenever the tab writer, the precon, or an h
    #    solve runs (tab/p/h are scalar-spin-taste regardless of which
    #    families operations selects — the vector structure of
    #    Vec-Scalar-Vec lives in the block kernels, not the solves). No
    #    mass axis. Local families bind no gauge; vec_onelink binds
    #    shift_gauge_name (gauge_apbc — D7). apply_g5="false" throughout
    #    (the upstream XML's setting).
    needed_families = set(needed_ref) | {family for (family, _, _) in needed_h}
    if tab_needed or need_precon or need_noise_t0:
        needed_families.add(Gamma.SCALAR_LOCAL)
    spintaste_names: t.Dict[Gamma, str] = {}
    for gamma in (Gamma.SCALAR_LOCAL, Gamma.VEC_LOCAL, Gamma.VEC_ONELINK):
        if gamma not in needed_families:
            continue
        name = f"spintaste_{gamma.name.lower()}"
        spintaste_names[gamma] = name
        modules[name] = hadmods.spin_taste(
            name=name,
            gammas=gamma.gamma_string,
            gauge="" if gamma.local else config.shift_gauge_name,
            apply_g5="false",
        )
        schedule.append(name)

    # 9. Full-volume noise + the batch RandomWall source (demand-driven).
    #    The ``_vec`` companions (noise_fv_vec, noise_t{t0}_vec) are
    #    implicit C++ outputs of the color-diagonal noise modules,
    #    published whenever colorDiag=true — nothing to emit for them.
    if need_noise_fv:
        modules["noise_fv"] = hadmods.full_volume_noise(
            name="noise_fv", nsrc=str(batch.noise)
        )
        schedule.append("noise_fv")
    if need_noise_t0:
        modules[noise_batch] = hadmods.noise_rw(
            name=noise_batch,
            nsrc=str(batch.noise),
            t0=str(batch.t0),
            tstep=str(batch.t_step),
            noise="noise_fv",
        )
        schedule.append(noise_batch)

    # 10. tab writer + loader (scalar family only): the ⟨ℓ|η⟩ overlap
    #     table the precon reconstructs from. Writer → loader is a FILE
    #     dependency with no environment edge — schedule order is the only
    #     execution contract (D10). The writer re-runs only when its own
    #     file is bad; the loader rides with the precon.
    tab_stem = config.output_config.tab.filestem
    if tab_needed:
        modules["mf_tab"] = hadmods.meson_field_v2(
            name="mf_tab",
            block=str(config.blocksize),
            gammas=spintaste_names[Gamma.SCALAR_LOCAL],
            low_modes=low_modes_defl,
            left="",
            right=noise_fv_vec,
            output=tab_stem,
            cb_pairs_left=cbpairs_l,
            cb_pairs_right=cbpairs_r,
        )
        schedule.append("mf_tab")
    if need_precon:
        modules["mfload_tab"] = hadmods.load_meson_field(
            name="mfload_tab",
            file=f"{tab_stem}.@traj@/G1_G1_0_0_0.h5",
            dataset="G1_G1_0_0_0",
        )
        schedule.append("mfload_tab")

    # 11. The batched precon guess p: ONE StagLMAMesonFieldProp spanning
    #     the whole source window with a2a_batch="true" (single-key
    #     output, batch-guess form consumable by canonical StagGaugeProp).
    #     The batch data contract is pinned by construction (D12) —
    #     upstream deliberately does not cross-check it: tA=t0,
    #     tB=t0+(n_slices-1)·t_step, tStep=t_step, nNoise=noise (the
    #     RandomWall nSrc), labels=G1_G1 == the scalar SpinTaste module's
    #     effective label, projector absent.
    if need_precon:
        modules[precon] = hadmods.lma_meson_field_prop_v2(
            name=precon,
            action=action_defl,
            low_modes=low_modes_defl,
            meson_field="mfload_tab",
            gammas=spintaste_names[Gamma.SCALAR_LOCAL],
            labels="G1_G1",
            ta=str(batch.t0),
            tb=str(batch.tb),
            tstep=str(batch.t_step),
            noise=noise_fv_vec,
            n_noise=str(batch.noise),
            a2a_batch="true",
        )
        schedule.append(precon)

    # 12. h solves: one canonical StagGaugeProp per needed operations
    #     mass, sourcing the batch _vec columns and guessing p (the low
    #     subspace is deliberately left in source and solution — it is
    #     removed at contraction, i.e. by the downstream task).
    for mass_label in h_masses:
        name = f"quark_h_mass_{mass_label}_t{batch.t0}"
        modules[name] = hadmods.quark_prop_v2(
            name=name,
            source=f"{noise_batch}_vec",
            solver=f"sib_solver_mass_{mass_label}",
            guess=precon,
            gammas=spintaste_names[Gamma.SCALAR_LOCAL],
        )
        schedule.append(name)

    # 13. Six leg-pair blocks per family (flattened stems, D8): reference
    #     pairs once at defl_mass (mass-free stem token), h pairs per
    #     operations mass (_m<label> token). Pure-high pairs (np/nh)
    #     carry empty lowModes and no CB pairs; low-side pairs bind both
    #     CB pair modules (the emitter requires them together). Blocks
    #     append after every module they reference (append order IS the
    #     schedule, D10).
    blocks_stem = config.output_config.file.filestem
    for op in config.op_list:
        gamma = op.gamma
        nick = _FAMILY_NICKNAMES[gamma]
        if not needed_ref.get(gamma) and not any(
            (gamma, leg_pair, mass_label) in needed_h
            for mass_label in config.masses
            for leg_pair in _H_LEG_PAIRS
        ):
            # No block of this family is demanded — the family's
            # SpinTaste module was not emitted either.
            continue
        gammas_ref = spintaste_names[gamma]
        for leg_pair in _REFERENCE_LEG_PAIRS:
            if leg_pair not in needed_ref.get(gamma, ()):
                continue
            pure_high = leg_pair in _PURE_HIGH_LEG_PAIRS
            name = f"mf_{nick}_{leg_pair}_t{batch.t0}"
            modules[name] = hadmods.meson_field_v2(
                name=name,
                block=str(config.blocksize),
                gammas=gammas_ref,
                low_modes="" if pure_high else low_modes_defl,
                left=noise_fv_vec if leg_pair.startswith("n") else "",
                right=precon if leg_pair.endswith("p") else "",
                output=blocks_stem.format(leg_pair=leg_pair, mass=""),
                cb_pairs_left="" if pure_high else cbpairs_l,
                cb_pairs_right="" if pure_high else cbpairs_r,
            )
            schedule.append(name)
        for mass_label in config.masses:
            for leg_pair in _H_LEG_PAIRS:
                if (gamma, leg_pair, mass_label) not in needed_h:
                    continue
                pure_high = leg_pair in _PURE_HIGH_LEG_PAIRS
                name = f"mf_{nick}_{leg_pair}_mass_{mass_label}_t{batch.t0}"
                modules[name] = hadmods.meson_field_v2(
                    name=name,
                    block=str(config.blocksize),
                    gammas=gammas_ref,
                    low_modes="" if pure_high else low_modes_defl,
                    left=noise_fv_vec if leg_pair.startswith("n") else "",
                    right=f"quark_h_mass_{mass_label}_t{batch.t0}",
                    output=blocks_stem.format(
                        leg_pair=leg_pair, mass=f"_m{mass_label}"
                    ),
                    cb_pairs_left="" if pure_high else cbpairs_l,
                    cb_pairs_right="" if pure_high else cbpairs_r,
                )
                schedule.append(name)

    # Deduplicate schedule: keep first occurrence of each module name
    deduplicated_schedule = list(dict.fromkeys(schedule))

    return HadronsInput(modules=modules, schedule=deduplicated_schedule)


def create_outfile_catalog(config: SIBMFConfig) -> pd.DataFrame:
    catalogs = [
        gauge.create_outfile_catalog(config.gauge_config),
        epack.create_outfile_catalog(config.epack_config),
        _sib_outfile_catalog(config),
    ]
    return pd.concat(catalogs, ignore_index=True)


def build_aggregator_params(config: SIBMFConfig, average: bool) -> t.Dict:
    """Aggregation has no meaning for a meson-field producer task (the
    ``aggregate`` CLI raises on empty params — the correct behavior here).
    """
    return {}


_PAIR_KEYS = ["leg_pair", "mass", "gamma"]


def _col(row: pd.Series, key: str, default: t.Any = None) -> t.Any:
    """Read a (possibly NaN / absent) catalog column from a merged row."""
    if key not in row:
        return default
    val = row[key]
    return default if pd.isna(val) else val


def _compare_h5_file(
    filepath_a: str,
    filepath_b: str,
    gamma: str,
    *,
    rtol: float,
    atol: float,
) -> t.Tuple[float, float, bool]:
    """Compare the ``{gamma}_0_0_0/a2aMatrix`` datasets in two SIB files.

    The compound ``{re, im}`` dataset is read as complex128 and compared
    element-wise (the same tolerance form as highmode_v2's correlator
    comparison; ``astype`` reads either float32 or float64 producer
    fields). Returns ``(max_abs_diff, max_rel_diff, within_tolerance)``;
    a shape mismatch reports infinite diffs and within=False.
    """
    max_abs = 0.0
    max_rel = 0.0
    within = True
    ds = f"{gamma}_0_0_0"
    try:
        with h5py.File(filepath_a, "r") as fa, h5py.File(filepath_b, "r") as fb:
            for f, path in ((fa, filepath_a), (fb, filepath_b)):
                if ds not in f or "a2aMatrix" not in f[ds]:
                    raise ValueError(
                        f"dataset {ds!r}/a2aMatrix not found in {path!r}"
                    )
            a = fa[ds]["a2aMatrix"][()]
            b = fb[ds]["a2aMatrix"][()]
            ac = a["re"].astype(np.float64) + 1j * a["im"].astype(np.float64)
            bc = b["re"].astype(np.float64) + 1j * b["im"].astype(np.float64)
            if ac.shape != bc.shape:
                return float("inf"), float("inf"), False
            diff = np.abs(ac - bc)
            max_abs = max(max_abs, float(np.max(diff)) if diff.size else 0.0)
            denom = np.abs(bc)
            nonzero = denom > 0
            if np.any(nonzero):
                rel = diff[nonzero] / denom[nonzero]
                max_rel = max(max_rel, float(np.max(rel)))
            within = within and bool(np.all(diff <= atol + rtol * denom))
    except (OSError, KeyError) as e:
        raise ValueError(
            f"Failed to read HDF5 output for comparison "
            f"({filepath_a!r} vs {filepath_b!r}): {e}"
        ) from e
    return max_abs, max_rel, within


def compare_outputs(
    config_a: SIBMFConfig,
    config_b: SIBMFConfig,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> pd.DataFrame:
    """Compare SIB block/tab outputs between two configs.

    Builds both SIB catalogs, pairs expected files by
    ``(leg_pair, mass, gamma)`` — the tab row pairs on its NaN leg/mass
    keys and gamma ``G1_G1`` — loads each pair's
    ``{gamma}_0_0_0/a2aMatrix`` compound dataset, and reports per-file
    max abs/rel diff plus whether the pair is within tolerance.

    Returns a DataFrame with columns: ``leg_pair, mass, gamma,
    filepath_a, filepath_b, max_abs_diff, max_rel_diff,
    within_tolerance, status`` where status is one of ``compared``,
    ``missing_file``.
    """
    logger = utils.get_logger()

    catalog_a = _sib_outfile_catalog(config_a)
    catalog_b = _sib_outfile_catalog(config_b)

    paired = catalog_a.merge(
        catalog_b, on=_PAIR_KEYS, how="outer", suffixes=("_a", "_b"), indicator=True
    )

    rows = []
    for _, r in paired.iterrows():
        filepath_a = _col(r, "filepath_a")
        filepath_b = _col(r, "filepath_b")
        exists_a = bool(_col(r, "exists_a", False))
        exists_b = bool(_col(r, "exists_b", False))

        base = {
            "leg_pair": r["leg_pair"],
            "mass": r["mass"],
            "gamma": r["gamma"],
            "filepath_a": filepath_a,
            "filepath_b": filepath_b,
        }

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
                f"Missing SIB output leg_pair={r['leg_pair']!r} "
                f"mass={r['mass']!r} gamma={r['gamma']!r} "
                f"(exists_a={exists_a}, exists_b={exists_b})."
            )
            continue

        max_abs, max_rel, within = _compare_h5_file(
            filepath_a, filepath_b, r["gamma"], rtol=rtol, atol=atol
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
        "leg_pair",
        "mass",
        "gamma",
        "filepath_a",
        "filepath_b",
        "max_abs_diff",
        "max_rel_diff",
        "within_tolerance",
        "status",
    ]
    return pd.DataFrame(rows, columns=columns)


# Sub-block registrations: leaves get the default route (absorbs the
# builder's _preprocessor slice into fields); strict handler lookup
# returns None for them (the CgConfig precedent,
# pyfm/tasks/hadrons/highmode_v2/config.py:341-357).
register_task("hadrons_sib_mf_batch", SIBBatchConfig, validate=validate_batch)
register_task("hadrons_sib_mf_solver", SIBSolverConfig, validate=validate_solver)
register_task("hadrons_sib_mf_output", SIBOutputConfig)

# Register SIBMFConfig with sib_mf-owned hooks throughout.
register_task(
    "hadrons_sib_mf",
    SIBMFConfig,
    create_outfile_catalog,
    build_input_params,
    build_aggregator_params,
    compare_outputs,
    route_params,
    validate=validate_config,
)
