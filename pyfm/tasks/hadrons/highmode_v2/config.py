"""Config tree for ``hadrons_lma_new`` high-mode entries (ADR 0001).

Sub-blocks of one ``tasks.high_modes`` entry, each owning one
responsibility (ADR Decision 5), composed under the entry composite
:class:`LMAHighModeConfig` (one per keyed or unkeyed entry):

* :class:`SourcesConfig` — wall-source enumeration (grid or biased),
* :class:`LowModesConfig` — low-mode method
  (``none`` / ``solve`` / ``meson_field``),
* :class:`CgConfig` — CG solver family, residual ladder, guess policy,
* :class:`OutputConfig` — correlator output files and cross-term mode.

Leaves are plain ``SimpleConfig``s registered with ``register_task`` so the
builder's default route hook absorbs their ``_preprocessor`` slice into
fields (the ``MesonLoaderConfig``/``contract`` precedent,
``pyfm/tasks/contract/mesonloader.py``); the composites route their own
nested blocks. Registrations are import side effects, so importing this
module is enough for a full builder pass over the tree.
"""

import random
import typing as t
from dataclasses import fields

from pydantic import Field
from pydantic.dataclasses import dataclass

from pyfm import utils
from pyfm.domain import (
    CompositeConfig,
    MassDict,
    OpList,
    Outfile,
    SerializableEnum,
    SimpleConfig,
)
from pyfm.tasks.hadrons.types import SolveCrossTerms, SourceRef
from pyfm.tasks.register import register_task


class CacheMode(SerializableEnum):
    """Per-entry LH-cache behavior (``low_modes.meson_field.cache``).

    ``BUILD_AND_LOAD`` writes the cache, then loads it and builds
    correlators; ``BUILD_ONLY`` writes it and stops (no quarks, solvers or
    correlators); ``LOAD`` reads an existing cache and builds correlators.
    """

    BUILD_AND_LOAD = 0
    BUILD_ONLY = 1
    LOAD = 2


class PreconMode(SerializableEnum):
    """CG guess-vector policy (``cg.precon``), independent of contraction pairs.

    ``CHAIN`` guesses the low-mode propagator for the loosest residual and
    the previous CG solve for each tighter one; ``EACH`` guesses the
    low-mode propagator for every solve; ``NONE`` provides no guess. The
    low-mode (``ranLL``) solve itself never takes a guess under any mode.
    """

    CHAIN = 0
    EACH = 1
    NONE = 2


class LowModeMethod(SerializableEnum):
    """How an entry produces its low-mode (``ranLL``) propagators.

    ``NONE`` skips them entirely (pure-CG entry); ``SOLVE`` builds the
    ``MSolver::StagLMA`` solver over the shared epack; ``MESON_FIELD``
    loads precomputed meson-field files and rebuilds the propagators
    eagerly with ``StagLMAMesonFieldProp`` producers.
    """

    NONE = 0
    SOLVE = 1
    MESON_FIELD = 2


@dataclass(frozen=True)
class GridSourceConfig(SimpleConfig):
    """Uniform ``tstart..tstop`` stride-``dt`` wall-source grid."""

    tstart: int
    tstop: int
    dt: int

    @property
    def tsources(self) -> t.List[int]:
        return list(range(self.tstart, self.tstop + 1, self.dt))


@dataclass(frozen=True)
class BiasedSourceConfig(SimpleConfig):
    """Seeded biased (TSM) wall-source draws.

    ``seed`` may embed formatting tokens (``seed_{series}_{cfg}``): the
    builder's partial-format pass resolves them per config (the same
    mechanism that resolves ``runid`` templates), so the draws are a pure
    function of the stored seed string — generation, completion checks and
    aggregation all re-derive the same list.
    """

    n: int
    seed: str
    replace: bool = True


@dataclass(frozen=True)
class SourcesConfig(CompositeConfig):
    """Entry-owned wall sources: ``time``/``noise`` plus exactly one mode.

    ``grid_config`` and ``biased_config`` are mutually exclusive; a source
    block absent from the routing table leaves both unset and
    ``validate_sources`` reports the misconfiguration.
    """

    time: int
    noise: int = 1
    grid_config: GridSourceConfig | None = None
    biased_config: BiasedSourceConfig | None = None

    @property
    def tsource_range(self) -> t.List[int]:
        """Source times: dt-spaced by default, else ``n`` seeded draws.

        In bias mode the draws default to **with replacement** (duplicate
        time slices are distinct sources); ``replace=False`` samples
        without replacement via ``rng.sample`` (requires ``n <= time``).
        Either way they are a pure function of the stored ``seed`` string,
        so generation, completion checks, and aggregation re-derive the
        same list.
        """
        if self.biased_config is not None:
            rng = random.Random(self.biased_config.seed)
            if self.biased_config.replace:
                return [rng.randrange(self.time) for _ in range(self.biased_config.n)]
            if self.biased_config.n > self.time:
                raise ValueError(
                    f"biased.n ({self.biased_config.n}) exceeds the time extent "
                    f"({self.time}); without-replacement sampling "
                    "(replace: false) requires n <= time."
                )
            return rng.sample(range(self.time), self.biased_config.n)
        if self.grid_config is None:
            raise ValueError(
                "SourcesConfig has neither grid_config nor biased_config; "
                "sources requires exactly one of `grid` and `biased`."
            )
        return self.grid_config.tsources

    @property
    def source_labels(self) -> t.List[str]:
        """Per-source module-name suffixes: ``t{tsource}`` (dt) or ``n{block}`` (bias)."""
        if self.biased_config is not None:
            return [f"n{i}" for i in range(self.biased_config.n)]
        return [f"t{tsrc}" for tsrc in self.tsource_range]

    @property
    def source_axis(self) -> t.List[str]:
        """``{tsource}`` replacement values: bare times (dt) or block labels (bias).

        Unique by construction in both modes, so catalogs, the resume gate,
        and aggregator replacement axes never double-count a source.
        """
        if self.biased_config is not None:
            return self.source_labels
        return [str(tsrc) for tsrc in self.tsource_range]

    @property
    def source_refs(self) -> t.List[SourceRef]:
        """Config-owned enumeration of all sources (see ``SourceRef``)."""
        return [
            SourceRef(label=label, axis=axis, t0=t0)
            for label, axis, t0 in zip(
                self.source_labels, self.source_axis, self.tsource_range
            )
        ]


def route_sources(params: t.Dict) -> t.Dict:
    """Route ``grid``/``biased`` onto their subconfig slices.

    Field keys (``time``, ``noise``) ride along as plain params — from the
    ``sources`` mapping itself or from the parent's shared params (the
    builder passes those down unchanged, so a shared ``time`` reaches this
    block even when the YAML only says ``sources: {grid: {...}}``).
    ``grid: true`` is accepted as an empty grid whose window fields then
    come entirely from shared params (``tstart``/``tstop``/``dt``); an
    explicit ``false`` counts as absent.
    """
    prep = params.pop("_preprocessor", {})
    child_prep: t.Dict[str, t.Dict] = {}
    grid = prep.pop("grid", None)
    if grid is not None and grid is not False:
        child_prep["grid_config"] = grid if isinstance(grid, dict) else {}
    biased = prep.pop("biased", None)
    if biased is not None and biased is not False:
        child_prep["biased_config"] = biased if isinstance(biased, dict) else {}
    return params | prep | {"_preprocessor": child_prep}


def validate_sources(config: SourcesConfig) -> None:
    """Validate SourcesConfig after construction: exactly one source mode."""
    if (config.grid_config is None) == (config.biased_config is None):
        raise ValueError(
            "sources requires exactly one of `grid` and `biased`; got "
            f"grid={config.grid_config is not None}, "
            f"biased={config.biased_config is not None}."
        )
    if config.noise < 1:
        raise ValueError(f"noise must be a positive integer; got {config.noise}.")
    if config.biased_config is not None:
        biased = config.biased_config
        if biased.n < 1:
            raise ValueError(f"biased.n must be a positive integer; got {biased.n}.")
        if not biased.seed:
            raise ValueError(
                "biased.seed is required (it seeds the deterministic "
                "time-slice draws)."
            )
        if not biased.replace and biased.n > config.time:
            raise ValueError(
                f"biased.n ({biased.n}) exceeds the time extent "
                f"({config.time}); without-replacement sampling "
                "(replace: false) requires n <= time."
            )


@dataclass(frozen=True)
class MesonFieldConfig(SimpleConfig):
    """``low_modes.meson_field`` block: cache file target and behavior.

    ``file`` is a files-label reference resolved by the builder against the
    job's ``files:`` block (the removed load-mode files-entry precedent); the
    filestem must carry ``{mass}`` (one cache per mass — checked in
    ``validate_low_modes``).
    """

    file: Outfile
    blocksize: int = 12
    cache: CacheMode = CacheMode.LOAD


@dataclass(frozen=True)
class LowModesConfig(CompositeConfig):
    """Entry low-modes block: exactly one of none / solve / meson_field.

    YAML arrives either as a scalar (``low_modes: solve``) — pre-wrapped by
    the entry's route hook into ``{"method": ...}`` — or as a mapping whose
    ``meson_field`` key carries the cache block.
    """

    method: LowModeMethod
    meson_field_config: MesonFieldConfig | None = None


def route_low_modes(params: t.Dict) -> t.Dict:
    """Route ``meson_field`` onto its subconfig slice; pin the method.

    A ``meson_field`` mapping in the slice becomes the
    :class:`MesonFieldConfig` slice and pins ``method=meson_field``
    (explicit ``method`` keys are overridden — the block's presence IS the
    method). Any other slice key rides along as a plain field (the scalar
    ``solve``/``none`` forms arrive pre-wrapped as ``method``).
    """
    prep = params.pop("_preprocessor", {})
    child_prep: t.Dict[str, t.Dict] = {}
    meson_field = prep.pop("meson_field", None)
    if meson_field is not None:
        child_prep["meson_field_config"] = meson_field
        prep["method"] = "meson_field"
    return params | prep | {"_preprocessor": child_prep}


def validate_low_modes(config: LowModesConfig) -> None:
    """Validate LowModesConfig after construction."""
    if config.method == LowModeMethod.MESON_FIELD:
        if config.meson_field_config is None:
            raise ValueError(
                "low_modes method 'meson_field' requires a meson_field block "
                "(file, blocksize, cache) in the sources of that method."
            )
        if "{mass}" not in config.meson_field_config.file.filestem:
            raise ValueError(
                "low_modes.meson_field.file filestem must carry the {mass} "
                "token (one cache per mass); got "
                f"{config.meson_field_config.file.filestem!r}."
            )
    elif config.meson_field_config is not None:
        raise ValueError(
            "low_modes block carries a meson_field entry but method="
            f"{config.method.name}; meson_field is only valid with the "
            "meson_field method."
        )


@dataclass(frozen=True)
class CgConfig(SimpleConfig):
    """Entry CG block: solver family, residual ladder, guess policy.

    ``precon`` replaces ``HighModeConfig.chain_cg_solves`` (ADR Decision 5):
    guess choice is now independent of which pairs are contracted.
    """

    solver: str = "mpcg"
    residual: t.List[float] = Field(default=[1e-8])
    precon: PreconMode = PreconMode.CHAIN


def validate_cg(config: CgConfig) -> None:
    """Validate CgConfig after construction."""
    if config.solver not in {"mpcg", "rb", "cg"}:
        raise ValueError(
            "cg.solver must be one of 'mpcg', 'rb', or 'cg'; got "
            f"{config.solver!r}."
        )
    if not config.residual:
        raise ValueError("cg.residual must list at least one residual.")


@dataclass(frozen=True)
class OutputConfig(SimpleConfig):
    """Entry output block: correlator files, overwrite policy, cross terms.

    ``file`` is a files-label reference resolved by the builder against the
    job's ``files:`` block (the old ``high_modes`` field, renamed).
    """

    file: Outfile
    overwrite: bool = False
    solve_cross_terms: SolveCrossTerms = SolveCrossTerms.DIAGONAL


# Sub-block registrations: leaves get the default route (absorbs the
# builder's _preprocessor slice into fields); composites carry their own.
# Keys mirror the MesonLoaderConfig precedent (a sub-config is not a
# runnable task — get_task_handler(strict=True) returns None for these).
register_task("hadrons_lma_grid_source", GridSourceConfig)
register_task("hadrons_lma_biased_source", BiasedSourceConfig)
register_task(
    "hadrons_lma_sources",
    SourcesConfig,
    route_params=route_sources,
    validate=validate_sources,
)
register_task("hadrons_lma_meson_field", MesonFieldConfig)
register_task(
    "hadrons_lma_low_modes",
    LowModesConfig,
    route_params=route_low_modes,
    validate=validate_low_modes,
)
register_task("hadrons_lma_cg", CgConfig, validate=validate_cg)
register_task("hadrons_lma_output", OutputConfig)


_ACTION_NAME = "stag_mass_{mass}"
_SOLVER_NAME = "stag_{solver}_mass_{mass}"
_LOW_MODES_NAME = "evecs_mass_{mass}"
_SHIFT_GAUGE_NAME = "gauge_apbc"


@dataclass(frozen=True)
class LMAHighModeConfig(CompositeConfig):
    """One keyed ``tasks.high_modes`` entry (ADR Decisions 3 and 5).

    The label prefixes every module the entry owns (``module_name``) and is
    available as a ``{label}`` formatting token; ``""`` (an unkeyed single
    entry) keeps module names byte-identical to the old list-based schema.
    The label arrives as a plain scalar param in the entry's DICT per-key
    params (injected by ``lma_new``'s routing), so it reaches this config,
    every sub-block, and the formatting map unchanged.

    Replaces ``HighModeConfig`` for ``hadrons_lma_new``: solver choice,
    source enumeration and output targets decompose into the Slice-3
    sub-blocks; the solve-cross machinery (``get_solver_labels`` /
    ``admits_solve_pair``) is ported verbatim with the flag mapping
    ``skip_low_modes`` → ``low_modes_config.method is NONE`` and
    ``skip_cg`` → ``cg_config is None``, and ``solve_cross_terms`` now
    lives on ``output_config``.
    """

    mass: MassDict
    operations: OpList
    action_name: str
    solver_name: str
    low_modes_name: str
    sources_config: SourcesConfig
    low_modes_config: LowModesConfig
    label: str = ""
    mass_cross_terms: bool = False
    shift_gauge_name: str | None = None
    cg_config: CgConfig | None = None
    output_config: OutputConfig | None = None
    split_mpi_layout: str | None = None
    subgrid_ranks: int | None = None

    def module_name(self, base: str) -> str:
        """Entry-owned module name: ``{label}_{base}`` (``''`` keeps ``base``)."""
        return f"{self.label}_{base}" if self.label else base

    @property
    def noise_name(self) -> str:
        """Full-volume noise module name (label-prefixed ``noise_fv``).

        The old configurable ``noise_name`` field is deliberately gone:
        the prefix IS the noise identity (Hadrons seeds from
        ``runId + module_name + traj``, ADR Decision 4).
        """
        return self.module_name("noise_fv")

    @property
    def use_meson_field(self) -> bool:
        return self.low_modes_config.method is LowModeMethod.MESON_FIELD

    @property
    def op_list(self) -> t.List[OpList.Op]:
        """Get list of gamma operations."""
        return self.operations.op_list

    @property
    def masses(self) -> t.List[str]:
        return self.operations.mass

    def get_mass_labels(self, op: OpList.Op, skip_cross: bool = False) -> t.List[str]:
        mass_labels = [self.mass.to_string(m, True) for m in op.mass]
        if not skip_cross and self.mass_cross_terms:
            # Canonical cross-mass order: raw-key ascending. This matches
            # contraction_gen's `mlabel1 < mlabel2` guard (which puts the
            # smaller raw key on the quark; TwoPointOp.mass_label joins
            # quark-first), so the catalog/resume/aggregation axis and the
            # emitted filenames agree for any op.mass listing order.
            cross_labels = [
                f"{self.mass.to_string(lo, True)}_m{self.mass.to_string(hi, True)}"
                for i, a in enumerate(op.mass)
                for j in range(i)
                for lo, hi in [sorted((op.mass[j], a))]
            ]
            mass_labels += cross_labels
        return mass_labels

    def get_solver_labels(self, skip_cross: bool = False) -> t.List[str]:
        solver_labels = []
        if self.low_modes_config.method is not LowModeMethod.NONE:
            solver_labels.append("ranLL")

        if self.cg_config is not None:
            residuals = self.cg_config.residual
            if len(residuals) == 1:
                solver_labels.append("ama")
            else:
                solver_labels += [f"ama_{r}" for r in residuals]

        if skip_cross:
            return solver_labels

        diagonals = [s for s in solver_labels if self.admits_solve_pair(s, s)]
        cross_labels = [
            f"{quark}_{antiquark}"
            for quark in solver_labels
            for antiquark in solver_labels
            if quark != antiquark and self.admits_solve_pair(quark, antiquark)
        ]
        return diagonals + cross_labels

    @property
    def effective_solve_cross_terms(self) -> SolveCrossTerms:
        """The configured solve-cross mode after degenerate-solver collapse.

        With either solver class absent (``low_modes`` ``none`` / no ``cg``
        block) the mode is ignored and every base pair is admitted — the
        effective mode is DIAGONAL. An absent ``output`` block (cache
        ``build_only`` entries) has no cross terms to configure either.
        """
        if (
            self.low_modes_config.method is LowModeMethod.NONE
            or self.cg_config is None
            or self.output_config is None
        ):
            return SolveCrossTerms.DIAGONAL
        return self.output_config.solve_cross_terms

    def admits_solve_pair(self, quark: str, antiquark: str) -> bool:
        """Whether a contraction pairing quark-side solver ``quark`` with
        antiquark-side solver ``antiquark`` belongs to the configured
        solve-cross mode (diagonal pairs included). Ported verbatim from
        ``HighModeConfig.admits_solve_pair`` — see its docstring for the
        DIAGONAL/ALL/TIERED admission table.
        """
        mode = self.effective_solve_cross_terms

        def is_low(label: str) -> bool:
            return label == "ranLL"

        if quark == antiquark:
            if mode == SolveCrossTerms.TIERED:
                return is_low(quark)
            return True
        low_quark = is_low(quark)
        low_antiquark = is_low(antiquark)
        if low_quark and not low_antiquark:
            return mode in (SolveCrossTerms.ALL, SolveCrossTerms.TIERED)
        if low_antiquark and not low_quark:
            return mode == SolveCrossTerms.ALL
        return False


def route_params(params: t.Dict) -> t.Dict:
    """Route the entry's ``_preprocessor`` slice onto fields and sub-blocks.

    Mirrors ``highmode/strategy.py``'s ``HighModeConfig`` route (field /
    operations split, split-grid both-or-neither strip) plus
    ``lmi.route_params``' name-default layering, and adds the sub-block
    routing D1 requires:

    * ``sources`` → ``_preprocessor["sources_config"]`` (mapping required),
    * ``low_modes`` → ``_preprocessor["low_modes_config"]`` — a scalar
      (``solve``/``none``) is pre-wrapped as ``{"method": ...}``; absent
      defaults to ``solve`` (the epack is always present under
      ``hadrons_lma_new``, so low modes are solvable by default — the old
      ``skip_low_modes`` derivation's equivalent),
    * ``cg`` → ``_preprocessor["cg_config"]`` (absent → ``None`` via the
      builder's optional-SIMPLE rule; ``cg: true`` → empty block, all
      defaults),
    * ``output`` → ``_preprocessor["output_config"]`` (mapping required).

    Non-field keys (``gamma``, ``mass`` for the OpList) land in
    ``operations`` exactly as before — the legacy ``cross_terms``
    translation is deliberately NOT ported (ADR Decision 1: old
    ``hadrons_lma_new`` YAML breaks; use ``mass_cross_terms`` and
    ``output.solve_cross_terms``).
    """
    prep = params.pop("_preprocessor", {})

    # Split-grid is opt-in and both-or-neither: a partial config (exactly
    # one of `split_mpi_layout`/`subgrid_ranks` set) is meaningless --
    # Hadrons needs the global <split> to define the subgrids that
    # <subgrid> tags reference. Strip both and warn so the job falls back
    # to non-split behavior (ported from highmode/strategy.py).
    split_mpi_layout = prep.get("split_mpi_layout")
    subgrid_ranks = prep.get("subgrid_ranks")
    if (split_mpi_layout is None) != (subgrid_ranks is None):
        utils.get_logger().warning(
            "Split-grid requires both `split_mpi_layout` and `subgrid_ranks`; "
            "only one was provided. Stripping both and falling back to "
            "non-split behavior."
        )
        prep.pop("split_mpi_layout", None)
        prep.pop("subgrid_ranks", None)

    # Name-template defaults (lmi.route_params' high_modes_defaults).
    prep = dict(
        action_name=_ACTION_NAME,
        low_modes_name=_LOW_MODES_NAME,
        solver_name=_SOLVER_NAME,
        shift_gauge_name=_SHIFT_GAUGE_NAME,
    ) | prep

    child_prep: t.Dict[str, t.Dict] = {}
    sources = prep.pop("sources", None)
    if sources is not None and sources is not False:
        if not isinstance(sources, dict):
            raise TypeError(
                f"sources must be a mapping; got {type(sources).__name__}."
            )
        child_prep["sources_config"] = sources
    low_modes = prep.pop("low_modes", None)
    if low_modes is None or low_modes is False:
        low_modes = {"method": "solve"}
    elif not isinstance(low_modes, dict):
        low_modes = {"method": low_modes}
    child_prep["low_modes_config"] = low_modes
    cg = prep.pop("cg", None)
    if cg is not None and cg is not False:
        child_prep["cg_config"] = cg if isinstance(cg, dict) else {}
    output = prep.pop("output", None)
    if output is not None and output is not False:
        if not isinstance(output, dict):
            raise TypeError(
                f"output must be a mapping; got {type(output).__name__}."
            )
        child_prep["output_config"] = output

    # Get field names from LMAHighModeConfig, excluding 'mass'
    # - 'mass' comes from top-level params (MassDict)
    # !NOTE: Don't squash params['mass']
    config_fields = {f.name for f in fields(LMAHighModeConfig) if f.name != "mass"}

    return (
        params
        | {
            "operations": {
                k: v for k, v in prep.items() if k not in config_fields
            },
        }
        | {k: v for k, v in prep.items() if k in config_fields}
        | {"_preprocessor": child_prep}
    )


def validate_config(config: LMAHighModeConfig) -> None:
    """Validate LMAHighModeConfig after construction (design doc rules).

    Entry:
    - ``low_modes`` ``none`` with no ``cg`` block → error (nothing to solve).
    - ``cg.precon != none`` with ``low_modes`` ``none`` → error (guesses
      reference the low-mode propagator).
    - ``cache: build_only`` with ``cg`` or ``output`` present → error.
    - Any non-``build_only`` state without ``output`` → error.
    - ``meson_field`` with ``sources.biased.replace: true`` → error (a
      repeated slice projects the same full-volume noise, giving an
      identical propagator).
    - Non-local operators require ``shift_gauge_name``; ``subgrid_ranks``
      must be positive.
    """
    method = config.low_modes_config.method
    has_cg = config.cg_config is not None
    has_output = config.output_config is not None
    meson_field = config.low_modes_config.meson_field_config

    if method is LowModeMethod.NONE and not has_cg:
        raise ValueError(
            "high_modes entry has low_modes 'none' and no cg block — there "
            "would be nothing to solve. Provide a cg block or use low_modes "
            "'solve'/'meson_field'."
        )
    if (
        method is LowModeMethod.NONE
        and has_cg
        and config.cg_config.precon is not PreconMode.NONE
    ):
        raise ValueError(
            "cg.precon != 'none' requires low modes (the guess chain "
            "references the low-mode propagator); got low_modes 'none' with "
            f"cg.precon {config.cg_config.precon.name.lower()}."
        )

    cache_build_only = (
        meson_field is not None and meson_field.cache is CacheMode.BUILD_ONLY
    )
    if cache_build_only and (has_cg or has_output):
        raise ValueError(
            "low_modes.meson_field.cache 'build_only' writes the cache and "
            "stops — it cannot be combined with cg or output blocks."
        )
    if not has_output and not cache_build_only:
        raise ValueError(
            "high_modes entry requires an output block (output.file) unless "
            "it is a cache 'build_only' entry."
        )

    if (
        meson_field is not None
        and config.sources_config.biased_config is not None
        and config.sources_config.biased_config.replace
    ):
        raise ValueError(
            "low_modes 'meson_field' cannot use biased sources with "
            "replace: true — a repeated slice projects the same full-volume "
            "noise, giving an identical propagator. Set "
            "sources.biased.replace: false."
        )

    has_nonlocal_ops = any(not op.gamma.local for op in config.operations.op_list)
    if has_nonlocal_ops and config.shift_gauge_name is None:
        raise ValueError(
            "Non-local operators detected, but shift_gauge_name is not set."
        )

    if config.subgrid_ranks is not None and config.subgrid_ranks <= 0:
        raise ValueError(
            f"subgrid_ranks must be a positive integer; got {config.subgrid_ranks}."
        )

    effective = config.effective_solve_cross_terms
    configured = (
        config.output_config.solve_cross_terms
        if config.output_config is not None
        else SolveCrossTerms.DIAGONAL
    )
    if effective != configured:
        triggers = ", ".join(
            trigger
            for trigger, enabled in (
                ("low_modes 'none'", method is LowModeMethod.NONE),
                ("no cg block", config.cg_config is None),
            )
            if enabled
        )
        utils.get_logger().warning(
            f"solve_cross_terms={configured.name} downgraded to "
            f"{effective.name}: {triggers} "
            f"{'are' if ', ' in triggers else 'is'} set — cross modes require "
            "both solver classes (low modes and CG); only same-solver "
            "(diagonal) pairs are admitted."
        )


register_task(
    "hadrons_lma_highmode",
    LMAHighModeConfig,
    route_params=route_params,
    validate=validate_config,
)
