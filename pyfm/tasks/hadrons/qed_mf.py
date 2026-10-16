"""Perturbative-QED HVP diagram-component producer task (``hadrons_qed_mf``).

Sibling composite of ``hadrons_sib_mf`` driving the HadronsMILC
stochastic-QED A2A module surface (``feature/opt-a2a-general-spin-taste``):
the SIB deflation chain shared verbatim (full-volume noise → tab →
batched precon → per-mass ``h`` solves) plus one ``MGauge::StochEmFunc``
stochastic photon source feeding three block families —

* vec: ``MContraction::StagA2AMesonField`` with the conjugate-expanded
  ±p momentum list on the external legs (momentum lives here only; the
  insertion side is pinned to zero), one HDF5 per gamma×momentum;
* ja: ``MContraction::StagA2AASlashMesonField`` — summed A_slash J·A
  local insertion, gammas from the four-entry J.A SpinTaste module;
* j1l: ``MContraction::StagA2AASlashOneLinkMesonField`` — per-component
  J_kA_k insertion through photon-dressed links (W~_k = A_k·U_k, base
  gauge).

EM families are emitted as per-sample worlds (split_em): one module per
sample with ``nEmFields=1`` and per-sample seed strings
``{seed}_n{i}`` — SHARED across ja/j1l so both insertion types of world
i consume identical photons (upstream cross-module reproducibility).
The task produces terminal artifacts only — momentum-projected files
cannot be reloaded upstream — so aggregation has no meaning here and
the downstream QED-HVP contraction is a future task (the sib_mf
producer/consumer precedent).

Block table: ``ll``/``nl`` reference pairs at ``defl_mass`` and
``lh``/``nh`` per operations mass, for every family; EM blocks
additionally carry the ``em_index`` axis. Unlike sib_mf, ``nl`` is
emitted for every family — sib's scalar-``nl`` derivation is the
zero-momentum tab identity, and momentum projection destroys it. The
derivable ``lp``/``np`` pairs are NOT emitted (exact offline arithmetic
from the stored ``ll``/``tab``/evals). Emission is demand-driven: the
resume gate catalogs the block/tab outputs and narrows the chain to the
missing ones. ``h`` solves are shared across EM worlds — photon dressing
happens inside the meson-field kernels, so no block needs a
photon-dressed solve.
"""

import copy
import typing as t

import h5py
import numpy as np
import pandas as pd
from pydantic.dataclasses import dataclass

from pyfm import utils
from pyfm.domain import CompositeConfig, Gamma, MassDict, OpList, Outfile, SimpleConfig
from pyfm.domain.blockkeys import (
    EM_FAMILY_JA,
    EM_FAMILY_J1L,
    ZERO_MOM as _ZERO_MOM,
    aslash_file_key as _aslash_file_key,
    em_components as _em_components,
    em_file_key,
    em_index_suffix as _em_index_suffix,
    expand_momenta as _expand_momenta,
    mass_token as _mass_token,
    mom_token as _mom_token,
    vec_file_key as _vec_file_key,
    VEC_FAMILY_TOKEN as _VEC_FAMILY_TOKEN,
)
from pyfm.tasks.hadrons.types import HadronsInput
from pyfm.tasks.register import register_task

from . import gauge, epack
from .lma_new import _merge_modules
import pyfm.tasks.hadrons.modules as hadmods

# Name-template defaults layered by route_params (lma_new conventions).
_ACTION_NAME = "stag_mass_{mass}"
_LOW_MODES_NAME = "evecs_mass_{mass}"
_SHIFT_GAUGE_NAME = "gauge_apbc"

# The vec-family Γ set (mirrors _SIB_FAMILIES, sib_mf.py:91): the
# momentum-carrying external-leg blocks. Spin-taste structure lives in
# the meson-field kernels, not the solves.
_QED_FAMILIES = frozenset({Gamma.SCALAR_LOCAL, Gamma.VEC_LOCAL, Gamma.VEC_ONELINK})

# Leg-pair block kinds (pyfm-side letters, the sib convention): l = low
# modes (eig), n = full-volume noise, h = unprojected high solve.
# Reference pairs at defl_mass; h pairs per operations mass. Unlike
# sib_mf, nl is emitted for EVERY family: sib's scalar-nl derivation is
# the zero-momentum tab identity and momentum projection destroys it.
# lp/np stay dropped (exact offline arithmetic from ll/tab/evals).
_REFERENCE_LEG_PAIRS = ("ll", "nl")
_H_LEG_PAIRS = ("lh", "nh")
# Pairs with no eigenvector leg on either side: empty lowModes, no CB
# pairs.
_PURE_HIGH_LEG_PAIRS = frozenset({"nh"})


def _em_file_key(
    config: QEDMFConfig, family: str, world: int, component: int | None
) -> str:
    """World+channel-composed file_key of one EM block file."""
    return em_file_key(family, config.em_seeds(family)[world], component)


@dataclass(frozen=True)
class QEDBatchConfig(SimpleConfig):
    """``qed.batch`` block: the A2A batch source knob surface.

    Identical to the sib batch contract: structural full-volume coverage
    (``t0=0``, ``tStep=1``, one wall per timeslice) means only ``noise``
    is configurable; ``time`` absorbs from shared/job params when
    omitted (the builder's shared-param layering) and pins the batched
    guess's ``tA=0``/``tB=time-1``/``tStep=1``.
    """

    noise: int
    time: int


def validate_batch(config: QEDBatchConfig) -> None:
    """Validate QEDBatchConfig after construction."""
    if config.noise < 1:
        raise ValueError(f"qed.batch.noise must be >= 1; got {config.noise}.")


@dataclass(frozen=True)
class QEDSolverConfig(SimpleConfig):
    """``qed.solver`` block: the per-mass ``h`` solve (single residual —
    the block topology has exactly one ``h`` leg per mass, the sib
    argument)."""

    solver: str = "mpcg"
    residual: float = 1e-8


def validate_solver(config: QEDSolverConfig) -> None:
    """Validate QEDSolverConfig after construction."""
    if config.solver not in {"mpcg", "rb", "cg"}:
        raise ValueError(
            "qed.solver.solver must be one of 'mpcg', 'rb', or 'cg'; got "
            f"{config.solver!r}."
        )


@dataclass(frozen=True)
class QEDEmConfig(SimpleConfig):
    """``qed.em`` block: the stochastic-photon knob surface.

    ``gauge``/``zm_scheme``/``improvement`` pass through to one
    ``MGauge::StochEmFunc`` (``gauge`` is the ``PhotonR::Gauge`` scheme
    string, NOT a module name; upstream parses the enums and improvement
    via ``strToVec<Real>``). ``n_em`` is the default per-family
    photon-sample count (per-family ``n_em`` overrides ride the
    aslash/onelink blocks); ``seed`` is the base ``EmSeedString``,
    composed per sample to ``{seed}_n{i}`` and SHARED across both EM
    families so the local J·A and one-link J_kA_k insertions of world i
    consume identical photons (upstream cross-module reproducibility).
    The default carries ``{series}``/``{cfg}`` so the photon realization
    changes per trajectory.
    """

    gauge: str = "feynman"
    zm_scheme: str = "qedL"
    improvement: str = ""
    n_em: int = 1
    seed: str = "qed_{series}_{cfg}"


def validate_em(config: QEDEmConfig) -> None:
    """Validate QEDEmConfig after construction."""
    if config.n_em < 1:
        raise ValueError(f"qed.em.n_em must be >= 1; got {config.n_em}.")
    if not config.seed:
        raise ValueError("qed.em.seed must be a non-empty string.")


@dataclass(frozen=True)
class QEDFamilyConfig(SimpleConfig):
    """``qed.aslash`` / ``qed.onelink`` block: EM-family enable + override.

    ``n_em`` overrides ``qed.em.n_em`` for this family; ``None``
    inherits. The block's PRESENCE in the routing table enables the
    family (bool-or-block normalization in route_params).
    """

    n_em: int | None = None


def validate_family(config: QEDFamilyConfig) -> None:
    """Validate QEDFamilyConfig after construction."""
    if config.n_em is not None and config.n_em < 1:
        raise ValueError(f"EM family n_em must be >= 1; got {config.n_em}.")


@dataclass(frozen=True)
class QEDOutputConfig(SimpleConfig):
    """``qed.output`` block: block and tab targets.

    ``file`` is the block files label — its filestem carries
    ``{family}``/``{leg_pair}``/``{mass}``/``{em_index}`` tokens
    (``{em_index}`` renders ``'_em<i>'`` for EM worlds, ``''`` for vec)
    and must NOT contain the substring "meson" (the
    ``Outfile.from_param`` ordered-substring routing sends it to the
    ``cfg_qed_h5`` token ext ``.{cfg}/{file_key}.h5``). ``tab`` is the
    ⟨ℓ|η⟩ overlap table label and MUST contain "meson" (routes to
    ``cfg_gamma_h5``, the zero-momentum ``{gamma}_0_0_0.h5`` grammar).
    """

    file: Outfile
    tab: Outfile
    overwrite: bool = False


@dataclass(frozen=True)
class QEDMFConfig(CompositeConfig):
    """Own config tree for ``hadrons_qed_mf`` (sibling of ``hadrons_sib_mf``).

    A genuinely new class, not a subclass or reuse of a sibling task's
    config class (``register.py``'s ``_config_to_task_key`` reverse-lookup
    keys by class object; reuse would steal the sibling's registration).
    ``operations`` selects the vec Γ families and the h-solve masses
    shared by ALL families; ``aslash_config``/``onelink_config`` enable
    the EM insertion families (``None`` = off — route omits the key and
    the builder's optional-subconfig path leaves the field ``None``);
    ``em_config`` is required iff any EM family is enabled;
    ``momentum`` is the vec-side list (±p expanded); ``defl_mass`` names
    the single ``ModifyEigenPackMILC`` shift every low-side artifact
    builds from.
    """

    gauge_config: gauge.GaugeConfig
    epack_config: epack.EpackConfig
    batch_config: QEDBatchConfig
    solver_config: QEDSolverConfig
    output_config: QEDOutputConfig
    mass: MassDict
    operations: OpList
    em_config: QEDEmConfig | None = None
    aslash_config: QEDFamilyConfig | None = None
    onelink_config: QEDFamilyConfig | None = None
    # Inner component arity is intentionally open ((int, ...)): pydantic
    # would reject a wrong-arity entry BEFORE validate_config could give
    # its loud "exactly 3 components" message, so the arity check lives
    # in validate_config (the plan's test contract).
    momentum: t.Tuple[t.Tuple[int, ...], ...] = ()
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

    def em_families(self) -> t.List[str]:
        """Enabled EM module families in emission order."""
        fams = []
        if self.aslash_config is not None:
            fams.append(EM_FAMILY_JA)
        if self.onelink_config is not None:
            fams.append(EM_FAMILY_J1L)
        return fams

    def n_em(self, family: str) -> int:
        """Per-family photon-sample count (family override or em.n_em)."""
        override = {
            EM_FAMILY_JA: (
                self.aslash_config.n_em if self.aslash_config else None
            ),
            EM_FAMILY_J1L: (
                self.onelink_config.n_em if self.onelink_config else None
            ),
        }[family]
        return override if override is not None else self.em_config.n_em

    def em_seeds(self, family: str) -> t.List[str]:
        """Per-sample ``EmSeedString``s ``{seed}_n{i}`` — shared across
        families (identical photons for ja/j1l of the same world);
        ``{series}``/``{cfg}`` in the base seed resolve here."""
        return [
            self.format_string(f"{self.em_config.seed}_n{i}")
            for i in range(self.n_em(family))
        ]

    @property
    def mom_elems(self) -> t.List[str]:
        """±p-expanded vec-side momentum elems (D3); empty config means
        the single zero momentum."""
        return _expand_momenta(self.momentum) or ["0 0 0"]


def route_params(params: t.Dict) -> t.Dict:
    """Route the ``qed:`` section onto sub-blocks and plain fields.

    The ``qed`` mapping lifts: ``em``/``batch``/``solver``/``output`` onto
    their subconfig slices, ``aslash``/``onelink`` (bool-or-block;
    ``true``/``{}`` enable, ``false``/absent leave the field ``None`` —
    the key is OMITTED from the routing table so the builder's
    optional-subconfig path applies) onto the family subconfigs,
    ``operations`` onto the OpList field, and the remaining keys
    (``momentum``, ``defl_mass``, ``blocksize``, ``shift_gauge_name``)
    onto plain composite fields. ``tasks.gauge``/``tasks.epack`` ride onto
    their child slices with the shared name-template defaults. Every
    other ``tasks:`` key and every unknown ``qed`` key is rejected
    loudly.
    """
    # Deep-copy: this hook pops keys out of the nested qed block (and the
    # gauge/epack slices below), and create_task is called once per cfgno
    # on the same YAML tree (lma_new.normalize_params' rationale).
    preprocessor_params = copy.deepcopy(params.pop("_preprocessor", {}))

    qed = preprocessor_params.pop("qed", {})
    if not isinstance(qed, dict):
        raise TypeError(
            f"tasks.qed must be a mapping; got {type(qed).__name__}."
        )

    def _block(value) -> t.Dict:
        """A sub-block mapping; ``true`` counts as an empty block (defaults)."""
        return value if isinstance(value, dict) else {}

    def _opt_block(value):
        """Optional sub-block: absent/``false`` -> ``None`` (disabled);
        ``true``/``{}``/mapping -> the block mapping."""
        if value is None or value is False:
            return None
        return _block(value)

    child_preprocessor: t.Dict[str, t.Dict] = dict(
        gauge_config=dict(action_name=_ACTION_NAME),
        epack_config=dict(
            action_name=_ACTION_NAME,
            low_modes_name=_LOW_MODES_NAME,
        ),
        batch_config=_block(qed.pop("batch", {})),
        solver_config=_block(qed.pop("solver", {})),
        output_config=(output := _block(qed.pop("output", {}))),
    )
    _check_output_label(output)
    if (em := _opt_block(qed.pop("em", None))) is not None:
        child_preprocessor["em_config"] = em
    for label, raw in (
        ("aslash_config", qed.pop("aslash", None)),
        ("onelink_config", qed.pop("onelink", None)),
    ):
        if (block := _opt_block(raw)) is not None:
            child_preprocessor[label] = block

    operations = qed.pop("operations", {})
    if not isinstance(operations, dict):
        raise TypeError(
            f"qed.operations must be a mapping; got {type(operations).__name__}."
        )

    unknown_tasks = sorted(
        k for k in preprocessor_params if k not in ("gauge", "epack")
    )
    if unknown_tasks:
        raise ValueError(
            f"Unknown tasks keys for hadrons_qed_mf: {unknown_tasks}; "
            "allowed task blocks are gauge, epack, qed."
        )
    for k, v in preprocessor_params.items():
        child_preprocessor[f"{k}_config"] |= v

    qed_fields = {"momentum", "defl_mass", "blocksize", "shift_gauge_name"}
    unknown_qed = sorted(k for k in qed if k not in qed_fields)
    if unknown_qed:
        raise ValueError(
            f"Unknown qed section keys: {unknown_qed}. Allowed keys are "
            "operations, em, aslash, onelink, momentum, batch, solver, "
            "output, defl_mass, blocksize, shift_gauge_name."
        )

    params = dict(shift_gauge_name=_SHIFT_GAUGE_NAME) | params | qed
    return params | {"operations": operations, "_preprocessor": child_preprocessor}


def _label_ext(label: str) -> str | None:
    """The ``Outfile.from_param`` routing outcome for a file label.

    Mirrors the ordered-substring ``name_keys`` walk in
    ``pyfm/domain/outfiles.py`` — needed at ROUTE time because a label
    absent from ``files:`` otherwise dies in pydantic before
    ``validate_config`` could reject the misrouting loudly.
    """
    name_keys = {
        "links": "cfg",
        "meson": "cfg_gamma_h5",
        "qed": "cfg_qed_h5",
        "eig": "cfg_bin",
        "a2a_vec": "cfg_bin",
        "contract_legacy": "cfg_pickle",
        "contract": "cfg_h5",
        "modes": "cfg_h5",
        "eval": "cfg_h5",
        "edir": "cfg_bin_multi",
        "scidac": "raw",
    }
    for k, ext in name_keys.items():
        if k in label:
            return ext
    return None


def _check_output_label(output: t.Dict) -> None:
    """Loud routing grammar (D6) on the RAW output labels, at route time.

    ``file`` must route to ``cfg_qed_h5`` (contains 'qed' but not
    'meson'), ``tab`` to ``cfg_gamma_h5`` (contains 'meson'). Labels that
    resolve to neither ext (or to no known ext at all) are rejected here
    with the same message ``validate_config`` gives built Outfiles.
    """
    for side, want, fixup in (
        (
            "file",
            "cfg_qed_h5",
            "pick a label containing 'qed' but NOT 'meson' (e.g. 'qed_blocks').",
        ),
        (
            "tab",
            "cfg_gamma_h5",
            "the tab label must contain 'meson' (e.g. 'qed_meson_tab').",
        ),
    ):
        label = output.get(side)
        if isinstance(label, str) and _label_ext(label) != want:
            raise ValueError(
                f"qed.output.{side} label {label!r} does not route to the "
                f"{want} ext — {fixup}"
            )


def validate_config(config: QEDMFConfig) -> None:
    """Validate QEDMFConfig after construction (cross-section checks).

    Per-block state was validated when each leaf was built. Here:
    ``operations`` non-empty with per-op masses; ``defl_mass`` and every
    operations mass exist in the MassDict; op gammas must be QED
    families; EM-family enablement and the em block must agree (both or
    neither); momentum entries have exactly 3 components; output labels
    route to the intended exts.
    """
    if not config.op_list:
        raise ValueError(
            "qed.operations has no operations — provide an operations: block "
            "with at least one vec-family gamma (scalar_local / vec_local / "
            "vec_onelink) and one mass (e.g. operations: {gamma: [vec_local], "
            "mass: [l]})."
        )
    for op in config.op_list:
        if not op.mass:
            raise ValueError(
                f"qed.operations gamma {op.gamma.name.lower()!r} has no mass — "
                "every operation needs at least one mass (the h solves and "
                "lh/nh blocks run per operations mass; e.g. operations: "
                "{gamma: [vec_local], mass: [l]})."
            )
    if config.defl_mass not in config.mass:
        raise ValueError(
            f"qed.defl_mass {config.defl_mass!r} is not present in the "
            f"mass parameters ({sorted(config.mass.keys())}); it names the "
            "single ModifyEigenPackMILC shift every low-side artifact "
            "builds from."
        )
    for m in config.masses:
        if m not in config.mass:
            raise ValueError(
                f"qed.operations mass {m!r} is not present in the mass "
                f"parameters ({sorted(config.mass.keys())})."
            )
    bad = [
        op.gamma.name.lower() for op in config.op_list
        if op.gamma not in _QED_FAMILIES
    ]
    if bad:
        raise ValueError(
            f"qed.operations gammas {bad} are not QED families; allowed: "
            "scalar_local, vec_local, vec_onelink."
        )
    has_em_families = bool(config.em_families())
    if has_em_families and config.em_config is None:
        raise ValueError(
            "EM families enabled (aslash/onelink) require the qed.em block "
            "(gauge, zm_scheme, improvement, n_em, seed)."
        )
    if not has_em_families and config.em_config is not None:
        raise ValueError(
            "qed.em block provided but no EM family is enabled — enable "
            "aslash: and/or onelink: or drop the em block."
        )
    for p in config.momentum:
        if len(p) != 3:
            raise ValueError(
                "qed.momentum entries must have exactly 3 components; got "
                f"{list(p)}."
            )
    # Output-label routing grammar (D6): checked on the routing OUTCOME.
    # By validate time the global token ({cfg}) is already resolved, so
    # the surviving TASK tokens identify the ext: file must carry
    # {file_key} (the cfg_qed_h5 token ext), tab must carry {gamma}
    # (cfg_gamma_h5's zero-momentum grammar). Outfile.from_param routes
    # by ordered substring, so a mislabeled entry silently misroutes;
    # make it loud (the contract/sib.py token-presence precedent).
    if "{file_key}" not in config.output_config.file.ext:
        raise ValueError(
            "qed.output.file label does not route to the cfg_qed_h5 token "
            "ext — pick a label containing 'qed' but NOT 'meson' (e.g. "
            "'qed_blocks')."
        )
    if "{gamma}" not in config.output_config.tab.ext:
        raise ValueError(
            "qed.output.tab label does not route to the cfg_gamma_h5 ext — "
            "the tab label must contain 'meson' (e.g. 'qed_meson_tab')."
        )


def _block_filepath(
    config: QEDMFConfig,
    family: str,
    leg_pair: str,
    mass: str | None,
    file_key: str,
    em_index: int | None = None,
) -> str:
    """Formatted output path of one block file (global tokens — {cfg},
    {series}, {eigs}, {noise} — are resolved at build time; this fills
    the task tokens)."""
    return config.output_config.file.filename.format(
        family=family,
        leg_pair=leg_pair,
        mass=_mass_token(mass),
        em_index=_em_index_suffix(em_index),
        file_key=file_key,
    )


def _tab_filepath(config: QEDMFConfig) -> str:
    """Formatted output path of the scalar ⟨ℓ|η⟩ overlap table."""
    return config.output_config.tab.filename.format(
        gamma=Gamma.SCALAR_LOCAL.gamma_list[0]
    )


def _block_axes(
    family: str, leg_pair: str, mass_token: str, em_index: str, file_key: str
) -> t.Dict[str, t.List[str]]:
    """One catalog yield's replacement axes (single-row product)."""
    return {
        "family": [family],
        "leg_pair": [leg_pair],
        "mass": [mass_token],
        "em_index": [em_index],
        "file_key": [file_key],
    }


def _leg_pair_axes(config: QEDMFConfig):
    """(leg_pair, mass_token) pairs: reference pairs mass-free, h pairs
    per operations mass (the sib block-table shape, D8)."""
    return [(lp, "") for lp in _REFERENCE_LEG_PAIRS] + [
        (lp, _mass_token(m)) for m in config.masses for lp in _H_LEG_PAIRS
    ]


def _qed_outfile_catalog(config: QEDMFConfig) -> pd.DataFrame:
    """Catalog the QED outputs at (family, leg_pair, mass, em_index,
    file_key) granularity — the axes the resume gate narrows on and
    compare_outputs pairs on.

    Vec rows: one per (Γ, momentum) per leg pair — the ±p expansion rides
    the file_key (``{GammaName}_{p}``); gamma/momentum/component/seed are
    recoverable from file_key (only filename tokens survive the catalog
    machinery's replacement filter, so no extra descriptive columns).
    EM rows: one per (world, channel) with the per-sample seed baked into
    file_key. The tab row is scalar-only (G1_G1, zero momentum) on the
    meson-routed ext. Module granularity: one vec module writes every
    gamma×momentum file of its (family, leg_pair, mass) group; one EM
    module every channel of its (family, leg_pair, mass, world).
    """
    block_outfile = config.output_config.file
    tab_outfile = config.output_config.tab

    def generate_outfile_formatting():
        for op in config.op_list:
            fam = _VEC_FAMILY_TOKEN[op.gamma]
            for leg_pair, mass in _leg_pair_axes(config):
                for gamma in op.gamma.gamma_list:
                    for mom in config.mom_elems:
                        yield (
                            _block_axes(
                                fam, leg_pair, mass, "",
                                _vec_file_key(gamma, mom),
                            ),
                            block_outfile,
                        )
        for family in config.em_families():
            for leg_pair, mass in _leg_pair_axes(config):
                for world in range(config.n_em(family)):
                    for component in _em_components(family):
                        yield (
                            _block_axes(
                                family,
                                leg_pair,
                                mass,
                                _em_index_suffix(world),
                                _em_file_key(config, family, world, component),
                            ),
                            block_outfile,
                        )
        yield ({"gamma": [Gamma.SCALAR_LOCAL.gamma_list[0]]}, tab_outfile)

    return utils.io.catalog_files(generate_outfile_formatting())


def _needed_blocks(
    config: QEDMFConfig, bad_files: t.Set[str]
) -> t.Tuple[
    t.Dict[Gamma, t.Set[str]],
    t.Set[t.Tuple[Gamma, str, str]],
    t.Set[t.Tuple[str, str, t.Optional[str], int]],
]:
    """Demand sets from the resume gate's bad-file list.

    Module granularity: a bad file anywhere in a vec group's gamma×momentum
    files re-runs that whole (family, leg_pair, mass) module; a bad EM
    file re-runs its (family, leg_pair, mass, world) module. Returns
    ``(needed_vec_ref, needed_vec_h, needed_em)``: reference leg pairs
    per vec Γ family, h-legged tuples per vec family, and EM world tuples
    (mass ``None`` = the reference pairs).
    """
    needed_vec_ref: t.Dict[Gamma, t.Set[str]] = {}
    needed_vec_h: t.Set[t.Tuple[Gamma, str, str]] = set()
    needed_em: t.Set[t.Tuple[str, str, t.Optional[str], int]] = set()

    for op in config.op_list:
        fam = _VEC_FAMILY_TOKEN[op.gamma]
        for leg_pair in _REFERENCE_LEG_PAIRS:
            if any(
                _block_filepath(
                    config, fam, leg_pair, None, _vec_file_key(g, m)
                )
                in bad_files
                for g in op.gamma.gamma_list
                for m in config.mom_elems
            ):
                needed_vec_ref.setdefault(op.gamma, set()).add(leg_pair)
        for mass_label in config.masses:
            for leg_pair in _H_LEG_PAIRS:
                if any(
                    _block_filepath(
                        config, fam, leg_pair, mass_label,
                        _vec_file_key(g, m),
                    )
                    in bad_files
                    for g in op.gamma.gamma_list
                    for m in config.mom_elems
                ):
                    needed_vec_h.add((op.gamma, leg_pair, mass_label))

    for family in config.em_families():
        for leg_pair, mass_label in [
            *[(lp, None) for lp in _REFERENCE_LEG_PAIRS],
            *[(lp, m) for m in config.masses for lp in _H_LEG_PAIRS],
        ]:
            for world in range(config.n_em(family)):
                if any(
                    _block_filepath(
                        config,
                        family,
                        leg_pair,
                        mass_label,
                        _em_file_key(config, family, world, comp),
                        em_index=world,
                    )
                    in bad_files
                    for comp in _em_components(family)
                ):
                    needed_em.add((family, leg_pair, mass_label, world))
    return needed_vec_ref, needed_vec_h, needed_em


def build_input_params(config: QEDMFConfig) -> HadronsInput:
    """Generate input parameters for the perturbative-QED HVP task.

    Emission order IS the execution contract under the naive scheduler,
    so sections append in dependency order: base gauge → epack actions →
    epack → defl-mass shift → actions (sp gauge when mpcg) → [resume
    gate] → CB pairs → em → h-solve solvers → SpinTaste → ll blocks →
    noise → tab writer/loader → precon → h solves → remaining blocks.
    The schedule is deduplicated first-occurrence (lma_new convention).
    The chain and blocks are demand-driven: the gate catalogs the
    block/tab outputs and emits only the missing ones plus the modules
    they transitively reference. ``output.overwrite`` skips the gate.
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

    # 3. Single mass shift at defl_mass: every low-side artifact (tab,
    #    the ll/nl blocks, the CB pairs) builds from this one shifted
    #    pack (the sib section-3 convention).
    mass_shifts_input = epack.build_epack_mass_shifts(
        config.epack_config, [config.defl_mass]
    )
    _merge_modules(modules, mass_shifts_input.modules)
    schedule += mass_shifts_input.schedule

    # 4. dp masses = operations masses ∪ {defl_mass} (the CB pairs bind
    #    stag_mass_{defl_mass}); sp masses + the sp gauge only when the
    #    h solver is mixed-precision.
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

    # 5. Resume gate: demand sets for the QED chain. With overwrite, or
    #    when every output file is present and good-sized, the chain
    #    narrows to what the missing blocks transitively reference. EM
    #    worlds share the chain (photon dressing is kernel-side).
    if config.output_config.overwrite:
        needed_vec_ref = {
            op.gamma: set(_REFERENCE_LEG_PAIRS) for op in config.op_list
        }
        needed_vec_h = {
            (op.gamma, leg_pair, m)
            for op in config.op_list
            for m in config.masses
            for leg_pair in _H_LEG_PAIRS
        }
        needed_em = {
            (family, leg_pair, mass_label, world)
            for family in config.em_families()
            for leg_pair, mass_label in [
                *[(lp, None) for lp in _REFERENCE_LEG_PAIRS],
                *[(lp, m) for m in config.masses for lp in _H_LEG_PAIRS],
            ]
            for world in range(config.n_em(family))
        }
        tab_needed = True
    else:
        bad_files = set(utils.io.get_bad_files(_qed_outfile_catalog(config)))
        needed_vec_ref, needed_vec_h, needed_em = _needed_blocks(config, bad_files)
        tab_needed = _tab_filepath(config) in bad_files

    # Chain demand from the block demand sets (the sib derivation,
    # extended by the EM axis): h solves run per mass demanded by ANY
    # family; noise feeds the tab/precon/h chain and every nl leg; CB
    # pairs feed every low-sided consumer (tab, and any demanded block
    # that is not pure-high); the em module runs iff any EM block will.
    vec_ref_pairs = {lp for pairs in needed_vec_ref.values() for lp in pairs}
    nl_demanded = "nl" in vec_ref_pairs or any(
        lp == "nl" for _, lp, _, _ in needed_em
    )
    need_h_chain = bool(needed_vec_h) or any(
        lp in _H_LEG_PAIRS for _, lp, _, _ in needed_em
    )
    vec_low_side = bool(vec_ref_pairs) or any(
        lp != "nh" for _, lp, _ in needed_vec_h
    )
    em_low_side = any(lp != "nh" for _, lp, _, _ in needed_em)
    need_cbpairs = tab_needed or vec_low_side or em_low_side
    need_noise_fv = tab_needed or nl_demanded or need_h_chain
    need_noise_batch = need_h_chain
    need_precon = need_h_chain
    need_em_module = bool(needed_em)
    h_masses = [
        m for m in config.masses
        if any(mm == m for _, _, mm in needed_vec_h)
        or any(mm == m for _, _, mm, _ in needed_em if mm is not None)
    ]

    action_defl = config.gauge_config.action_name.format(mass=config.defl_mass)
    low_modes_defl = config.epack_config.low_modes_name.format(mass=config.defl_mass)
    cbpairs_l = f"cbpairs_l_mass_{config.defl_mass}"
    cbpairs_r = f"cbpairs_r_mass_{config.defl_mass}"
    precon = "precon_batch"
    noise_batch = "noise_batch"
    noise_fv_vec = "noise_fv_vec"

    # 6. CB pairs at defl_mass: two DISTINCT MUtilities::EigenPackCBPairs
    #    instances (the C++ setup requires cbPairsLeft != cbPairsRight),
    #    shared by the tab writer and every low-sided block of every
    #    family. Emitted only when a low-side consumer will run.
    if need_cbpairs:
        for name in (cbpairs_l, cbpairs_r):
            modules[name] = hadmods.eigen_pack_cb_pairs(
                name=name, eigen_pack=low_modes_defl, action=action_defl
            )
            schedule.append(name)

    # 7. em module (demand-driven): ONE StochEmFunc serves every EM world
    #    — the stochastic draws live in the consumers (seeded there by
    #    EmSeedString) and the _weight companion is cached once, so a
    #    single module costs nothing extra. Emission order IS the
    #    execution contract: em strictly precedes every EM block module.
    if need_em_module:
        em = config.em_config
        modules["em"] = hadmods.stoch_em_func(
            name="em",
            gauge=em.gauge,
            zm_scheme=em.zm_scheme,
            improvement=em.improvement,
        )
        schedule.append("em")

    # 8. h-solve solver modules, one per mass whose h solve will run
    #    (names carry no solver-label substrings — nothing sorts this
    #    schedule).
    for mass_label in h_masses:
        action = config.gauge_config.action_name.format(mass=mass_label)
        name = f"qed_solver_mass_{mass_label}"
        resid = str(config.solver_config.residual)
        match config.solver_config.solver:
            case "rb":
                modules[name] = hadmods.rb_cg(name=name, action=action, residual=resid)
            case "cg":
                modules[name] = hadmods.cg(name=name, action=action, residual=resid)
            case "mpcg":
                modules[name] = hadmods.mixed_precision_cg(
                    name=name,
                    outer_action=action,
                    inner_action=f"i{action}",
                    residual=resid,
                )
            case _:
                raise ValueError(
                    f"Unknown QED solver: {config.solver_config.solver}"
                )
        schedule.append(name)

    # 9. SpinTaste modules — per vec family plus the scalar/JA roles.
    #    Each vec block module writes one file per gamma of ITS bound
    #    SpinTaste module under a family-scoped stem (the catalog
    #    grammar), so families get their OWN modules — a union module
    #    would write every family's gammas into every family's stem (the
    #    sib union scheme relies on family-flattened stems, which the
    #    QED grammar precludes; cascade amendment at slice-5
    #    verification). Local families bind no gauge; vo binds the shift
    #    gauge; apply_g5="false" throughout (the upstream XML's
    #    setting). ``spintaste_ja`` binds the BASE thin gauge (taste
    #    phases — never the APBC shift) and publishes the J.A
    #    directional spec: FOURVEC_ONELINK's gamma string IS the
    #    upstream gammas_ja spec.
    if tab_needed or need_precon:
        modules["spintaste_scalar"] = hadmods.spin_taste(
            name="spintaste_scalar",
            gammas=Gamma.SCALAR_LOCAL.gamma_string,
            gauge="",
            apply_g5="false",
        )
        schedule.append("spintaste_scalar")

    needed_families = set(needed_vec_ref) | {f for (f, _, _) in needed_vec_h}
    for g in sorted(needed_families, key=lambda fam: fam.name):
        fam_token = _VEC_FAMILY_TOKEN[g]
        modules[f"spintaste_{fam_token}"] = hadmods.spin_taste(
            name=f"spintaste_{fam_token}",
            gammas=g.gamma_string,
            gauge=(
                config.shift_gauge_name if g == Gamma.VEC_ONELINK else ""
            ),
            apply_g5="false",
        )
        schedule.append(f"spintaste_{fam_token}")
    if any(f == EM_FAMILY_JA for f, _, _, _ in needed_em):
        modules["spintaste_ja"] = hadmods.spin_taste(
            name="spintaste_ja",
            gammas=Gamma.FOURVEC_ONELINK.gamma_string,
            # the BASE thin gauge module name (gauge.py's _GAUGE), never
            # the APBC shift the vo module binds
            gauge="gauge",
            apply_g5="false",
        )
        schedule.append("spintaste_ja")

    # 9b. Block-emission helpers + ll blocks first (cheap banking — the
    #     sib 8b rationale). Vec blocks: one module per (family,
    #     leg_pair, mass) carrying the FULL ±p-expanded momentum list
    #     (module-granularity restart over momenta — chunking deferred,
    #     D11); stems and names are family-scoped (the catalog grammar).
    #     EM blocks: one module per (family, leg_pair, mass, world) with
    #     nEmFields=1 and the world's seed (split_em, D4) — the seed
    #     string is shared across ja/j1l of the same world so both
    #     insertion types consume identical photons (D5). ll modules
    #     reference only lowModes, the CB pairs, and their SpinTaste
    #     module, so nothing but the eigenvector pack is resident when
    #     they run and the cheap outputs are banked before the long
    #     solve chain (sections 10-13 below).
    def _vec_block_name(family_token, leg_pair, mass_label):
        name = f"mf_{family_token}_{leg_pair}"
        if mass_label:
            name += f"_mass_{mass_label}"
        return name

    def _output_stem(family_token, leg_pair, mass_label, em_index):
        return config.output_config.file.filestem.format(
            family=family_token,
            leg_pair=leg_pair,
            mass=_mass_token(mass_label),
            em_index=_em_index_suffix(em_index),
        )

    def _emit_vec_block(gamma, leg_pair, mass_label):
        fam = _VEC_FAMILY_TOKEN[gamma]
        name = _vec_block_name(fam, leg_pair, mass_label)
        pure_high = leg_pair in _PURE_HIGH_LEG_PAIRS
        modules[name] = hadmods.meson_field_v2(
            name=name,
            block=str(config.blocksize),
            gammas=f"spintaste_{fam}",
            low_modes="" if pure_high else low_modes_defl,
            left=noise_fv_vec if leg_pair.startswith("n") else "",
            right=(
                f"quark_h_mass_{mass_label}"
                if leg_pair.endswith("h")
                else ""
            ),
            output=_output_stem(fam, leg_pair, mass_label, None),
            cb_pairs_left="" if pure_high else cbpairs_l,
            cb_pairs_right="" if pure_high else cbpairs_r,
            mom=config.mom_elems,
        )
        schedule.append(name)

    def _em_block_name(family, leg_pair, mass_label, world):
        name = f"mf_{family}_{leg_pair}"
        if mass_label:
            name += f"_mass_{mass_label}"
        return f"{name}_em{world}"

    def _emit_em_block(family, leg_pair, mass_label, world):
        name = _em_block_name(family, leg_pair, mass_label, world)
        pure_high = leg_pair in _PURE_HIGH_LEG_PAIRS
        seed = config.em_seeds(family)[world]
        low = "" if pure_high else low_modes_defl
        cbl = "" if pure_high else cbpairs_l
        cbr = "" if pure_high else cbpairs_r
        left = noise_fv_vec if leg_pair.startswith("n") else ""
        right = (
            f"quark_h_mass_{mass_label}"
            if leg_pair.endswith("h")
            else ""
        )
        stem = _output_stem(family, leg_pair, mass_label, world)
        if family == EM_FAMILY_JA:
            modules[name] = hadmods.aslash_meson_field_v2(
                name=name,
                block=str(config.blocksize),
                gammas="spintaste_ja",
                low_modes=low,
                left=left,
                right=right,
                output=stem,
                em_fn="em",
                n_em_fields="1",
                em_seed_string=seed,
                mom=_ZERO_MOM,
                cb_pairs_left=cbl,
                cb_pairs_right=cbr,
            )
        else:
            modules[name] = hadmods.aslash_onelink_meson_field(
                name=name,
                block=str(config.blocksize),
                gauge="gauge",
                low_modes=low,
                left=left,
                right=right,
                output=stem,
                em_fn="em",
                n_em_fields="1",
                em_seed_string=seed,
                mom=_ZERO_MOM,
                cb_pairs_left=cbl,
                cb_pairs_right=cbr,
            )
        schedule.append(name)

    def _em_sort_key(entry):
        family, leg_pair, mass_label, world = entry
        return (family, leg_pair, mass_label or "", world)

    for gamma in sorted(needed_vec_ref, key=lambda g: g.name):
        for leg_pair in sorted(needed_vec_ref[gamma]):
            if leg_pair == "ll":
                _emit_vec_block(gamma, leg_pair, None)
    for entry in sorted(needed_em, key=_em_sort_key):
        family, leg_pair, mass_label, world = entry
        if leg_pair == "ll":
            _emit_em_block(family, leg_pair, mass_label, world)

    # 10. Full-volume noise + the batch RandomWall source (demand-driven;
    #     shared across EM worlds — solves are photon-independent). The
    #     ``_vec`` companions are implicit C++ outputs of the
    #     color-diagonal noise modules — nothing to emit for them.
    if need_noise_fv:
        modules["noise_fv"] = hadmods.full_volume_noise(
            name="noise_fv", nsrc=str(config.batch_config.noise)
        )
        schedule.append("noise_fv")
    if need_noise_batch:
        modules[noise_batch] = hadmods.noise_rw(
            name=noise_batch,
            nsrc=str(config.batch_config.noise),
            t0="0",
            tstep="1",
            noise="noise_fv",
        )
        schedule.append(noise_batch)

    # 11. tab writer + loader (scalar family only): the ⟨ℓ|η⟩ overlap
    #     table the precon reconstructs from. Writer → loader is a FILE
    #     dependency with no environment edge — schedule order is the
    #     only execution contract. The writer re-runs only when its own
    #     file is bad; the loader rides with the precon.
    tab_stem = config.output_config.tab.filestem
    if tab_needed:
        modules["mf_tab"] = hadmods.meson_field_v2(
            name="mf_tab",
            block=str(config.blocksize),
            gammas="spintaste_scalar",
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

    # 12. The batched precon guess p: ONE StagLMAMesonFieldProp spanning
    #     the whole source window with a2a_batch="true" (single-key
    #     output, batch-guess form consumable by canonical StagGaugeProp).
    #     The batch data contract is pinned by construction — upstream
    #     deliberately does not cross-check it: tA=0, tB=time-1, tStep=1,
    #     nNoise=noise (the RandomWall nSrc), labels=G1_G1.
    if need_precon:
        modules[precon] = hadmods.lma_meson_field_prop_v2(
            name=precon,
            action=action_defl,
            low_modes=low_modes_defl,
            meson_field="mfload_tab",
            gammas="spintaste_scalar",
            labels="G1_G1",
            ta="0",
            tb=str(config.batch_config.time - 1),
            tstep="1",
            noise=noise_fv_vec,
            n_noise=str(config.batch_config.noise),
            a2a_batch="true",
        )
        schedule.append(precon)

    # 13. h solves: one canonical StagGaugeProp per needed operations
    #     mass, sourcing the batch _vec columns and guessing p (the low
    #     subspace is deliberately left in source and solution — removed
    #     at contraction by the downstream task). Shared across every
    #     family and EM world.
    for mass_label in h_masses:
        name = f"quark_h_mass_{mass_label}"
        modules[name] = hadmods.quark_prop_v2(
            name=name,
            source=f"{noise_batch}_vec",
            solver=f"qed_solver_mass_{mass_label}",
            guess=precon,
            gammas="spintaste_scalar",
        )
        schedule.append(name)

    # 14. Remaining block modules, appended after every module they
    #     reference (append order IS the schedule): vec nl/lh/nh groups
    #     and EM nl/lh/nh worlds (ll blocks were banked at 9b).
    for gamma in sorted(needed_vec_ref, key=lambda g: g.name):
        for leg_pair in sorted(needed_vec_ref[gamma]):
            if leg_pair != "ll":
                _emit_vec_block(gamma, leg_pair, None)
    for gamma, leg_pair, mass_label in sorted(
        needed_vec_h, key=lambda e: (e[0].name, e[1], e[2])
    ):
        _emit_vec_block(gamma, leg_pair, mass_label)
    for entry in sorted(needed_em, key=_em_sort_key):
        family, leg_pair, mass_label, world = entry
        if leg_pair != "ll":
            _emit_em_block(family, leg_pair, mass_label, world)

    # Deduplicate schedule: keep first occurrence of each module name
    deduplicated_schedule = list(dict.fromkeys(schedule))

    return HadronsInput(modules=modules, schedule=deduplicated_schedule)


def create_outfile_catalog(config: QEDMFConfig) -> pd.DataFrame:
    catalogs = [
        gauge.create_outfile_catalog(config.gauge_config),
        epack.create_outfile_catalog(config.epack_config),
        _qed_outfile_catalog(config),
    ]
    return pd.concat(catalogs, ignore_index=True)


def build_aggregator_params(config: QEDMFConfig, average: bool) -> t.Dict:
    """Aggregation has no meaning for a terminal meson-field producer
    task (the ``aggregate`` CLI raises on empty params — the correct
    behavior here; momentum-projected outputs cannot be reloaded)."""
    return {}


def _pair_keys() -> t.List[str]:
    """Catalog columns compare_outputs pairs on."""
    return ["family", "leg_pair", "mass", "em_index", "file_key"]


def _col(row: pd.Series, key: str, default: t.Any = None) -> t.Any:
    """Read a (possibly NaN / absent) catalog column from a merged row."""
    if key not in row:
        return default
    val = row[key]
    return default if pd.isna(val) else val


def _compare_h5_file(
    filepath_a: str,
    filepath_b: str,
    dataset: str,
    *,
    rtol: float,
    atol: float,
) -> t.Tuple[float, float, bool]:
    """Compare the ``{dataset}/a2aMatrix`` datasets in two QED files.

    The HDF5 dataset group equals the filename stem for every family
    grammar (vec ``{GammaName}_{p}``, aslash ``{seed}_{a}_{p}``, one-link
    ``{seed}_{j}_{k}_{p}``) — one uniform reader keyed on file_key. The
    compound ``{re, im}`` dataset is read as complex128 and compared
    element-wise (the sib tolerance form; ``astype`` reads either float32
    or float64 producer fields). Returns ``(max_abs_diff, max_rel_diff,
    within_tolerance)``; a shape mismatch reports infinite diffs and
    within=False.
    """
    max_abs = 0.0
    max_rel = 0.0
    within = True
    try:
        with h5py.File(filepath_a, "r") as fa, h5py.File(filepath_b, "r") as fb:
            for f, path in ((fa, filepath_a), (fb, filepath_b)):
                if dataset not in f or "a2aMatrix" not in f[dataset]:
                    raise ValueError(
                        f"dataset {dataset!r}/a2aMatrix not found in {path!r}"
                    )
            a = fa[dataset]["a2aMatrix"][()]
            b = fb[dataset]["a2aMatrix"][()]
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
    config_a: QEDMFConfig,
    config_b: QEDMFConfig,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> pd.DataFrame:
    """Compare QED block/tab outputs between two configs.

    Builds both QED catalogs, pairs expected files by (family, leg_pair,
    mass, em_index, file_key) — the tab row pairs through its NaN block
    keys (pandas outer merge matches NaN keys) and compares the
    ``G1_G1_0_0_0`` dataset — loads each pair's ``{dataset}/a2aMatrix``
    compound dataset, and reports per-file max abs/rel diff plus whether
    the pair is within tolerance. With seeds pinned in config the
    comparison can be exact; a differing ``em.seed`` base is a config
    error, not a numeric diff — it raises (the photon realizations would
    be incomparable).
    """
    logger = utils.get_logger()

    seed_a = config_a.em_config.seed if config_a.em_config else None
    seed_b = config_b.em_config.seed if config_b.em_config else None
    if seed_a != seed_b:
        raise ValueError(
            "compare_outputs requires the same qed.em.seed on both sides; "
            f"got {seed_a!r} and {seed_b!r} — differing seeds pin different "
            "photon realizations (config error, not a numeric diff)."
        )

    catalog_a = _qed_outfile_catalog(config_a)
    catalog_b = _qed_outfile_catalog(config_b)

    paired = catalog_a.merge(
        catalog_b, on=_pair_keys(), how="outer",
        suffixes=("_a", "_b"), indicator=True,
    )

    rows = []
    for _, r in paired.iterrows():
        filepath_a = _col(r, "filepath_a")
        filepath_b = _col(r, "filepath_b")
        exists_a = bool(_col(r, "exists_a", False))
        exists_b = bool(_col(r, "exists_b", False))
        file_key = _col(r, "file_key", "")
        # The tab row carries no block axes; its dataset is the
        # zero-momentum G1_G1 grammar (the _tab_filepath stem).
        dataset = file_key if file_key else "G1_G1_0_0_0"

        base = {
            "family": _col(r, "family", ""),
            "leg_pair": _col(r, "leg_pair", ""),
            "mass": _col(r, "mass", ""),
            "em_index": _col(r, "em_index", ""),
            "file_key": file_key,
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
                f"Missing QED output family={base['family']!r} "
                f"leg_pair={base['leg_pair']!r} file_key={file_key!r} "
                f"(exists_a={exists_a}, exists_b={exists_b})."
            )
            continue

        max_abs, max_rel, within = _compare_h5_file(
            filepath_a, filepath_b, dataset, rtol=rtol, atol=atol
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
        "family", "leg_pair", "mass", "em_index", "file_key",
        "filepath_a", "filepath_b",
        "max_abs_diff", "max_rel_diff", "within_tolerance", "status",
    ]
    return pd.DataFrame(rows, columns=columns)


# Sub-block registrations: leaves get the default route (absorbs the
# builder's _preprocessor slice into fields); strict handler lookup
# returns None for them (the CgConfig precedent,
# pyfm/tasks/hadrons/highmode_v2/config.py:341-357).
register_task("hadrons_qed_mf_batch", QEDBatchConfig, validate=validate_batch)
register_task("hadrons_qed_mf_solver", QEDSolverConfig, validate=validate_solver)
register_task("hadrons_qed_mf_output", QEDOutputConfig)
register_task("hadrons_qed_mf_em", QEDEmConfig, validate=validate_em)
register_task("hadrons_qed_mf_family", QEDFamilyConfig, validate=validate_family)

# Register QEDMFConfig with qed_mf-owned hooks throughout (the full
# handler form: catalog + build + aggregator + compare — a fresh import
# re-registers with every callable).
register_task(
    "hadrons_qed_mf",
    QEDMFConfig,
    create_outfile_catalog,
    build_input_params,
    build_aggregator_params,
    compare_outputs,
    route_params=route_params,
    validate=validate_config,
)
