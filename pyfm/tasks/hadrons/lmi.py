import typing as t
import pandas as pd

from pydantic.dataclasses import dataclass

from pyfm import utils
from pyfm.tasks.hadrons.types import HadronsInput
from pyfm.domain import CompositeConfig
from pyfm.tasks.register import register_task

from . import gauge, meson, epack, highmode
from .types import HighModeConfig


@dataclass(frozen=True)
class LMIConfig(CompositeConfig):
    gauge_config: gauge.GaugeConfig
    epack_config: epack.EpackConfig
    meson_config: t.List[meson.MesonConfig]
    high_modes_config: t.List[HighModeConfig]
    skip_epack: bool = False
    skip_meson: bool = False
    skip_high_modes: bool = False

    @property
    def split_mpi_layout(self) -> str | None:
        """Re-expose the split-grid MPI layout from the high-modes subconfigs.

        ``split_mpi_layout`` lives on :class:`HighModeConfig`; this property
        lets ``inputgen.write_input_file`` read it uniformly from a composite
        ``LMIConfig`` (standalone ``HighModeConfig`` exposes it as a direct
        field). With a list of high-mode configs the first entry that sets a
        layout wins; conflicting layouts are a hard misconfiguration (one
        ``<split>`` per XML job). ``None`` when no entry opts in or the list
        is empty.
        """
        layouts = [
            hm.split_mpi_layout
            for hm in self.high_modes_config
            if hm.split_mpi_layout is not None
        ]
        if not layouts:
            return None
        if any(layout != layouts[0] for layout in layouts[1:]):
            raise ValueError(
                "Conflicting split_mpi_layout values across high_modes_config "
                f"entries: {layouts!r}; a job admits exactly one <split>."
            )
        return layouts[0]


_OPTIONAL_CONFIGS = ["meson", "high_modes", "epack"]
_LIST_CONFIGS = ["high_modes", "meson"]


def normalize_params(params: t.Dict) -> t.Dict:
    """Normalize LMIConfig input: derive ``skip_*`` flags and canonicalize
    the ``high_modes``/``meson`` task blocks to lists.

    Both ``tasks.high_modes`` and ``tasks.meson`` accept a single mapping
    (one entry) or a list of mappings (multi-entry); the single-mapping form
    is coerced to a one-entry list here. ``skip_high_modes``/``skip_meson``
    are derived from each canonicalized list's emptiness — an absent block
    and an explicit ``[]`` are equivalent. The incoming ``_preprocessor``
    slice is *inspected* (and, for these two keys, canonically replaced)
    here — ``route_params`` owns its consumption.
    """
    incoming = params.get("_preprocessor", {})
    skip_flags = {
        f"skip_{k}": True
        for k in _OPTIONAL_CONFIGS
        if k not in _LIST_CONFIGS and k not in incoming
    }

    canonical: t.Dict[str, t.List] = {}
    for key in _LIST_CONFIGS:
        raw = incoming.get(key)
        if raw is None:
            entries = []
        elif isinstance(raw, dict):
            entries = [raw]
        elif isinstance(raw, list):
            entries = raw
        else:
            raise TypeError(
                f"tasks.{key} must be a mapping or a list of mappings; got "
                f"{type(raw).__name__}."
            )
        if not all(isinstance(entry, dict) for entry in entries):
            raise TypeError(
                f"tasks.{key} entries must all be mappings; got "
                f"{[type(entry).__name__ for entry in entries]}."
            )
        skip_flags[f"skip_{key}"] = not entries
        if raw is not None and not isinstance(raw, list):
            canonical[key] = entries

    if canonical:
        params = params | {"_preprocessor": incoming | canonical}
    return params | skip_flags


def route_params(params: t.Dict) -> t.Dict:
    """Route per-subtask input to the child configs, layering in name defaults.

    ``high_modes`` and ``meson`` are LIST subconfigs: every canonicalized
    entry is layered over its section's shared defaults and routed as its
    own slice under ``_preprocessor["high_modes_config"]``/
    ``_preprocessor["meson_config"]``.
    """

    ACTION_NAME = "stag_mass_{mass}"
    SOLVER_NAME = "stag_{solver}_mass_{mass}"
    LOW_MODES_NAME = "evecs_mass_{mass}"
    SHIFT_GAUGE_NAME = "gauge_apbc"

    preprocessor_params = params.pop("_preprocessor", {})

    high_modes_defaults = dict(
        action_name=ACTION_NAME,
        low_modes_name=LOW_MODES_NAME,
        solver_name=SOLVER_NAME,
        shift_gauge_name=SHIFT_GAUGE_NAME,
        skip_low_modes="epack" not in preprocessor_params,
    )
    meson_defaults = dict(
        action_name=ACTION_NAME,
        shift_gauge_name=SHIFT_GAUGE_NAME,
        low_modes_name=LOW_MODES_NAME,
    )

    hm_entries = preprocessor_params.get("high_modes", [])
    if isinstance(hm_entries, dict):
        hm_entries = [hm_entries]

    meson_entries = preprocessor_params.get("meson", [])
    if isinstance(meson_entries, dict):
        meson_entries = [meson_entries]

    child_preprocessor = dict(
        gauge_config=dict(action_name=ACTION_NAME),
        epack_config=dict(
            action_name=ACTION_NAME,
            low_modes_name=LOW_MODES_NAME,
        ),
        meson_config=[meson_defaults | entry for entry in meson_entries],
        high_modes_config=[
            high_modes_defaults | entry for entry in hm_entries
        ],
    )

    for k, v in preprocessor_params.items():
        if k in ("high_modes", "meson"):
            continue  # already expanded into the lists above
        child_preprocessor[f"{k}_config"] |= v

    return params | dict(_preprocessor=child_preprocessor)


def validate_shared_config(config: CompositeConfig) -> None:
    """Schema-agnostic LMIConfig-shaped validation, shared by ``lmi.py``
    and ``lma_new.py`` (duck-typed against field names, not ``isinstance``
    -gated — the same reuse contract ``grid/lma.py``'s ``GridLMAConfig``
    already relies on for these hooks).

    Validates that if epack is skipped, meson must also be skipped. Warns
    when two high-mode entries bind the same files label: their outputs
    share a filestem namespace, and only distinct source labels (e.g. bias
    block labels) keep the files distinct.
    """
    for k in ["meson", "high_modes", "epack"]:
        if getattr(config, f"skip_{k}", False):
            utils.get_logger().debug(f"Skipping {k} step")

    if config.skip_epack and not config.skip_meson:
        raise ValueError("Epack parameters must be set to perform meson calculation")

    stems = [hm.high_modes.filestem for hm in config.high_modes_config]
    for i, stem in enumerate(stems):
        if stem in stems[:i]:
            utils.get_logger().warning(
                f"high_modes_config entry {i} binds the same files label "
                f"(filestem {stem!r}) as an earlier entry; their outputs "
                "share a filestem namespace. Add a second files entry (e.g. "
                "files.bias_modes) and set `high_modes: bias_modes` in that "
                "tasks.high_modes list entry to segregate them."
            )


def validate_config(config: LMIConfig) -> None:
    """Validate LMIConfig after construction and postprocessing.

    ``hadrons_lmi`` is on a deprecation path: it targets HadronsMILC's
    develop-schema Legacy modules. The schema-agnostic checks (epack/meson
    consistency, filestem collisions) live in :func:`validate_shared_config`,
    reused unchanged by ``lma_new.py``.
    """
    utils.get_logger().warning(
        "hadrons_lmi is on a deprecation path: it targets HadronsMILC's "
        "develop-schema Legacy modules (StagGaugePropLegacy/StagMesonLegacy/"
        "StagA2AMesonFieldLegacy). Use hadrons_lma_new for new work "
        "targeting the current (SpinTaste-module-based) HadronsMILC API."
    )

    validate_shared_config(config)


def build_input_params(config: LMIConfig) -> HadronsInput:
    """Generate input parameters for the full LMI task.

    Orchestrates gauge module generation with submodule computation, ensuring that
    gauge action modules are generated only when needed by the submodules that use them.
    High-mode entries are emitted in list order; a single sp-gauge covers every
    mixed-precision entry (module merges are idempotent, the schedule is
    deduplicated below).
    """
    modules = {}
    schedule = []

    # 1. Always start with base gauge
    base_gauge = gauge.build_base_gauge(config.gauge_config)
    modules |= base_gauge.modules
    schedule += base_gauge.schedule

    # 2. EPACK section: generate actions then compute
    if not config.skip_epack:
        epack_masses = config.epack_config.masses
        actions = gauge.build_action_modules(
            config.gauge_config, dp_masses=epack_masses
        )
        modules |= actions.modules
        schedule += actions.schedule

        epack_input = epack.build_input_params(config.epack_config)
        modules |= epack_input.modules
        schedule += epack_input.schedule

        # Handle epack mass shifts for meson and every high-mode entry.
        epack_mass_shifts = []
        if not config.skip_meson:
            for mc in config.meson_config:
                epack_mass_shifts.extend(mc.masses)
        for hm in config.high_modes_config:
            epack_mass_shifts.extend(hm.masses)

        if epack_mass_shifts:
            mass_shifts_input = epack.build_epack_mass_shifts(
                config.epack_config, epack_mass_shifts
            )
            modules |= mass_shifts_input.modules
            schedule += mass_shifts_input.schedule

    # 3. MESON section: generate actions then compute, once per entry.
    if not config.skip_meson:
        meson_masses = []
        for mc in config.meson_config:
            meson_masses.extend(mc.masses)
        actions = gauge.build_action_modules(
            config.gauge_config, dp_masses=meson_masses
        )
        modules |= actions.modules
        schedule += actions.schedule

        for mc in config.meson_config:
            meson_input = meson.build_input_params(mc)
            modules |= meson_input.modules
            schedule += meson_input.schedule

    # 4. HIGHMODE section: generate actions then compute, once per entry.
    # Per-entry sp masses stay solver-scoped (an entry's masses join the sp
    # set only when that entry itself solves mixed-precision), exactly as the
    # former high_modes/bias siblings behaved.
    if not config.skip_high_modes:
        entry_sp_masses = [
            hm.masses if hm.solver == "mpcg" else []
            for hm in config.high_modes_config
        ]
        if any(entry_sp_masses):
            sp_gauge = gauge.build_sp_gauge(config.gauge_config)
            modules |= sp_gauge.modules
            schedule += sp_gauge.schedule

        for hm, sp_masses in zip(config.high_modes_config, entry_sp_masses):
            actions = gauge.build_action_modules(
                config.gauge_config, dp_masses=hm.masses, sp_masses=sp_masses
            )
            modules |= actions.modules
            schedule += actions.schedule

            hm_input = highmode.build_input_params(hm)
            modules |= hm_input.modules
            schedule += hm_input.schedule

    # Deduplicate schedule: keep first occurrence of each module name
    deduplicated_schedule = list(dict.fromkeys(schedule))

    return HadronsInput(modules=modules, schedule=deduplicated_schedule)


def create_outfile_catalog(config: LMIConfig) -> pd.DataFrame:
    catalogs = [
        gauge.create_outfile_catalog(config.gauge_config),
        epack.create_outfile_catalog(config.epack_config),
    ]
    for mc in config.meson_config:
        catalogs.append(meson.create_outfile_catalog(mc))
    # skip_high_modes means build_input_params never schedules any
    # high-modes correlator module (lma_new.py's cache-only carve-out only
    # ever calls highmode_v2.build_input_params in its noise-only reduced
    # form) — so no entry's correlator files are ever written, and
    # including their rows here would permanently poison
    # validator.has_good_output's completion mask.
    if not config.skip_high_modes:
        for hm in config.high_modes_config:
            if not hm.op_list:
                continue  # degenerate entry: excluded, as the old sibling guard did
            catalogs.append(highmode.create_outfile_catalog(hm))
    return pd.concat(catalogs, ignore_index=True)


def build_aggregator_params(config: LMIConfig, average: bool) -> t.Dict:
    params: t.Dict = {}
    if config.skip_high_modes:
        return params
    for i, hm in enumerate(config.high_modes_config):
        # Entry 0 keeps today's unprefixed run keys (single-entry aggregation
        # byte-identical); later entries namespace under hm{i}_ so run keys
        # never clobber when (gamma, mass, dset) combos repeat.
        run_prefix = "" if i == 0 else f"hm{i}_"
        entry = highmode.build_aggregator_params(hm, average, run_prefix=run_prefix)
        params = {
            **params,
            **entry,
            "run": params.get("run", []) + entry.get("run", []),
        }
    return params


def compare_outputs(
    config_a: LMIConfig, config_b: LMIConfig, *, rtol: float = 1e-9, atol: float = 1e-12
) -> pd.DataFrame:
    """Compare LMI outputs between two configs.

    High-mode entries are compared pairwise by list position; both configs
    must present the same number of entries. Currently delegates to the
    high-mode correlator comparison only (the primary output); meson/epack
    comparison is deferred.
    """
    if config_a.skip_high_modes or config_b.skip_high_modes:
        raise ValueError(
            "compare_outputs requires both configs to compute high_modes "
            "(skip_high_modes must be False on both sides)."
        )
    if len(config_a.high_modes_config) != len(config_b.high_modes_config):
        raise ValueError(
            "compare_outputs requires the same number of high_modes_config "
            f"entries on both sides; got {len(config_a.high_modes_config)} and "
            f"{len(config_b.high_modes_config)}."
        )
    reports = [
        highmode.compare_outputs(hm_a, hm_b, rtol=rtol, atol=atol)
        for hm_a, hm_b in zip(
            config_a.high_modes_config, config_b.high_modes_config
        )
    ]
    return pd.concat(reports, ignore_index=True)


# Register LMIConfig with all handlers
register_task(
    "hadrons_lmi",
    LMIConfig,
    create_outfile_catalog,
    build_input_params,
    build_aggregator_params,
    compare_outputs,
    normalize_params,
    route_params,
    validate=validate_config,
)
