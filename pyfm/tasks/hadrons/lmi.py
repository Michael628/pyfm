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
    meson_config: meson.MesonConfig
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


def normalize_params(params: t.Dict) -> t.Dict:
    """Normalize LMIConfig input: derive ``skip_*`` flags and canonicalize
    the ``high_modes`` task block to a list.

    ``tasks.high_modes`` accepts a single mapping (one entry) or a list of
    mappings (multi-entry); the single-mapping form is coerced to a one-entry
    list here (``DiagramConfig.mesons`` precedent, ``contract/diagram.py``).
    ``skip_high_modes`` is derived from the canonicalized list's emptiness —
    an absent block and an explicit ``high_modes: []`` are equivalent. The
    incoming ``_preprocessor`` slice is *inspected* (and, for ``high_modes``,
    canonically replaced) here — ``route_params`` owns its consumption.
    """
    incoming = params.get("_preprocessor", {})
    skip_flags = {
        f"skip_{k}": True
        for k in _OPTIONAL_CONFIGS
        if k != "high_modes" and k not in incoming
    }

    hm_raw = incoming.get("high_modes")
    if hm_raw is None:
        hm_entries = []
    elif isinstance(hm_raw, dict):
        hm_entries = [hm_raw]
    elif isinstance(hm_raw, list):
        hm_entries = hm_raw
    else:
        raise TypeError(
            "tasks.high_modes must be a mapping or a list of mappings; got "
            f"{type(hm_raw).__name__}."
        )
    if not all(isinstance(entry, dict) for entry in hm_entries):
        raise TypeError(
            "tasks.high_modes entries must all be mappings; got "
            f"{[type(entry).__name__ for entry in hm_entries]}."
        )
    skip_flags["skip_high_modes"] = not hm_entries

    if hm_raw is not None and not isinstance(hm_raw, list):
        params = params | {"_preprocessor": incoming | {"high_modes": hm_entries}}
    return params | skip_flags


def route_params(params: t.Dict) -> t.Dict:
    """Route per-subtask input to the child configs, layering in name defaults.

    ``high_modes`` is a LIST subconfig: every canonicalized entry is layered
    over the shared defaults and routed as its own slice under
    ``_preprocessor["high_modes_config"]``. A ``tasks.bias`` key now fails
    loudly — the bias sibling is gone; bias is a list entry carrying
    ``nbias``/``bias_seed``.
    """

    ACTION_NAME = "stag_mass_{mass}"
    SOLVER_NAME = "stag_{solver}_mass_{mass}"
    LOW_MODES_NAME = "evecs_mass_{mass}"
    SHIFT_GAUGE_NAME = "gauge_apbc"

    # Incoming slice holds per-subtask input keyed by subtask name.
    preprocessor_params = params.pop("_preprocessor", {})

    # Shared per-entry defaults for the high-mode children
    high_modes_defaults = dict(
        action_name=ACTION_NAME,
        low_modes_name=LOW_MODES_NAME,
        solver_name=SOLVER_NAME,
        shift_gauge_name=SHIFT_GAUGE_NAME,
        skip_low_modes="epack" not in preprocessor_params,
    )

    # Set defaults for child configs. high_modes_config is a list of
    # per-entry slices (normalize_params canonicalized the shape; the
    # isinstance guard keeps route total for direct callers).
    entries = preprocessor_params.get("high_modes", [])
    if isinstance(entries, dict):
        entries = [entries]
    child_preprocessor = dict(
        gauge_config=dict(action_name=ACTION_NAME),
        epack_config=dict(
            action_name=ACTION_NAME,
            low_modes_name=LOW_MODES_NAME,
        ),
        meson_config=dict(
            action_name=ACTION_NAME,
            shift_gauge_name=SHIFT_GAUGE_NAME,
            low_modes_name=LOW_MODES_NAME,
        ),
        high_modes_config=[
            high_modes_defaults | entry for entry in entries
        ],
    )

    # Update child processor with corresponding params passed to parent
    for k, v in preprocessor_params.items():
        if k == "high_modes":
            continue  # already expanded into the list above
        child_preprocessor[f"{k}_config"] |= v

    return params | dict(_preprocessor=child_preprocessor)


def validate_config(config: LMIConfig) -> None:
    """Validate LMIConfig after construction and postprocessing.

    Validates that if epack is skipped, meson must also be skipped. Warns when
    two high-mode entries bind the same files label: their outputs share a
    filestem namespace, and only distinct source labels (e.g. bias block
    labels) keep the files distinct.
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

        # Handle epack mass shifts for meson and every high-mode entry
        epack_mass_shifts = []
        if not config.skip_meson:
            epack_mass_shifts.extend(config.meson_config.masses)
        for hm in config.high_modes_config:
            epack_mass_shifts.extend(hm.masses)

        if epack_mass_shifts:
            mass_shifts_input = epack.build_epack_mass_shifts(
                config.epack_config, epack_mass_shifts
            )
            modules |= mass_shifts_input.modules
            schedule += mass_shifts_input.schedule

    # 3. MESON section: generate actions then compute
    if not config.skip_meson:
        meson_masses = config.meson_config.masses
        actions = gauge.build_action_modules(
            config.gauge_config, dp_masses=meson_masses
        )
        modules |= actions.modules
        schedule += actions.schedule

        meson_input = meson.build_input_params(config.meson_config)
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
        meson.create_outfile_catalog(config.meson_config),
    ]
    for hm in config.high_modes_config:
        if not hm.op_list:
            continue  # degenerate entry: excluded, as the old sibling guard did
        catalogs.append(highmode.create_outfile_catalog(hm))
    return pd.concat(catalogs, ignore_index=True)


def build_aggregator_params(config: LMIConfig, average: bool) -> t.Dict:
    params: t.Dict = {}
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
