import copy
import typing as t

import pandas as pd
from pydantic import Field
from pydantic.dataclasses import dataclass

from pyfm import utils
from pyfm.domain import CompositeConfig
from pyfm.tasks.hadrons.types import HadronsInput
from pyfm.tasks.register import register_task

from . import gauge, meson, epack, meson_v2, highmode_v2
from .highmode_v2.config import CacheMode, LMAHighModeConfig


@dataclass(frozen=True)
class LMANewConfig(CompositeConfig):
    """Own config tree for ``hadrons_lma_new`` (ADR 0001).

    Sections are presence-driven: an absent ``tasks.meson`` block builds
    an empty list, an absent or empty ``tasks.high_modes`` an empty dict —
    there are no stored ``skip_*`` flags and no legacy cache-build flag
    (ADR Decision 2). ``epack_config`` is required: both jobs of a cache
    workflow load or generate the eigenpack (the cache saves the
    meson-field stencil work, not eigenvector I/O), so the two jobs of a
    cache workflow must share ``runid`` for noise reproducibility.

    A genuinely new class, not a subclass or reuse of the legacy LMI
    config class (``register.py``'s ``_config_to_task_key`` reverse-lookup
    keys by class object; reusing a sibling task's config class verbatim
    would silently steal its registration).
    """

    gauge_config: gauge.GaugeConfig
    epack_config: epack.EpackConfig
    meson_config: t.List[meson.MesonConfig] = Field(default_factory=list)
    high_modes_config: t.Dict[str, LMAHighModeConfig] = Field(default_factory=dict)

    @property
    def split_mpi_layout(self) -> str | None:
        """Re-expose the split-grid MPI layout from the high-modes entries.

        The first entry that sets a layout wins; conflicting layouts are a
        hard misconfiguration (one ``<split>`` per XML job). ``None`` when
        no entry opts in or the dict is empty.
        """
        layouts = [
            hm.split_mpi_layout
            for hm in self.high_modes_config.values()
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


def normalize_params(params: t.Dict) -> t.Dict:
    """Normalize LMANewConfig input: canonicalize the task blocks.

    ``tasks.meson`` accepts a single mapping or a list of mappings (the
    single-mapping form is coerced to a one-entry list, matching the
    legacy sibling task's convention). ``tasks.high_modes`` accepts a
    keyed mapping — ``entries: {label: {...}}`` plus any number of shared
    defaults layered under each entry — or, without the ``entries``
    marker, a single unkeyed entry that gets label ``""`` (D5).
    """
    incoming = params.get("_preprocessor", {})
    # Deep-copy: the route hooks below pop keys out of these nested blocks
    # (``sources.grid``/``biased``, ``low_modes.meson_field``), and the
    # ``shared | entry`` layering is shallow — without the copy the pops
    # would mutate the caller's params dict, breaking a second
    # ``create_task`` on the same YAML tree.
    incoming = copy.deepcopy(incoming)
    canonical: t.Dict[str, t.List | t.Dict] = {}

    meson_raw = incoming.get("meson")
    if isinstance(meson_raw, dict):
        canonical["meson"] = [meson_raw]
    elif meson_raw is not None and not isinstance(meson_raw, list):
        raise TypeError(
            f"tasks.meson must be a mapping or a list of mappings; got "
            f"{type(meson_raw).__name__}."
        )

    hm_raw = incoming.get("high_modes")
    if hm_raw is not None:
        if not isinstance(hm_raw, dict):
            raise TypeError(
                "tasks.high_modes must be a mapping (keyed entries or one "
                f"unkeyed entry); got {type(hm_raw).__name__}."
            )
        entries = hm_raw.get("entries")
        if entries is None:
            hm_entries = {"": hm_raw}
        else:
            if not isinstance(entries, dict):
                raise TypeError(
                    "tasks.high_modes.entries must be a mapping of label to "
                    f"entry; got {type(entries).__name__}."
                )
            if not all(isinstance(entry, dict) for entry in entries.values()):
                raise TypeError(
                    "tasks.high_modes.entries values must all be mappings."
                )
            shared = {k: v for k, v in hm_raw.items() if k != "entries"}
            hm_entries = {label: shared | entry for label, entry in entries.items()}
        canonical["high_modes"] = hm_entries

    if canonical:
        params = params | {"_preprocessor": incoming | canonical}
    return params


def route_params(params: t.Dict) -> t.Dict:
    """Route per-subtask input to the child configs, layering name defaults.

    ``high_modes`` is a DICT subconfig: the per-key routing table
    (``_preprocessor["high_modes_config"]``) carries each entry's mapping,
    and the per-key params (``params["high_modes_config"][label]``) carry
    ONLY the label — a plain scalar param that reaches the entry, every
    sub-block, and the formatting map (D6), keeping the OpList ``mass``
    key away from the ``MassDict`` param path.
    """

    ACTION_NAME = "stag_mass_{mass}"
    LOW_MODES_NAME = "evecs_mass_{mass}"
    SHIFT_GAUGE_NAME = "gauge_apbc"

    preprocessor_params = params.pop("_preprocessor", {})

    meson_defaults = dict(
        action_name=ACTION_NAME,
        shift_gauge_name=SHIFT_GAUGE_NAME,
        low_modes_name=LOW_MODES_NAME,
    )

    meson_entries = preprocessor_params.get("meson", [])
    if isinstance(meson_entries, dict):
        meson_entries = [meson_entries]

    hm_entries = preprocessor_params.get("high_modes", {})

    child_preprocessor = dict(
        gauge_config=dict(action_name=ACTION_NAME),
        epack_config=dict(
            action_name=ACTION_NAME,
            low_modes_name=LOW_MODES_NAME,
        ),
        meson_config=[meson_defaults | entry for entry in meson_entries],
        high_modes_config=hm_entries,
    )

    for k, v in preprocessor_params.items():
        if k in ("high_modes", "meson"):
            continue  # already expanded into the routing tables above
        child_preprocessor[f"{k}_config"] |= v

    def entry_params(label: str, entry: t.Dict) -> t.Dict:
        """Per-key DICT params: the label plus derived formatting tokens (D6).

        ``nbias`` re-derives from the entry's biased-sources block so
        output filestems may carry the ``{nbias}`` namespace token (the
        legacy ``HighModeConfig.nbias`` field's role): it rides into the
        builder's format-key map like ``label`` does, so the config
        builder resolves it in every ``Outfile`` at build time.
        """
        params = {"label": label}
        sources = entry.get("sources")
        if isinstance(sources, dict):
            biased = sources.get("biased")
            if isinstance(biased, dict) and biased.get("n") is not None:
                params["nbias"] = biased["n"]
        return params

    return params | {
        # Per-key DICT params: label (+ derived tokens) as plain scalars (D6).
        "high_modes_config": {
            label: entry_params(label, entry) for label, entry in hm_entries.items()
        },
    } | dict(_preprocessor=child_preprocessor)


def build_input_params(config: LMANewConfig) -> HadronsInput:
    """Generate input parameters for the full LMA-new task.

    Section order (design doc): base gauge → epack actions → epack →
    epack mass shifts (meson + every entry) → user meson entries →
    high-mode entries (per-entry actions, then the entry's own emission).
    Sections are presence-driven — no stored ``skip_*`` flags (ADR
    Decision 2); ``epack`` is required. Cache ``build_only`` entries skip
    their action modules (the writer needs only the epack low modes and
    the entry's noise) but still join the epack mass-shift set.
    """
    modules = {}
    schedule = []

    # 1. Always start with base gauge
    base_gauge = gauge.build_base_gauge(config.gauge_config)
    modules |= base_gauge.modules
    schedule += base_gauge.schedule

    # 2. EPACK section (required)
    epack_masses = config.epack_config.masses
    actions = gauge.build_action_modules(config.gauge_config, dp_masses=epack_masses)
    modules |= actions.modules
    schedule += actions.schedule

    epack_input = epack.build_input_params(config.epack_config)
    modules |= epack_input.modules
    schedule += epack_input.schedule

    # Handle epack mass shifts for meson and every high-mode entry.
    epack_mass_shifts = [m for mc in config.meson_config for m in mc.masses]
    for hm in config.high_modes_config.values():
        epack_mass_shifts.extend(hm.masses)
    if epack_mass_shifts:
        mass_shifts_input = epack.build_epack_mass_shifts(
            config.epack_config, epack_mass_shifts
        )
        modules |= mass_shifts_input.modules
        schedule += mass_shifts_input.schedule

    # 3. MESON section (presence-driven)
    if config.meson_config:
        meson_masses = [m for mc in config.meson_config for m in mc.masses]
        actions = gauge.build_action_modules(config.gauge_config, dp_masses=meson_masses)
        modules |= actions.modules
        schedule += actions.schedule

        for mc in config.meson_config:
            meson_input = meson_v2.build_input_params(mc)
            modules |= meson_input.modules
            schedule += meson_input.schedule

    # 4. HIGHMODE section, once per keyed entry. Per-entry sp masses stay
    # solver-scoped (an entry's masses join the sp set only when that
    # entry itself solves mixed-precision); build_only entries contribute
    # none (they have no cg block).
    entry_sp_masses = [
        hm.masses
        if hm.cg_config is not None and hm.cg_config.solver == "mpcg"
        else []
        for hm in config.high_modes_config.values()
    ]
    if any(entry_sp_masses):
        sp_gauge = gauge.build_sp_gauge(config.gauge_config)
        modules |= sp_gauge.modules
        schedule += sp_gauge.schedule

    for hm, sp_masses in zip(config.high_modes_config.values(), entry_sp_masses):
        build_only = (
            hm.low_modes_config.meson_field_config is not None
            and hm.low_modes_config.meson_field_config.cache is CacheMode.BUILD_ONLY
        )
        if not build_only:
            actions = gauge.build_action_modules(
                config.gauge_config, dp_masses=hm.masses, sp_masses=sp_masses
            )
            modules |= actions.modules
            schedule += actions.schedule

        hm_input = highmode_v2.build_input_params(hm)
        modules |= hm_input.modules
        schedule += hm_input.schedule

    # Deduplicate schedule: keep first occurrence of each module name
    deduplicated_schedule = list(dict.fromkeys(schedule))

    return HadronsInput(modules=modules, schedule=deduplicated_schedule)


def create_outfile_catalog(config: LMANewConfig) -> pd.DataFrame:
    catalogs = [
        gauge.create_outfile_catalog(config.gauge_config),
        epack.create_outfile_catalog(config.epack_config),
    ]
    for mc in config.meson_config:
        catalogs.append(meson.create_outfile_catalog(mc))
    for hm in config.high_modes_config.values():
        # build_only entries catalog nothing (strategy returns an empty
        # frame); degenerate empty-op entries likewise contribute no rows.
        catalogs.append(highmode_v2.create_outfile_catalog(hm))
    return pd.concat(catalogs, ignore_index=True)


def build_aggregator_params(config: LMANewConfig, average: bool) -> t.Dict:
    params: t.Dict = {}
    for hm in config.high_modes_config.values():
        entry = highmode_v2.build_aggregator_params(hm, average)
        params = {
            **params,
            **entry,
            "run": params.get("run", []) + entry.get("run", []),
        }
    return params


def compare_outputs(
    config_a: LMANewConfig, config_b: LMANewConfig, *, rtol: float = 1e-9, atol: float = 1e-12
) -> pd.DataFrame:
    """Compare LMA-new outputs between two configs.

    High-mode entries are compared pairwise by LABEL (not position); both
    configs must present the same label set. Delegates to the high-mode
    correlator comparison only (the primary output); meson/epack
    comparison is deferred.
    """
    labels_a = set(config_a.high_modes_config)
    labels_b = set(config_b.high_modes_config)
    if labels_a != labels_b:
        raise ValueError(
            "compare_outputs requires the same high_modes entry labels on "
            f"both sides; got {sorted(labels_a)} and {sorted(labels_b)}."
        )
    reports = [
        highmode_v2.compare_outputs(
            config_a.high_modes_config[label],
            config_b.high_modes_config[label],
            rtol=rtol,
            atol=atol,
        )
        for label in sorted(labels_a)
    ]
    if not reports:
        return pd.DataFrame(
            columns=[
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
        )
    return pd.concat(reports, ignore_index=True)


def validate_config(config: LMANewConfig) -> None:
    """Validate LMANewConfig after construction.

    Cross-entry checks only — per-entry state was validated when each
    ``LMAHighModeConfig`` was built: (1) conflicting ``split_mpi_layout``
    across entries (one ``<split>`` per XML job — enforced by the property,
    touched here so the misconfiguration surfaces at build time);
    (2) a warning when two entries bind the same output files label
    (their correlators share a filestem namespace — bind a distinct files
    entry per keyed entry).
    """
    _ = config.split_mpi_layout

    stems = [
        hm.output_config.file.filestem
        for hm in config.high_modes_config.values()
        if hm.output_config is not None
    ]
    for i, stem in enumerate(stems):
        if stem in stems[:i]:
            utils.get_logger().warning(
                f"high_modes entry binding files label with filestem {stem!r} "
                "is shared with another entry; their outputs share a filestem "
                "namespace. Bind a distinct files entry per keyed entry."
            )


# Register LMANewConfig with lma_new-owned hooks throughout (ADR Decision 1,
# D7): no legacy imports, no config-synthesis hook (the cache writer is
# emitted per entry inside highmode_v2.build_input_params).
register_task(
    "hadrons_lma_new",
    LMANewConfig,
    create_outfile_catalog,
    build_input_params,
    build_aggregator_params,
    compare_outputs,
    normalize_params,
    route_params,
    validate=validate_config,
)
