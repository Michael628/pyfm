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
from .highmode_v2.config import LMAHighModeConfig
from .highmode_v2.phases import validate_phase_sizes

# Label given to an unkeyed ``tasks.high_modes`` entry. Non-empty so the
# entry's cache writer (``{label}_mf_*``) never aliases the ``meson:``
# stanza's unprefixed ``mf_*`` modules.
DEFAULT_HM_LABEL = "hm"


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

    # Program-level split phasing, shared by every entry: a phase count
    # (``subgrid_phases``) XOR explicit phase sizes counted in sources (a
    # ref counts its entry's ``noise``). ``subgrid_last_phase_size``
    # defaults to ``subgrid_phase_size`` and only applies beside it.
    subgrid_phases: int | None = None
    subgrid_phase_size: int | None = None
    subgrid_last_phase_size: int | None = None

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

    @property
    def subgrid_ranks(self) -> int | None:
        """Agreed subgrid count across high-modes entries (or ``None``).

        Entries must agree: one ``<split>`` per XML job means one subgrid
        geometry, and the program-level phasing knobs and the pooled
        planner key off a single ``subgrid_ranks``. Mirrors
        ``split_mpi_layout``'s conflict detection; ``None`` when no entry
        opts in.
        """
        ranks = [
            hm.subgrid_ranks
            for hm in self.high_modes_config.values()
            if hm.subgrid_ranks is not None
        ]
        if not ranks:
            return None
        if any(r != ranks[0] for r in ranks[1:]):
            raise ValueError(
                "Conflicting subgrid_ranks values across high_modes_config "
                f"entries: {ranks!r}; a job admits exactly one subgrid "
                "geometry."
            )
        return ranks[0]


def normalize_params(params: t.Dict) -> t.Dict:
    """Normalize LMANewConfig input: canonicalize the task blocks.

    ``tasks.meson`` accepts a single mapping or a list of mappings (the
    single-mapping form is coerced to a one-entry list, matching the
    legacy sibling task's convention). ``tasks.high_modes`` accepts a
    keyed mapping — ``entries: {label: {...}}`` plus any number of shared
    defaults layered under each entry — or, without the ``entries``
    marker, a single unkeyed entry that gets label ``DEFAULT_HM_LABEL``
    (``"hm"``) so its modules never alias the ``meson:`` stanza's.
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
            hm_entries = {DEFAULT_HM_LABEL: hm_raw}
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

    # Program-level phasing knobs are plain LMANewConfig fields, not
    # config sections: pop them before the ``{k}_config`` loop below
    # (which would otherwise KeyError treating them as section keys).
    _PROGRAM_KNOBS = (
        "subgrid_phases",
        "subgrid_phase_size",
        "subgrid_last_phase_size",
    )
    program_knobs = {
        key: preprocessor_params.pop(key)
        for key in _PROGRAM_KNOBS
        if key in preprocessor_params
    }

    meson_defaults = dict(
        action_name=ACTION_NAME,
        shift_gauge_name=SHIFT_GAUGE_NAME,
        low_modes_name=LOW_MODES_NAME,
    )

    meson_entries = preprocessor_params.get("meson", [])
    if isinstance(meson_entries, dict):
        meson_entries = [meson_entries]

    hm_entries = preprocessor_params.get("high_modes", {})

    # The phasing knobs are program-level: an entry carrying the retired
    # per-entry key gets a pointer, not an unknown-field error.
    for label, entry in hm_entries.items():
        if entry.get("subgrid_phases") is not None:
            raise ValueError(
                f"high_modes entry {label!r} carries subgrid_phases; the "
                "phasing knobs are program-level — set subgrid_phases (a "
                "count) or subgrid_phase_size on tasks.hadrons_lma_new, "
                "not on an entry."
            )

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

    return params | program_knobs | {
        # Per-key DICT params: label (+ derived tokens) as plain scalars (D6).
        "high_modes_config": {
            label: entry_params(label, entry) for label, entry in hm_entries.items()
        },
    } | dict(_preprocessor=child_preprocessor)


def _merge_modules(modules: t.Dict, new: t.Dict) -> None:
    """Merge ``new`` into ``modules`` in place, rejecting name collisions.

    A name already present must map to an identical module — the benign
    shared case (``stag_mass_*`` actions, per-mass ``cbpairs_*``). Any
    other collision would silently replace the earlier module's options
    (Hadrons runs one module per name), so it raises instead.
    """
    for name, module in new.items():
        if name in modules and modules[name] != module:
            raise ValueError(
                f"Module name collision in hadrons_lma_new input: {name!r} is "
                "emitted twice with different options."
            )
    modules |= new


def build_input_params(config: LMANewConfig) -> HadronsInput:
    """Generate input parameters for the full LMA-new task.

    Section order (design doc): base gauge → epack actions → epack →
    epack mass shifts (meson + every entry) → user meson entries →
    high-mode entries (per-entry actions, then the entry's own emission).
    Sections are presence-driven — no stored ``skip_*`` flags (ADR
    Decision 2); ``epack`` is required. Every entry — ``build_only``
    included — gets its per-mass dp action modules: the cache writer's
    ``EigenPackCBPairs`` modules reference ``stag_mass_<mass>`` by name,
    so the action must exist even when no solver consumes it.
    ``build_only`` entries contribute no sp masses (no cg block) and
    stop after the writer; they still join the epack mass-shift set.
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

    # Handle epack mass shifts for meson and every high-mode entry.
    epack_mass_shifts = [m for mc in config.meson_config for m in mc.masses]
    for hm in config.high_modes_config.values():
        epack_mass_shifts.extend(hm.masses)
    if epack_mass_shifts:
        mass_shifts_input = epack.build_epack_mass_shifts(
            config.epack_config, epack_mass_shifts
        )
        _merge_modules(modules, mass_shifts_input.modules)
        schedule += mass_shifts_input.schedule

    # 3. MESON section (presence-driven)
    if config.meson_config:
        meson_masses = [m for mc in config.meson_config for m in mc.masses]
        actions = gauge.build_action_modules(config.gauge_config, dp_masses=meson_masses)
        _merge_modules(modules, actions.modules)
        schedule += actions.schedule

        for mc in config.meson_config:
            meson_input = meson_v2.build_input_params(mc)
            _merge_modules(modules, meson_input.modules)
            schedule += meson_input.schedule

    # 4. HIGHMODE section, once per keyed entry. Per-entry sp masses stay
    # solver-scoped (an entry's masses join the sp set only when that
    # entry itself solves mixed-precision); build_only entries contribute
    # none (they have no cg block).
    hms = list(config.high_modes_config.values())
    entry_sp_masses = [
        hm.masses
        if hm.cg_config is not None and hm.cg_config.solver == "mpcg"
        else []
        for hm in hms
    ]
    if any(entry_sp_masses):
        sp_gauge = gauge.build_sp_gauge(config.gauge_config)
        _merge_modules(modules, sp_gauge.modules)
        schedule += sp_gauge.schedule

    # Split entries with refs to run are emitted TOGETHER, after the
    # per-entry sections, as one pooled split program (phases outer,
    # entries inner); build_only split entries emit no tagged modules
    # and stay per-entry.
    emitting_split = [
        hm
        for hm in hms
        if hm.subgrid_ranks is not None
        and not highmode_v2.is_cache_build_only(hm)
    ]
    pooled_ids = {id(hm) for hm in emitting_split}

    for hm, sp_masses in zip(hms, entry_sp_masses):
        # Unconditional — build_only entries need their dp actions too:
        # the writer's EigenPackCBPairs modules reference stag_mass_<mass>
        # (sp_masses is [] for them: no cg block, nothing solver-scoped).
        actions = gauge.build_action_modules(
            config.gauge_config, dp_masses=hm.masses, sp_masses=sp_masses
        )
        _merge_modules(modules, actions.modules)
        schedule += actions.schedule

        if id(hm) in pooled_ids:
            continue  # emitted below as part of the pooled split program
        hm_input = highmode_v2.build_input_params(hm)
        _merge_modules(modules, hm_input.modules)
        schedule += hm_input.schedule

    if emitting_split:
        _, placements, n_phases = highmode_v2.pooled_placement(
            emitting_split,
            config.subgrid_ranks,
            n_phases=config.subgrid_phases,
            phase_size=config.subgrid_phase_size,
            last_size=config.subgrid_last_phase_size,
        )
        split_input = highmode_v2.build_split_program(
            emitting_split, placements, n_phases
        )
        _merge_modules(modules, split_input.modules)
        schedule += split_input.schedule

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
        # frame). Empty-op entries are rejected by validate_config.
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

    Cross-entry checks — per-entry state was validated when each
    ``LMAHighModeConfig`` was built: (1) conflicting ``split_mpi_layout``
    across entries (one ``<split>`` per XML job) and conflicting
    ``subgrid_ranks`` (one subgrid geometry a program), enforced by the
    properties and touched here so the misconfigurations surface at
    build time; (2) the program-level phasing knobs: ``subgrid_phases``
    (a count) XOR ``subgrid_phase_size``, ``subgrid_last_phase_size``
    only beside the size and both positive multiples of the agreed
    ``subgrid_ranks`` — and, when any knob is set, every entry split
    (pooled phasing cannot place an unsplit entry's refs); (3) a warning
    when two entries bind the same output files label.
    """
    _ = config.split_mpi_layout
    _ = config.subgrid_ranks

    count = config.subgrid_phases
    size = config.subgrid_phase_size
    last = config.subgrid_last_phase_size

    if count is not None and size is not None:
        raise ValueError(
            "subgrid_phases and subgrid_phase_size are mutually "
            f"exclusive; got subgrid_phases={count}, "
            f"subgrid_phase_size={size}."
        )
    if last is not None and size is None:
        raise ValueError(
            "subgrid_last_phase_size requires subgrid_phase_size (it "
            "defaults to subgrid_phase_size when absent)."
        )
    if count is not None and count < 1:
        raise ValueError(
            f"subgrid_phases must be a positive integer; got {count}."
        )

    if count is not None or size is not None:
        if not config.high_modes_config:
            raise ValueError(
                "The subgrid phasing knobs require high_modes entries to "
                "phase."
            )
        unsplit = [
            label
            for label, hm in config.high_modes_config.items()
            if hm.subgrid_ranks is None
        ]
        if unsplit:
            raise ValueError(
                "The subgrid phasing knobs require every high_modes entry "
                "to be split (split_mpi_layout + subgrid_ranks); unsplit "
                f"entries: {sorted(unsplit)}."
            )
        if size is not None:
            validate_phase_sizes(
                size, last if last is not None else size, config.subgrid_ranks
            )

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


# Note: subgrid_phases counts larger than the pooled run refs are
# NOT rejected here — the pooled ref count is only known at emission
# (entry_run_refs' resume gate). plan_phase_counts raises "exceeds
# the pooled source refs" at build_input_params time; that deferred
# failure is intentional.


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
