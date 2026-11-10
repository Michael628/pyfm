"""HadronsMILC SpinTaste-API high-mode task support for ``hadrons_lma_new``.

Fully separate from the legacy ``highmode/`` package (ADR 0001 / D4):
``config`` owns the ``LMAHighModeConfig`` tree and its build hooks,
``strategy`` the per-entry emission / catalog / aggregation / comparison,
``twopoint`` the op algebra and module emission, and ``phases`` the pure
pooled split-phase planning."""

from pyfm.tasks.hadrons.highmode_v2 import config, phases, twopoint
from pyfm.tasks.hadrons.highmode_v2.strategy import (
    build_aggregator_params,
    build_input_params,
    build_lma_meson_field_chain,
    build_split_program,
    compare_outputs,
    create_outfile_catalog,
    entry_run_refs,
    is_cache_build_only,
    needed_ranll_gammas,
    pooled_placement,
    reorder_cross_contractions,
    sort_schedule,
)

__all__ = [
    "config",
    "phases",
    "twopoint",
    "build_aggregator_params",
    "build_input_params",
    "build_lma_meson_field_chain",
    "build_split_program",
    "compare_outputs",
    "create_outfile_catalog",
    "entry_run_refs",
    "is_cache_build_only",
    "needed_ranll_gammas",
    "pooled_placement",
    "reorder_cross_contractions",
    "sort_schedule",
]
