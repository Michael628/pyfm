"""HadronsMILC SpinTaste-API high-mode task support for ``hadrons_lma_new``.

Fully separate from the legacy ``highmode/`` package (ADR 0001 / D4):
``config`` owns the ``LMAHighModeConfig`` tree and its build hooks,
``strategy`` the per-entry emission / catalog / aggregation / comparison,
and ``twopoint`` the op algebra and module emission."""

from pyfm.tasks.hadrons.highmode_v2 import config, twopoint
from pyfm.tasks.hadrons.highmode_v2.strategy import (
    build_aggregator_params,
    build_input_params,
    build_lma_meson_field_chain,
    compare_outputs,
    create_outfile_catalog,
    needed_ranll_gammas,
    sort_schedule,
)

__all__ = [
    "config",
    "twopoint",
    "build_aggregator_params",
    "build_input_params",
    "build_lma_meson_field_chain",
    "compare_outputs",
    "create_outfile_catalog",
    "needed_ranll_gammas",
    "sort_schedule",
]
