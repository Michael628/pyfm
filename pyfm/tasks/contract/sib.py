"""SIB three-point contraction task (``contract_sib``).

Task 2 of the connected-SIB effort: contracts the ``hadrons_sib_mf``
split-noise outputs into the connected HVP three-point correlator
V(t1)·S(t2)·V(t3), decomposed into the eight L/H junction terms
(``SIB_TERM_LABELS``) with derived p-side blocks (the pair-basis identity,
``pyfm/a2a/sib_derive.py``) and world-selection noise handling. The task
emits raw per-term correlator outputs; per-term world-count normalization
and the eight-term sum into C3 happen at aggregation
(``term_normalize`` + ``sum`` actions) so the number of noise sources can
change between runs without invalidating raw outputs.

Module layout mirrors the ``contract`` task trio (contraction/diagram/
mesonloader): normalize/route reuse the ContractConfig implementations
(the diagrams/diagram_params selection logic is config-class agnostic);
everything else is SIB-owned.
"""

import typing as t

import pandas as pd

from pyfm import utils
from pyfm.a2a.types import SIBContractConfig, SIBDiagramConfig, SIB_TERM_LABELS
from pyfm.domain import Gamma
from pyfm.tasks.contract.contraction import normalize_params, route_params
from pyfm.tasks.register import register_task

# The Gamma families the block table covers (mirrors sib_mf's _SIB_FAMILIES).
_SIB_FAMILIES = frozenset(
    {Gamma.SCALAR_LOCAL, Gamma.VEC_LOCAL, Gamma.VEC_ONELINK}
)


def _sib_term_factors(noise: int) -> t.Dict[str, float]:
    """World-selection normalization factor per term (aggregation-side).

    One noise junction sums N diagonal world pairs; two junctions sum N(N-1)
    ordered off-diagonal pairs; the nnn term sums N(N-1)(N-2) pairwise
    distinct triples. Raw kernel outputs carry none of these factors (D7) —
    the aggregator applies them via the ``term_normalize`` action before
    summing the terms into C3. When the world selection for a term is
    empty (e.g. nnn with N=2: no pairwise-distinct triple exists), the
    factor is 0 — the term contributes nothing to the sum, which the raw
    kernel output reflects as an all-zero correlator.
    """
    n = noise

    def _factor(n_worlds_needed: int) -> float:
        if noise < n_worlds_needed:
            return 0.0
        denom = 1.0
        for k in range(n_worlds_needed):
            denom *= n - k
        return 1.0 / denom

    return {
        "lll": 1.0,
        "nll": _factor(1),
        "lnl": _factor(1),
        "lln": _factor(1),
        "nnl": _factor(2),
        "nln": _factor(2),
        "lnn": _factor(2),
        "nnn": _factor(3),
    }


def _oplist_input(operations) -> dict:
    """Serialize an OpList for the run-side input YAML (per-gamma form).

    ``OpList.from_dict`` accepts ``{gamma_name: {"mass": [...]}}`` per-op
    blocks, which round-trips per-op mass tuples exactly.
    """
    return {
        op.gamma.name.lower(): {"mass": list(op.mass)} for op in operations.op_list
    }


def _diagram_input(diagram: SIBDiagramConfig) -> t.Dict[str, t.Any]:
    """Run-side params for one SIB diagram.

    Outfiles pass through as built — the config builder's formatting pass
    has already resolved every formatting token (series/eigs/noise/cfg),
    leaving exactly the run-time tokens: the block grammar
    ``{leg_pair}/{mass}/{n_index}/{hp_index}/{gamma}`` for the accessor and
    ``{mass}/{gamma}`` on the correlator outfile (terms ride inside as
    frame labels).
    """
    return {
        "contraction_type": diagram.contraction_type.name,
        "operations": _oplist_input(diagram.operations),
        "mass": diagram.mass._asdict(),
        "defl_mass": diagram.defl_mass,
        "noise": diagram.noise,
        "t0": diagram.t0,
        "t_step": diagram.t_step,
        "n_slices": diagram.n_slices,
        "blocks": diagram.blocks,
        "tab": diagram.tab,
        "evalfile": diagram.evalfile,
        "outfile": diagram.outfile,
    }


def build_input_params(config: SIBContractConfig) -> t.Dict[str, t.Any]:
    """Emit the contraction-run input YAML (the inputgen "contract" branch
    serializes this dict; `pyfm contract run` rebuilds SIBContractConfig
    from it with ``normalized=True``)."""
    input_yaml = {
        "diagrams": {},
        "logging_level": config.logging_level,
        "runid": config.runid,
        "time": config.time,
    }
    for dlabel, diagram in config.diagrams.items():
        input_yaml["diagrams"][dlabel] = _diagram_input(diagram)
    return input_yaml


def _diagram_catalog(diagram: SIBDiagramConfig) -> pd.DataFrame:
    """Catalog the correlator outputs at (gamma, mass, term) granularity."""

    def generate_outfile_formatting():
        for op in diagram.op_list:
            for gamma_name in op.gamma.gamma_list:
                for mass_label in op.mass:
                    yield (
                        {
                            "gamma": [gamma_name],
                            "mass": [diagram.mass.to_string(mass_label, True)],
                            "term": list(SIB_TERM_LABELS),
                        },
                        diagram.outfile,
                    )

    return utils.io.catalog_files(generate_outfile_formatting())


def create_outfile_catalog(config: SIBContractConfig) -> pd.DataFrame:
    df = [_diagram_catalog(d) for d in config.diagrams.values()]
    return pd.concat(df, ignore_index=True)


def _diagram_aggregator(
    diagram: SIBDiagramConfig, time: int, average: bool
) -> t.Dict:
    """Aggregation params for one SIB diagram's per-term correlator files.

    Loads the raw per-term (T,T,T) outputs (dict_labels term/gamma, array
    order t1/t2/t3), applies the world-count factors via ``term_normalize``
    (a processor action; the factors come from the producer's noise count),
    sums the eight terms into C3, and writes the processed frame. The
    ``--average`` pathway is not supported for three-point data in v1 (the
    pandas time_average action is 2D-only); the flag is accepted and
    ignored with a warning.
    """
    if average:
        utils.get_logger().warning(
            "SIB three-point aggregation ignores --average in v1: the "
            "pandas time_average action is two-point-only."
        )
    infile = diagram.outfile
    suffix = "_avg" if average else ""
    outfile_stem = utils.io.get_processed_filename(
        infile.filestem, remove=["series"], suffix=suffix
    )

    t_order = ["t1", "t2", "t3"]
    actions = {
        "term_normalize": _sib_term_factors(diagram.noise),
        "term_sum": ["term"],
        # gamma/mass stay COLUMNS — the write step groups the processed
        # output by the outfile's {gamma}/{mass} tokens (one file per
        # group), like the two-point flow.
        "index": ["series.cfg", *t_order],
    }

    return {
        "diagram": {
            "logging_level": diagram.logging_level,
            "load_files": {
                "filestem": infile.filename,
                "regex": {"series": "[a-z]", "cfg": "[0-9]+"},
                "replacements": {"mass": _diagram_mass_tokens(diagram)},
                # {gamma} (and any other path-constant token) is recovered
                # from the written paths by wildcard expansion — the
                # aggregator's documented recipe (aggregator.py:310).
                "wildcard_fill": True,
                "dict_labels": ["term", "gamma"],
                "array_order": t_order,
                "array_labels": {t: f"0..{time - 1}" for t in t_order},
            },
            "out_files": {"filestem": outfile_stem},
            "actions": actions,
        }
    }


def _diagram_mass_tokens(diagram: SIBDiagramConfig) -> t.List[str]:
    """Mass token values the loader may find in outfile stems."""
    return [diagram.mass.to_string(m, True) for m in diagram.correlator_masses]


def build_aggregator_params(
    config: SIBContractConfig, average: bool
) -> t.Dict[str, t.Any]:
    agg_params = {"run": []}
    for dlabel, diagram in config.diagrams.items():
        agg_params[dlabel] = _diagram_aggregator(
            diagram, config.time, average
        )["diagram"]
        agg_params["run"].append(dlabel)
    return agg_params


def validate_sib_diagram(config: SIBDiagramConfig) -> None:
    """Validate one SIB diagram after construction."""
    if config.contraction_type.name != "SIB":
        raise ValueError(
            f"contract_sib diagram has contraction_type "
            f"{config.contraction_type.name!r}; only SIB is valid here."
        )
    if not config.op_list:
        raise ValueError(
            "SIB diagram has no operations — provide operations: {gamma: "
            "[scalar_local/vec_local/vec_onelink], mass: [...]}."
        )
    for op in config.op_list:
        if op.gamma not in _SIB_FAMILIES:
            raise ValueError(
                f"SIB operations gamma {op.gamma.name.lower()!r} is not an "
                "SIB family; allowed: scalar_local, vec_local, vec_onelink."
            )
        if not op.mass:
            raise ValueError(
                f"SIB operations gamma {op.gamma.name.lower()!r} has no mass."
            )
    for m in [*config.correlator_masses, config.defl_mass]:
        if m not in config.mass:
            raise ValueError(
                f"mass label {m!r} is not in the mass parameters "
                f"({sorted(config.mass.keys())})."
            )
    if config.noise < 2:
        raise ValueError(
            f"SIB contraction requires noise >= 2 (got {config.noise}): "
            "the cross-noise world selection needs at least two "
            "independent realizations."
        )
    if config.t0 < 0 or config.t_step < 1 or config.n_slices < 1:
        raise ValueError(
            f"SIB batch window invalid: t0={config.t0}, t_step="
            f"{config.t_step}, n_slices={config.n_slices}."
        )
    # The split-noise layout is required (D6): reject shared-mode stems
    # loudly — the derivation layer addresses per-world files.
    stem = config.blocks.filestem
    for token in ("{leg_pair}", "{mass}", "{n_index}", "{hp_index}"):
        if token not in stem:
            raise ValueError(
                f"SIB blocks filestem {stem!r} lacks the {token} token — "
                "the contraction requires the producer's split-noise "
                "output layout (sib.output.split_noise: true)."
            )
    if "{n_index}" not in config.tab.filestem:
        raise ValueError(
            f"SIB tab filestem {config.tab.filestem!r} lacks the "
            "{n_index} token — per-world tabs are the derivation input."
        )
    if "{gamma}" not in config.blocks.ext:
        raise ValueError(
            f"SIB blocks ext {config.blocks.ext!r} lacks the {{gamma}} "
            "token (the cfg_gamma_h5 grammar)."
        )
    for token in ("{mass}", "{gamma}"):
        if token not in config.outfile.filestem:
            raise ValueError(
                f"SIB outfile filestem {config.outfile.filestem!r} lacks "
                f"the {token} token (one correlator file per mass/gamma; "
                "terms ride inside as frame labels)."
            )


def validate_config(config: SIBContractConfig) -> None:
    """Validate the composite: non-empty diagrams, batch windows in extent."""
    if len(config.diagrams) == 0:
        raise ValueError("SIBContractConfig.diagrams must not be empty")
    for dlabel, diagram in config.diagrams.items():
        tb = diagram.t0 + (diagram.n_slices - 1) * diagram.t_step
        if not (0 <= diagram.t0 <= tb < config.time):
            raise ValueError(
                f"diagram {dlabel!r}: batch window [t0={diagram.t0}, "
                f"tB={tb}] stride {diagram.t_step} exceeds the lattice "
                f"time extent {config.time}."
            )


# Sub-config registration: SIBDiagramConfig gets the default route (absorbs
# the builder's _preprocessor slice); strict handler lookup returns None
# for it (the hadrons_sib_mf_batch precedent).
register_task("contract_sib_diagram", SIBDiagramConfig, validate=validate_sib_diagram)

# Register SIBContractConfig with contract_sib-owned hooks throughout.
register_task(
    "contract_sib",
    SIBContractConfig,
    build_input_params,
    build_aggregator_params,
    create_outfile_catalog,
    normalize_params,
    route_params,
    validate=validate_config,
)
