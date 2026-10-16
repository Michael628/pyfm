"""QED four-point contraction task (``contract_qed``).

Task 2 of the QED-HVP effort: contracts the ``hadrons_qed_mf`` outputs
into the connected four-point QED-HVP correlator (V·J·V·J / V·J·J·V),
decomposed into the sixteen L/H junction terms (``QED_TERM_LABELS``)
with fermion-world axis slicing and same-photon-world EM insertions
(``pyfm/a2a/qed_derive.py``, ``pyfm/a2a/contractions.py::qed_conn_4pt``).
The task emits raw per-(gamma, mass, momentum, em-world) per-term
outputs; world-count and photon normalization (1/((N)_k · n_em)) and the
em/term folds happen at aggregation (``term_normalize`` +
``term_sum``) so noise/world counts can change between runs without
invalidating raw outputs (D7).

Module layout mirrors ``contract_sib``: the diagram-selection half of
normalize and the whole of route reuse ``contraction.py``; QED owns a
normalize WRAPPER that additionally expands the ``em_seed`` template
into per-world ``em_seeds`` (series/cfg resolve here at broad-param
time; the run-side YAML carries the pre-expanded list, D10).
Everything else is QED-owned.
"""

import typing as t

import pandas as pd

from pyfm import utils
from pyfm.a2a.types import QEDContractConfig, QEDDiagramConfig, QED_TERM_LABELS
from pyfm.domain import Gamma
from pyfm.domain.blockkeys import EM_FAMILY_JA, EM_FAMILY_J1L
from pyfm.tasks.contract.contraction import normalize_params, route_params
from pyfm.tasks.register import register_task

# The vec Γ families the block table covers (mirrors qed_mf's _QED_FAMILIES).
_QED_FAMILIES = frozenset(
    {Gamma.SCALAR_LOCAL, Gamma.VEC_LOCAL, Gamma.VEC_ONELINK}
)
_EM_FAMILIES = frozenset({EM_FAMILY_JA, EM_FAMILY_J1L})


def normalize_qed_params(params: t.Dict) -> t.Dict:
    """Broad→canonical normalize: shared diagram selection + seed expansion.

    Runs the shared selection logic (``contraction.normalize_params``),
    then expands each selected diagram's ``em_seed`` template + ``n_em``
    into the pre-expanded ``em_seeds`` list (D10). series/cfg are present
    in the broad params (the task builder injects them per trajectory)
    and resolve HERE, so neither the run-side YAML nor the rebuilt
    aggregator config ever carries a template. The expansion builds NEW
    diagram dicts — the shared YAML tree is never mutated (create_task
    runs once per cfgno on the same tree; in-place pops would freeze the
    first cfg's seeds into every later one).
    """
    combined = params | params.get("_preprocessor", {})
    series = combined.get("series")
    cfg = combined.get("cfg")
    shared_seed = combined.get("em_seed")
    shared_n_em = combined.get("n_em")

    normalized = normalize_params(params)
    diagrams = normalized["diagrams"]

    expanded = {}
    for label, diagram in diagrams.items():
        if "em_seeds" in diagram:
            if "em_seed" in diagram:
                raise ValueError(
                    f"contract_qed diagram {label!r} carries both em_seed "
                    "and em_seeds — provide one (em_seed expands via n_em)."
                )
            expanded[label] = diagram
            continue
        template = diagram.get("em_seed", shared_seed)
        n_em = diagram.get("n_em", shared_n_em)
        if template is None or n_em is None:
            raise ValueError(
                f"contract_qed diagram {label!r} needs em_seed (template) "
                "and n_em to expand em_seeds — provide them on the diagram "
                "or in shared params."
            )
        if series is None or cfg is None:
            raise ValueError(
                "em_seed expansion requires series/cfg in the task params "
                "(the task builder injects them per trajectory)."
            )
        base = template.format(series=series, cfg=cfg)
        d = {k: v for k, v in diagram.items() if k != "em_seed"}
        d["em_seeds"] = [f"{base}_n{i}" for i in range(int(n_em))]
        expanded[label] = d
    normalized["diagrams"] = expanded
    return normalized


def _oplist_input(operations) -> dict:
    """Serialize an OpList for the run-side input YAML (per-gamma form)."""
    return {
        op.gamma.name.lower(): {"mass": list(op.mass)}
        for op in operations.op_list
    }


def _diagram_input(diagram: QEDDiagramConfig) -> t.Dict[str, t.Any]:
    """Run-side params for one QED diagram.

    Outfiles pass through as built — the builder's formatting pass has
    resolved every generation token (series/eigs/noise/cfg), leaving the
    run-time tokens: the block grammar
    ``{family}/{leg_pair}/{mass}/{em_index}`` + ``{file_key}`` ext for
    the accessor and ``{gamma}/{mass}/{momentum}/{em}`` on the
    correlator outfile (terms ride inside as frame labels).
    """
    return {
        "contraction_type": diagram.contraction_type.name,
        "ordering": list(diagram.ordering),
        "em": list(diagram.em),
        "operations": _oplist_input(diagram.operations),
        "mass": diagram.mass._asdict(),
        "defl_mass": diagram.defl_mass,
        "noise": diagram.noise,
        "n_em": diagram.n_em,
        "em_seeds": list(diagram.em_seeds),
        "momentum": [list(p) for p in diagram.momentum],
        "blocks": diagram.blocks,
        "tab": diagram.tab,
        "evalfile": diagram.evalfile,
        "outfile": diagram.outfile,
    }


def build_input_params(config: QEDContractConfig) -> t.Dict[str, t.Any]:
    """Emit the contraction-run input YAML (the inputgen "contract" branch
    serializes this dict; `pyfm contract run` rebuilds QEDContractConfig
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


def _diagram_catalog(diagram: QEDDiagramConfig) -> pd.DataFrame:
    """Catalog the correlator outputs at (gamma, mass, momentum, em, term)
    granularity — the raw-output axes (D4)."""

    def generate_outfile_formatting():
        for op in diagram.op_list:
            for gamma_name in op.gamma.gamma_list:
                for mass_label in op.mass:
                    for p_token in diagram.output_momenta:
                        yield (
                            {
                                "gamma": [gamma_name],
                                "mass": [
                                    diagram.mass.to_string(mass_label, True)
                                ],
                                "momentum": [p_token],
                                "em": [str(w) for w in range(diagram.n_em)],
                                "term": list(QED_TERM_LABELS),
                            },
                            diagram.outfile,
                        )

    return utils.io.catalog_files(generate_outfile_formatting())


def create_outfile_catalog(config: QEDContractConfig) -> pd.DataFrame:
    df = [_diagram_catalog(d) for d in config.diagrams.values()]
    return pd.concat(df, ignore_index=True)


def _qed_term_factors(noise: int, n_em: int) -> t.Dict[str, float]:
    """Per-term normalization factors (aggregation-side): 1/((N)_k · n_em).

    k = the term's noise-junction count; (N)_k the falling factorial over
    fermion-noise worlds; n_em the photon-world count, folded into the
    same factor (D5). Empty selections (k > N) give 0 — the raw kernel
    output is identically zero and must contribute nothing.
    """

    def _falling(k: int) -> float:
        if noise < k:
            return 0.0
        denom = 1.0
        for i in range(k):
            denom *= noise - i
        return 1.0 / denom

    return {
        term: _falling(term.count("n")) / n_em for term in QED_TERM_LABELS
    }


def _diagram_aggregator(
    diagram: QEDDiagramConfig, time: int, average: bool
) -> t.Dict:
    """Aggregation params for one QED diagram's raw per-term files.

    Loads the raw per-(gamma, mass, momentum, em) outputs (dict_labels
    term + path-recovered momentum/em via wildcard_fill; array order
    t1..t4), applies 1/((N)_k · n_em) via ``term_normalize``, folds em
    and term via ``term_sum`` (momentum survives — D3), indexes, and
    writes the processed frame. The processed stem strips the literal
    ``_em{em}`` fragment (validation enforces its presence — D4/D5).
    ``--average`` is rejected loudly (D12): the pandas time_average
    action is hard 2D-only and a 4D frame would be silently mangled.
    """
    if average:
        raise ValueError(
            "--average is not supported for QED four-point aggregation: "
            "the pandas time_average action is two-point-only."
        )
    infile = diagram.outfile
    outfile_stem = utils.io.get_processed_filename(
        infile.filestem, remove=["series"]
    )
    if "_em{em}" not in outfile_stem:
        raise ValueError(
            f"QED outfile filestem {infile.filestem!r} lacks the literal "
            "'_em{em}' fragment — the processed stem is derived by "
            "stripping it after the em fold."
        )
    outfile_stem = outfile_stem.replace("_em{em}", "")

    t_order = ["t1", "t2", "t3", "t4"]
    actions = {
        "term_normalize": _qed_term_factors(diagram.noise, diagram.n_em),
        "term_sum": ["em", "term"],
        "index": ["series.cfg", *t_order],
    }

    return {
        "diagram": {
            "logging_level": diagram.logging_level,
            "load_files": {
                "filestem": infile.filename,
                "regex": {"series": "[a-z]", "cfg": "[0-9]+"},
                "replacements": {"mass": _diagram_mass_tokens(diagram)},
                "wildcard_fill": True,
                "dict_labels": ["term", "momentum", "em"],
                "array_order": t_order,
                "array_labels": {t: f"0..{time - 1}" for t in t_order},
            },
            "out_files": {"filestem": outfile_stem},
            "actions": actions,
        }
    }


def _diagram_mass_tokens(diagram: QEDDiagramConfig) -> t.List[str]:
    """Mass token values the loader may find in outfile stems."""
    return [
        diagram.mass.to_string(m, True) for m in diagram.correlator_masses
    ]


def build_aggregator_params(
    config: QEDContractConfig, average: bool
) -> t.Dict[str, t.Any]:
    agg_params = {"run": []}
    for dlabel, diagram in config.diagrams.items():
        agg_params[dlabel] = _diagram_aggregator(
            diagram, config.time, average
        )["diagram"]
        agg_params["run"].append(dlabel)
    return agg_params


def validate_qed_diagram(config: QEDDiagramConfig) -> None:
    """Validate one QED diagram after construction."""
    if config.contraction_type.name != "QED":
        raise ValueError(
            f"contract_qed diagram has contraction_type "
            f"{config.contraction_type.name!r}; only QED is valid here."
        )
    if len(config.ordering) != 4 or config.ordering[0] != "vec":
        raise ValueError(
            f"QED ordering {list(config.ordering)!r} must have four entries "
            "starting at 'vec' (the canonical ring start)."
        )
    if sorted(config.ordering) != ["em", "em", "vec", "vec"]:
        raise ValueError(
            f"QED ordering {list(config.ordering)!r} must contain exactly "
            "two 'vec' and two 'em' vertices."
        )
    for f in config.em:
        if f not in _EM_FAMILIES:
            raise ValueError(
                f"QED em family {f!r} is not an EM family; allowed: "
                f"{sorted(_EM_FAMILIES)}."
            )
    if not config.op_list:
        raise ValueError(
            "QED diagram has no operations — provide operations: {gamma: "
            "[scalar_local/vec_local/vec_onelink], mass: [...]}."
        )
    for op in config.op_list:
        if op.gamma not in _QED_FAMILIES:
            raise ValueError(
                f"QED operations gamma {op.gamma.name.lower()!r} is not a "
                "QED family; allowed: scalar_local, vec_local, vec_onelink."
            )
        if not op.mass:
            raise ValueError(
                f"QED operations gamma {op.gamma.name.lower()!r} has no mass."
            )
    for m in [*config.correlator_masses, config.defl_mass]:
        if m not in config.mass:
            raise ValueError(
                f"mass label {m!r} is not in the mass parameters "
                f"({sorted(config.mass.keys())})."
            )
    if config.noise < 2:
        raise ValueError(
            f"QED contraction requires noise >= 2 (got {config.noise}): "
            "the cross-noise world selection needs at least two "
            "independent realizations."
        )
    if config.n_em < 1:
        raise ValueError(f"QED n_em must be >= 1 (got {config.n_em}).")
    if len(config.em_seeds) != config.n_em:
        raise ValueError(
            f"em_seeds has {len(config.em_seeds)} entries but n_em = "
            f"{config.n_em} — one seed per photon world."
        )
    # Token grammar (the validate_sib_diagram transpose)
    stem = config.blocks.filestem
    for token in ("{family}", "{leg_pair}", "{mass}", "{em_index}"):
        if token not in stem:
            raise ValueError(
                f"QED blocks filestem {stem!r} lacks the {token} token — "
                "the contraction requires the producer's qed block output "
                "layout."
            )
    if "{file_key}" not in config.blocks.ext:
        raise ValueError(
            f"QED blocks ext {config.blocks.ext!r} lacks the {{file_key}} "
            "token (the cfg_qed_h5 grammar)."
        )
    if "{gamma}" not in config.tab.ext:
        raise ValueError(
            f"QED tab ext {config.tab.ext!r} lacks the {{gamma}} token "
            "(the cfg_gamma_h5 grammar)."
        )
    for token in ("{mass}", "{gamma}", "{momentum}", "{em}"):
        if token not in config.outfile.filestem:
            raise ValueError(
                f"QED outfile filestem {config.outfile.filestem!r} lacks "
                f"the {token} token (one raw correlator file per mass/"
                "gamma/momentum/em-world; terms ride inside as frame "
                "labels)."
            )
    if "_em{em}" not in config.outfile.filestem:
        raise ValueError(
            f"QED outfile filestem {config.outfile.filestem!r} lacks the "
            "literal '_em{em}' fragment — aggregation strips exactly that "
            "fragment when folding the em axis."
        )


def validate_config(config: QEDContractConfig) -> None:
    """Validate the composite: non-empty diagrams."""
    if len(config.diagrams) == 0:
        raise ValueError("QEDContractConfig.diagrams must not be empty")


# Sub-config registration: QEDDiagramConfig gets the default route; strict
# handler lookup returns None for it (the contract_sib_diagram precedent).
register_task(
    "contract_qed_diagram", QEDDiagramConfig, validate=validate_qed_diagram
)

# Register QEDContractConfig with contract_qed-owned hooks throughout.
# NOTE: the normalize hook MUST pass as a kwarg — register_task routes
# positional functions by __name__ against {normalize_params, route_params,
# postprocess_config}, so a positional ``normalize_qed_params`` would be
# silently dropped (the register.py _HOOK_NAME_MAP trap).
register_task(
    "contract_qed",
    QEDContractConfig,
    build_input_params,
    build_aggregator_params,
    create_outfile_catalog,
    normalize_params=normalize_qed_params,
    route_params=route_params,
    validate=validate_config,
)
