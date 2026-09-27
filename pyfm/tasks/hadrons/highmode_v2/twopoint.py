import typing as t

from pyfm.tasks.hadrons.types import HadronsInput
import pyfm.tasks.hadrons.modules as hadmods
from pyfm.domain import Gamma
from pyfm.tasks.hadrons.types import HighModeConfig, SourceRef
from pyfm.tasks.hadrons.highmode.twopoint import (
    _AXIAL_GAMMAS,
    quark_gen,
    contraction_gen,
)


def build_spintaste_modules(
    config: HighModeConfig,
) -> t.Tuple[t.Dict[str, t.Dict], t.Dict[Gamma, str]]:
    """Config-scoped SpinTaste module set, built once and shared across
    ``build_quarks``/``build_contractions``/``build_lma_meson_field_chain``.

    One module per unique SOLVED gamma (``quark_gen``'s required set, keyed
    by ``Gamma`` value alone — SpinTaste modules carry no mass/solver axis),
    named ``spintaste_{glabel}``. This module set only has to satisfy
    ``sinkGammas``/``source`` key matching (``Meson::checkKeys`` — see
    ``MContraction/Meson.hpp``): the ``sink`` (``con.antiquark``) side is
    matched separately via ``MUtilities::GammaMapElement`` in
    ``build_contractions``, bypassing ``checkKeys`` for that side entirely
    (a bare object is "used unchanged for every gamma", no key-set check).

    Every axial ``op.gamma`` (``contraction_gen``'s *requested* gamma, which
    ``quark_gen`` already substitutes away on the solve side, e.g.
    ``AXIAL_VEC_LOCAL`` -> ``VEC_LOCAL``) gets its OWN additional module:
    same physics (its own raw gamma string, ``apply_g5="true"`` —
    unchanged from today's convention) but ``labels`` overridden to the
    shared quark-side module's own labels, so the axial correlator's
    ``sinkGammas`` keys land on the SAME propagator map entries its
    (non-axial) source solve actually published (see the design's
    SpinTaste ``labels`` wiring decision, grounded in
    ``StagGamma::getLabelName()``'s fold-invariance: the default label of
    a module built from a gamma's own raw ``gamma_string`` is exactly that
    gamma's own ``gamma_list``, regardless of ``apply_g5``).
    """
    modules: t.Dict[str, t.Dict] = {}
    names: t.Dict[Gamma, str] = {}

    def _add(gamma: Gamma, labels: str = "") -> None:
        if gamma in names:
            return
        name = f"spintaste_{gamma.name.lower()}"
        names[gamma] = name
        modules[name] = hadmods.spin_taste(
            name=name,
            gammas=gamma.gamma_string,
            gauge="" if gamma.local else config.shift_gauge_name,
            apply_g5="true",
            labels=labels,
        )

    for op in quark_gen(config):
        _add(op.gamma)

    for op, con in contraction_gen(config):
        if op.gamma in _AXIAL_GAMMAS:
            _add(op.gamma, labels=" ".join(con.quark.gamma.gamma_list))

    return modules, names


def build_quarks(
    config: HighModeConfig,
    run_refs: t.List[SourceRef],
    spintaste_names: t.Dict[Gamma, str],
) -> HadronsInput:
    """Canonical-schema counterpart of ``highmode.twopoint.build_quarks``.

    Same demand-driven emission (``quark_gen``, load-mode ``ranLL`` skip,
    split-grid subgrid tagging) — only the module call differs:
    ``quark_prop_v2``'s ``gammas`` references the shared SpinTaste module
    (``spintaste_names[op.gamma]``) instead of an inline gamma string.
    """
    modules = {}
    for ref in run_refs:
        for op in set(quark_gen(config)):
            if op.solver == "ranLL" and config.low_mode_method == "load":
                # Load mode: no GaugeProp middleman — the eager producer
                # emitted by build_lma_meson_field_chain already creates
                # the outputs this module would have solved for.
                continue
            glabel = op.gamma.name.lower()
            quark = f"quark_{op.solver}_{glabel}_mass_{op.mass}_{ref.label}"
            source = f"noise_{ref.label}"
            solver = config.solver_name.format(solver=op.solver, mass=op.mass)

            if op.precon:
                guess = f"quark_{op.precon}_{glabel}_mass_{op.mass}_{ref.label}"
            else:
                guess = ""

            if config.subgrid_ranks is not None and "ama" in op.solver:
                subgrid = ref.t0 % config.subgrid_ranks
            else:
                subgrid = None

            modules[quark] = hadmods.quark_prop_v2(
                name=quark,
                source=source,
                solver=solver,
                guess=guess,
                gammas=spintaste_names[op.gamma],
                subgrid=subgrid,
            )

    return HadronsInput(modules=modules, schedule=list(modules.keys()))


def build_contractions(
    config: HighModeConfig,
    run_refs: t.List[SourceRef],
    spintaste_names: t.Dict[Gamma, str],
) -> HadronsInput:
    """Canonical-schema counterpart of ``highmode.twopoint.build_contractions``.

    Same naming/split-grid logic — only the module call differs:
    ``prop_contract_v2``'s ``sink_gammas`` references the shared SpinTaste
    module keyed by the ORIGINAL requested ``op.gamma`` (axial or not) —
    ``spintaste_names`` already carries the axial-relabeled module for
    axial ops (built by ``build_spintaste_modules``), so this lookup is
    uniform regardless of axial-ness. ``sink`` is NOT the antiquark
    propagator directly: ``con_set.antiquark`` is always single-gamma
    (``PION_LOCAL``/``IDENTITY``) but ``StagGaugeProp`` still publishes it
    as a (one-entry) ``TGammaMap``, and ``Meson::checkKeys`` requires every
    ``sinkGammas`` key present in ANY param that is itself a ``TGammaMap``
    — impossible to satisfy for multi-component ``sinkGammas``
    (``VEC``/``FOURVEC``/axial ops) by relabeling alone. A
    ``gamma_map_element`` bridge extracts the antiquark's one entry (its
    own default label) onto a bare object, so ``sink`` skips ``checkKeys``
    entirely (bare params are "used unchanged for every gamma" — exactly
    the antiquark's actual role: the same single propagator contracted
    against every sink-gamma component).
    """
    modules = {}
    antiquark_elements: t.Dict[str, str] = {}
    for ref in run_refs:
        for op, con_set in set(contraction_gen(config)):
            glabel = op.gamma.name.lower()
            quark_glabel = con_set.quark.gamma.name.lower()
            antiquark_glabel = con_set.antiquark.gamma.name.lower()
            mlabel1 = con_set.quark.mass
            mlabel2 = con_set.antiquark.mass
            quark = (
                f"quark_{con_set.quark.solver}_{quark_glabel}_mass_{mlabel1}_{ref.label}"
            )
            antiquark = f"quark_{con_set.antiquark.solver}_{antiquark_glabel}_mass_{mlabel2}_{ref.label}"

            if antiquark not in antiquark_elements:
                element = f"{antiquark}_elem"
                antiquark_elements[antiquark] = element
                modules[element] = hadmods.gamma_map_element(
                    name=element,
                    map=antiquark,
                    label=con_set.antiquark.gamma.gamma_list[0],
                )
            antiquark_element = antiquark_elements[antiquark]

            mass_output = con_set.mass_label(config.mass)
            solver_label = con_set.solver_label

            if mlabel1 == mlabel2:
                mass_label = f"mass_{mlabel1}"
            else:
                mass_label = f"mass_{mlabel1}_mass_{mlabel2}"

            output = config.high_modes.filestem.format(
                mass=mass_output, dset=solver_label, gamma_label=glabel, tsource=ref.axis
            )

            if config.subgrid_ranks is not None and (
                "ama" in con_set.quark.solver
                or "ama" in con_set.antiquark.solver
            ):
                subgrid = ref.t0 % config.subgrid_ranks
            else:
                subgrid = None

            name = f"corr_{solver_label}_{glabel}_{mass_label}_{ref.label}"
            modules[name] = hadmods.prop_contract_v2(
                name=name,
                source=quark,
                sink=antiquark_element,
                sink_fn="sink",
                source_shift=f"noise_{ref.label}_shift",
                sink_gammas=spintaste_names[op.gamma],
                output=output,
                subgrid=subgrid,
            )
    return HadronsInput(modules=modules, schedule=list(modules.keys()))
