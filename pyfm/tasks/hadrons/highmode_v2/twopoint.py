"""Canonical-schema two-point generators, fully separate from the legacy
``highmode/`` package (ADR 0001, design D4).

Owns the op algebra (``TwoPointOp``, ``_AXIAL_GAMMAS``,
``contraction_gen``, ``quark_gen``) typed on ``LMAHighModeConfig``, plus
the SpinTaste-module builder and quark/contraction emission. The guess
chain comes from the explicit ``cg.precon`` setting (D8), independent of
which pairs are contracted; every module name the entry owns is routed
through ``config.module_name`` (D6), so an unkeyed (label ``""``) entry
reproduces the legacy module-name grammar byte-for-byte while keyed
entries get statistically independent noise and collision-free modules.
"""

import itertools
import typing as t

from pyfm.domain import Gamma, MassDict, OpList
from pyfm.tasks.hadrons.types import HadronsInput, SourceRef
import pyfm.tasks.hadrons.modules as hadmods
from pyfm.tasks.hadrons.highmode_v2.config import (
    LMAHighModeConfig,
    PreconMode,
)


_AXIAL_GAMMAS = frozenset(
    {
        Gamma.AXIAL_VEC_ONELINK,
        Gamma.AXIAL_VEC_LOCAL,
        Gamma.AXIAL_FOURVEC_ONELINK,
        Gamma.AXIAL_FOURVEC_LOCAL,
    }
)

# G5 hermiticity for connected two-point functions
# -----------------------------------------------
# Every propagator here is solved with apply_g5=True, so the requested
# op.gamma is effectively multiplied by gamma5. The contract partner (the
# "antiquark" side of a TwoPointOp) is chosen to exploit this:
#   * PION_LOCAL (= G5_G5) becomes the identity once gamma5 is applied.
#   * IDENTITY (= G1_G1) becomes gamma5 once applied.
# Pairing a non-axial operator with a PION_LOCAL antiquark therefore yields the
# standard g5-hermitic contraction. Axial operators instead pair with an
# IDENTITY antiquark, and because the quark side reuses the same non-axial
# VEC/FOURVEC propagator (see quark_gen / contraction_gen), a single VEC solve
# produces both the vector and the axial correlators depending on which
# antiquark it is contracted against.


class TwoPointOp(t.NamedTuple):
    class Op(t.NamedTuple):
        gamma: Gamma
        mass: str
        solver: str
        apply_g5: bool
        precon: str | None = None

    quark: Op
    antiquark: Op
    sink: Op

    def mass_label(self, masses: MassDict) -> str:
        return "_m".join(
            dict.fromkeys(
                masses.to_string(m, True)
                for m in [self.quark.mass, self.antiquark.mass]
            )
        )

    @property
    def solver_label(self) -> str:
        return "_".join(dict.fromkeys([self.quark.solver, self.antiquark.solver]))


def sink_name(config: LMAHighModeConfig) -> str:
    """Entry-owned sink module name (design doc Labels section)."""
    return config.module_name("sink")


def noise_rw_name(config: LMAHighModeConfig, ref: SourceRef) -> str:
    """Entry-owned per-source noise module name."""
    return config.module_name(f"noise_{ref.label}")


def solver_module_name(config: LMAHighModeConfig, solver: str, mass: str) -> str:
    """Entry-owned CG/low-mode solver module name.

    The user's ``solver_name`` template is filled per solve (``label`` is
    offered to the template too — a template may reference it explicitly,
    on top of the build-time partial resolution) and the label is ALSO
    always prepended to the formatted result, so two entries with
    identical templates still produce disjoint solver modules even if
    neither template mentions ``{label}`` (D6).
    """
    return config.module_name(
        config.solver_name.format(solver=solver, mass=mass, label=config.label)
    )


def meson_field_producer_name(
    config: LMAHighModeConfig, gamma_label: str, mass: str, ref: SourceRef
) -> str:
    """Registered ``StagLMAMesonFieldProp`` module name for (mass, gamma, ref).

    One producer per source slice (``tA=tB=ref.t0``). A single-slice
    instance publishes its ``TGammaMap`` under its own module name, so
    this is also the propagator object contractions and the ama guess
    chain reference. Grid labels (``t{t0}``) already carry the time;
    biased labels (``n{i}``) get ``_t{t0}`` appended, and keep ``n{i}``
    so with-replacement draws of the same ``t0`` stay distinct.
    """
    base = config.module_name(f"quark_ranLL_{gamma_label}_mass_{mass}")
    if config.sources_config.biased_config is None:
        return f"{base}_{ref.label}"
    return f"{base}_{ref.label}_t{ref.t0}"


def quark_name(
    config: LMAHighModeConfig, solver: str, gamma_label: str, mass: str, ref: SourceRef
) -> str:
    """Entry-owned quark-propagator module/reference name.

    Shared by ``build_quarks`` (module definition) and
    ``build_contractions`` (reference, rebuilt independently for both the
    quark and antiquark side) so the two can never drift. The ``ranLL``
    solver under ``meson_field`` low modes has no GaugeProp middleman —
    ``build_quarks`` skips it entirely — so any reference to it (an ama
    guess, or either side of a contraction) instead resolves to the
    per-slice meson-field producer (:func:`meson_field_producer_name`),
    whose module name is its published output.
    """
    if solver == "ranLL" and config.use_meson_field:
        return meson_field_producer_name(config, gamma_label, mass, ref)
    return config.module_name(f"quark_{solver}_{gamma_label}_mass_{mass}_{ref.label}")


def contraction_gen(
    config: LMAHighModeConfig,
) -> t.Iterator[t.Tuple[OpList.Op, TwoPointOp]]:
    """Generates required contractions for the requested two-point functions.

    Defaults to g5 hermiticity by pairing each operator with a PION_LOCAL
    antiquark; axial gammas are the exception and pair with IDENTITY. The quark
    side reuses the non-axial VEC/FOURVEC propagator so a single solve serves
    both axial and non-axial correlators. See the module-level G5_HERMITICITY
    note and quark_gen for the matching propagator set. Ported verbatim from
    ``highmode/twopoint.py`` (only the config type changed).
    """
    solver_labels = config.get_solver_labels(skip_cross=True)
    for op in config.operations:
        for slabel1, slabel2, mlabel1, mlabel2 in itertools.product(
            solver_labels,
            solver_labels,
            op.mass,
            op.mass,
        ):
            if mlabel1 < mlabel2:
                continue

            if not config.mass_cross_terms and mlabel1 != mlabel2:
                continue

            # slabel1 drives the antiquark, slabel2 the quark, and dset names
            # are quark-first (TwoPointOp.solver_label) — so the pair check
            # takes (quark=slabel2, antiquark=slabel1) and TIERED keeps the
            # pairs whose dset is `ranLL_ama`.
            if not config.admits_solve_pair(slabel2, slabel1):
                continue

            common1 = dict(
                apply_g5=True,
                mass=mlabel1,
                solver=slabel1,
            )
            common2 = dict(
                apply_g5=True,
                mass=mlabel2,
                solver=slabel2,
            )
            # Set antiquark: the g5-hermiticity contract partner. Axial gammas
            # pair with IDENTITY (which is gamma5 once apply_g5 is applied);
            # everything else pairs with PION_LOCAL (the identity under g5).
            # See the module-level G5_HERMITICITY note.
            is_axial = op.gamma in _AXIAL_GAMMAS
            antiquark = TwoPointOp.Op(
                gamma=Gamma.IDENTITY if is_axial else Gamma.PION_LOCAL, **common1
            )
            # Set quark: axial gammas reuse the non-axial counterpart
            # propagator. Paired with the IDENTITY antiquark above this yields
            # the axial correlator from the very same VEC/FOURVEC solve used for
            # the non-axial correlator. Every other gamma solves its own gamma.
            match op.gamma:
                case Gamma.AXIAL_VEC_LOCAL:
                    quark = TwoPointOp.Op(gamma=Gamma.VEC_LOCAL, **common2)
                case Gamma.AXIAL_FOURVEC_LOCAL:
                    quark = TwoPointOp.Op(gamma=Gamma.FOURVEC_LOCAL, **common2)
                case Gamma.AXIAL_VEC_ONELINK:
                    quark = TwoPointOp.Op(gamma=Gamma.VEC_ONELINK, **common2)
                case Gamma.AXIAL_FOURVEC_ONELINK:
                    quark = TwoPointOp.Op(gamma=Gamma.FOURVEC_ONELINK, **common2)
                case _:
                    quark = TwoPointOp.Op(gamma=op.gamma, **common2)
            # Set sink
            sink = TwoPointOp.Op(gamma=op.gamma, **common2)

            yield op, TwoPointOp(
                quark=quark,
                antiquark=antiquark,
                sink=sink,
            )


def quark_gen(config: LMAHighModeConfig) -> t.Iterator[TwoPointOp.Op]:
    """Generates exactly the propagators the requested contractions consume.

    Demand-driven: the propagator set is derived from ``contraction_gen``'s
    emitted quark/antiquark sides, so a solve is emitted iff some contraction
    references it. Under TIERED the HH diagonal is dropped, leaving the LH
    cross (``ranLL_ama``) as the only contraction consuming an ama (CG)
    propagator — its antiquark side uses the contract gamma
    (PION_LOCAL/IDENTITY) only, so the op-gamma CG solves have no consumer
    and are skipped.

    The guess chain is the explicit ``cg.precon`` setting (D8), independent
    of which pairs are contracted — ``chain`` (the old chained-solves
    default) guesses the nearest earlier emitted base solver; ``each`` (the
    old ``False``) guesses ``ranLL`` directly;
    ``none`` provides no guess. ``ranLL`` itself never takes a guess, and a
    guess never references a skipped module (per (gamma, mass) the required
    set is a prefix of the base solver list). Emission order is
    deterministic and mass-major, base-solver order preserved within a mass
    so precons precede their consumers.
    """
    solver_labels = config.get_solver_labels(skip_cross=True)

    required: t.Dict[t.Tuple[str, Gamma, str], None] = {}
    for _, con in contraction_gen(config):
        for side in (con.quark, con.antiquark):
            required.setdefault((side.solver, side.gamma, side.mass))

    for solver, gamma, mass in sorted(
        required, key=lambda k: (k[2], solver_labels.index(k[0]), k[1].name)
    ):
        precon = None
        if solver != "ranLL" and config.cg_config is not None:
            mode = config.cg_config.precon
            if mode is PreconMode.CHAIN:
                # Chained: nearest earlier base solver whose
                # same-(gamma, mass) propagator is also emitted (byte-identical
                # to the old chained-solves path).
                for earlier in reversed(solver_labels[: solver_labels.index(solver)]):
                    if (earlier, gamma, mass) in required:
                        precon = earlier
                        break
            elif mode is PreconMode.EACH and ("ranLL", gamma, mass) in required:
                # Independent: every CG solve guesses ranLL directly. The LL
                # diagonal is always admitted, so an emitted CG solve's
                # same-(gamma, mass) ranLL is emitted too; without low modes
                # no guess is provided.
                precon = "ranLL"
        yield TwoPointOp.Op(
            gamma=gamma, mass=mass, solver=solver, apply_g5=True, precon=precon
        )


def build_spintaste_modules(
    config: LMAHighModeConfig,
) -> t.Tuple[t.Dict[str, t.Dict], t.Dict[Gamma, str]]:
    """Config-scoped SpinTaste module set, built once and shared across
    ``build_quarks``/``build_contractions`` and the meson-field producers.

    One module per unique SOLVED gamma (``quark_gen``'s required set, keyed
    by ``Gamma`` value alone — SpinTaste modules carry no mass/solver axis),
    named ``spintaste_{glabel}`` under ``config.module_name``. This module
    set only has to satisfy ``sinkGammas``/``source`` key matching
    (``Meson::checkKeys`` — see ``MContraction/Meson.hpp``): the ``sink``
    (``con.antiquark``) side is matched separately via
    ``MUtilities::GammaMapElement`` in ``build_contractions``, bypassing
    ``checkKeys`` for that side entirely.

    Every axial ``op.gamma`` (``contraction_gen``'s *requested* gamma, which
    ``quark_gen`` already substitutes away on the solve side) gets its OWN
    additional module: same physics (its own raw gamma string,
    ``apply_g5="true"``) but ``labels`` overridden to the shared quark-side
    module's own labels, so the axial correlator's ``sinkGammas`` keys land
    on the SAME propagator map entries its (non-axial) source solve actually
    published (grounded in ``StagGamma::getLabelName()``'s fold-invariance).
    """
    modules: t.Dict[str, t.Dict] = {}
    names: t.Dict[Gamma, str] = {}

    def _add(gamma: Gamma, labels: str = "") -> None:
        if gamma in names:
            return
        name = config.module_name(f"spintaste_{gamma.name.lower()}")
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
    config: LMAHighModeConfig,
    run_refs: t.List[SourceRef],
    spintaste_names: t.Dict[Gamma, str],
) -> HadronsInput:
    """Canonical-schema quark emission.

    Same demand-driven emission (``quark_gen``, meson-field ``ranLL`` skip,
    split-grid subgrid tagging) as the previous version — every entry-owned
    name (quark module, per-source noise, solver module, guess reference)
    is routed through ``config.module_name``, and the solver module name is
    the user's ``solver_name`` template (label-resolved at build time)
    filled per solve and then prefixed (D6).
    """
    modules = {}
    for ref in run_refs:
        for op in set(quark_gen(config)):
            if op.solver == "ranLL" and config.use_meson_field:
                # Meson-field mode: no GaugeProp middleman — the eager
                # producer emitted by the strategy's meson-field chain
                # already creates the outputs this module would have
                # solved for.
                continue
            glabel = op.gamma.name.lower()
            quark = quark_name(config, op.solver, glabel, op.mass, ref)
            source = noise_rw_name(config, ref)
            solver = solver_module_name(config, op.solver, op.mass)

            if op.precon:
                guess = quark_name(config, op.precon, glabel, op.mass, ref)
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
    config: LMAHighModeConfig,
    run_refs: t.List[SourceRef],
    spintaste_names: t.Dict[Gamma, str],
) -> HadronsInput:
    """Canonical-schema contraction emission.

    Same naming/split-grid logic as the previous version with entry-owned
    names (contraction module, antiquark element, per-source noise shift)
    routed through ``config.module_name``. The correlator OUTPUT filestem
    is deliberately NOT prefixed: output files are user-owned (bind a
    distinct files label per entry, e.g. ``files.bias_modes``) — only
    module names act as noise identity (D6). ``sink`` is a
    ``gamma_map_element`` bridge, not the raw antiquark TGammaMap — see
    ``hadmods.gamma_map_element`` for the checkKeys bypass rationale.
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
            quark = quark_name(config, con_set.quark.solver, quark_glabel, mlabel1, ref)
            antiquark = quark_name(
                config, con_set.antiquark.solver, antiquark_glabel, mlabel2, ref
            )

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

            output = config.output_config.file.filestem.format(
                mass=mass_output, dset=solver_label, gamma_label=glabel, tsource=ref.axis
            )

            if config.subgrid_ranks is not None and (
                "ama" in con_set.quark.solver
                or "ama" in con_set.antiquark.solver
            ):
                subgrid = ref.t0 % config.subgrid_ranks
            else:
                subgrid = None

            name = config.module_name(
                f"corr_{solver_label}_{glabel}_{mass_label}_{ref.label}"
            )
            modules[name] = hadmods.prop_contract_v2(
                name=name,
                source=quark,
                sink=antiquark_element,
                sink_fn=sink_name(config),
                source_shift=f"{noise_rw_name(config, ref)}_shift",
                sink_gammas=spintaste_names[op.gamma],
                output=output,
                subgrid=subgrid,
            )
    return HadronsInput(modules=modules, schedule=list(modules.keys()))
