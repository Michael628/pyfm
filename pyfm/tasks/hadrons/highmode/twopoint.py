import typing as t
import itertools

from pyfm.tasks.hadrons.types import HadronsInput
import pyfm.tasks.hadrons.modules as hadmods
from pyfm.domain import OpList, Gamma, MassDict
from pyfm.tasks.hadrons.types import CrossTerms, HighModeConfig


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


def quark_gen(config: HighModeConfig) -> t.Iterator[TwoPointOp.Op]:
    """Generates required propagators for the requested two-point functions

    Note:
    - PION_LOCAL requires only the identity gamma operation (equivalent to G5_G5 with apply_g5=True)
    - (AXIAL_)VEC and (AXIAL_)FOURVEC operations require a vector/four-vector gamma solve paired with a (identity)G5_G5 solve
    - Axial gammas contract against IDENTITY and reuse the non-axial VEC/FOURVEC
      solve; every other gamma contracts against PION_LOCAL. See the module-level
      G5_HERMITICITY note.
    """
    solver_labels = config.get_solver_labels(skip_cross=True)
    guess_solver_labels = solver_labels[:-1].copy()
    guess_solver_labels.insert(0, None)
    solver_zip = list(zip(solver_labels, guess_solver_labels))

    for op in config.operations:
        for (slabel, slabel_guess), mlabel in itertools.product(solver_zip, op.mass):
            common = dict(
                apply_g5=True,
                mass=mlabel,
                solver=slabel,
                precon=slabel_guess,
            )
            # Only axial gammas are *translated*: they contract against
            # IDENTITY (instead of PION_LOCAL) and their vector solve uses the
            # non-axial counterpart gamma. Every other gamma contracts with
            # PION_LOCAL and solves its own gamma.
            is_axial = op.gamma in _AXIAL_GAMMAS

            # First (contract) solve: IDENTITY for axial gammas, PION_LOCAL
            # otherwise.
            yield TwoPointOp.Op(
                gamma=Gamma.IDENTITY if is_axial else Gamma.PION_LOCAL, **common
            )

            # Second solve: axial gammas are translated to their non-axial
            # counterpart; every other gamma solves its own gamma.
            match op.gamma:
                case Gamma.AXIAL_VEC_LOCAL:
                    yield TwoPointOp.Op(gamma=Gamma.VEC_LOCAL, **common)
                case Gamma.AXIAL_FOURVEC_LOCAL:
                    yield TwoPointOp.Op(gamma=Gamma.FOURVEC_LOCAL, **common)
                case Gamma.AXIAL_VEC_ONELINK:
                    yield TwoPointOp.Op(gamma=Gamma.VEC_ONELINK, **common)
                case Gamma.AXIAL_FOURVEC_ONELINK:
                    yield TwoPointOp.Op(gamma=Gamma.FOURVEC_ONELINK, **common)
                case _:
                    yield TwoPointOp.Op(gamma=op.gamma, **common)


def contraction_gen(
    config: HighModeConfig,
) -> t.Iterator[t.Tuple[OpList.Op, TwoPointOp]]:
    """Generates required contractions for the requested two-point functions.

    Defaults to g5 hermiticity by pairing each operator with a PION_LOCAL
    antiquark; axial gammas are the exception and pair with IDENTITY. The quark
    side reuses the non-axial VEC/FOURVEC propagator so a single solve serves
    both axial and non-axial correlators. See the module-level G5_HERMITICITY
    note and quark_gen for the matching propagator set.
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

            if (
                config.cross_terms not in (CrossTerms.MASS, CrossTerms.ALL)
                and mlabel1 != mlabel2
            ):
                continue

            if (
                config.cross_terms not in (CrossTerms.SOLVE, CrossTerms.ALL)
                and slabel1 != slabel2
            ):
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


def build_quarks(config: HighModeConfig, run_tsources: t.List[str]) -> HadronsInput:
    modules = {}
    for tsource in run_tsources:
        for op in set(quark_gen(config)):
            glabel = op.gamma.name.lower()
            quark = f"quark_{op.solver}_{glabel}_mass_{op.mass}_t{tsource}"
            source = f"noise_t{tsource}"
            solver = config.solver_name.format(solver=op.solver, mass=op.mass)

            if op.precon:
                guess = f"quark_{op.precon}_{glabel}_mass_{op.mass}_t{tsource}"
            else:
                guess = ""

            # Split-grid: tag CG-solve propagators (solver segment "ama"/"ama_{r}")
            # with their subgrid index; ranLL (low-mode/LMA solve) is excluded.
            if config.subgrid_ranks is not None and "ama" in op.solver:
                subgrid = int(tsource) % config.subgrid_ranks
            else:
                subgrid = None

            modules[quark] = hadmods.quark_prop(
                name=quark,
                source=source,
                solver=solver,
                guess=guess,
                gammas=op.gamma.gamma_string,
                apply_g5=str(op.apply_g5).lower(),
                gauge="" if op.gamma.local else config.shift_gauge_name,
                subgrid=subgrid,
            )

    return HadronsInput(modules=modules, schedule=list(modules.keys()))


def build_contractions(
    config: HighModeConfig, run_tsources: t.List[str]
) -> HadronsInput:
    modules = {}

    for tsource in run_tsources:
        for op, con_set in set(contraction_gen(config)):
            glabel = op.gamma.name.lower()
            quark_glabel = con_set.quark.gamma.name.lower()
            antiquark_glabel = con_set.antiquark.gamma.name.lower()
            mlabel1 = con_set.quark.mass
            mlabel2 = con_set.antiquark.mass
            quark = (
                f"quark_{con_set.quark.solver}_{quark_glabel}_mass_{mlabel1}_t{tsource}"
            )
            antiquark = f"quark_{con_set.antiquark.solver}_{antiquark_glabel}_mass_{mlabel2}_t{tsource}"

            mass_output = con_set.mass_label(config.mass)
            solver_label = con_set.solver_label

            if mlabel1 == mlabel2:
                mass_label = f"mass_{mlabel1}"
            else:
                mass_label = f"mass_{mlabel1}_mass_{mlabel2}"

            output = config.high_modes.filestem.format(
                mass=mass_output, dset=solver_label, gamma_label=glabel, tsource=tsource
            )

            # Split-grid: tag contractions that source a CG-solve propagator
            # (either quark or antiquark side) to that CG subgrid. A cross-term
            # contraction (e.g. corr_ranLL_ama) sits at a single tsource, so both
            # quarks share the same tsource % subgrid_ranks; the ranLL propagator
            # is scattered onto the subgrid by Hadrons at runtime.
            if config.subgrid_ranks is not None and (
                "ama" in con_set.quark.solver
                or "ama" in con_set.antiquark.solver
            ):
                subgrid = int(tsource) % config.subgrid_ranks
            else:
                subgrid = None

            name = f"corr_{solver_label}_{glabel}_{mass_label}_t{tsource}"
            modules[name] = hadmods.prop_contract(
                name=name,
                source=quark,
                sink=antiquark,
                sink_fn="sink",
                source_shift=f"noise_t{tsource}_shift",
                source_gammas=con_set.quark.gamma.gamma_string,
                sink_gammas=con_set.sink.gamma.gamma_string,
                apply_g5=str(con_set.sink.apply_g5).lower(),
                gauge="" if con_set.quark.gamma.local else config.shift_gauge_name,
                output=output,
                subgrid=subgrid,
            )
    return HadronsInput(modules=modules, schedule=list(modules.keys()))
