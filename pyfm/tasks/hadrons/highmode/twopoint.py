import typing as t
import itertools

from pyfm.tasks.hadrons.types import HadronsInput
import pyfm.tasks.hadrons.modules as hadmods
from pyfm.domain import OpList, Gamma, MassDict
from pyfm.tasks.hadrons.types import HighModeConfig, SourceRef


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
    """Generates exactly the propagators the requested contractions consume.

    Demand-driven: the propagator set is derived from ``contraction_gen``'s
    emitted quark/antiquark sides, so a solve is emitted iff some contraction
    references it. Under TIERED the HH diagonal is dropped, leaving the LH
    cross (``ranLL_ama``) as the only contraction consuming an ama (CG)
    propagator — its antiquark side uses the contract gamma
    (PION_LOCAL/IDENTITY) only, so the op-gamma CG solves (e.g.
    ``quark_ama_vec_local_*``) have no consumer and are skipped. Under
    DIAGONAL/ALL every base propagator remains required and the emitted set
    is identical to the previous label-list-driven generator.

    The precon (guess) chain is preserved among the remaining solvers: each
    emitted solve's guess is the nearest earlier base solver whose
    same-(gamma, mass) propagator is also emitted — the demand-driven
    refinement of the previous ``[None] + solver_labels[:-1]`` zip. Per
    (gamma, mass) the required set is a prefix of the base solver list (the
    HH diagonal drops wholesale), so a guess never references a skipped
    module. With ``chain_cg_solves=False`` (independent mode) every CG solve
    guesses ``ranLL`` directly instead — or nothing, when low modes are
    skipped (``ranLL`` itself never takes a guess in either mode).
    Emission order is deterministic and mass-major: every solve and
    gamma for one mass is emitted before moving on to the next mass (the
    schedule builder consumes this order), base-solver order preserved
    within a mass so precons precede their consumers.

    Note:
    - PION_LOCAL requires only the identity gamma operation (equivalent to G5_G5 with apply_g5=True)
    - (AXIAL_)VEC and (AXIAL_)FOURVEC operations require a vector/four-vector gamma solve paired with a (identity)G5_G5 solve
    - Axial gammas contract against IDENTITY and reuse the non-axial VEC/FOURVEC
      solve; every other gamma contracts against PION_LOCAL. See the module-level
      G5_HERMITICITY note.
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
        if config.chain_cg_solves or solver == "ranLL":
            # Chained (default): nearest earlier base solver whose
            # same-(gamma, mass) propagator is also emitted. Unchanged from
            # the previous behavior (ranLL hits an empty prefix and stays
            # guessless), so the default path is byte-identical.
            for earlier in reversed(solver_labels[: solver_labels.index(solver)]):
                if (earlier, gamma, mass) in required:
                    precon = earlier
                    break
        elif ("ranLL", gamma, mass) in required:
            # Independent: every CG solve guesses ranLL directly. The LL
            # diagonal is always admitted, so an emitted CG solve's
            # same-(gamma, mass) ranLL is emitted too; without low modes no
            # guess is provided.
            precon = "ranLL"
        yield TwoPointOp.Op(
            gamma=gamma, mass=mass, solver=solver, apply_g5=True, precon=precon
        )


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


def build_quarks(config: HighModeConfig, run_refs: t.List[SourceRef]) -> HadronsInput:
    modules = {}
    for ref in run_refs:
        for op in set(quark_gen(config)):
            glabel = op.gamma.name.lower()
            quark = f"quark_{op.solver}_{glabel}_mass_{op.mass}_{ref.label}"
            source = f"noise_{ref.label}"
            solver = config.solver_name.format(solver=op.solver, mass=op.mass)

            if op.precon:
                guess = f"quark_{op.precon}_{glabel}_mass_{op.mass}_{ref.label}"
            else:
                guess = ""

            # Split-grid: tag CG-solve propagators (solver segment "ama"/"ama_{r}")
            # with their subgrid index; ranLL (low-mode/LMA solve) is excluded.
            if config.subgrid_ranks is not None and "ama" in op.solver:
                subgrid = ref.t0 % config.subgrid_ranks
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


def build_contractions(config: HighModeConfig, run_refs: t.List[SourceRef]) -> HadronsInput:
    modules = {}

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

            mass_output = con_set.mass_label(config.mass)
            solver_label = con_set.solver_label

            if mlabel1 == mlabel2:
                mass_label = f"mass_{mlabel1}"
            else:
                mass_label = f"mass_{mlabel1}_mass_{mlabel2}"

            output = config.high_modes.filestem.format(
                mass=mass_output, dset=solver_label, gamma_label=glabel, tsource=ref.axis
            )

            # Split-grid: tag contractions that source a CG-solve propagator
            # (either quark or antiquark side) to that CG subgrid. A cross-term
            # contraction (e.g. corr_ranLL_ama) sits at a single source, so both
            # quarks share the same t0 % subgrid_ranks; the ranLL propagator
            # is scattered onto the subgrid by Hadrons at runtime.
            if config.subgrid_ranks is not None and (
                "ama" in con_set.quark.solver
                or "ama" in con_set.antiquark.solver
            ):
                subgrid = ref.t0 % config.subgrid_ranks
            else:
                subgrid = None

            name = f"corr_{solver_label}_{glabel}_{mass_label}_{ref.label}"
            modules[name] = hadmods.prop_contract(
                name=name,
                source=quark,
                sink=antiquark,
                sink_fn="sink",
                source_shift=f"noise_{ref.label}_shift",
                source_gammas=con_set.quark.gamma.gamma_string,
                sink_gammas=con_set.sink.gamma.gamma_string,
                apply_g5=str(con_set.sink.apply_g5).lower(),
                gauge="" if con_set.quark.gamma.local else config.shift_gauge_name,
                output=output,
                subgrid=subgrid,
            )
    return HadronsInput(modules=modules, schedule=list(modules.keys()))
