import typing as t
import itertools

from pyfm.tasks.hadrons.types import HadronsInput
import pyfm.tasks.hadrons.modules as hadmods
from pyfm.domain import OpList, Gamma, MassDict
from pyfm.tasks.hadrons.types import CrossTerms, HighModeConfig


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
    - PION_LOCAL requires only the identity gamm operation (equivalent to G5_G5 with apply_g5=True)
    - (AXIAL_)VEC and (AXIAL_)FOURVEC operations require a vector/four-vector gamma solve paired with a (identity)G5_G5 solve
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
            match op.gamma:
                case Gamma.PION_LOCAL:
                    yield TwoPointOp.Op(gamma=op.gamma, **common)
                case (
                    Gamma.AXIAL_VEC_ONELINK
                    | Gamma.AXIAL_VEC_LOCAL
                    | Gamma.AXIAL_FOURVEC_ONELINK
                    | Gamma.AXIAL_FOURVEC_LOCAL
                ):
                    yield TwoPointOp.Op(gamma=Gamma.IDENTITY, **common)
                case (
                    Gamma.VEC_ONELINK
                    | Gamma.VEC_LOCAL
                    | Gamma.FOURVEC_ONELINK
                    | Gamma.FOURVEC_LOCAL
                ):
                    yield TwoPointOp.Op(gamma=Gamma.PION_LOCAL, **common)
                case _:
                    raise ValueError(f"Unexpected Gamma value: {op.gamma}")

            match op.gamma:
                case Gamma.VEC_LOCAL | Gamma.AXIAL_VEC_LOCAL:
                    yield TwoPointOp.Op(gamma=Gamma.VEC_LOCAL, **common)
                case Gamma.FOURVEC_LOCAL | Gamma.AXIAL_FOURVEC_LOCAL:
                    yield TwoPointOp.Op(gamma=Gamma.FOURVEC_LOCAL, **common)
                case Gamma.AXIAL_VEC_ONELINK | Gamma.VEC_ONELINK:
                    yield TwoPointOp.Op(gamma=Gamma.VEC_ONELINK, **common)
                case Gamma.AXIAL_FOURVEC_ONELINK | Gamma.FOURVEC_ONELINK:
                    yield TwoPointOp.Op(gamma=Gamma.FOURVEC_ONELINK, **common)
                case _:
                    pass


def contraction_gen(
    config: HighModeConfig,
) -> t.Iterator[t.Tuple[OpList.Op, TwoPointOp]]:
    """Generates required contractions for the requested two-point functions
    See get_quark_list notes for additional information
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
            # Set antiquark
            match op.gamma:
                case (
                    Gamma.PION_LOCAL
                    | Gamma.VEC_ONELINK
                    | Gamma.VEC_LOCAL
                    | Gamma.FOURVEC_ONELINK
                    | Gamma.FOURVEC_LOCAL
                ):
                    antiquark = TwoPointOp.Op(gamma=Gamma.PION_LOCAL, **common1)
                case (
                    Gamma.AXIAL_VEC_ONELINK
                    | Gamma.AXIAL_VEC_LOCAL
                    | Gamma.AXIAL_FOURVEC_ONELINK
                    | Gamma.AXIAL_FOURVEC_LOCAL
                ):
                    antiquark = TwoPointOp.Op(gamma=Gamma.IDENTITY, **common1)
                case _:
                    raise ValueError(f"Unexpected Gamma value: {op.gamma}")
            # Set quark
            match op.gamma:
                case Gamma.PION_LOCAL:
                    quark = antiquark
                case Gamma.VEC_LOCAL | Gamma.AXIAL_VEC_LOCAL:
                    quark = TwoPointOp.Op(gamma=Gamma.VEC_LOCAL, **common2)
                case Gamma.FOURVEC_LOCAL | Gamma.AXIAL_FOURVEC_LOCAL:
                    quark = TwoPointOp.Op(gamma=Gamma.FOURVEC_LOCAL, **common2)
                case Gamma.AXIAL_VEC_ONELINK | Gamma.VEC_ONELINK:
                    quark = TwoPointOp.Op(gamma=Gamma.VEC_ONELINK, **common2)
                case Gamma.AXIAL_FOURVEC_ONELINK | Gamma.FOURVEC_ONELINK:
                    quark = TwoPointOp.Op(gamma=Gamma.FOURVEC_ONELINK, **common2)
                case _:
                    pass
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

            modules[quark] = hadmods.quark_prop(
                name=quark,
                source=source,
                solver=solver,
                guess=guess,
                gammas=op.gamma.gamma_string,
                apply_g5=str(op.apply_g5).lower(),
                gauge="" if op.gamma.local else config.shift_gauge_name,
            )

    return HadronsInput(modules=modules, schedule=list(modules.keys()))


def build_contractions(
    config: HighModeConfig, run_tsources: t.List[str]
) -> HadronsInput:
    modules = {}
    solver_labels = config.get_solver_labels(skip_cross=True)

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
            )
    return HadronsInput(modules=modules, schedule=list(modules.keys()))
