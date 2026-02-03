import typing as t
import itertools

from pyfm.tasks.hadrons.types import HadronsInput
import pyfm.tasks.hadrons.modules as hadmods
from pyfm.domain import OpList, Gamma
from pyfm.tasks.hadrons.types import HighModeConfig


class TwoPointOp(t.NamedTuple):
    class Op(t.NamedTuple):
        gamma: Gamma
        mass: t.Tuple[str, ...]
        apply_g5: bool

    quark: Op
    antiquark: Op
    sink: Op


def quark_gen(op_list: OpList) -> t.Iterator[TwoPointOp.Op]:
    """Generates required propagators for the requested two-point functions

    Note:
    - PION_LOCAL requires only the identity gamm operation (equivalent to G5_G5 with apply_g5=True)
    - (AXIAL_)VEC operation requires a vec gamma solve paired with a (identity)G5_G5 solve
    """
    for op in op_list:
        match op.gamma:
            case Gamma.PION_LOCAL:
                yield TwoPointOp.Op(gamma=op.gamma, mass=op.mass, apply_g5=True)
            case Gamma.AXIAL_VEC_ONELINK | Gamma.AXIAL_VEC_LOCAL:
                yield TwoPointOp.Op(gamma=Gamma.IDENTITY, mass=op.mass, apply_g5=True)
            case Gamma.VEC_ONELINK | Gamma.VEC_LOCAL:
                yield TwoPointOp.Op(gamma=Gamma.PION_LOCAL, mass=op.mass, apply_g5=True)
            case _:
                raise ValueError(f"Unexpected Gamma value: {op.gamma}")

        match op.gamma:
            case Gamma.VEC_LOCAL | Gamma.AXIAL_VEC_LOCAL:
                yield TwoPointOp.Op(gamma=Gamma.VEC_LOCAL, mass=op.mass, apply_g5=True)
            case Gamma.AXIAL_VEC_ONELINK | Gamma.VEC_ONELINK:
                yield TwoPointOp.Op(
                    gamma=Gamma.VEC_ONELINK, mass=op.mass, apply_g5=True
                )
            case _:
                pass


def contraction_gen(op_list: OpList) -> t.Tuple[OpList.Op, t.Iterator[TwoPointOp]]:
    """Generates required contractions for the requested two-point functions
    See get_quark_list notes for additional information
    """
    for op in op_list:
        # Set antiquark
        match op.gamma:
            case Gamma.PION_LOCAL | Gamma.VEC_ONELINK | Gamma.VEC_LOCAL:
                antiquark = TwoPointOp.Op(
                    gamma=Gamma.PION_LOCAL, mass=op.mass, apply_g5=True
                )
            case Gamma.AXIAL_VEC_ONELINK | Gamma.AXIAL_VEC_LOCAL:
                antiquark = TwoPointOp.Op(
                    gamma=Gamma.IDENTITY, mass=op.mass, apply_g5=True
                )
            case _:
                raise ValueError(f"Unexpected Gamma value: {op.gamma}")
        # Set quark
        match op.gamma:
            case Gamma.PION_LOCAL:
                quark = antiquark
            case Gamma.VEC_LOCAL | Gamma.AXIAL_VEC_LOCAL:
                quark = TwoPointOp.Op(
                    gamma=Gamma.VEC_LOCAL, mass=op.mass, apply_g5=True
                )
            case Gamma.AXIAL_VEC_ONELINK | Gamma.VEC_ONELINK:
                quark = TwoPointOp.Op(
                    gamma=Gamma.VEC_ONELINK, mass=op.mass, apply_g5=True
                )
            case _:
                pass
        # Set sink
        sink = TwoPointOp.Op(gamma=op.gamma, mass=op.mass, apply_g5=True)

        yield op, TwoPointOp(
            quark=quark,
            antiquark=antiquark,
            sink=sink,
        )


def build_quarks(config: HighModeConfig, run_tsources: t.List[str]) -> HadronsInput:
    modules = {}
    solver_labels = config.get_solver_labels()
    guess_solver_labels = solver_labels[:-1].copy()
    guess_solver_labels.insert(0, "")
    solver_zip = list(zip(solver_labels, guess_solver_labels))

    for op in set(quark_gen(config.operations)):
        glabel = op.gamma.name.lower()
        for tsource, (slabel, slabel_guess), mlabel in itertools.product(
            run_tsources, solver_zip, op.mass
        ):
            quark = f"quark_{slabel}_{glabel}_mass_{mlabel}_t{tsource}"
            source = f"noise_t{tsource}"
            solver = config.solver_name.format(solver=slabel, mass=mlabel)

            if slabel_guess:
                guess = f"quark_{slabel_guess}_{glabel}_mass_{mlabel}_t{tsource}"
            else:
                guess = ""

            modules[quark] = hadmods.quark_prop(
                name=quark,
                source=source,
                solver=solver,
                guess=guess,
                gammas=op.gamma.gamma_string,
                apply_g5=str(op.apply_g5).lower(),
                gauge="" if op.gamma.local else "gauge",
            )

    return HadronsInput(modules=modules, schedule=list(modules.keys()))


def build_contractions(
    config: HighModeConfig, run_tsources: t.List[str]
) -> HadronsInput:
    modules = {}
    solver_labels = config.get_solver_labels()

    for op, con_set in set(contraction_gen(config.operations)):
        glabel = op.gamma.name.lower()
        quark_glabel = con_set.quark.gamma.name.lower()
        antiquark_glabel = con_set.antiquark.gamma.name.lower()
        for tsource, slabel, m1label, m2label in itertools.product(
            run_tsources, solver_labels, con_set.quark.mass, con_set.antiquark.mass
        ):
            if m1label < m2label:
                continue

            if not config.cross_terms and m1label != m2label:
                continue

            quark = f"quark_{slabel}_{quark_glabel}_mass_{m1label}_t{tsource}"
            antiquark = f"quark_{slabel}_{antiquark_glabel}_mass_{m2label}_t{tsource}"
            m1out = config.mass.to_string(m1label, True)
            m2out = config.mass.to_string(m2label, True)

            if m1label == m2label:
                mass_label = f"mass_{m1label}"
                mass_output = m1out
            else:
                mass_label = f"mass_{m1label}_mass_{m2label}"
                mass_output = f"{m1out}_m{m2out}"

            output = config.high_modes.filestem.format(
                mass=mass_output, dset=slabel, gamma_label=glabel, tsource=tsource
            )

            name = f"corr_{slabel}_{glabel}_{mass_label}_t{tsource}"
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
