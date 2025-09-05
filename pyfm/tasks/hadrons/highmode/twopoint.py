import typing as t
import itertools

from pyfm.domain import HadronsInput, OpList, hadmods, Gamma
from .domain import HighModeConfig


class TwoPointOp(t.NamedTuple):
    class Op(t.NamedTuple):
        gamma: Gamma
        mass: t.Tuple[str, ...]
        apply_g5: bool

    quark: Op
    antiquark: Op
    sink: Op


def get_quark_list(op_list: OpList) -> t.Iterator[TwoPointOp.Op]:
    for op in op_list:
        match op.gamma:
            case Gamma.PION_LOCAL:
                yield TwoPointOp.Op(gamma=op.gamma, mass=op.mass, apply_g5=True)
            case Gamma.AXIAL_VEC_ONELINK | Gamma.AXIAL_VEC_LOCAL:
                yield TwoPointOp.Op(gamma=Gamma.IDENTITY, mass=op.mass, apply_g5=True)
                yield TwoPointOp.Op(gamma=op.gamma, mass=op.mass, apply_g5=True)
            case Gamma.VEC_ONELINK | Gamma.VEC_LOCAL:
                yield TwoPointOp.Op(gamma=op.gamma, mass=op.mass, apply_g5=True)
                yield TwoPointOp.Op(gamma=Gamma.PION_LOCAL, mass=op.mass, apply_g5=True)
            case _:
                raise ValueError(f"Unexpected Gamma value: {op.gamma}")


def get_contraction_list(op_list: OpList) -> t.Iterator[TwoPointOp]:
    for op in op_list:
        match op.gamma:
            case Gamma.PION_LOCAL:
                quark = TwoPointOp.Op(gamma=op.gamma, mass=op.mass, apply_g5=True)
                antiquark = quark
                sink = quark
            case Gamma.AXIAL_VEC_ONELINK | Gamma.AXIAL_VEC_LOCAL:
                quark = TwoPointOp.Op(gamma=op.gamma, mass=op.mass, apply_g5=True)
                antiquark = TwoPointOp.Op(
                    gamma=Gamma.IDENTITY, mass=op.mass, apply_g5=True
                )
                sink = TwoPointOp.Op(gamma=op.gamma, mass=op.mass, apply_g5=False)
            case Gamma.VEC_ONELINK | Gamma.VEC_LOCAL:
                quark = TwoPointOp.Op(gamma=op.gamma, mass=op.mass, apply_g5=True)
                antiquark = TwoPointOp.Op(
                    gamma=Gamma.PION_LOCAL, mass=op.mass, apply_g5=True
                )
                sink = quark
            case _:
                raise ValueError(f"Unexpected Gamma value: {op.gamma}")

        yield TwoPointOp(
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

    for op in set(get_quark_list(config.operations)):
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

    for op_set in set(get_contraction_list(config.operations)):
        glabel = op_set.quark.gamma.name.lower()
        antiquark_glabel = op_set.antiquark.gamma.name.lower()
        for tsource, slabel, m1label, m2label in itertools.product(
            run_tsources, solver_labels, op_set.quark.mass, op_set.antiquark.mass
        ):
            if m1label < m2label:
                continue
            quark = f"quark_{slabel}_{glabel}_mass_{m1label}_t{tsource}"
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
                sink_func="sink",
                source_shift=f"noise_t{tsource}_shift",
                source_gammas=op_set.quark.gamma.gamma_string,
                sink_gammas=op_set.sink.gamma.gamma_string,
                apply_g5=str(op_set.sink.apply_g5).lower(),
                gauge="" if op_set.quark.gamma.local else config.shift_gauge_name,
                output=output,
            )
    return HadronsInput(modules=modules, schedule=list(modules.keys()))
