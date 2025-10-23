import typing as t
from pyfm.domain import HadronsInput, OpList, hadmods
from pyfm.tasks.hadrons.highmode.domain import HighModeConfig


def build_quarks(
    config: HighModeConfig,
    op: OpList.Op,
    param_iter: t.List[t.Tuple[str, str, t.Tuple[str, str], str]],
) -> HadronsInput:
    modules = {}

    for glabel, tsource, (slabel, slabel_guess), mlabel in param_iter:
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
            apply_g5="true",
            gauge="" if op.gamma.local else "gauge",
        )

        # sib = Gamma.G1_G1
        # quark_g1 = quark+sib.name
        # modules.append(hadmods.seq_gamma(
        #     name=quark_g1,
        #     q=quark+glabel,
        #     ta='0',
        #     tb=submit_conf_dict['time'],
        #     gammas=sib.gamma_string,
        #     apply_g5='false',
        #     gauge="",
        #     mom='0 0 0'
        # ))

        quark_M = quark + "_M"
        guess_M = guess + "_M" if guess else guess
        modules[quark_M] = hadmods.quark_prop(
            name=quark_M,
            source=quark + glabel,
            solver=solver,
            guess=guess_M,
            gammas="",
            apply_g5="false",
            gauge="",
        )

    return HadronsInput(modules=modules, schedule=list(modules.keys()))


def build_contractions(
    config: HighModeConfig, op: OpList.Op, param_iter: t.Iterable
) -> HadronsInput:
    modules = {}
    for glabel, tsource, slabel, (m1label, m1out), (m2label, m2out) in param_iter:
        quark1 = f"quark_{slabel}_{glabel}_mass_{m1label}_t{tsource}_M"
        quark2 = f"quark_{slabel}_G5_G5_mass_{m2label}_t{tsource}G5_G5"

        if m1label == m2label:
            mass_label = f"mass_{m1label}"
            mass_output = m1out
        else:
            mass_label = f"mass_{m1label}_mass_{m2label}"
            mass_output = f"{m1out}_m{m2out}"

        output = config.high_modes.filestem.format(
            mass=mass_label, dset=slabel, gamma_label=glabel, tsource=tsource
        )

        name = f"corr_{slabel}_{glabel}_mass_{mass_label}_t{tsource}"
        modules[name] = hadmods.prop_contract(
            name=name,
            source=quark1,
            sink=quark2,
            sink_func="sink",
            source_shift=f"noise_t{tsource}_shift",
            source_gammas="",
            sink_gammas=op.gamma.gamma_string,
            apply_g5="true",
            gauge="" if op.gamma.local else "gauge",
            output=output,
        )

    return HadronsInput(modules=modules, schedule=list(modules.keys()))
