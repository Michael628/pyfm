import pandas as pd
from dataclasses import fields
from pydantic import Field
from pydantic.dataclasses import dataclass
from pyfm.nanny.tasks.hadrons.components import hadmods
from pyfm.nanny.tasks.hadrons import SubmitHadronsConfig
from pyfm.nanny.tasks.hadrons.components import ComponentBase
from pyfm import OpList, utils

import functools
import itertools
import typing as t


@dataclass
class HighModeHadronsComponent(ComponentBase):
    _operations: OpList
    skip_cg: bool = False
    solver: str = "mpcg"
    has_eigs: bool = False
    solve_strategy: str = "2pt"
    cg_residual: t.List[float] = Field(default=[1e-8])

    @classmethod
    def from_dict(cls, kwargs):
        params: t.Dict = {"_operations": OpList.from_dict(kwargs)}
        for f in fields(cls):
            if item := kwargs.get(f.name, None):
                params[f.name] = item

        if cg_residual := params.get("cg_residual", None):
            assert isinstance(cg_residual, list)
            cg_residual = sorted((float(x) for x in cg_residual), reverse=True)
            params["cg_residual"] = cg_residual

        return cls(**params)

    @property
    def operations(self):
        return self._operations.op_list

    @property
    def mass(self):
        return self._operations.mass

    def get_file_catalog(self, submit_config: SubmitHadronsConfig) -> pd.DataFrame:
        """
        Generates a dataframe containing information on each file that would be generated
        from this component based on the submission configuration (see utils.catalog_files).

        Args:
            submit_config (SubmitHadronsConfig): Configuration object containing
                misc. Hadrons parameters (see class definition).

        Returns:
            pd.DataFrame: file data generated using the provided configuration.
        """

        def generate_outfile_formatting():
            res = {"tsource": list(map(str, submit_config.tsource_range)), "dset": []}
            if self.has_eigs:
                res["dset"].append("ranLL")
            if not self.skip_cg:
                res["dset"].append("ama")

            for op in self.operations:
                res["gamma_label"] = op.gamma.name.lower()
                res["mass"] = [submit_config.mass_out_label[m] for m in op.mass]
                yield res, submit_config.files["high_modes"]

        outfile_generator = generate_outfile_formatting()
        replacements = submit_config.string_dict()

        return utils.catalog_files(outfile_generator, replacements)

    def build_quark_strategy(self, op: OpList.Op, param_iter: t.Iterable) -> t.Dict:
        raise NotImplementedError

    def build_contract_strategy(
        self, op: OpList.Op, param_iter: t.Iterable, out_filestem: t.Callable
    ) -> t.Dict:
        raise NotImplementedError

    def input_params(self, submit_config: SubmitHadronsConfig) -> t.Dict:
        submit_conf_dict = submit_config.string_dict()
        modules = {}

        if self.solver == "mpcg":
            print(self.mass)
            for mass_label in self.mass:
                name = f"istag_mass_{mass_label}"
                mass = str(submit_config.mass[mass_label])
                modules[name] = hadmods.action_float(
                    name=name,
                    mass=mass,
                    gauge_fat="gauge_fatf",
                    gauge_long="gauge_longf",
                )

        if not submit_config.overwrite_sources:
            cf = self.get_file_catalog(submit_config)
            missing_files = cf[cf["exists"] == False]
            run_tsources = []
            for tsource in submit_config.tsource_range:
                if any(missing_files["tsource"] == str(tsource)):
                    run_tsources.append(str(tsource))
        else:
            run_tsources = list(map(str, submit_config.tsource_range))

        modules["sink"] = hadmods.sink(name="sink", mom="0 0 0")

        for tsource in run_tsources:
            name = f"noise_t{tsource}"
            modules[name] = hadmods.noise_rw(
                name=name,
                nsrc=submit_conf_dict["noise"],
                t0=tsource,
                tstep=submit_conf_dict["time"],
            )

        solver_labels = []
        residuals = self.cg_residual
        if not self.skip_cg:
            if len(residuals) == 1:
                solver_labels.append("ama")
            else:
                solver_labels += [f"ama_{r}" for r in residuals]

        for mass_label in self.mass:
            if self.has_eigs:
                name = f"stag_ranLL_mass_{mass_label}"
                modules[name] = hadmods.lma_solver(
                    name=name,
                    action=f"stag_mass_{mass_label}",
                    low_modes=f"evecs_mass_{mass_label}",
                )

            for resid, sl in zip(map(str, residuals), solver_labels):
                name = f"stag_{sl}_mass_{mass_label}"

                if self.solver == "rb":
                    name = f"stag_ama_mass_{mass_label}"
                    modules[name] = hadmods.rb_cg(
                        name=name,
                        action=f"stag_mass_{mass_label}",
                        residual=resid,
                    )
                else:
                    name = f"stag_ama_mass_{mass_label}"
                    modules[name] = hadmods.mixed_precision_cg(
                        name=name,
                        outer_action=f"stag_mass_{mass_label}",
                        inner_action=f"istag_mass_{mass_label}",
                        residual=resid,
                    )

        if self.has_eigs:
            solver_labels.insert(0, "ranLL")

        guess_solver_labels = solver_labels[:-1].copy()
        guess_solver_labels.insert(0, None)

        assert "high_modes" in submit_config.files

        high_path = submit_config.files["high_modes"].filestem
        high_partial = functools.partial(high_path.format, **submit_conf_dict)

        for op in self.operations:
            glabel = op.gamma.name.lower()
            solver_zip = list(zip(solver_labels, guess_solver_labels))
            quark_iter = itertools.product([glabel], run_tsources, solver_zip, op.mass)

            modules |= self.build_quark_strategy(op, quark_iter)

            def m1_ge_m2(x):
                return x[-2][0] >= x[-1][0]

            out_masses = map(lambda x: submit_config.mass_out_label[x], op.mass)
            mass_zip = list(zip(op.mass, out_masses))
            contract_iter = filter(
                m1_ge_m2,
                itertools.product(
                    [glabel], run_tsources, solver_labels, mass_zip, mass_zip
                ),
            )

            modules |= self.build_contract_strategy(op, contract_iter, high_partial)

        return modules


def build_quark_sib(op: OpList.Op, param_iter: t.Iterable) -> t.Dict:
    modules = {}

    for glabel, tsource, (slabel, slabel_guess), mlabel in param_iter:
        quark = f"quark_{slabel}_{glabel}_mass_{mlabel}_t{tsource}"
        source = f"noise_t{tsource}"
        solver = f"stag_{slabel}_mass_{mlabel}"
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

    return modules


def build_quark_2pt(op: OpList.Op, param_iter: t.Iterable) -> t.Dict:
    modules = {}

    for glabel, tsource, (slabel, slabel_guess), mlabel in param_iter:
        quark = f"quark_{slabel}_{glabel}_mass_{mlabel}_t{tsource}"
        source = f"noise_t{tsource}"
        solver = f"stag_{slabel}_mass_{mlabel}"

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

    return modules


def build_contract_2pt(
    op: OpList.Op, param_iter: t.Iterable, out_filestem: t.Callable
) -> t.Dict:
    modules = {}

    for glabel, tsource, slabel, (m1label, m1out), (m2label, m2out) in param_iter:
        quark1 = f"quark_{slabel}_{glabel}_mass_{m1label}_t{tsource}"
        quark2 = f"quark_{slabel}_pion_local_mass_{m2label}_t{tsource}"

        if m1label == m2label:
            mass_label = f"mass_{m1label}"
            mass_output = m1out
        else:
            mass_label = f"mass_{m1label}_mass_{m2label}"
            mass_output = f"{m1out}_m{m2out}"

        output = out_filestem(
            mass=mass_output, dset=slabel, gamma_label=glabel, tsource=tsource
        )

        name = f"corr_{slabel}_{glabel}_{mass_label}_t{tsource}"
        modules[name] = hadmods.prop_contract(
            name=name,
            source=quark1,
            sink=quark2,
            sink_func="sink",
            source_shift=f"noise_t{tsource}_shift",
            source_gammas=op.gamma.gamma_string,
            sink_gammas=op.gamma.gamma_string,
            apply_g5="true",
            gauge="" if op.gamma.local else "gauge",
            output=output,
        )
    return modules


def build_contract_sib(
    op: OpList.Op, param_iter: t.Iterable, out_filestem: t.Callable
) -> t.Dict:
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

        output = out_filestem(
            mass=mass_output, dset=slabel, gamma_label=glabel, tsource=tsource
        )

        output = out_filestem(
            mass=mass_label, dset=slabel, gamma=glabel, tsource=tsource
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

    return modules


def build_quark_strategy(
    component: HighModeHadronsComponent, op: OpList.Op, param_iter: t.Iterable
) -> t.Dict:
    if component.solve_strategy == "2pt":
        return build_quark_2pt(op, param_iter)
    elif component.solve_strategy == "sib":
        return build_quark_sib(op, param_iter)
    else:
        raise ValueError(f"Unknown solve_strategy: {component.solve_strategy}")


HighModeHadronsComponent.build_quark_strategy = build_quark_strategy


def build_contract_strategy(
    component: HighModeHadronsComponent,
    op: OpList.Op,
    param_iter: t.Iterable,
    out_filestem: t.Callable,
) -> t.Dict:
    if component.solve_strategy == "2pt":
        return build_contract_2pt(op, param_iter, out_filestem)
    elif component.solve_strategy == "sib":
        return build_contract_sib(op, param_iter, out_filestem)
    else:
        raise ValueError(f"Unknown solve_strategy: {component.solve_strategy}")


HighModeHadronsComponent.build_contract_strategy = build_contract_strategy
