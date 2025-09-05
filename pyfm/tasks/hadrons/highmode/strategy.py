import typing as t
import re
import pandas as pd
import itertools
from pyrsistent import thaw, m

from pyfm.domain import (
    HadronsInput,
    OpList,
    hadmods,
)
from .domain import HighModeConfig, CorrelatorStrategy
from . import sib, twopoint

from pyfm import utils


def create_file_catalog(config: HighModeConfig) -> pd.DataFrame:
    def generate_outfile_formatting():
        res = {"tsource": list(map(str, config.tsource_range)), "dset": []}
        if not config.skip_low_modes:
            res["dset"].append("ranLL")
        if not config.skip_cg:
            res["dset"].append("ama")

        for op in config.op_list:
            res["gamma_label"] = op.gamma.name.lower()
            res["mass"] = [config.mass[m] for m in op.mass]
            yield res, config.high_modes

    outfile_generator = generate_outfile_formatting()

    return utils.io.catalog_files(outfile_generator)


def build_input_params(config: HighModeConfig) -> HadronsInput:
    modules = {}
    schedule = []

    if not config.overwrite:
        df = create_file_catalog(config)
        missing_files = df[df["exists"] == False]
        run_tsources = []
        for tsource in config.tsource_range:
            if any(missing_files["tsource"] == str(tsource)):
                run_tsources.append(str(tsource))
    else:
        run_tsources = list(map(str, config.tsource_range))

    modules["sink"] = hadmods.sink(name="sink", mom="0 0 0")
    schedule.append("sink")

    quark_schedule = []
    for tsource in run_tsources:
        name = f"noise_t{tsource}"
        modules[name] = hadmods.noise_rw(
            name=name,
            nsrc=str(config.noise),
            t0=tsource,
            tstep=str(config.time),
        )
        quark_schedule.append(name)

    for mass_label in config.masses:
        action = config.action_name.format(mass=mass_label)
        if not config.skip_low_modes:
            name = config.solver_name.format(solver="ranLL", mass=mass_label)
            low_modes = config.low_modes_name.format(mass=mass_label)
            modules[name] = hadmods.lma_solver(
                name=name,
                action=action,
                low_modes=low_modes,
            )
            schedule.append(name)

        cg_solver_labels: t.List = [s for s in config.get_solver_labels() if "ama" in s]
        for resid, sl in zip(map(str, config.residual), cg_solver_labels):
            name = config.solver_name.format(solver=sl, mass=mass_label)

            if config.solver == "rb":
                modules[name] = hadmods.rb_cg(
                    name=name,
                    action=action,
                    residual=resid,
                )
            else:
                inner_action = f"i{action}"
                modules[name] = hadmods.mixed_precision_cg(
                    name=name,
                    outer_action=action,
                    inner_action=inner_action,
                    residual=resid,
                )
            schedule.append(name)

    quark_inputs = build_quark_strategy(config, run_tsources)
    modules |= quark_inputs.modules
    quark_schedule += quark_inputs.schedule

    contract_inputs = build_contract_strategy(config, run_tsources)
    modules |= contract_inputs.modules
    quark_schedule += contract_inputs.schedule

    schedule += sort_schedule(config, quark_schedule)

    return HadronsInput(modules=modules, schedule=schedule)


def sort_schedule(config: HighModeConfig, module_names: t.List[str]) -> t.List[str]:
    gammas = ["pion_local", "vec_local", "vec_onelink"]

    def gamma_order(name):
        for i, gamma in enumerate(gammas):
            if gamma in name:
                return i
        return -1

    def mass_order(name):
        for i, mass in enumerate(config.mass.keys()):
            if f"mass_{mass}" in name:
                return i
        return -1

    def mixed_mass_last(name):
        return len(re.findall(r"_mass", name))

    def tslice_order(name):
        time = re.findall(r"_t(\d+)", name)
        if len(time):
            return int(time[0])
        else:
            return -1

    sorted_modules = sorted(module_names, key=gamma_order)
    sorted_modules = sorted(sorted_modules, key=mass_order)
    sorted_modules = sorted(sorted_modules, key=mixed_mass_last)
    sorted_modules = sorted(sorted_modules, key=tslice_order)

    return sorted_modules


def build_quark_strategy(
    config: HighModeConfig, run_tsources: t.List[str]
) -> HadronsInput:
    match config.correlator_strategy:
        case CorrelatorStrategy.TWOPOINT:
            return twopoint.build_quarks(config, run_tsources)
        case CorrelatorStrategy.SIB:
            return sib.build_quarks(config, run_tsources)
        case _:
            raise ValueError(
                f"Unknown correlator_strategy: {config.correlator_strategy}"
            )


def build_contract_strategy(
    config: HighModeConfig, run_tsources: t.List[str]
) -> HadronsInput:
    match config.correlator_strategy:
        case CorrelatorStrategy.SIB:
            return sib.build_contractions(config, run_tsources)
        case CorrelatorStrategy.TWOPOINT:
            return twopoint.build_contractions(config, run_tsources)
        case _:
            raise ValueError(
                f"Unknown correlator_strategy: {config.correlator_strategy}"
            )


def create_outfile_catalog(config: HighModeConfig) -> pd.DataFrame:
    def generate_outfile_formatting():
        solver_labels = config.get_solver_labels()
        res = {"tsource": list(map(str, config.tsource_range)), "dset": solver_labels}

        for op in config.op_list:
            res["gamma_label"] = op.gamma.name.lower()
            res["mass"] = [config.mass.to_string(m, True) for m in op.mass]
            yield res, config.high_modes

    outfile_generator = generate_outfile_formatting()

    df = utils.io.catalog_files(outfile_generator)

    return df


def build_aggregator_params(
    config: HighModeConfig,
) -> t.Dict:
    proc_params = m()

    outfile = (
        config.high_modes.filestem.replace("correlators", "processed/{format}")
        .replace("_{series}", "")
        .replace("_t{tsource}", "")
    ) + ".h5"

    infile_stem = config.high_modes.filename

    e_rep = m().evolver()
    e_rep["tsource"] = list(map(str, config.tsource_range))

    solver_labels = config.get_solver_labels()

    run_list = []
    for op in config.op_list:
        gamma_label = op.gamma.name.lower()
        e_rep["gamma_label"] = gamma_label
        for mass, dset in itertools.product(op.mass, solver_labels):
            mass_label = config.mass.to_string(mass, True)
            file_label = f"{gamma_label}_{mass_label}_{dset}"
            run_list.append(file_label)
            e_rep["mass"] = mass_label
            e_rep["dset"] = dset
            replacements = e_rep.persistent()

            h5_datasets = {
                g: f"/meson/meson_{i}/corr" for i, g in enumerate(op.gamma.gamma_list)
            }

            array_params = {
                "order": ["t"],
                "labels": {"t": f"0..{config.time - 1}"},
            }

            proc_params = proc_params.set(
                file_label,
                {
                    "logging_level": config.logging_level,
                    "load_files": {
                        "filestem": infile_stem,
                        "regex": {"series": "[a-z]", "cfg": "[0-9]+"},
                        "replacements": thaw(replacements),
                        "name": "gamma",
                        "datasets": h5_datasets,
                        **array_params,
                    },
                    "out_files": {"filestem": outfile},
                },
            )
            e_rep = replacements.evolver()
    proc_params = proc_params.set("run", run_list)

    return dict(thaw(proc_params))
