import itertools
import os.path
import re
import typing as t
from dataclasses import fields

import pandas as pd
from pydantic.dataclasses import dataclass
from pydantic import Field

from pyfm import Gamma, utils
from pyfm.nanny import TaskBase
from pyfm.nanny.tasks.hadrons.components import hadmods
from pyfm.nanny.tasks.hadrons import SubmitHadronsConfig


# ============LMI Task Configuration===========
@dataclass
class LMITask(TaskBase):
    # ============Epack===========
    @dataclass
    class EpackTask(TaskBase):
        load: bool
        multifile: bool = False
        save_eigs: bool = False
        save_evals: bool = True
        residual: float = 1e-8

    # ============Operator List===========
    @dataclass
    class OpList(TaskBase):
        """Configuration for a list of gamma operations.

        Attributes
        ----------
        operations: list
            Gamma operations to be performed, usually for meson fields or high mode solves.
        """

        @dataclass
        class Op:
            """Parameters for a gamma operation and associated masses."""

            gamma: Gamma
            mass: t.List[str]

        operations: t.List[Op]

        @classmethod
        def from_dict(cls, kwargs) -> "LMITask.OpList":
            """Creates a new instance of OpList from a dictionary.

            Note
            ----
            Valid dictionary input formats:

            kwargs = {
              'gamma': ['op1','op2','op3'],
              'mass': ['m1','m2']
            }

            or

            kwargs = {
              'op1': {
                'mass': ['m1']
              },
              'op2': {
                'mass': ['m2','m3']
              }
            }

            """
            params = utils.deep_copy_dict(kwargs)

            if "mass" not in params:
                operations = []
                for key, val in params.items():
                    mass = val["mass"]
                    if isinstance(mass, str):
                        mass = [mass]
                    gamma = Gamma[key.upper()]
                    operations.append(cls.Op(gamma=gamma, mass=mass))
            else:
                gammas = params["gamma"]
                mass = params["mass"]
                if isinstance(mass, str):
                    mass = [mass]
                if isinstance(gammas, str):
                    gammas = [gammas]
                operations = [cls.Op(gamma=Gamma[g.upper()], mass=mass) for g in gammas]
            params["operations"] = operations
            return cls(**params)

        @property
        def mass(self):
            res: t.Set = set()
            for op in self.operations:
                for m in op.mass:
                    res.add(m)

            return list(res)

    @dataclass
    class HighModes(OpList):
        skip_cg: bool = False
        solver: str = "mpcg"
        cg_residual: t.List[float] = Field(default=[1e-8])

        def __post_init__(self):
            assert all([isinstance(x, float) for x in self.cg_residual])
            self.cg_residual = sorted((x for x in self.cg_residual), reverse=True)

    epack: t.Optional[EpackTask] = None
    meson: t.Optional[OpList] = None
    high_modes: t.Optional[HighModes] = None

    @classmethod
    def from_dict(cls, kwargs) -> "LMITask":
        """Creates a new instance of LMITaskConfig from a dictionary."""
        params = utils.deep_copy_dict(kwargs)
        for f in fields(cls):
            field_name = f.name
            field_type = f.type.__args__[0]
            if field_name in kwargs:
                params[field_name] = field_type.from_dict(kwargs[field_name])

        return cls(**params)

    @property
    def mass(self):
        """Returns list of labels for masses required by task components."""
        res = []

        if self.epack and not self.epack.load:
            res.append("zero")
        if self.meson:
            res += self.meson.mass
        if self.high_modes:
            res += self.high_modes.mass

        return list(set(res))


# ============Functions for building params and checking outfiles===========
def input_params(
    task_config: LMITask,
    submit_config: SubmitHadronsConfig,
) -> t.Tuple[t.List[t.Dict], t.Optional[t.List[str]]]:
    def build_schedule(module_names: t.List[str]) -> t.List[str]:
        gammas = ["pion_local", "vec_local", "vec_onelink"]

        def pop_conditional(mi, cond):
            indices = [i for i, item in enumerate(mi) if cond(item)]
            # Pop in reverse order but return in original order
            return [mi.pop(i) for i in indices[::-1]][::-1]

        def get_mf_inputs(x):
            is_action = x["type"].endswith("ImprovedStaggeredMILC")
            is_evec = "ModifyEigenPack" in x["type"]
            has_sea_mass = "mass_l" in x["name"]
            return (is_action or is_evec) and has_sea_mass

        dp_gauges = pop_conditional(module_names, lambda x: "LoadIldg" in x["type"])
        sp_gauges = pop_conditional(
            module_names, lambda x: "PrecisionCast" in x["type"]
        )
        meson_fields = pop_conditional(
            module_names, lambda x: "A2AMesonField" in x["type"]
        )
        meson_field_inputs = pop_conditional(module_names, get_mf_inputs)

        indep_mass_tslice = pop_conditional(
            module_names,
            lambda x: ("mass" not in x["name"] or "mass_zero" in x["name"])
            and "_t" not in x["name"],
        )

        sorted_modules = dp_gauges + indep_mass_tslice
        sorted_modules += meson_field_inputs + meson_fields
        sorted_modules += sp_gauges

        def gamma_order(x):
            for i, gamma in enumerate(gammas):
                if gamma in x["name"]:
                    return i
            return -1

        def mass_order(x):
            for i, mass in enumerate(submit_config.mass.keys()):
                if f"mass_{mass}" in x["name"]:
                    return i
            return -1

        def mixed_mass_last(x):
            return len(re.findall(r"_mass", x["name"]))

        def tslice_order(x):
            time = re.findall(r"_t(\d+)", x["name"])
            if len(time):
                return int(time[0])
            else:
                return -1

        module_names = sorted(module_names, key=gamma_order)
        module_names = sorted(module_names, key=mass_order)
        module_names = sorted(module_names, key=mixed_mass_last)
        module_names = sorted(module_names, key=tslice_order)

        sorted_modules += module_names

        return [m["name"] for m in sorted_modules]

    outfile_dict = submit_config.files

    submit_conf_dict = submit_config.string_dict()

    if not submit_config.overwrite_sources:
        if task_config.high_modes:
            cf = catalog_files(task_config, submit_config)
            missing_files = cf[cf["exists"] != True]
            run_tsources = []
            for tsource in submit_config.tsource_range:
                if any(missing_files["tsource"] == str(tsource)):
                    run_tsources.append(str(tsource))

        if task_config.meson:
            bf = bad_files(task_config, submit_config)
            meson_template = outfile_dict["meson_ll"].filename
            for i, op in enumerate(task_config.meson.operations[:]):
                for j, mass_label in enumerate(op.mass[:]):
                    meson_files = [
                        meson_template.format(
                            mass=submit_config.mass_out_label[mass_label],
                            gamma=g,
                            **submit_conf_dict,
                        )
                        for g in op.gamma.gamma_list
                    ]
                    if not any([mf in bf for mf in meson_files]):
                        op.mass.pop(j)

                if not op.mass:
                    task_config.meson.operations.pop(i)

    else:
        run_tsources = list(map(str, submit_config.tsource_range))

    gauge_filepath = outfile_dict["gauge_links"].filestem.format(**submit_conf_dict)
    gauge_fat_filepath = outfile_dict["fat_links"].filestem.format(**submit_conf_dict)
    gauge_long_filepath = outfile_dict["long_links"].filestem.format(**submit_conf_dict)

    modules = [
        hadmods.load_gauge("gauge", gauge_filepath),
        hadmods.load_gauge("gauge_fat", gauge_fat_filepath),
        hadmods.load_gauge("gauge_long", gauge_long_filepath),
        hadmods.cast_gauge("gauge_fatf", "gauge_fat"),
        hadmods.cast_gauge("gauge_longf", "gauge_long"),
    ]

    for mass_label in task_config.mass:
        name = f"stag_mass_{mass_label}"
        mass = str(submit_config.mass[mass_label])
        modules.append(
            hadmods.action(
                name=name, mass=mass, gauge_fat="gauge_fat", gauge_long="gauge_long"
            )
        )

    if (
        task_config.high_modes
        and not task_config.high_modes.skip_cg
        and task_config.high_modes.solver == "mpcg"
    ):
        for mass_label in task_config.high_modes.mass:
            name = f"istag_mass_{mass_label}"
            mass = str(submit_config.mass[mass_label])
            modules.append(
                hadmods.action_float(
                    name=name,
                    mass=mass,
                    gauge_fat="gauge_fatf",
                    gauge_long="gauge_longf",
                )
            )

    if task_config.epack:
        epack_path = ""
        multifile = str(task_config.epack.multifile).lower()
        residual = str(task_config.epack.residual)
        if task_config.epack.load or task_config.epack.save_eigs:
            epack_path = outfile_dict["eig"].filestem.format(**submit_conf_dict)

        # Load or generate eigenvectors
        if task_config.epack.load:
            modules.append(
                hadmods.epack_load(
                    name="epack",
                    filestem=epack_path,
                    size=submit_conf_dict["eigs"],
                    multifile=multifile,
                )
            )
        else:
            modules.append(hadmods.op("stag_op", "stag_mass_zero"))
            modules.append(
                hadmods.irl(
                    name="epack",
                    op="stag_op_schur",
                    alpha=submit_conf_dict["alpha"],
                    beta=submit_conf_dict["beta"],
                    npoly=submit_conf_dict["npoly"],
                    nstop=submit_conf_dict["nstop"],
                    nk=submit_conf_dict["nk"],
                    nm=submit_conf_dict["nm"],
                    multifile=multifile,
                    residual=residual,
                    output=epack_path,
                )
            )

        # Shift mass of eigenvalues
        for mass_label in task_config.mass:
            if mass_label == "zero":
                continue
            mass = str(submit_config.mass[mass_label])
            modules.append(
                hadmods.epack_modify(
                    name=f"evecs_mass_{mass_label}", eigen_pack="epack", mass=mass
                )
            )

        if task_config.epack.save_evals:
            eval_path = outfile_dict["eval"].filestem.format(**submit_conf_dict)
            modules.append(
                hadmods.eval_save(
                    name="eval_save", eigen_pack="epack", output=eval_path
                )
            )

    if task_config.meson:
        meson_template = outfile_dict["meson_ll"].filestem
        for op in task_config.meson.operations:
            op_type = op.gamma.name.lower()
            gauge = "" if op.gamma == Gamma.LOCAL else "gauge"
            for mass_label in op.mass:
                output = meson_template.format(
                    mass=submit_config.mass_out_label[mass_label], **submit_conf_dict
                )
                modules.append(
                    hadmods.meson_field(
                        name=f"mf_{op_type}_mass_{mass_label}",
                        action=f"stag_mass_{mass_label}",
                        block=submit_conf_dict["blocksize"],
                        gammas=op.gamma.gamma_string,
                        apply_g5="false",
                        gauge=gauge,
                        low_modes=f"evecs_mass_{mass_label}",
                        left="",
                        right="",
                        output=output,
                    )
                )

    if task_config.high_modes:

        def m1_eq_m2(x):
            return x[-2] == x[-1]

        def m1_ge_m2(x):
            return x[-2] >= x[-1]

        modules.append(hadmods.sink(name="sink", mom="0 0 0"))

        for tsource in run_tsources:
            modules.append(
                hadmods.noise_rw(
                    name=f"noise_t{tsource}",
                    nsrc=submit_conf_dict["noise"],
                    t0=tsource,
                    tstep=submit_conf_dict["time"],
                )
            )

        solver_labels = []
        residuals = task_config.high_modes.cg_residual
        if task_config.high_modes and not task_config.high_modes.skip_cg:
            if len(residuals) == 1:
                solver_labels.append("ama")
            else:
                solver_labels += [f"ama_{r}" for r in residuals]

        for mass_label in task_config.high_modes.mass:
            if task_config.epack:
                modules.append(
                    hadmods.lma_solver(
                        name=f"stag_ranLL_mass_{mass_label}",
                        action=f"stag_mass_{mass_label}",
                        low_modes=f"evecs_mass_{mass_label}",
                    )
                )

            for resid, sl in zip(map(str, residuals), solver_labels):
                name = f"stag_{sl}_mass_{mass_label}"

                if task_config.high_modes.solver == "rb":
                    modules.append(
                        hadmods.rb_cg(
                            name=name,
                            action=f"stag_mass_{mass_label}",
                            residual=resid,
                        )
                    )
                else:
                    modules.append(
                        hadmods.mixed_precision_cg(
                            name=name,
                            outer_action=f"stag_mass_{mass_label}",
                            inner_action=f"istag_mass_{mass_label}",
                            residual=resid,
                        )
                    )

        if task_config.epack:
            solver_labels.insert(0, "ranLL")

        high_path = outfile_dict["high_modes"].filestem

        for op in task_config.high_modes.operations:
            glabel = op.gamma.name.lower()
            quark_iter = list(
                itertools.product(run_tsources, solver_labels, op.mass, op.mass)
            )

            for tsource, slabel, mlabel, _ in filter(m1_eq_m2, quark_iter):
                quark = f"quark_{slabel}_{glabel}_mass_{mlabel}_t{tsource}"
                source = f"noise_t{tsource}"
                solver = f"stag_{slabel}_mass_{mlabel}"

                if guess_index := solver_labels.index(slabel):
                    guess_index -= 1
                    guess = f"quark_{solver_labels[guess_index]}_{glabel}_mass_{mlabel}_t{tsource}"
                else:
                    guess = ""

                modules.append(
                    hadmods.quark_prop(
                        name=quark,
                        source=source,
                        solver=solver,
                        guess=guess,
                        gammas=op.gamma.gamma_string,
                        apply_g5="true",
                        gauge="" if op.gamma.local else "gauge",
                    )
                )

            for tsource, slabel, m1label, m2label in filter(m1_ge_m2, quark_iter):
                quark1 = f"quark_{slabel}_{glabel}_mass_{m1label}_t{tsource}"
                quark2 = f"quark_{slabel}_pion_local_mass_{m2label}_t{tsource}"

                if m1label == m2label:
                    mass_label = f"mass_{m1label}"
                    mass_output = f"{submit_config.mass_out_label[m1label]}"
                else:
                    mass_label = f"mass_{m1label}_mass_{m2label}"
                    mass_output = (
                        f"{submit_config.mass_out_label[m1label]}"
                        f"_m{submit_config.mass_out_label[m2label]}"
                    )

                output = high_path.format(
                    mass=mass_output,
                    dset=slabel,
                    gamma_label=glabel,
                    tsource=tsource,
                    **submit_conf_dict,
                )

                modules.append(
                    hadmods.prop_contract(
                        name=f"corr_{slabel}_{glabel}_{mass_label}_t{tsource}",
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
                )

    module_info = [m["id"] for m in modules]
    schedule = build_schedule(module_info)

    return modules, schedule


def catalog_files(
    task_config: LMITask,
    submit_config: SubmitHadronsConfig,
) -> pd.DataFrame:
    def generate_outfile_formatting():
        outfile_dict = submit_config.files
        if task_config.epack:
            if task_config.epack.save_eigs:
                if task_config.epack.multifile:
                    yield (
                        {"eig_index": list(range(int(submit_config.eigs)))},
                        outfile_dict["eigdir"],
                    )
                else:
                    yield {}, outfile_dict["eig"]
            if task_config.epack.save_eigs:
                yield {}, outfile_dict["eval"]

        if task_config.meson:
            res: t.Dict = {}
            for op in task_config.meson.operations:
                res["gamma"] = op.gamma.gamma_list
                res["mass"] = [submit_config.mass_out_label[m] for m in op.mass]
                yield res, outfile_dict["meson_ll"]

        if task_config.high_modes:
            res = {"tsource": list(map(str, submit_config.tsource_range)), "dset": []}
            if task_config.epack:
                res["dset"].append("ranLL")
            if not task_config.high_modes.skip_cg:
                residuals = task_config.high_modes.cg_residual
                if len(residuals) == 1:
                    res["dset"].append("ama")
                else:
                    res["dset"] += [f"ama_{r}" for r in residuals]

            for op in task_config.high_modes.operations:
                res["gamma_label"] = op.gamma.name.lower()
                res["mass"] = [submit_config.mass_out_label[m] for m in op.mass]
                yield res, outfile_dict["high_modes"]

    def build_row(filepath: str, repls: t.Dict[str, str]) -> t.Dict[str, str]:
        repls["filepath"] = filepath
        return repls

    outfile_generator = generate_outfile_formatting()
    replacements = submit_config.string_dict()

    df = []
    for task_replacements, outfile_config in outfile_generator:
        outfile = outfile_config.filestem + outfile_config.ext
        filekeys = utils.format_keys(outfile)
        replacements.update(task_replacements)
        files = utils.process_files(
            outfile,
            processor=build_row,
            replacements={k: v for k, v in replacements.items() if k in filekeys},
        )
        dict_of_rows = {
            k: [file[k] for file in files] for k in files[0] if len(files) > 0
        }

        new_df = pd.DataFrame(dict_of_rows)
        new_df["good_size"] = outfile_config.good_size
        new_df["exists"] = new_df["filepath"].apply(os.path.exists)
        new_df["file_size"] = None
        new_df.loc[new_df["exists"], "file_size"] = new_df[new_df["exists"]][
            "filepath"
        ].apply(os.path.getsize)
        df.append(new_df)

    df = pd.concat(df, ignore_index=True)

    return df


def bad_files(
    task_config: LMITask,
    submit_config: SubmitHadronsConfig,
) -> t.List[str]:
    df = catalog_files(task_config, submit_config)
    return list(df[(df["file_size"] >= df["good_size"]) != True]["filepath"])


def processing_params(
    task_config: LMITask,
    submit_config: SubmitHadronsConfig,
) -> t.Dict:
    proc_params = {"run": []}
    outfile_dict = submit_config.files
    infile_stem = outfile_dict["high_modes"].filename
    outfile = outfile_dict["high_modes"].filestem
    filekeys = utils.format_keys(infile_stem)
    outfile = outfile.replace("correlators", "dataframes")
    outfile = outfile.replace("_{series}", "")
    outfile = outfile.replace("_t{tsource}", "")
    outfile += ".h5"
    replacements = {
        k: v for k, v in submit_config.string_dict().items() if k in filekeys
    }
    replacements["tsource"] = list(map(str, submit_config.tsource_range))

    solver_labels = []
    if task_config.epack:
        solver_labels.append("ranLL")
    if task_config.high_modes:
        if not task_config.high_modes.skip_cg:
            residuals = task_config.high_modes.cg_residual
            if len(residuals) == 1:
                solver_labels.append("ama")
            else:
                solver_labels += [f"ama_{r}" for r in residuals]

        for op in task_config.high_modes.operations:
            gamma_label = op.gamma.name.lower()
            replacements["gamma_label"] = gamma_label
            for m, dset in itertools.product(op.mass, solver_labels):
                mass_label = submit_config.mass_out_label[m]
                file_label = f"{gamma_label}_{mass_label}_{dset}"
                proc_params["run"].append(file_label)
                replacements["mass"] = mass_label
                replacements["dset"] = dset

                h5_datasets = {
                    g: f"/meson/meson_{i}/corr"
                    for i, g in enumerate(op.gamma.gamma_list)
                }
                array_params = {
                    g: {"order": ["t"], "labels": {"t": f"0..{submit_config.time - 1}"}}
                    for g in h5_datasets.keys()
                }
                proc_params[file_label] = {
                    "logging_level": getattr(submit_config, "logging_level", "INFO"),
                    "load_files": {
                        "filestem": infile_stem,
                        "regex": {"series": "[a-z]", "cfg": "[0-9]+"},
                        "replacements": utils.deep_copy_dict(replacements),
                        "h5_params": {"name": "gamma", "datasets": h5_datasets},
                        "array_params": array_params,
                    },
                    "out_files": {"filestem": outfile, "type": "dataframe"},
                }

    return proc_params


def get_task_factory():
    return LMITask.from_dict


if __name__ == "__main__":
    from pyfm import utils
    from pyfm.nanny.spawnjob import make_inputs

    param = utils.load_param("params.yaml")
    param["job_setup"]["lmi"] = {
        "run": "run.slurm",
        "params": {
            "ext": "1e8",
            "run_id": "LMI-RW-series-{series}-2000-eigs-{noise}-noise",
            "blocksize": 1000,
        },
        "tasks": {
            "epack": {
                "load": False,
                "save_eigs": True,
                "save_evals": True,
                "multifile": False,
            },
            "meson": {
                "gamma": [
                    "local",
                    "onelink",
                ],
                "mass": [
                    "l",
                ],
            },
            "high_modes": {
                "gamma": [
                    "pion_local",
                    "vec_local",
                    "vec_onelink",
                ],
                "mass": [
                    "l",
                ],
            },
        },
        "io": "lmi-test",
        "wall_time": "12:00:00",
    }
    cfgno_steps = [("a.108", None)]
    make_inputs(param, "lmi", cfgno_steps)
    # param = utils.load_param("params.yaml")
    #
    # jc = config.get_job_config(param, "lmi")
    # sc = config.get_submit_config(param, jc, series="a", cfg="100")
    #
    # stuff = config.input_params(jc)
