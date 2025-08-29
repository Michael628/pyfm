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
from pyfm.nanny.tasks.hadrons import HadronsTaskBase, SubmitHadronsConfig
from pyfm.nanny.tasks.hadrons.components import gauge, eig, highmode

# TODO: Change modules into a dictionary instead of a list
# TODO: Move functions into LMITask class and use `self` instead of task_config


# ============LMI Task Configuration===========
@dataclass
class LMITask(TaskBase):
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

    gauge_component: t.Optional[gauge.GaugeHadronsComponent] = None
    epack_component: t.Optional[eig.EigHadronsComponent] = None
    meson_component: t.Optional[OpList] = None
    high_modes_component: t.Optional[highmode.HighModeHadronsComponent] = None

    @classmethod
    def from_dict(cls, kwargs) -> "LMITask":
        """Creates a new instance of LMITaskConfig from a dictionary."""
        # assert "highmode" in kwargs
        params = {}
        params["gauge_component"] = gauge.GaugeHadronsComponent.from_dict(
            kwargs.get("gauge", {})
        )
        has_eigs = "epack" in kwargs
        if hc := kwargs.get("high_modes", None):
            hc = highmode.HighModeHadronsComponent.from_dict(
                hc | {"has_eigs": has_eigs}
            )
        params["high_modes_component"] = hc

        if mc := kwargs.get("meson", None):
            mc = cls.OpList.from_dict(mc)
            params["meson_component"] = mc

        # if hc := kwargs.get("high_modes", None):
        #     hc = cls.HighModes.from_dict(hc)

        params["epack_component"] = None
        if has_eigs:
            masses = set(hc.mass) if hc else set()
            masses |= set(mc.mass) if mc else set()
            masses = {"masses": list(masses)} if masses else {}
            params["epack_component"] = eig.EigHadronsComponent.from_dict(
                kwargs["epack"] | masses
            )

        return cls(**params)

    @property
    def mass(self):
        """Returns list of labels for masses required by task components."""
        res = []

        if self.epack_component and not self.epack_component.load:
            res.append("zero")
        if self.meson_component:
            res += self.meson_component.mass
        if self.high_modes_component:
            res += self.high_modes_component.mass

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
        if task_config.meson_component:
            bf = bad_files(task_config, submit_config)
            meson_template = outfile_dict["meson_ll"].filename
            meson_ops = task_config.meson_component.operations[:]
            for i, op in sorted(enumerate(meson_ops), reverse=True):
                for j, mass_label in sorted(enumerate(op.mass[:]), reverse=True):
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
                    task_config.meson_component.operations.pop(i)

    temp = task_config.gauge_component.input_params(submit_config)
    modules = list(temp.values())

    for mass_label in task_config.mass:
        name = f"stag_mass_{mass_label}"
        mass = str(submit_config.mass[mass_label])
        modules.append(
            hadmods.action(
                name=name, mass=mass, gauge_fat="gauge_fat", gauge_long="gauge_long"
            )
        )

    if task_config.epack_component is not None:
        temp = task_config.epack_component.input_params(submit_config)
        modules += temp.values()

    if task_config.high_modes_component is not None:
        temp = task_config.high_modes_component.input_params(submit_config)
        modules += temp.values()

    if task_config.meson_component:
        meson_template = outfile_dict["meson_ll"].filestem
        for op in task_config.meson_component.operations:
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

    module_info = [m["id"] for m in modules]
    schedule = build_schedule(module_info)

    return modules, schedule


def catalog_files(
    task_config: LMITask,
    submit_config: SubmitHadronsConfig,
) -> pd.DataFrame:
    """
    Generates a catalog of files based on the task and submission configurations.

    Args:
        task_config (LMITask): The configuration for the LMI task, which includes
            components like epack, meson, and high modes.
        submit_config (SubmitHadronsConfig): The submission configuration, which
            includes file paths, eigenvalues, mass labels, and other parameters.

    Returns:
        pd.DataFrame: A DataFrame containing details about the generated files,
        including their paths and associated metadata.

    Notes:
        - The function uses nested generators to yield file formatting details
          for different components (e.g., epack, meson, high modes).
        - The `utils.catalog_files` function is used to process the generated
          file details and replacements into a DataFrame.
    """

    def generate_outfile_formatting():
        """
        A generator function that yields file formatting details for different
        components of the task configuration.

        Yields:
            Tuple[Dict[str, Any], str]: A dictionary of replacements and the
            corresponding output file path for each component.
        """

        outfile_dict = submit_config.files
        if task_config.epack_component:
            if task_config.epack_component.save_eigs:
                if task_config.epack_component.multifile:
                    yield (
                        {"eig_index": list(range(int(submit_config.eigs)))},
                        outfile_dict["eigdir"],
                    )
                else:
                    yield {}, outfile_dict["eig"]
            if task_config.epack_component.save_eigs:
                yield {}, outfile_dict["eval"]

        if task_config.meson_component:
            res: t.Dict = {}
            for op in task_config.meson_component.operations:
                res["gamma"] = op.gamma.gamma_list
                res["mass"] = [submit_config.mass_out_label[m] for m in op.mass]
                yield res, outfile_dict["meson_ll"]

        if task_config.high_modes_component:
            res = {"tsource": list(map(str, submit_config.tsource_range)), "dset": []}
            if task_config.epack_component:
                res["dset"].append("ranLL")
            if not task_config.high_modes_component.skip_cg:
                residuals = task_config.high_modes_component.cg_residual
                if len(residuals) == 1:
                    res["dset"].append("ama")
                else:
                    res["dset"] += [f"ama_{r}" for r in residuals]

            for op in task_config.high_modes_component.operations:
                res["gamma_label"] = op.gamma.name.lower()
                res["mass"] = [submit_config.mass_out_label[m] for m in op.mass]
                yield res, outfile_dict["high_modes"]

    outfile_generator = generate_outfile_formatting()
    replacements = submit_config.string_dict()

    df = utils.catalog_files(outfile_generator, replacements)

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
    if task_config.epack_component:
        solver_labels.append("ranLL")
    if task_config.high_modes_component:
        if not task_config.high_modes_component.skip_cg:
            residuals = task_config.high_modes_component.cg_residual
            if len(residuals) == 1:
                solver_labels.append("ama")
            else:
                solver_labels += [f"ama_{r}" for r in residuals]

        for op in task_config.high_modes_component.operations:
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
                    "order": ["t"],
                    "labels": {"t": f"0..{submit_config.time - 1}"},
                }

                proc_params[file_label] = {
                    "logging_level": getattr(submit_config, "logging_level", "INFO"),
                    "load_files": {
                        "filestem": infile_stem,
                        "regex": {"series": "[a-z]", "cfg": "[0-9]+"},
                        "replacements": utils.deep_copy_dict(replacements),
                        "name": "gamma",
                        "datasets": h5_datasets,
                        **array_params,
                    },
                    "out_files": {"filestem": outfile, "type": "dataframe"},
                }

    return proc_params


def get_task_factory():
    return LMITask.from_dict
