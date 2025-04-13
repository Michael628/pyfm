# This file generates xml parameters for the HadronsMILC app.
# Tasks performed:
#
# 1: Load eigenvectors
# 2: Generate noise sources
# 3: Solve low-mode propagation of sources
# 4: Solve CG on result of step 3
# 5: Subtract 3 from 4
# 6: Save result of 5 to disk
import logging
import typing as t

from pydantic.dataclasses import dataclass

from pyfm.nanny.config import OutfileList
from pyfm.nanny.tasks.hadrons import HadronsTaskBase, SubmitHadronsConfig
from pyfm.nanny.tasks.hadrons.components import gauge, eig, highmode


@dataclass
class SeqSIBTask(HadronsTaskBase):
    gauge_component: gauge.GaugeHadronsComponent
    highmode_component: highmode.HighModeHadronsComponent
    eig_component: t.Optional[eig.EigHadronsComponent] = None

    @classmethod
    def from_dict(cls, kwargs):
        assert "gauge" in kwargs
        assert "highmode" in kwargs
        params = {}
        params["gauge_component"] = gauge.GaugeHadronsComponent.from_dict(
            kwargs["gauge"]
        )
        has_eigs = "eig" in kwargs
        hc = highmode.HighModeHadronsComponent.from_dict(
            kwargs["highmode"] | {"has_eigs": has_eigs}
        )
        params["highmode_component"] = hc
        params["eig_component"] = None

        if has_eigs:
            params["eig_component"] = eig.EigHadronsComponent.from_dict(
                kwargs["eig"] | {"masses": hc.mass}
            )

        return cls(**params)

    def input_params(
        self,
        submit_config: SubmitHadronsConfig,
        outfile_config_list: OutfileList,
    ) -> t.Tuple[t.List[t.Dict], t.Optional[t.List[str]]]:
        modules = self.gauge_component.input_params(
            submit_config, outfile_config_list
        ) | self.highmode_component.input_params(submit_config, outfile_config_list)
        if self.eig_component is not None:
            modules |= self.eig_component.input_params(
                submit_config, outfile_config_list
            )

        # TODO: Get schedule in requisite order.
        # TODO: initialize highmode_component with proper strategy.
        schedule = list(modules.keys())
        mod_list = list(modules.values())

        return mod_list, schedule


# TODO: This assumes it's being handed the hadrons module <id> tag, i.e. containing both name and type.
# Could replace with name only syntax parsing? Or pass while id subdict.
def build_schedule(module_names: t.List[str]) -> t.List[str]:
    gammas = ["pion_local", "vec_local", "vec_onelink"]

    def pop_conditional(mi, cond):
        """Pop all items in mi that match cond"""
        indices = [i for i, item in enumerate(mi) if cond(item)]
        # Pop in reverse order but return in original order
        return [mi.pop(i) for i in indices[::-1]][::-1]

    # HACK: Assumes low mode meson fields only
    def get_mf_inputs(x):
        """match meson field inputs"""
        is_action = x["type"].endswith("ImprovedStaggeredMILC")
        is_evec = "ModifyEigenPack" in x["type"]
        has_sea_mass = "mass_l" in x["name"]
        return (is_action or is_evec) and has_sea_mass

    # Pop gauge modules
    dp_gauges = pop_conditional(module_names, lambda x: "LoadIldg" in x["type"])
    # Pop single precision gauge modules
    sp_gauges = pop_conditional(module_names, lambda x: "PrecisionCast" in x["type"])
    # Pop meson field modules
    meson_fields = pop_conditional(module_names, lambda x: "A2AMesonField" in x["type"])
    # Pop inputs for meson fields
    meson_field_inputs = pop_conditional(module_names, get_mf_inputs)

    # Pop modules that are indep of mass and time slices
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

    # sort by tslice > mixed mass > mass > gammas
    module_names = sorted(module_names, key=gamma_order)
    module_names = sorted(module_names, key=mass_order)
    module_names = sorted(module_names, key=mixed_mass_last)
    module_names = sorted(module_names, key=tslice_order)

    sorted_modules += module_names

    return [m["name"] for m in sorted_modules]


def bad_files(
    task_config: HadronsTaskBase,
    submit_config: SubmitHadronsConfig,
    outfile_config_list: OutfileList,
) -> t.List[str]:
    logging.warning(
        "Check completion succeeds automatically. No implementation of bad_files function in `hadrons_a2a_vectors.py`."
    )
    return []


def get_task_factory():
    return SeqSIBTask.from_dict
