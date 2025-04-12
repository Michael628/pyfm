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

from pyfm import Gamma, utils
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

    @property
    def has_eigs(self) -> bool:
        if self.eig_task is not None:
            return True
        return False

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
        schedule = [m["id"]["name"] for m in modules]

        return modules, schedule


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
