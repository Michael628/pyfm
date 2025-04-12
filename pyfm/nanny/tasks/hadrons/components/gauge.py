import typing as t

from pydantic.dataclasses import dataclass

from pyfm import Gamma, utils
from pyfm.nanny.tasks.hadrons.components import ComponentBase
from pyfm.nanny.config import OutfileList
from pyfm.nanny.tasks.hadrons.components import hadmods
from pyfm.nanny.tasks.hadrons import SubmitHadronsConfig


@dataclass
class GaugeHadronsComponent(ComponentBase):
    free: bool = False

    def input_params(
        self,
        submit_config: SubmitHadronsConfig,
        outfile_config_list: OutfileList,
    ) -> t.Dict:
        submit_conf_dict = submit_config.string_dict()
        modules = {}
        for name in ["gauge", "gauge_fat", "gauge_long"]:
            if self.free:
                modules[name] = [hadmods.unit_gauge(name)]
            else:
                gauge_outfile = getattr(outfile_config_list, name)
                assert isinstance(gauge_outfile, OutfileList.Outfile)
                gauge_filepath = gauge_outfile.filestem.format(**submit_conf_dict)

                modules[name] = hadmods.load_gauge(name, gauge_filepath)
        modules["gauge_fatf"] = hadmods.cast_gauge("gauge_fatf", "gauge_fat")
        modules["gauge_longf"] = hadmods.cast_gauge("gauge_longf", "gauge_long")

        return modules
