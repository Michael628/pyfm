from pydantic.dataclasses import dataclass
from pydantic import Field
from pyfm.nanny.tasks.hadrons.components import ComponentBase
import typing as t
from pyfm.nanny.config import OutfileList
from pyfm.nanny.tasks.hadrons.components import hadmods
from pyfm.nanny.tasks.hadrons import SubmitHadronsConfig


@dataclass
class EigHadronsComponent(ComponentBase):
    load: bool
    multifile: bool = False
    save_eigs: bool = False
    save_evals: bool = True
    masses: t.List[str] = Field(default_factory=list)

    def update(self, **kwargs):
        """Add parent's masses to this task."""
        if masses := kwargs.get("mass", []):
            for m in masses:
                if m not in self.masses:
                    self.masses.append(m)

    def input_params(
        self, submit_config: SubmitHadronsConfig, outfile_config_list: OutfileList
    ) -> t.Dict:
        submit_conf_dict = submit_config.string_dict()
        epack_path = ""
        multifile = str(self.multifile).lower()
        modules = {}
        if self.load or self.save_eigs:
            assert outfile_config_list.eig is not None
            epack_path = outfile_config_list.eig.filestem.format(**submit_conf_dict)

        # Load or generate eigenvectors
        if self.load:
            modules["epack"] = hadmods.epack_load(
                name="epack",
                filestem=epack_path,
                size=submit_conf_dict["eigs"],
                multifile=multifile,
            )
        else:
            modules["stag_op"] = hadmods.op("stag_op", "stag_mass_zero")
            modules["epack"] = hadmods.irl(
                name="epack",
                op="stag_op_schur",
                alpha=submit_conf_dict["alpha"],
                beta=submit_conf_dict["beta"],
                npoly=submit_conf_dict["npoly"],
                nstop=submit_conf_dict["nstop"],
                nk=submit_conf_dict["nk"],
                nm=submit_conf_dict["nm"],
                multifile=multifile,
                output=epack_path,
            )

        # Shift mass of eigenvalues
        for mass_label in self.masses:
            if mass_label == "zero":
                continue
            mass = str(submit_config.mass[mass_label])
            name = f"evecs_mass_{mass_label}"
            modules[name] = hadmods.epack_modify(
                name=name, eigen_pack="epack", mass=mass
            )

        if self.save_evals:
            assert outfile_config_list.eval is not None
            eval_path = outfile_config_list.eval.filestem.format(**submit_conf_dict)
            modules["eval_save"] = hadmods.eval_save(
                name="eval_save", eigen_pack="epack", output=eval_path
            )
        return modules
