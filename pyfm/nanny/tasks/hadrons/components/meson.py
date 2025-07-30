import pandas as pd
from pydantic.dataclasses import dataclass
from pyfm.nanny.tasks.hadrons.components import ComponentBase
from pyfm.nanny.tasks.hadrons.components import hadmods
from pyfm.nanny.tasks.hadrons import SubmitHadronsConfig
from pyfm import OpList, Gamma, utils
import typing as t


@dataclass
class MesonHadronsComponent(ComponentBase):
    """Component for handling meson field generation in Hadrons tasks.
    
    This component manages the creation of meson fields with various gamma
    operations and mass combinations for low-mode inversions.
    """
    
    _operations: OpList
    
    @classmethod
    def from_dict(cls, kwargs: t.Dict[str, t.Any]) -> "MesonHadronsComponent":
        """Create MesonHadronsComponent from dictionary configuration.
        
        Parameters
        ----------
        kwargs : dict
            Configuration dictionary that can be parsed by OpList.from_dict
            
        Returns
        -------
        MesonHadronsComponent
            Configured meson component instance
        """
        return cls(_operations=OpList.from_dict(kwargs))
    
    @property
    def operations(self) -> t.List[OpList.Op]:
        """Get list of gamma operations."""
        return self._operations.op_list
    
    @property
    def mass(self) -> t.List[str]:
        """Get list of unique mass labels required by all operations."""
        return self._operations.mass
    
    def get_file_catalog(self, submit_config: SubmitHadronsConfig) -> pd.DataFrame:
        """Generate catalog of meson field output files.
        
        Parameters
        ----------
        submit_config : SubmitHadronsConfig
            Submission configuration containing file paths and parameters
            
        Returns
        -------
        pd.DataFrame
            DataFrame with file information including paths and metadata
        """
        def generate_outfile_formatting():
            """Generator for meson field file formatting parameters."""
            for op in self.operations:
                res = {
                    "gamma": op.gamma.gamma_list,
                    "mass": [submit_config.mass_out_label[m] for m in op.mass]
                }
                yield res, submit_config.files["meson_ll"]
        
        outfile_generator = generate_outfile_formatting()
        replacements = submit_config.string_dict()
        
        return utils.catalog_files(outfile_generator, replacements)
    
    def get_bad_files(self, submit_config: SubmitHadronsConfig) -> t.List[str]:
        """Get list of meson field files that don't meet size requirements.
        
        Parameters
        ----------
        submit_config : SubmitHadronsConfig
            Submission configuration
            
        Returns
        -------
        List[str]
            List of file paths that are incomplete or corrupted
        """
        df = self.get_file_catalog(submit_config)
        return list(df[(df["file_size"] >= df["good_size"]) != True]["filepath"])
    
    def filter_existing_operations(self, submit_config: SubmitHadronsConfig) -> None:
        """Remove operations that have already completed successfully.
        
        This method modifies the component in-place, removing mass entries
        and operations that already have valid output files.
        
        Parameters
        ----------
        submit_config : SubmitHadronsConfig
            Submission configuration to check against
        """
        if submit_config.overwrite_sources:
            return
            
        bad_files = self.get_bad_files(submit_config)
        outfile_dict = submit_config.files
        submit_conf_dict = submit_config.string_dict()
        meson_template = outfile_dict["meson_ll"].filename
        
        # Work backwards through operations to safely remove items
        for i, op in sorted(enumerate(self.operations[:]), reverse=True):
            for j, mass_label in sorted(enumerate(op.mass[:]), reverse=True):
                # Check if all gamma files for this mass exist and are valid
                meson_files = [
                    meson_template.format(
                        mass=submit_config.mass_out_label[mass_label],
                        gamma=g,
                        **submit_conf_dict,
                    )
                    for g in op.gamma.gamma_list
                ]
                
                # If none of the files are bad, remove this mass from the operation
                if not any([mf in bad_files for mf in meson_files]):
                    op.mass.pop(j)
            
            # If operation has no masses left, remove the entire operation
            if not op.mass:
                self._operations.op_list.pop(i)
    
    def input_params(self, submit_config: SubmitHadronsConfig) -> t.Dict[str, t.Dict]:
        """Generate Hadrons module parameters for meson field creation.
        
        Parameters
        ----------
        submit_config : SubmitHadronsConfig
            Submission configuration containing paths and parameters
            
        Returns
        -------
        Dict[str, Dict]
            Dictionary of module configurations keyed by module name
        """
        modules = {}
        
        if not self.operations:
            return modules
            
        outfile_dict = submit_config.files
        submit_conf_dict = submit_config.string_dict()
        meson_template = outfile_dict["meson_ll"].filestem
        
        for op in self.operations:
            op_type = op.gamma.name.lower()
            gauge = "" if op.gamma == Gamma.LOCAL else "gauge"
            
            for mass_label in op.mass:
                output = meson_template.format(
                    mass=submit_config.mass_out_label[mass_label], 
                    **submit_conf_dict
                )
                
                module_name = f"mf_{op_type}_mass_{mass_label}"
                modules[module_name] = hadmods.meson_field(
                    name=module_name,
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
        
        return modules