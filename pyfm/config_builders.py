import typing as t
from dataclasses import dataclass, field

from pyfm import ConfigBuilder


@dataclass
class OpList:
    """Configuration for a list of gamma operations.

    Attributes
    ----------
    op_list: list
        Gamma operations to be performed, usually for meson fields or high mode solves.
    """

    @dataclass
    class Op:
        """Parameters for a gamma operation and associated masses."""

        gamma: Gamma
        mass: t.List[str]

    op_list: t.List[Op]

    @classmethod
    def from_dict(cls, kwargs) -> "OpList":
        """Creates a new instance of OpList from a dictionary.

        Note
        ----
        Ignores input keys that do not match format.

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
        if "mass" not in kwargs:
            op_list = []
            for key, val in kwargs.items():
                if isinstance(val, dict) and "mass" in val:
                    mass = val["mass"]
                    if isinstance(mass, str):
                        mass = [mass]
                    gamma = Gamma[key.upper()]
                    op_list.append(cls.Op(gamma=gamma, mass=mass))
        else:
            assert "gamma" in kwargs
            assert "mass" in kwargs
            gammas = kwargs["gamma"]
            mass = kwargs["mass"]
            if isinstance(mass, str):
                mass = [mass]
            if isinstance(gammas, str):
                gammas = [gammas]
            op_list = [cls.Op(gamma=Gamma[g.upper()], mass=mass) for g in gammas]

        return cls(op_list=op_list)

    @property
    def mass(self):
        res: t.Set = set()
        for op in self.op_list:
            for m in op.mass:
                res.add(m)

        return list(res)


@dataclass(frozen=True)
class SmearConfig:
    """Immutable configuration for smear tasks."""

    ens: str
    time: int
    space: int
    node_geometry: str
    files: t.Dict[str, t.Any]
    unsmeared_file: str
    series: str | None = None
    cfg: str | None = None


class SmearConfigBuilder(ConfigBuilder[SmearConfig]):
    """Builder for smear task configurations."""

    def with_ensemble_params(self, ens: str, space: int, time: int):
        return (
            self.with_field("ens", ens)
            .with_field("space", space)
            .with_field("time", time)
        )

    def with_geometry(self, node_geometry: str):
        return self.with_field("node_geometry", node_geometry)

    def with_files(self, files_config: t.Dict[str, t.Any]):
        return self.with_field("files", files_config)

    def with_unsmeared_file(self, file_path: str):
        return self.with_field("unsmeared_file", file_path)

    def build(self) -> SmearConfig:
        required = ["ens", "time", "space", "node_geometry", "files"]
        missing = [f for f in required if f not in self._config_data]
        if missing:
            raise ValueError(f"Missing required fields for SmearConfig: {missing}")

        return SmearConfig(**self._config_data)


@dataclass(frozen=True)
class DiagramConfig:
    """Configuration for a single a2a meson field contraction diagram."""

    gamma_label: str
    contraction_type: str
    gammas: t.List[str]
    mass: str
    symmetric: bool = False
    meson_mass: t.Optional[str] = None
    high_count: t.Optional[int] = None
    high_label: t.Optional[str] = None
    low_max: t.Optional[int] = None
    low_label: t.Optional[str] = None
    mesonKey: t.Optional[str] = None
    emseedstring: t.Optional[str] = None
    perms: t.Optional[t.List[str]] = None
    n_em: t.Optional[int] = None
    evalfile: t.Optional[str] = None
    outfile: t.Optional[str] = None
    mesonfiles: t.Optional[t.Union[t.List[str], str]] = None


@dataclass(frozen=True)
class ContractConfig:
    """Immutable configuration for contract tasks."""

    ens: str
    time: int
    diagrams: t.Dict[str, DiagramConfig]
    files: t.Dict[str, t.Any]
    hardware: str = "cpu"
    logging_level: str = "INFO"
    overwrite_correlators: bool = True
    series: str | None = None
    cfg: str | None = None


class ContractConfigBuilder(ConfigBuilder[ContractConfig]):
    """Builder for contract task configurations."""

    def __init__(self):
        super().__init__()
        self._diagrams: t.Dict[str, DiagramConfig] = {}

    def with_ensemble_params(self, ens: str, time: int):
        return self.with_field("ens", ens).with_field("time", time)

    def with_hardware(self, hardware: str):
        return self.with_field("hardware", hardware)

    def with_logging_level(self, level: str):
        return self.with_field("logging_level", level)

    def with_files(self, files_config: t.Dict[str, t.Any]):
        return self.with_field("files", files_config)

    def with_diagram(self, name: str, diagram_config: DiagramConfig):
        self._diagrams[name] = diagram_config
        return self

    def with_diagrams_from_dict(self, diagrams_dict: t.Dict[str, t.Dict[str, t.Any]]):
        for name, config in diagrams_dict.items():
            diagram = DiagramConfig(**config)
            self._diagrams[name] = diagram
        return self

    def build(self) -> ContractConfig:
        """Build the final ContractConfig."""
        required = ["ens", "time", "files"]
        missing = [f for f in required if f not in self._config_data]
        if missing:
            raise ValueError(f"Missing required fields for ContractConfig: {missing}")

        self._config_data["diagrams"] = self._diagrams
        return ContractConfig(**self._config_data)


@dataclass(frozen=True)
class HadronsComponent:
    """Base class for hadrons components."""

    pass


@dataclass(frozen=True)
class GaugeComponent(HadronsComponent):
    """Gauge component configuration."""

    pass


@dataclass(frozen=True)
class EigComponent(HadronsComponent):
    """Eigenvalue component configuration."""

    load: bool = True
    save_evals: bool = False


@dataclass(frozen=True)
class MesonComponent(HadronsComponent):
    """Meson component configuration."""

    gamma: t.List[str]
    mass: t.List[str]


@dataclass(frozen=True)
class HighModeComponent(HadronsComponent):
    """High mode component configuration."""

    gamma: t.List[str]
    mass: t.List[str]
    has_eigs: bool = False


@dataclass(frozen=True)
class HadronsConfig:
    """Immutable configuration for hadrons tasks."""

    ens: str
    time: int
    files: t.Dict[str, t.Any]
    components: t.Dict[str, HadronsComponent]
    task_type: str = "lmi"  # Default task type
    mass: t.Dict[str, float] = field(default_factory=dict)
    tstart: int = 0
    tstop: int = 1
    dt: int = 1
    noise: int = 1
    eigs: int = 1000
    sourceeigs: int = 1000
    alpha: float = 0.009
    beta: int = 24
    npoly: int = 81
    nstop: int = 1000
    nk: int = 1030
    nm: int = 1600
    cg_residual: float = 1e-8
    eigresid: float = 1e-8
    blocksize: int = 500
    overwrite_sources: bool = False
    run_id: str = "LMI-RW-series-{series}-{eigs}-eigs-{noise}-noise"
    series: str | None = None
    cfg: str | None = None


class HadronsConfigBuilder(ConfigBuilder[HadronsConfig]):
    """Builder for complex hadrons configurations with components."""

    def __init__(self):
        super().__init__()
        self._components: t.Dict[str, HadronsComponent] = {}

    def with_ensemble_params(self, ens: str, time: int):
        """Set ensemble parameters."""
        return self.with_field("ens", ens).with_field("time", time)

    def with_files(self, files_config: t.Dict[str, t.Any]):
        """Set file configurations."""
        return self.with_field("files", files_config)

    def with_mass_params(self, mass_dict: t.Dict[str, float]):
        """Set mass parameters."""
        return self.with_field("mass", mass_dict)

    def with_gauge_component(self, **kwargs):
        """Add gauge component."""
        self._components["gauge"] = GaugeComponent(**kwargs)
        return self

    def with_eig_component(self, load: bool = True, save_evals: bool = False):
        """Add eigenvalue component."""
        self._components["epack"] = EigComponent(load=load, save_evals=save_evals)
        return self

    def with_meson_component(self, gamma: t.List[str], mass: t.List[str]):
        """Add meson component."""
        self._components["meson"] = MesonComponent(gamma=gamma, mass=mass)
        return self

    def with_high_modes_component(self, gamma: t.List[str], mass: t.List[str]):
        """Add high modes component."""
        has_eigs = "epack" in self._components
        self._components["high_modes"] = HighModeComponent(
            gamma=gamma, mass=mass, has_eigs=has_eigs
        )
        return self

    def build(self) -> HadronsConfig:
        """Build the final HadronsConfig."""
        required = ["ens", "time", "cfg", "files"]
        missing = [f for f in required if f not in self._config_data]
        if missing:
            raise ValueError(f"Missing required fields for HadronsConfig: {missing}")

        self._config_data["components"] = self._components
        return HadronsConfig(**self._config_data)


class TaskConfigFactory:
    """Factory for creating task configurations from YAML."""

    _builders = {
        "smear": SmearConfigBuilder,
        "contract": ContractConfigBuilder,
        "hadrons": HadronsConfigBuilder,
    }

    @classmethod
    def create_config(
        cls,
        job_type: str,
        yaml_data: t.Dict[str, t.Any],
        series: str,
        cfg: str,
        task_type: t.Optional[str] = None,
    ) -> t.Union[SmearConfig, ContractConfig, HadronsConfig]:
        """Create configuration from YAML data."""
        if job_type not in cls._builders:
            raise ValueError(
                f"Unknown job type: {job_type}. Available: {list(cls._builders.keys())}"
            )

        builder = cls._builders[job_type]()

        # Store task_type for hadrons subtasks
        if job_type == "hadrons" and task_type:
            builder.with_field("task_type", task_type)

        # Load common submit parameters
        builder.with_yaml_section(yaml_data, "submit_params")

        # Load job-specific parameters
        builder.with_yaml_section(yaml_data, f"{job_type}_params")

        # Add runtime parameters
        builder.with_field("series", series).with_field("cfg", cfg)

        # Load file configurations
        if "files" in yaml_data:
            builder.with_files(cls._process_files_config(yaml_data["files"]))

        # Job-specific assembly
        if job_type == "hadrons":
            cls._assemble_hadrons_components(builder, yaml_data)
        elif job_type == "contract":
            cls._assemble_contract_diagrams(builder, yaml_data)
        elif job_type == "smear":
            cls._assemble_smear_tasks(builder, yaml_data, series, cfg)

        return builder.build()

    @classmethod
    def _process_files_config(
        cls, files_config: t.Dict[str, t.Any]
    ) -> t.Dict[str, t.Any]:
        """Process file configurations, handling home directory and file stems."""
        processed = {}
        home = files_config.get("home", "")

        for key, config in files_config.items():
            if key == "home":
                continue

            if isinstance(config, dict) and "filestem" in config:
                processed[key] = {
                    "file_path": home,
                    "filestem": config["filestem"],
                    "good_size": config.get("good_size", 0),
                }
            else:
                processed[key] = config

        return processed

    @classmethod
    def _assemble_hadrons_components(
        cls, builder: HadronsConfigBuilder, yaml_data: t.Dict[str, t.Any]
    ):
        """Assemble hadrons components from YAML."""
        # Load hadrons parameters
        if "hadrons_params" in yaml_data:
            hadrons_params = yaml_data["hadrons_params"]
            if "mass" in hadrons_params:
                builder.with_mass_params(hadrons_params["mass"])

            # Set other hadrons parameters
            for key, value in hadrons_params.items():
                if key != "mass":
                    builder.with_field(key, value)

        # Load job setup tasks
        job_setup = yaml_data.get("job_setup", {}).get("hadrons", {})
        tasks = job_setup.get("tasks", {})

        # Always add gauge component
        builder.with_gauge_component()

        if "epack" in tasks:
            epack_config = tasks["epack"]
            builder.with_eig_component(
                load=epack_config.get("load", True),
                save_evals=epack_config.get("save_evals", False),
            )

        if "meson" in tasks:
            meson_config = tasks["meson"]
            builder.with_meson_component(
                gamma=meson_config.get("gamma", []), mass=meson_config.get("mass", [])
            )

        if "high_modes" in tasks:
            high_modes_config = tasks["high_modes"]
            builder.with_high_modes_component(
                gamma=high_modes_config.get("gamma", []),
                mass=high_modes_config.get("mass", []),
            )

    @classmethod
    def _assemble_contract_diagrams(
        cls, builder: ContractConfigBuilder, yaml_data: t.Dict[str, t.Any]
    ):
        """Assemble contract diagrams from YAML."""
        # Load contract parameters
        if "contract_params" in yaml_data:
            contract_params = yaml_data["contract_params"]
            if "hardware" in contract_params:
                builder.with_hardware(contract_params["hardware"])
            if "logging_level" in contract_params:
                builder.with_logging_level(contract_params["logging_level"])

            # Load diagram parameters - don't add to config_data directly
            if "diagram_params" in contract_params:
                builder.with_diagrams_from_dict(contract_params["diagram_params"])
                # Remove from config_data to avoid passing to ContractConfig.__init__
                if "diagram_params" in builder._config_data:
                    del builder._config_data["diagram_params"]

    @classmethod
    def _assemble_smear_tasks(
        cls,
        builder: SmearConfigBuilder,
        yaml_data: t.Dict[str, t.Any],
        series: str,
        cfg: str,
    ):
        """Assemble smear task configuration from YAML."""
        # Load smear parameters
        if "smear_params" in yaml_data:
            smear_params = yaml_data["smear_params"]
            if "space" in smear_params:
                builder.with_field("space", smear_params["space"])
            if "node_geometry" in smear_params:
                builder.with_field("node_geometry", smear_params["node_geometry"])

        # Set up unsmeared file path from job setup
        job_setup = yaml_data.get("job_setup", {}).get("smear", {})
        tasks = job_setup.get("tasks", {})
        if "unsmeared_file" in tasks:
            unsmeared_template = tasks["unsmeared_file"]
            # Format the template with current parameters
            template_vars = builder.get_string_dict()
            template_vars.update({"series": series, "cfg": cfg})
            unsmeared_file = unsmeared_template.format(**template_vars)
            builder.with_unsmeared_file(unsmeared_file)
