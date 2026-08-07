import typing as t
from enum import auto

from pyfm.domain import Outfile, MassDict, SerializableEnum, CompositeConfig, SimpleConfig
from pydantic.dataclasses import dataclass


def get_comm():
    """Return the MPI communicator, or None if mpi4py is unavailable.

    Importing ``mpi4py.MPI`` runs ``MPI_Init``, which aborts on hosts without an
    MPI fabric (e.g. HPC login nodes). Import it lazily here so that merely
    importing this module -- as the task registry does on every CLI invocation
    -- never initializes MPI. Initialization happens only when contraction code
    first asks for the communicator.
    """
    try:
        from mpi4py import MPI
    except ImportError:
        return None
    return MPI.COMM_WORLD


class ContractType(SerializableEnum):
    TWOPOINT = auto()
    SIB = auto()
    PHOTEX = auto()
    SELFEN = auto()

    @property
    def npoint(self) -> int:
        match self:
            case ContractType.TWOPOINT:
                return 2

@dataclass(frozen=True)
class MesonLoaderConfig(SimpleConfig):
    class MassShift(t.NamedTuple):
        original: str
        updated: str | None = None
        milc_mass: bool = True

        @classmethod
        def from_dict(cls, kwargs) -> "MassShift":
            return cls(**kwargs)

    mass: MassDict
    file: Outfile
    mass_shift: MassShift
    evalfile: Outfile | None = None

    def __post_init__(self):
        for label in [self.mass_shift.original, self.mass_shift.updated]:
            if label is not None and label not in self.mass:
                raise ValueError(
                    f"Provided mass label ({label}) not present in mass param."
                )

        if self.mass_shift.updated is not None and self.evalfile is None:
            raise ValueError(
                f"No eigenvalue file provided when shifting mass to {self.mass_shift.updated}"
            )

    def get_mass_label(self, include_shift: bool = True) -> str:
        if include_shift and self.mass_shift.updated is not None:
            return self.mass.to_string(self.mass_shift.updated, True)
        else:
            return self.mass.to_string(self.mass_shift.original, True)


@dataclass(frozen=True)
class DiagramConfig(CompositeConfig):
    class MesonIndex(t.NamedTuple):
        max: int = -1
        min: int = 0

        @classmethod
        def from_dict(cls, kwargs) -> "MassShift":
            return cls(**kwargs)

    time: int
    contraction_type: ContractType
    mesons: t.List[MesonLoaderConfig]
    outfile: Outfile
    gammas: t.List[str]
    eig_range: MesonIndex | None = None
    stoch_range: MesonIndex | None = None
    symmetric: bool = False
    perms: t.List[str] | None = None
    stoch_seed_indices: t.List[str] | None = None
    efield_indices: t.List[str] | None = None

    def __post_init__(self):
        if self.eig_range is None and self.stoch_range is None:
            raise ValueError("Must provide either eig_range or stoch_range")

        if self.stoch_range is not None and self.stoch_seed_indices is None:
            raise ValueError("Must provide stoch_seed_indices when using stoch_range")

        if self.contraction_type.npoint != len(self.mesons):
            if len(self.mesons) == 1:
                _ = [self.mesons.append(self.mesons[0]) for _ in range(self.npoint - 1)]
            else:
                raise ValueError(
                    f"Expected 1 or {self.npoint} meson configs, got {len(self.mesons)}"
                )

    @property
    def npoint(self) -> int:
        return self.contraction_type.npoint

    @property
    def has_low(self) -> bool:
        return self.eig_range is not None

    @property
    def has_high(self) -> bool:
        return self.stoch_range is not None

    @property
    def mass_label(self) -> str:
        return "_m".join(dict.fromkeys(m.get_mass_label() for m in self.mesons))


@dataclass(frozen=True)
class ContractConfig(CompositeConfig):
    diagrams: t.Dict[str, DiagramConfig]
    time: int
    overwrite: bool = True
    hardware: str = "cpu"

    @property
    def comm_size(self) -> int:
        comm = get_comm()
        if comm:
            return comm.Get_size()
        return 1

    @property
    def rank(self) -> int:
        comm = get_comm()
        if comm:
            return comm.Get_rank()
        return 0
