from enum import Enum, auto
import typing as t
from pydantic.dataclasses import dataclass
from dataclasses import fields

from pydantic import Field

from pyfm.domain import (
    SimpleConfig,
    Outfile,
    OpList,
    MassDict,
    FromDictProtocol, SerializableEnum
)


class HadronsInput(t.NamedTuple):
    modules: t.Dict[str, t.Dict]
    schedule: t.List[str]


@dataclass(frozen=True)
class LanczosParams(FromDictProtocol):
    alpha: float
    beta: float
    npoly: int
    nstop: int
    nk: int
    nm: int
    residual: float = 1e-8

    def keys(self):
        return [field.name for field in fields(self)]

    def __getitem__(self, key):
        return getattr(self, key)

    def values(self):
        return [getattr(self, k) for k in self.keys()]

    def items(self):
        return [(k, getattr(self, k)) for k in self.keys()]

    def to_string(self) -> t.Dict:
        return {k: str(v) for k, v in self.items()}


class CrossTerms(SerializableEnum):
    NONE = 0
    MASS = 1
    SOLVE = 2
    ALL = 3

class CorrelatorStrategy(Enum):
    TWOPOINT = auto()
    SIB = auto()


@dataclass(frozen=True)
class HighModeConfig(SimpleConfig):
    mass: MassDict
    action_name: str
    solver_name: str
    low_modes_name: str
    operations: OpList
    high_modes: Outfile
    tstart: int
    tstop: int
    dt: int
    noise: int
    time: int
    cross_terms: CrossTerms = CrossTerms.NONE
    shift_gauge_name: str | None = None
    skip_low_modes: bool = False
    skip_cg: bool = False
    solver: str = "mpcg"
    overwrite: bool = False
    correlator_strategy: CorrelatorStrategy = CorrelatorStrategy.TWOPOINT
    residual: t.List[float] = Field(default=[1e-8])
    split_mpi_layout: str | None = None
    subgrid_ranks: int | None = None

    @property
    def tsource_range(self) -> t.List[int]:
        return list(range(self.tstart, self.tstop + 1, self.dt))

    @property
    def op_list(self) -> t.List[OpList.Op]:
        """Get list of gamma operations."""
        return self.operations.op_list

    @property
    def masses(self) -> t.List[str]:
        return self.operations.mass

    def get_mass_labels(self, op:OpList.Op, skip_cross: bool = False) -> t.List[str]:
        mass_labels = [self.mass.to_string(m, True) for m in op.mass]
        if not skip_cross and self.cross_terms in (CrossTerms.MASS, CrossTerms.ALL):
            cross_labels = [f"{mass_labels[j]}_m{a}" for i,a in enumerate(mass_labels) for j in range(i)]
            mass_labels += cross_labels
        return mass_labels

    def get_solver_labels(self, skip_cross: bool = False) -> t.List[str]:
        solver_labels = []
        if not self.skip_low_modes:
            solver_labels.append("ranLL")

        if not self.skip_cg:
            residuals = self.residual
            if len(residuals) == 1:
                solver_labels.append("ama")
            else:
                solver_labels += [f"ama_{r}" for r in residuals]

        if not skip_cross and self.cross_terms in (CrossTerms.SOLVE, CrossTerms.ALL):
            cross_labels = [f"{a}_{b}" for a in solver_labels for b in solver_labels if a != b]
            solver_labels += cross_labels

        return solver_labels
