from enum import Enum, auto
import random
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


class SourceRef(t.NamedTuple):
    """Identity of one wall source.

    ``label`` is the module-name suffix (``t{tsource}`` in dt mode, ``n{block}``
    in bias mode); ``axis`` is the ``{tsource}`` replacement value used by
    catalogs, filestems, and the aggregator (bare time strings in dt mode,
    block labels in bias mode); ``t0`` is the physical source time.
    """

    label: str
    axis: str
    t0: int


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
    nbias: int | None = None
    bias_seed: str | None = None
    bias_replace: bool = True

    @property
    def tsource_range(self) -> t.List[int]:
        """Source times: dt-spaced by default, else ``nbias`` seeded draws.

        In bias mode the draws default to **with replacement** (duplicate time
        slices are distinct sources); ``bias_replace=False`` samples without
        replacement via ``rng.sample`` (requires ``nbias <= time``). Either
        way they are a pure function of the stored ``bias_seed`` string, so
        generation, completion checks, and aggregation re-derive the same list.
        """
        if self.nbias is not None:
            if self.bias_seed is None:
                raise ValueError(
                    "bias_seed is required when nbias is set; refusing to draw "
                    "from an unseeded RNG."
                )
            rng = random.Random(self.bias_seed)
            if self.bias_replace:
                return [rng.randrange(self.time) for _ in range(self.nbias)]
            if self.nbias > self.time:
                raise ValueError(
                    f"nbias ({self.nbias}) exceeds the time extent "
                    f"({self.time}); without-replacement sampling "
                    "(bias_replace=False) requires nbias <= time."
                )
            return rng.sample(range(self.time), self.nbias)
        return list(range(self.tstart, self.tstop + 1, self.dt))

    @property
    def source_labels(self) -> t.List[str]:
        """Per-source module-name suffixes: ``t{tsource}`` (dt) or ``n{block}`` (bias)."""
        if self.nbias is not None:
            return [f"n{i}" for i in range(self.nbias)]
        return [f"t{t}" for t in self.tsource_range]

    @property
    def source_axis(self) -> t.List[str]:
        """``{tsource}`` replacement values: bare times (dt) or block labels (bias).

        Unique by construction in both modes, so catalogs, the resume gate, and
        aggregator replacement axes never double-count a source.
        """
        if self.nbias is not None:
            return self.source_labels
        return [str(t) for t in self.tsource_range]

    @property
    def source_refs(self) -> t.List[SourceRef]:
        """Config-owned enumeration of all sources (see ``SourceRef``)."""
        return [
            SourceRef(label=label, axis=axis, t0=t0)
            for label, axis, t0 in zip(
                self.source_labels, self.source_axis, self.tsource_range
            )
        ]

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
