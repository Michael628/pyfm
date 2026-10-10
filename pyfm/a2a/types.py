import typing as t
from enum import auto

from pyfm.domain import Outfile, MassDict, SerializableEnum, CompositeConfig, SimpleConfig
from pyfm.domain.ops import Gamma, OpList
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
            case ContractType.SIB:
                return 3
            case _:
                raise ValueError(f"npoint not defined for {self.name}")

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


SIB_TERM_LABELS = ("lll", "nll", "lnl", "lln", "nnl", "nln", "lnn", "nnn")
"""The eight SIB three-point terms, id'd by their junction letters.

Each term id concatenates the junction types (J12, J23, J31) between the
three fields (outer1@t1, scalar-middle@t2, outer3@t3): ``l`` = eig<->eig
junction (plain index contraction), ``n`` = noise junction (eta rows x h/p
columns, slice pinned to the eta-side field's time row). Field blocks per
term (h-legs carry the derived p-side subtraction; ``nl`` is emitted for
vector families and derived for the scalar family):

- ``lll``: (ll, ll, ll)                       — plain einsum
- ``nll``: (pure_lh, nl, ll)  summed over worlds at J12
- ``lnl``: (ll, pure_lh, nl)  summed over worlds at J23
- ``lln``: (nl, ll, pure_lh)  summed over worlds at J31
- ``nnl``: (pure_lh, pure_nh, nl)  summed over ordered world pairs i!=j
- ``nln``: (pure_nh, nl, pure_lh)  summed over ordered world pairs i!=j
- ``lnn``: (nl, pure_lh, pure_nh)  summed over ordered world pairs i!=j
- ``nnn``: (pure_nh, pure_nh, pure_nh) summed over pairwise-distinct triples

World-selection normalizers (applied at aggregation, not in the kernel):
``lll`` 1, one-n 1/N, two-n 1/(N(N-1)), ``nnn`` 1/(N(N-1)(N-2)).
"""


@dataclass(frozen=True)
class SIBDiagramConfig(SimpleConfig):
    """One SIB three-point diagram: V(t1)·S(t2)·V(t3) per outer GammaName.

    ``operations`` selects the outer Gamma families (scalar_local /
    vec_local / vec_onelink) and correlator masses, mirroring the producer's
    ``hadrons_sib_mf`` operations block. ``blocks``/``tab``/``evalfile`` are
    the producer's own file labels — the split-noise layout is REQUIRED
    (validation rejects stems lacking the ``{n_index}``/``{hp_index}`` world
    tokens): the contraction derives the removed lp/np/scalar-nl blocks
    offline via the pair-basis identity (``pyfm/a2a/sib_derive.py``).
    ``noise`` is the world count; h/p columns are slice-major ``3*t + c``
    over the full lattice (full-volume noise — upstream ``StagRandomWall``
    cannot express sub-extent windows), so the column block for time ``t``
    is ``3*t..3*t+3`` and shape checks enforce the width. ``outfile``
    targets the per-term correlator files and must carry
    ``{mass}`` and ``{gamma}`` tokens (terms ride inside as frame labels).
    """

    contraction_type: ContractType
    operations: OpList
    mass: MassDict
    blocks: Outfile
    tab: Outfile
    evalfile: Outfile
    outfile: Outfile
    noise: int
    defl_mass: str = "l"
    symmetric: bool = False

    @property
    def op_list(self) -> t.List[OpList.Op]:
        return self.operations.op_list

    @property
    def npoint(self) -> int:
        return self.contraction_type.npoint

    @property
    def correlator_masses(self) -> t.List[str]:
        """Mass labels this diagram contracts, in first-occurrence order."""
        seen: t.List[str] = []
        for op in self.op_list:
            for m in op.mass:
                if m not in seen:
                    seen.append(m)
        return seen


@dataclass(frozen=True)
class SIBContractConfig(CompositeConfig):
    """Config tree for the ``contract_sib`` task (sibling of ``contract``).

    A genuinely new class, not a reuse of ContractConfig: the registry's
    reverse map keys by class object, and the builder's DICT children are
    homogeneous — SIB diagrams build through SIBDiagramConfig's own hooks.
    """

    diagrams: t.Dict[str, SIBDiagramConfig]
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
