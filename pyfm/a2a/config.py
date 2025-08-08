import typing as t
from enum import Enum, auto

import pyfm
from pyfm.nanny import Outfile
from pyfm import config as c

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
except ImportError:
    COMM = None


class Diagrams(Enum):
    photex = auto()
    selfen = auto()


@c.dataclass_with_getters
class DiagramConfig(pyfm.ConfigBase):
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
    _has_high: bool = False
    _has_low: bool = False
    _npoint: int = -1
    evalfile: t.Optional[str] = None
    outfile: t.Optional[str] = None
    mesonfiles: t.Optional[t.Union[t.List[str] | str]] = None

    def __post_init__(self):
        if self.high_label and self.high_count:
            self._has_high = True

        if self.low_label:
            self._has_low = True

        npoint = {"conn_2pt": 2, "sib_conn_3pt": 3, "qed_conn_4pt": 4}
        self._npoint = npoint[self.contraction_type]

        self._meson_params = {
            "wmax_index": self.low_max,
            "vmax_index": self.low_max,
            "milc_mass": True,
        }

        if self.meson_mass and self.mass != self.meson_mass:
            self._meson_params["shift_mass"] = True
            self._meson_params["oldmass"] = float(f"0.{self.meson_mass}")
            self._meson_params["newmass"] = float(f"0.{self.mass}")
        else:
            self.meson_mass = self.mass

    def set_filenames(self, outfile_config: t.Dict[str, Outfile]) -> None:
        """Uses 'outfile_config' argument to replace provided parameters with
        filenames if the existing parameter matches a field in `outfile_config`."""

        def get_filename(s: str):
            if s in outfile_config:
                return outfile_config[s].filename
            else:
                return s

        self.mesonfiles = (
            get_filename(self.mesonfiles)
            if isinstance(self.mesonfiles, str)
            else [get_filename(m) for m in self.mesonfiles]
        )

        if self.evalfile:
            self.evalfile = get_filename(self.evalfile)

        self.outfile = get_filename(self.outfile)

    @classmethod
    def create(cls, **kwargs):
        obj_vars = kwargs.copy()

        obj = super().create(**obj_vars)
        # obj.outfile = obj_vars.pop('outfile')
        assert isinstance(obj.outfile, str)

        # obj.evalfile = obj_vars.pop('evalfile',None)
        if obj.evalfile:
            assert isinstance(obj.evalfile, str)

        # mesonfiles = obj_vars.pop('mesonfiles')
        if isinstance(obj.mesonfiles, str):
            obj.mesonfiles = [obj.mesonfiles]

        assert isinstance(obj.mesonfiles, t.List)

        if len(obj.mesonfiles) == 1:
            obj.mesonfiles = obj.mesonfiles * obj.npoint
        assert obj.npoint == len(obj.mesonfiles)

        return obj

    @property
    def meson_params(self):
        return self._meson_params

    def format_evalfile(self, **kwargs) -> None:
        self.meson_params["evalfile"] = self.evalfile.format(**kwargs)


@c.dataclass_with_getters
class RunContractConfig(pyfm.ConfigBase):
    time: int
    ens: str
    series: str
    cfg: str
    _diagrams: t.Optional[t.List[DiagramConfig]] = None
    _overwrite_correlators: bool = True
    _hardware: str = "cpu"
    _logging_level: str = "INFO"
    _comm_size: int = 1
    _rank: int = 0

    @classmethod
    def create(cls, **kwargs):
        obj_vars = kwargs.copy()
        diagrams = obj_vars.pop("diagrams")
        obj = super().create(**obj_vars)
        obj.diagrams = [
            DiagramConfig.create(**v, run_vars=obj.string_dict())
            for k, v in diagrams.items()
        ]
        if COMM:
            obj.rank = COMM.Get_rank()
            obj.comm_size = COMM.Get_size()

        return obj


def get_contract_config(params: t.Dict) -> RunContractConfig:
    return RunContractConfig.create(**params)
