import typing as t
from enum import Enum, auto
from pydantic.dataclasses import dataclass


class Gamma(Enum):
    G1_G1 = auto()
    G5_G5 = auto()
    GX_GX = auto()
    GY_GY = auto()
    GZ_GZ = auto()
    GX_G1 = auto()
    GY_G1 = auto()
    GZ_G1 = auto()
    G5X_G5X = auto()
    G5Y_G5Y = auto()
    G5Z_G5Z = auto()
    G5X_G5 = auto()
    G5Y_G5 = auto()
    G5Z_G5 = auto()
    AXIAL_VEC_ONELINK = auto()
    AXIAL_VEC_LOCAL = auto()
    VEC_ONELINK = auto()
    VEC_LOCAL = auto()
    PION_LOCAL = auto()
    IDENTITY = auto()
    ONELINK = VEC_ONELINK
    LOCAL = auto()

    @property
    def gamma_list(self) -> t.List[str]:
        match self:
            case Gamma.ONELINK | Gamma.VEC_ONELINK:
                return ["GX_G1", "GY_G1", "GZ_G1"]
            case Gamma.LOCAL:
                return ["G5_G5", "GX_GX", "GY_GY", "GZ_GZ"]
            case Gamma.AXIAL_VEC_LOCAL:
                return ["G5X_G5X", "G5Y_G5Y", "G5Z_G5Z"]
            case Gamma.AXIAL_VEC_ONELINK:
                return ["G5X_G5", "G5Y_G5", "G5Z_G5"]
            case Gamma.VEC_LOCAL:
                return ["GX_GX", "GY_GY", "GZ_GZ"]
            case Gamma.IDENTITY:
                return ["G5_G5"]
            case Gamma.PION_LOCAL:
                return ["G5_G5"]
            case _:
                return [self.name]
            # raise ValueError(f"Unexpected Gamma value: {self}")

    @property
    def gamma_string(self) -> str:
        gammas = self.gamma_list
        gammas = [f"({gamma})" for gamma in gammas]
        gammas = " ".join(gammas)
        gammas = gammas.replace("_", " ")
        return gammas

    @property
    def _local_gammas(self) -> t.List:
        return [
            Gamma.LOCAL,
            Gamma.PION_LOCAL,
            Gamma.VEC_LOCAL,
            Gamma.AXIAL_VEC_LOCAL,
            Gamma.IDENTITY,
            Gamma.G1_G1,
            Gamma.GX_GX,
            Gamma.GY_GY,
            Gamma.GZ_GZ,
            Gamma.G5_G5,
        ]

    @property
    def local(self) -> bool:
        if self in self._local_gammas:
            return True
        else:
            return False


@dataclass
class OpList:
    class Op(t.NamedTuple):
        gamma: Gamma
        mass: t.Tuple[str, ...]

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
                    op_list.append(cls.Op(gamma=gamma, mass=tuple(mass)))
        else:
            if "gamma" not in kwargs:
                raise ValueError(
                    "No gamma provided. Required for using OpList.from_dict."
                )
            gammas = kwargs["gamma"]
            mass = kwargs["mass"]
            if isinstance(mass, str):
                mass = [mass]
            if isinstance(gammas, str):
                gammas = [gammas]
            op_list = [cls.Op(gamma=Gamma[g.upper()], mass=tuple(mass)) for g in gammas]

        return cls(op_list=op_list)

    @property
    def mass(self):
        res: t.Set = set()
        for op in self.op_list:
            for m in op.mass:
                res.add(m)

        return list(res)

    def __iter__(self):
        return iter(self.op_list)
