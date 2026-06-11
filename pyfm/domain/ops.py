import typing as t
from enum import Enum, auto
from pydantic.dataclasses import dataclass

from pyfm import utils


@dataclass(frozen=True)
class MassDict:
    _items: t.Dict[str, float]

    @classmethod
    def from_dict(cls, kwargs) -> "MassDict":
        default = {"zero": 0.0}
        return cls(_items=default | kwargs)

    def __contains__(self, key):
        return key in self._items

    def __getitem__(self, key):
        return self._items[key]

    def keys(self):
        return self._items.keys()

    def values(self):
        return self._items.values()

    def items(self):
        return self._items.items()

    def to_string(self, mass_label: str, remove_prefix: bool = False) -> str:
        if remove_prefix:
            return str(self[mass_label]).removeprefix("0.")
        else:
            return str(self[mass_label])

    def _asdict(self) -> t.Dict:
        return self._items


class Gamma(Enum):
    G1_G1 = auto()
    G5_G5 = auto()
    GX_GX = auto()
    GY_GY = auto()
    GZ_GZ = auto()
    GT_GT = auto()
    GX_G1 = auto()
    GY_G1 = auto()
    GZ_G1 = auto()
    GT_G1 = auto()
    G5X_G5X = auto()
    G5Y_G5Y = auto()
    G5Z_G5Z = auto()
    G5T_G5T = auto()
    G5X_G5 = auto()
    G5Y_G5 = auto()
    G5Z_G5 = auto()
    G5T_G5 = auto()
    AXIAL_VEC_ONELINK = auto()
    AXIAL_VEC_LOCAL = auto()
    AXIAL_FOURVEC_ONELINK = auto()
    AXIAL_FOURVEC_LOCAL = auto()
    FOURVEC_ONELINK = auto()
    FOURVEC_LOCAL = auto()
    VEC_ONELINK = auto()
    VEC_LOCAL = auto()
    PION_LOCAL = auto()
    PSEUDO_SCALAR_LOCAL = PION_LOCAL
    SCALAR_LOCAL = auto()
    IDENTITY = SCALAR_LOCAL
    LOCAL = auto()
    ONELINK = auto()
    TWOLINK = auto()
    THREELINK = auto()
    FOURLINK = auto()

    @property
    def gamma_list(self) -> t.List[str]:
        match self:
            case Gamma.VEC_ONELINK:
                return ["GX_G1", "GY_G1", "GZ_G1"]
            case Gamma.FOURVEC_ONELINK:
                return ["GX_G1", "GY_G1", "GZ_G1", "GT_G1"]
            case Gamma.AXIAL_VEC_LOCAL:
                return ["G5X_G5X", "G5Y_G5Y", "G5Z_G5Z"]
            case Gamma.AXIAL_FOURVEC_LOCAL:
                return ["G5X_G5X", "G5Y_G5Y", "G5Z_G5Z", "G5T_G5T"]
            case Gamma.AXIAL_VEC_ONELINK:
                return ["G5X_G5", "G5Y_G5", "G5Z_G5"]
            case Gamma.AXIAL_FOURVEC_ONELINK:
                return ["G5X_G5", "G5Y_G5", "G5Z_G5", "G5T_G5"]
            case Gamma.VEC_LOCAL:
                return ["GX_GX", "GY_GY", "GZ_GZ"]
            case Gamma.FOURVEC_LOCAL:
                return ["GX_GX", "GY_GY", "GZ_GZ", "GT_GT"]
            case Gamma.IDENTITY:
                return ["G1_G1"]
            case Gamma.PION_LOCAL:
                return ["G5_G5"]
            case (
                Gamma.LOCAL
                | Gamma.ONELINK
                | Gamma.TWOLINK
                | Gamma.THREELINK
                | Gamma.FOURLINK
            ):
                raise ValueError(
                    f"{self.name} has no explicit gamma_list representation. See OpList.gamma_list instead."
                )
            case _:
                return [self.name]

    @property
    def gamma_string(self) -> str:
        gammas = self.gamma_list
        gammas = [f"({gamma})" for gamma in gammas]
        gammas = " ".join(gammas)
        gammas = gammas.replace("_", " ")
        return gammas

    @staticmethod
    def _local_gammas() -> t.List:
        return [
            Gamma.LOCAL,
            Gamma.PION_LOCAL,
            Gamma.VEC_LOCAL,
            Gamma.AXIAL_VEC_LOCAL,
            Gamma.IDENTITY,
            Gamma.G1_G1,
            Gamma.G5_G5,
            Gamma.GX_GX,
            Gamma.GY_GY,
            Gamma.GZ_GZ,
            Gamma.GT_GT,
            Gamma.G5X_G5X,
            Gamma.G5Y_G5Y,
            Gamma.G5Z_G5Z,
            Gamma.G5T_G5T,
            Gamma.FOURVEC_LOCAL,
            Gamma.AXIAL_FOURVEC_LOCAL,
        ]

    @staticmethod
    def _onelink_gammas() -> t.List:
        return [
            Gamma.ONELINK,
            Gamma.VEC_ONELINK,
            Gamma.AXIAL_VEC_ONELINK,
            Gamma.GX_G1,
            Gamma.GY_G1,
            Gamma.GZ_G1,
            Gamma.GT_G1,
            Gamma.G5X_G5,
            Gamma.G5Y_G5,
            Gamma.G5Z_G5,
            Gamma.G5T_G5,
            Gamma.FOURVEC_ONELINK,
            Gamma.AXIAL_FOURVEC_ONELINK,
        ]

    @staticmethod
    def _twolink_gammas() -> t.List:
        return [Gamma.TWOLINK]

    @staticmethod
    def _threelink_gammas() -> t.List:
        return [Gamma.THREELINK]

    @staticmethod
    def _fourlink_gammas() -> t.List:
        return [Gamma.FOURLINK]

    @property
    def local(self) -> bool:
        if self in self._local_gammas():
            return True
        else:
            return False

    @property
    def shift(self) -> int:
        if self in self._local_gammas():
            return 0
        elif self in self._onelink_gammas():
            return 1
        elif self in self._twolink_gammas():
            return 2
        elif self in self._threelink_gammas():
            return 3
        elif self in self._fourlink_gammas():
            return 4
        else:
            raise ValueError(f"Cannot determine shift for gamma: {self}")


@dataclass
class OpList:
    class Op(t.NamedTuple):
        gamma: Gamma
        mass: t.Tuple[str, ...]

        def __eq__(self, gamma: Gamma) -> bool:
            if self.gamma == gamma:
                return True
            return False

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
        if "mass" in kwargs and "gamma" in kwargs:

            mass = kwargs["mass"]
            if isinstance(mass, str):
                mass = [mass]
            elif not isinstance(mass, list):
                raise ValueError("Mass must be a string or list of strings.")

            gammas = kwargs["gamma"]
            if isinstance(gammas, str):
                gammas = [gammas]
            elif not isinstance(gammas, list):
                raise ValueError("Gammas must be a string or list of strings.")

            op_list = [cls.Op(gamma=Gamma[g.upper()], mass=tuple(mass)) for g in gammas]
        else:
            op_list = []
            for gamma_enum in Gamma:
                if val := kwargs.get(gamma_enum.name.lower(), None):
                    if isinstance(val, dict) and "mass" in val:
                        mass = val["mass"]
                        if isinstance(mass, str):
                            mass = [mass]
                        op_list.append(cls.Op(gamma=gamma_enum, mass=tuple(mass)))

        if len(op_list) == 0:
            utils.get_logger().debug("Returning an empty Op List.")
            # raise ValueError("Valid operations not found in provided parameters.")

        return cls(op_list=op_list)

    @property
    def mass(self):
        res: t.Set = set()
        for op in self.op_list:
            for m in op.mass:
                res.add(m)

        return list(res)

    def group_by_mass_and_shift(
        self,
    ) -> t.Generator[t.Tuple[Op, t.List[Gamma]], None, None]:
        for m in self.mass:
            ops_with_mass_m = list(filter(lambda x: m in x.mass, self.op_list))
            for i, g in enumerate(
                [
                    Gamma.LOCAL,
                    Gamma.ONELINK,
                    Gamma.TWOLINK,
                    Gamma.THREELINK,
                    Gamma.FOURLINK,
                ]
            ):

                if mass_m_shift_i := list(
                    filter(lambda x: x.gamma.shift == i, ops_with_mass_m)
                ):
                    yield self.Op(gamma=g, mass=(m,)), [
                        op.gamma for op in mass_m_shift_i
                    ]

    def __iter__(self):
        return iter(self.op_list)
