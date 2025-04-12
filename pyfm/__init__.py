import logging
import sys
import typing as t
from pydantic.dataclasses import dataclass
from dataclasses import fields
from enum import Enum, auto
from pyfm import utils


class Gamma(Enum):
    ONELINK = auto()
    LOCAL = auto()
    VEC_ONELINK = auto()
    VEC_LOCAL = auto()
    PION_LOCAL = auto()
    G1_G1 = auto()
    GX_GX = auto()
    GY_GY = auto()
    GZ_GZ = auto()
    G5_G5 = auto()
    GX_G1 = auto()
    GY_G1 = auto()
    GZ_G1 = auto()

    @property
    def gamma_list(self) -> t.List[str]:
        if self in [Gamma.ONELINK, Gamma.VEC_ONELINK]:
            return ["GX_G1", "GY_G1", "GZ_G1"]
        if self == Gamma.LOCAL:
            return ["G5_G5", "GX_GX", "GY_GY", "GZ_GZ"]
        if self == Gamma.VEC_LOCAL:
            return ["GX_GX", "GY_GY", "GZ_GZ"]
        if self == Gamma.PION_LOCAL:
            return ["G5_G5"]
        else:
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


def setup_logging(logging_level: str):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)-5s - %(message)s",
        style="%",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger().setLevel(logging_level)


T = t.TypeVar("T", bound="ConfigBase")


class ObserverInterface:
    """Interface for observers that want to be notified of changes."""

    def update(self, **kwargs) -> None:
        """Update the observer with the new state of the subject."""
        raise NotImplementedError


class SubjectInterface:
    """Interface for objects that can be observed."""

    def register(self, observer: ObserverInterface) -> None:
        """Register an observer to be notified of changes."""
        raise NotImplementedError

    def unregister(self, observer: ObserverInterface) -> None:
        """Unregister an observer."""
        raise NotImplementedError

    def notify(self) -> None:
        """Notify all registered observers of a change."""
        raise NotImplementedError


class ConfigBase:
    @classmethod
    def create(cls: t.Type[T], **kwargs) -> T:
        """Creates a new instance of ConfigBase from a dictionary.

        Note
        ----
        Checks for dataclass fields stored in the class object.
        If kwargs match a dataclass field apart from a leading underscore,
        e.g. kwargs['mass'] and _mass, the value gets assigned to the underscored object attribute.
        """

        conflicts = [
            k
            for k in kwargs.keys()
            if k in kwargs and f"_{k}" in kwargs and not k.startswith("_")
        ]

        if any(conflicts):
            raise ValueError(
                f"Conflict in parameters. Both {conflicts[0]} and _{conflicts[0]} passed to {cls} `create`."
            )

        class_vars = [f.name for f in fields(cls)]
        obj_vars = {}
        new_vars = {}
        for k, v in kwargs.items():
            if k in class_vars:
                obj_vars[k] = v
            elif f"_{k}" in class_vars and not k.startswith("_"):
                obj_vars[f"_{k}"] = v
            elif k in cls.__dict__:
                raise ValueError(
                    f"Cannot overwrite existing {cls} param, `{k}`. Try relabeling `{k}`"
                )
            else:
                new_vars[k] = v

        obj = cls(**obj_vars)

        for k, v in new_vars.items():
            setattr(obj, k, v)

        return obj

    def string_dict(self):
        """Converts all attributes without leading underscore to strings or lists of strings.
        Dictionary attributes are removed from output.
        Returns a dictionary keyed by the attribute labels
        """
        res = {}
        for k, v in self.__dict__.items():
            if k.startswith("_") or isinstance(v, t.Dict):
                continue
            elif isinstance(v, t.List):
                res[k] = list(map(str, v))
            elif isinstance(v, bool):
                res[k] = str(v).lower()
            else:
                if v is not None:
                    res[k] = str(v)

        return res

    @property
    def public_dict(self):
        """Converts object attributes to dictionary, removing attributes with leading underscore
        Returns a dictionary keyed by the attribute labels
        """
        res = {}
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                res[k] = v

        return res


# ============Operator List===========
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
