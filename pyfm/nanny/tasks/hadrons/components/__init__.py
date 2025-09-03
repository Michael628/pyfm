from pydantic.dataclasses import dataclass
import typing as t


@dataclass
class ComponentBase:
    """Base class for hadrons components."""
    
    @classmethod
    def from_dict(cls, kwargs: t.Dict[str, t.Any]) -> "ComponentBase":
        return cls(**kwargs)
