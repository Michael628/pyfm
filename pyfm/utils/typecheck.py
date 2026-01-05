from typing import TypeVar, Union
import typing as t
from .logging import get_logger
from enum import Enum, auto


def extract_list_type(type_hint: t.Any) -> t.Type | None:
    if not hasattr(type_hint, "__origin__"):
        return None

    if type_hint.__origin__ is not list:
        return None

    if not hasattr(type_hint, "__args__"):
        get_logger().debug(
            f"List type hint {type_hint} does not have value type specified"
        )

    args = type_hint.__args__
    if len(args) < 1:
        get_logger().debug(
            f"List type hint {type_hint} does not have value type specified"
        )

    return args[0]


def extract_dict_value_type(type_hint: t.Any) -> t.Type | None:
    if not hasattr(type_hint, "__origin__"):
        return None

    if type_hint.__origin__ is not dict:
        return None

    if not hasattr(type_hint, "__args__"):
        get_logger().debug(
            f"Dictionary type hint {type_hint} does not have value type specified"
        )
        return None

    args = type_hint.__args__
    if len(args) < 2:
        get_logger().debug(
            f"Dictionary type hint {type_hint} does not have value type specified"
        )
        return None

    return args[1]


def extract_non_none_type(type_hint: t.Any) -> t.Any:
    # Handle typing.Union (legacy syntax)
    if hasattr(type_hint, "__origin__"):
        origin = getattr(type_hint, "__origin__", None)
        if origin is Union:
            args = getattr(type_hint, "__args__", ())
            return next((arg for arg in args if arg is not type(None)), type_hint)

    # Handle types.UnionType (Python 3.10+ | syntax)
    elif (
        hasattr(type_hint, "__args__") and str(type(type_hint)).find("UnionType") != -1
    ):
        args = getattr(type_hint, "__args__", ())
        return next((arg for arg in args if arg is not type(None)), type_hint)

    return type_hint


T = TypeVar("T")


def satisfies_protocol(cls: type[T] | None, protocol: type) -> bool:
    if cls is None:
        return False

    # Extract the actual type from union types
    actual_type = extract_non_none_type(cls)

    # Check if it's a valid type and satisfies the protocol
    return isinstance(actual_type, type) and issubclass(actual_type, protocol)


class ContainerType(t.NamedTuple):
    class Types(Enum):
        SIMPLE = auto()
        LIST = auto()
        DICT = auto()

    container: Types
    name: str
    type: t.Any


def iterate_container(
    iterable: t.Iterable[t.Tuple[str, t.Any]], cond: t.Callable[..., bool]
) -> t.Iterable[ContainerType]:
    for key, v in iterable:
        field_type = extract_non_none_type(v)
        if cond(field_type):
            yield ContainerType(
                name=key, container=ContainerType.Types.SIMPLE, type=field_type
            )
        elif list_type := extract_list_type(field_type):
            if cond(list_type):
                yield ContainerType(
                    name=key, container=ContainerType.Types.LIST, type=list_type
                )
        elif dict_type := extract_dict_value_type(field_type):
            if cond(dict_type):
                yield ContainerType(
                    name=key, container=ContainerType.Types.DICT, type=dict_type
                )


def get_container(item_type) -> ContainerType:
    field_type = extract_non_none_type(item_type)

    if list_type := extract_list_type(field_type):
        return ContainerType(
            container=ContainerType.Types.LIST, name=None, type=list_type
        )
    elif dict_type := extract_dict_value_type(field_type):
        return ContainerType(
            container=ContainerType.Types.DICT, name=None, type=dict_type
        )
    else:
        return ContainerType(
            container=ContainerType.Types.SIMPLE, name=None, type=field_type
        )
