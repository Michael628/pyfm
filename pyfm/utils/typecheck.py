from typing import TypeVar, Union
import typing as t

T = TypeVar("T")


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


def satisfies_protocol(cls: type[T] | None, protocol: type) -> bool:
    if cls is None:
        return False

    # Extract the actual type from union types
    actual_type = extract_non_none_type(cls)

    # Check if it's a valid type and satisfies the protocol
    return isinstance(actual_type, type) and issubclass(actual_type, protocol)
