import typing as t
from string import Formatter
from pyrsistent import freeze, thaw
from .logging import get_logger


def format_keys(format_string: str) -> t.List[str]:
    """
    Extract formatting variables from a given format string.

    This function parses the input `format_string` and identifies all unique
    formatting variables enclosed in curly braces `{}`.

    Args:
        format_string (str): The format string to parse.

    Returns:
        t.List[str]: A list of unique formatting variable names found in the format string.
    """

    key_list = list(
        {k[1] for k in Formatter().parse(format_string) if k[1] is not None}
    )

    return key_list


def make_string_dict(instance: t.Any) -> t.Dict[str, str | t.List[str]]:
    res = {}
    for k, v in instance.__dict__.items():
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


def process_params(**params) -> t.Dict:
    """
    Processes the input parameters by handling string and list values.

    - Strings containing ".." are converted into a range of integers.
      For example, "1..3" becomes [1, 2, 3].
    - Strings without ".." are wrapped in a list.
      For example, "value" becomes ["value"].
    - List values are retained as-is.
    - Non-string and non-list values are excluded from the output.

    Args:
        **params: Arbitrary keyword arguments representing parameters.

    Returns:
        A dictionary with processed parameters where strings are converted
        to lists or ranges, and non-string/non-list values are removed.
    """

    param_out = freeze(params)
    e_param = param_out.evolver()

    for key, val in params.items():
        if isinstance(val, str):
            if ".." in val:
                range_input = list(map(int, val.split("..")))
                range_input[1] += 1
                param_out = param_out.set(key, list(range(*range_input)))
            else:
                param_out = param_out.set(key, [val])
        elif not isinstance(val, t.List):
            get_logger().debug(f"Removing key: {key} from parameters.")
            param_out = param_out.remove(key)

    return thaw(param_out)
