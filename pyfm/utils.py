import pandas as pd
import copy
import itertools
import logging
import os
import re
import typing as t
from collections.abc import Mapping
from functools import partial
from string import Formatter

import yaml

procFn = t.Callable[[str, t.Any], t.Any]


class ReadOnlyDict(Mapping):
    def __init__(self, initial_data):
        self._data = dict(initial_data)

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._data})"

    def __setitem__(self, key, value):
        raise TypeError("This dictionary is read-only")

    def __delitem__(self, key):
        raise TypeError("This dictionary is read-only")


def deep_copy_dict(read_dict):
    """
    Recursively creates a writable deep copy of a dictionary.

    Args:
        read_dict (dict or ReadOnlyDict): The dictionary to be copied.
            It must be an instance of `dict` or `ReadOnlyDict`.

    Returns:
        dict: A writable deep copy of the input dictionary.

    Raises:
        ValueError: If the input is not an instance of `dict` or `ReadOnlyDict`.

    Notes:
        - Nested dictionaries (or ReadOnlyDict instances) are also deeply copied.
        - Non-dictionary values are copied using `copy.deepcopy`.
    """

    def is_dict(d):
        return isinstance(d, ReadOnlyDict) or isinstance(d, dict)

    if not is_dict(read_dict):
        raise ValueError("Input must be a ReadOnlyDict or dict instance")

    writable_copy = {}
    for key, value in read_dict.items():
        if is_dict(value):
            # Recursively copy nested ReadOnlyDict
            writable_copy[key] = deep_copy_dict(value)
        else:
            # For other types, just copy the value
            writable_copy[key] = copy.deepcopy(value)

    return writable_copy


def load_param(file):
    """Read the YAML parameter file"""

    param = yaml.safe_load(open(file, "r"))

    return param


# FIXME: This function is nearly the same as format_sanitize_dict
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

    param_out = deep_copy_dict(params)

    for key, val in params.items():
        if isinstance(val, str):
            if ".." in val:
                range_input = list(map(int, val.split("..")))
                range_input[1] += 1
                param_out[key] = list(range(*range_input))
            else:
                param_out[key] = [val]
        elif not isinstance(val, t.List):
            logging.debug(f"Removing key: {key} from parameters.")
            param_out.pop(key)

    return param_out


def format_sanitize_dict(params: t.Dict, include_keys: t.List[str] = []) -> t.Dict:
    """
    Convert all values in the input dictionary `params` to strings or lists of strings.

    - If a value is a list, each element is converted to a string.
    - If a value is not a list, it is converted to a single-element list containing the string representation.
    - Keys not in `include_keys` (if provided) are excluded from the output.

    Args:
        params (t.Dict): The input dictionary to sanitize.
        include_keys (t.List[str], optional): A list of keys to include in the output.
                                              If empty, all keys are included. Defaults to [].

    Returns:
        t.Dict: A sanitized dictionary with string or list of string values.
    """
    param_out = {}
    for k, v in params.items():
        if include_keys and k not in include_keys:
            logging.debug(f"Removing key: {k} from parameters.")
            continue
        if isinstance(v, t.List):
            param_out[k] = list(map(str, v))
        else:
            param_out[k] = [str(v)]

    return param_out


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


def file_regex_gen(filestem: partial, regex: t.Dict[str, str]):
    """
    Generates file paths by formatting a filestem with regex replacements
    and searching for matching files in the system.

    Args:
        filestem (partial): A partial function that generates file paths
            with placeholders for regex replacements.
        regex (Dict[str, str]): A dictionary where keys are placeholder names
            in the filestem, and values are regex patterns to match.

    Yields:
        Tuple[Dict[str, str], str]:
            - A dictionary of matched values corresponding to the regex patterns.
            - The full path of the file that matched the regex search.

    Notes:
        - If `regex` is empty, the function yields the result of `filestem()`
          without any replacements.
        - Assumes all regex matches occur in the file name, not in the directory path.
    """
    if len(regex) == 0:
        yield {}, filestem()
    else:
        # Build regex objects to catch each replacement
        regex_repl = {k: f"(?P<{k}>{val})" for k, val in regex.items()}
        file_pattern = filestem(**regex_repl)

        # FIXME: Assumes all regex matches occur in file name,
        # not in the directory path.
        directory, match = os.path.split(file_pattern)

        files: t.List[str] = os.listdir(directory)

        regex_pattern: re.Pattern = re.compile(match)

        for file in files:
            try:
                regex_repl = next(regex_pattern.finditer(file)).groupdict()
            except StopIteration:
                continue

            yield regex_repl, f"{directory}/{file}"


def string_replacement_gen(
    fstring: str, replacements: t.Dict[str, t.Union[str, t.List[str]]]
):
    """
    Generator for keyword replacements in a formatted string.

    Args:
        fstring (str): The format string containing placeholders for replacements.
        replacements (Dict[str, Union[str, List[str]]]): A dictionary where keys
            are placeholder names in the format string, and values are either
            a single replacement string or a list of replacement strings.

    Yields:
        Tuple[Dict[str, str], functools.partial]:
            - A dictionary of replacements applied to the format string.
            - A `functools.partial` object representing the partially formatted
              string. Calling this object will return the final string if no
              further replacements are needed.

    Notes:
        - If `replacements` is empty, the function yields the original format
          string without any replacements.
        - For each combination of replacements, the function generates a unique
          partially formatted string.
    """

    if len(replacements) == 0:
        yield {}, partial(fstring.format)
    else:
        keys, repls = zip(
            *(
                (k, map(str, r)) if isinstance(r, t.List) else (k, [str(r)])
                for k, r in replacements.items()
            )
        )

        for r in itertools.product(*repls):
            repl: t.Dict = dict(zip(keys, r))
            string_repl: partial = partial(fstring.format, **repl)

            yield repl, string_repl


def process_files(
    filestem: str,
    processor: procFn,
    replacements: t.Optional[t.Dict] = None,
    regex: t.Optional[t.Dict] = None,
) -> t.List:
    """
    Processes files by applying string and regex replacements, then passing
    the resulting files to a processor function.

    Args:
        filestem (str): The base file name or path with placeholders for replacements.
        processor (procFn): A function that processes each generated file.
            It takes the file name and a dictionary of replacements as arguments.
        replacements (Optional[Dict], optional): A dictionary of string replacements
            where keys are placeholders and values are replacement strings. Defaults to None.
        regex (Optional[Dict], optional): A dictionary of regex replacements where
            keys are placeholders and values are regex patterns. Defaults to None.

    Returns:
        List: A collection of results returned by the processor function.

    Raises:
        AssertionError: If the number of replacement keys does not match the
            combined number of string and regex replacements, or if any replacement
            key is missing in the provided dictionaries.

    Notes:
        - The function generates all possible combinations of string and regex
          replacements for the given filestem.
        - The processor function is expected to handle each generated file and
          return a result, which is collected into the final list.
        - If the processor raises a `StopIteration` exception with an argument,
          the argument is added to the collection, and processing stops.
    """
    repl_keys: t.List[str] = format_keys(filestem)

    str_repl: t.Dict = replacements if replacements else {}
    regex_repl: t.Dict = regex if regex else {}

    logging.debug(f"repl_keys: {sorted(repl_keys)}")
    logging.debug(f"str_repl keys: {sorted(str_repl.keys())}")
    logging.debug(f"regex_repl keys: {sorted(regex_repl.keys())}")
    assert len(repl_keys) == len(str_repl) + len(regex_repl)
    assert all(((k in str_repl or k in regex_repl) for k in repl_keys))

    collection: t.List = []

    def file_gen():
        for str_reps, repl_filename in string_replacement_gen(filestem, str_repl):
            for reg_reps, regex_filename in file_regex_gen(repl_filename, regex_repl):
                str_reps.update(reg_reps)
                yield regex_filename, deep_copy_dict(str_reps)

    for filename, reps in file_gen():
        try:
            new_result = processor(filename, reps)
        except StopIteration as e:
            if e.args:
                assert len(e.args) == 1
                collection.append(e.args[0])
            break
        collection.append(new_result)

    return collection


def catalog_files(outfile_generator, replacements) -> pd.DataFrame:
    def build_row(filepath: str, repls: t.Dict[str, str]) -> t.Dict[str, str]:
        repls["filepath"] = filepath
        return repls

    df = []
    for task_replacements, outfile_config in outfile_generator:
        assert outfile_config is not None
        outfile = outfile_config.filestem + outfile_config.ext
        filekeys = formatkeys(outfile)
        replacements.update(task_replacements)
        files = process_files(
            outfile,
            processor=build_row,
            replacements={k: v for k, v in replacements.items() if k in filekeys},
        )
        dict_of_rows = {
            k: [file[k] for file in files] for k in files[0] if len(files) > 0
        }

        new_df = pd.DataFrame(dict_of_rows)
        new_df["good_size"] = outfile_config.good_size
        new_df["exists"] = new_df["filepath"].apply(os.path.exists)
        new_df["file_size"] = None
        new_df.loc[new_df["exists"], "file_size"] = new_df[new_df["exists"]][
            "filepath"
        ].apply(os.path.getsize)
        df.append(new_df)

    df = pd.concat(df, ignore_index=True)

    return df
