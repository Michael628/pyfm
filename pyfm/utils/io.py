import itertools
import os
import re
import typing as t
from functools import partial
import glob


from pyfm.utils.string import format_keys

from pyrsistent import freeze, thaw

import yaml
from dict2xml import dict2xml as dxml
import pandas as pd

from pyfm.utils.logging import get_logger


procFn = t.Callable[[str, t.Any], t.Any]


def process_files(
    filestem: str,
    processor: procFn,
    replacements: t.Dict | None = None,
    regex: t.Dict | None = None,
    wildcard_fill: bool = False,
) -> t.List:
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
            glob_repl = {k: "*" for k in regex.keys()}

            files = glob.glob(filestem(**glob_repl))

            # Build regex objects to catch each replacement
            regex_repl = {k: f"(?P<{k}>{val})" for k, val in regex.items()}

            file_pattern = os.path.basename(filestem(**regex_repl))
            pattern_parser: re.Pattern = re.compile(file_pattern)

            for file in files:
                base = os.path.basename(file)
                try:
                    regex_repl = freeze(next(pattern_parser.finditer(base)).groupdict())
                except StopIteration:
                    continue

                yield regex_repl, file

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
            yield freeze({}), partial(fstring.format)
        else:
            keys, repls = zip(
                *(
                    (k, map(str, r)) if isinstance(r, t.List) else (k, [str(r)])
                    for k, r in replacements.items()
                )
            )

            for r in itertools.product(*repls):
                repl = freeze(dict(zip(keys, r)))
                string_repl: partial = partial(fstring.format, **repl)

                yield repl, string_repl

    repl_keys: t.List[str] = format_keys(filestem)

    str_repl: t.Dict = replacements if replacements else {}
    regex_repl: t.Dict = regex if regex else {}

    logger = get_logger()
    logger.debug(f"repl_keys: {sorted(repl_keys)}")
    logger.debug(f"str_repl keys: {sorted(str_repl.keys())}")
    logger.debug(f"regex_repl keys: {sorted(regex_repl.keys())}")
    missing_keys = set(repl_keys) - set(str_repl.keys()) - set(regex_repl.keys())
    if len(missing_keys) > 0:
        if wildcard_fill:
            logger.info(
                f"Adding wildcards to keys in replacements: {', '.join(sorted(missing_keys))}"
            )
            for k in missing_keys:
                regex_repl[k] = ".*"
        else:
            raise ValueError(f"Missing keys {', '.join(sorted(missing_keys))}")

    collection: t.List = []

    def file_gen():
        fs = os.path.expanduser(filestem)
        for str_reps, repl_filename in string_replacement_gen(fs, str_repl):
            for reg_reps, regex_filename in file_regex_gen(repl_filename, regex_repl):
                yield regex_filename, thaw(str_reps.update(reg_reps))

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


def load_param(file):
    """Read the YAML parameter file"""

    param = yaml.safe_load(open(file, "r"))

    return param


def write_plain_text(file_stem: str, contents: str, ext: str | None = None) -> str:
    filename = file_stem if not ext else f"{file_stem}.{ext}"
    filename = f"in/{filename}"
    os.makedirs("in/", exist_ok=True)
    with open(filename, "w") as f:
        f.write(contents)
    return filename


def write_schedule(input_file: str, schedule: t.List[str]) -> str:
    input_stem = os.path.splitext(input_file)[0]
    os.makedirs("schedules/", exist_ok=True)
    sched_file = f"schedules/{input_stem}.sched"
    with open(sched_file, "w") as f:
        f.write(str(len(schedule)) + "\n" + "\n".join(schedule))

    return sched_file


def write_xml(file_stem: str, contents: t.Dict[str, t.Any]) -> str:
    filename = f"in/{file_stem}.xml"
    contents_xml = dxml(contents)
    os.makedirs("in/", exist_ok=True)
    with open(filename, "w") as f:
        f.write(contents_xml)

    return filename


def catalog_files(
    outfile_generator, replacements: t.Dict[str, str] | None = None
) -> pd.DataFrame:
    """
    Catalogs system file data based on the provided outfile generator and replacements.

    Args:
        outfile_generator (Iterable[Tuple[Dict[str, str], OutfileConfig]]):
            An iterable that yields task-specific format key replacements and
            corresponding outfile data (filestem, expected file size, file extension).
        replacements (Dict[str, str]):
            A dictionary of replacement keys and their corresponding values.

    Returns:
        pd.DataFrame: A DataFrame containing details about the cataloged files,
        including their existence, size, and other metadata.

    Notes:
        - The function processes files using a custom processor (`build_row`)
          and formats keys based on the outfile configuration.
        - It calculates additional metadata such as file existence and size.
    """

    if replacements is None:
        replacements = {}

    def add_filepath(filepath: str, repls: t.Dict[str, str]) -> t.Dict[str, str]:
        repls["filepath"] = filepath
        return repls

    df = []
    for task_replacements, outfile_config in outfile_generator:
        assert outfile_config is not None
        outfile = outfile_config.filestem + outfile_config.ext
        filekeys = format_keys(outfile)
        replacements.update(task_replacements)
        files = process_files(
            outfile,
            processor=add_filepath,
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

    if len(df) == 0:
        return pd.DataFrame()
    else:
        return pd.concat(df, ignore_index=True)


def get_bad_files(df: pd.DataFrame) -> t.List[str]:
    return list(df[(df["file_size"] >= df["good_size"]) != True]["filepath"])
