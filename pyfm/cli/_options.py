"""Shared Click option decorators for the pyfm CLI.

These small factories each return a ``click.option`` decorator so command
modules stack only the options they need, replacing the hand-written,
drift-prone ``-p/--param-file`` / ``--logging-level`` / ``-j/--job`` /
``-f/--format`` declarations that were copy-pasted across the command groups.

The module is deliberately import-light (stdlib + ``click`` only): every
command module imports it, so pulling in numpy/pandas/h5py here would defeat
the lazy subcommand loading added in Phase 4 (``pyfm/cli/_lazy.py``).
"""
from __future__ import annotations

import click


def param_file_option(default: str = "params.yaml"):
    """``-p/--param-file`` -> ``param_file``.

    A file path (``click.Path(dir_okay=False)``); defaults to ``params.yaml``.
    """
    return click.option(
        "-p",
        "--param-file",
        type=click.Path(dir_okay=False),
        default=default,
        help="Path to YAML parameter file.",
    )


def logging_level_option(default: str = "INFO"):
    """``--logging-level`` -> ``logging_level`` (default ``INFO``)."""
    return click.option(
        "--logging-level",
        type=str,
        default=default,
        help="Logging verbosity (DEBUG, INFO, WARNING, ERROR).",
    )


def format_option(choices=("csv", "hdf5", "dict"), default: str = "csv"):
    """``-f/--format`` -> ``fmt`` as a ``click.Choice`` (default ``csv``).

    Callers pass ``choices=("csv", "hdf5")`` to withhold ``dict`` until it is
    backed downstream (Phase 5).
    """
    return click.option(
        "-f",
        "--format",
        "fmt",
        type=click.Choice(list(choices)),
        default=default,
        show_default=True,
        help="Output file format.",
    )


def job_option(required: bool = False, default=None, help: str = "Job step name."):
    """``-j/--job`` -> ``job``.

    ``required`` / ``default`` / ``help`` vary per command, so they are passed
    explicitly at each call site.

    ``default=None`` *omits* the ``default`` kwarg rather than passing it
    through. Since click 8.2 an explicit ``default=None`` counts as a real
    value (the ``UNSET`` sentinel marks "missing"), which silently defeats
    ``required=True``. Omitting the kwarg restores required-option errors and
    keeps optional options surfacing as ``None`` in the callback.
    """
    kwargs = {"type": str, "required": required, "help": help}
    if default is not None:
        kwargs["default"] = default
    return click.option("-j", "--job", **kwargs)
