"""Lazy subcommand loading for the ``pyfm`` CLI.

``LazyGroup`` defers importing a subcommand's module until that subcommand is
traversed (help, completion, or invocation), giving per-command module
isolation. (numpy/pandas/h5py remain eager via ``pyfm/__init__.py``; sympy is
deferred by ``contract.py``'s local imports — see Phase 4.)

Adapted from the Click "Complex Applications" lazy-loading recipe.
"""
from __future__ import annotations

import importlib
import typing as t

import click


class LazyGroup(click.Group):
    """A :class:`click.Group` that imports its subcommands on demand.

    ``lazy_subcommands`` maps a subcommand name to the import path of the
    target ``Command``/``Group`` object, using ``"module:attribute"``.
    """

    def __init__(self, *args: t.Any, lazy_subcommands: dict[str, str] | None = None, **kwargs: t.Any) -> None:
        super().__init__(*args, **kwargs)
        self.lazy_subcommands: dict[str, str] = lazy_subcommands or {}

    def list_commands(self, ctx: click.Context) -> list[str]:
        names = set(super().list_commands(ctx))
        names.update(self.lazy_subcommands)
        return sorted(names)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        if cmd_name in self.lazy_subcommands:
            return self._lazy_load(cmd_name)
        return super().get_command(ctx, cmd_name)

    def _lazy_load(self, cmd_name: str) -> click.Command:
        import_path = self.lazy_subcommands[cmd_name]
        modname, _, attr = import_path.partition(":")
        if not attr:
            raise ValueError(f"Lazy load target {import_path!r} must be 'module:attribute'.")
        mod = importlib.import_module(modname)
        cmd_obj = getattr(mod, attr)
        if not isinstance(cmd_obj, click.Command):
            raise TypeError(
                f"Lazy load target {import_path!r} is not a click.Command "
                f"(got {type(cmd_obj).__name__})."
            )
        self.commands[cmd_name] = cmd_obj
        del self.lazy_subcommands[cmd_name]
        return cmd_obj
