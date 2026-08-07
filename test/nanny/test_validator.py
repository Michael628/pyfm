"""Tests for validator functions has_good_output and get_outfiles.

These are the first real tests that exercise the validator's catalog-driven
completeness logic. They use direct Task construction with stub handlers —
no registry or YAML needed (pattern from test_taskbuilder_task.py).
"""
import pandas as pd
from pydantic.dataclasses import dataclass

from pyfm.nanny.taskbuilder import Task
from pyfm.nanny.validator import has_good_output, get_outfiles
from pyfm.domain.task_registry import TaskHandler
from pyfm.domain.protocols import OutfileCatalogProtocol
from pyfm.domain.conftypes import SimpleConfig


@dataclass(frozen=True)
class _StubConfig(SimpleConfig):
    pass


_KW = dict(formatting={}, logging_level="INFO", runid="test")


def _make_task(catalog_df):
    """Build a Task whose handler returns ``catalog_df`` from its catalog."""

    def _catalog(config):
        return catalog_df

    handler = TaskHandler(config_type=_StubConfig, create_outfile_catalog=_catalog)
    return Task(handler=handler, config=_StubConfig(**_KW), key="stub")


def _make_task_no_catalog():
    """Build a Task whose handler has no create_outfile_catalog."""
    handler = TaskHandler(config_type=_StubConfig)
    return Task(handler=handler, config=_StubConfig(**_KW), key="stub")


_COMPLETE_DF = pd.DataFrame({
    "filepath": ["/a", "/b"],
    "exists": [True, True],
    "file_size": [100, 200],
    "good_size": [50, 50],
})

_MISSING_DF = pd.DataFrame({
    "filepath": ["/a", "/b"],
    "exists": [True, False],
    "file_size": [100, None],
    "good_size": [50, 50],
})

_EMPTY_DF = pd.DataFrame(
    columns=["filepath", "good_size", "exists", "file_size"]
)


class TestHasGoodOutput:
    def test_all_complete_returns_true(self):
        assert has_good_output(_make_task(_COMPLETE_DF)) is True

    def test_missing_file_returns_false(self):
        assert has_good_output(_make_task(_MISSING_DF)) is False

    def test_empty_catalog_returns_false(self):
        # A zero-row catalog is incomplete — must route to XXfix, not pass
        # vacuously (the core completeness fix).
        assert has_good_output(_make_task(_EMPTY_DF)) is False

    def test_none_catalog_returns_false(self):
        # Handler without create_outfile_catalog -> get_outfiles returns None.
        # The guard must catch this before the mask computation (ordering fix).
        assert has_good_output(_make_task_no_catalog()) is False


class TestGetOutfilesProtocolGate:
    def test_returns_catalog_when_handler_satisfies_protocol(self):
        task = _make_task(_COMPLETE_DF)
        result = get_outfiles(task)
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_returns_none_when_handler_lacks_catalog(self):
        task = _make_task_no_catalog()
        assert get_outfiles(task) is None

    def test_protocol_is_domain_outfile_catalog_protocol(self):
        # The validator uses the domain OutfileCatalogProtocol, not a local one.
        task = _make_task(_COMPLETE_DF)
        assert isinstance(task.handler, OutfileCatalogProtocol)
        task_no_cat = _make_task_no_catalog()
        assert not isinstance(task_no_cat.handler, OutfileCatalogProtocol)
