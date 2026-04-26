"""Tests for the Task NamedTuple introduced in place of BoundTaskHandler."""
import typing as t
import pytest
import pandas as pd
from pydantic.dataclasses import dataclass

from pyfm.nanny.core import Task
from pyfm.domain.task_registry import TaskHandler
from pyfm.domain.conftypes import ConfigBase, SimpleConfig


# ---------------------------------------------------------------------------
# Minimal config stub — a concrete SimpleConfig with only base fields
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _StubConfig(SimpleConfig):
    pass


_STUB_CONFIG_KWARGS = dict(formatting={}, logging_level="INFO", runid="test-run")


# ---------------------------------------------------------------------------
# Task NamedTuple structural tests
# ---------------------------------------------------------------------------

class TestTaskNamedTuple:
    def _make_handler(self, **callables):
        return TaskHandler(config_type=_StubConfig, **callables)

    def test_fields(self):
        assert Task._fields == ("handler", "config", "key")

    def test_construction(self):
        handler = self._make_handler()
        config = _StubConfig(**_STUB_CONFIG_KWARGS)
        task = Task(handler=handler, config=config, key="test_task")
        assert task.handler is handler
        assert task.config is config
        assert task.key == "test_task"

    def test_is_namedtuple_unpackable(self):
        handler = self._make_handler()
        config = _StubConfig(**_STUB_CONFIG_KWARGS)
        task = Task(handler=handler, config=config, key="test_task")
        h, c, k = task
        assert h is handler
        assert c is config
        assert k == "test_task"

    def test_immutable(self):
        handler = self._make_handler()
        config = _StubConfig(**_STUB_CONFIG_KWARGS)
        task = Task(handler=handler, config=config, key="test_task")
        with pytest.raises(AttributeError):
            task.key = "other"  # type: ignore[misc]

    def test_handler_build_input_params_called_with_config(self):
        received = {}

        def _build_input(config):
            received["config"] = config
            return {"result": True}

        handler = self._make_handler(build_input_params=_build_input)
        config = _StubConfig(**_STUB_CONFIG_KWARGS)
        task = Task(handler=handler, config=config, key="hadrons_stub")

        result = task.handler.build_input_params(task.config)
        assert result == {"result": True}
        assert received["config"] is config

    def test_handler_build_aggregator_params_called_with_config(self):
        received = {}

        def _build_agg(config, average):
            received["average"] = average
            return {"run": ["a"]}

        handler = self._make_handler(build_aggregator_params=_build_agg)
        config = _StubConfig(**_STUB_CONFIG_KWARGS)
        task = Task(handler=handler, config=config, key="contract_stub")

        result = task.handler.build_aggregator_params(task.config, True)
        assert result == {"run": ["a"]}
        assert received["average"] is True

    def test_absent_callable_raises_attribute_error(self):
        """TaskHandler with no build_input_params raises AttributeError on access."""
        handler = self._make_handler()  # no build_input_params
        config = _StubConfig(**_STUB_CONFIG_KWARGS)
        task = Task(handler=handler, config=config, key="hadrons_stub")
        with pytest.raises(AttributeError):
            _ = task.handler.build_input_params


# ---------------------------------------------------------------------------
# Protocol isinstance checks on task.handler (validator.py pattern)
# ---------------------------------------------------------------------------

@t.runtime_checkable
class _TaskOutputProtocol(t.Protocol):
    def create_outfile_catalog(self) -> pd.DataFrame: ...


class TestHandlerProtocolCheck:
    def test_handler_without_outfile_catalog_fails_protocol(self):
        handler = TaskHandler(config_type=_StubConfig)
        config = _StubConfig(**_STUB_CONFIG_KWARGS)
        task = Task(handler=handler, config=config, key="hadrons_stub")
        assert not isinstance(task.handler, _TaskOutputProtocol)

    def test_handler_with_outfile_catalog_passes_protocol(self):
        def _catalog(config):
            return pd.DataFrame()

        handler = TaskHandler(config_type=_StubConfig, create_outfile_catalog=_catalog)
        config = _StubConfig(**_STUB_CONFIG_KWARGS)
        task = Task(handler=handler, config=config, key="hadrons_stub")
        assert isinstance(task.handler, _TaskOutputProtocol)

    def test_outfile_catalog_called_with_config(self):
        received = {}

        def _catalog(config):
            received["config"] = config
            return pd.DataFrame({"filepath": ["x"], "exists": [True], "file_size": [10], "good_size": [5]})

        handler = TaskHandler(config_type=_StubConfig, create_outfile_catalog=_catalog)
        config = _StubConfig(**_STUB_CONFIG_KWARGS)
        task = Task(handler=handler, config=config, key="hadrons_stub")

        result = task.handler.create_outfile_catalog(task.config)
        assert received["config"] is config
        assert not result.empty


# ---------------------------------------------------------------------------
# BoundTaskHandler no longer exported from core
# ---------------------------------------------------------------------------

def test_bound_task_handler_removed():
    """BoundTaskHandler must not exist in pyfm.nanny.core after the refactor."""
    import pyfm.nanny.core as core_module
    assert not hasattr(core_module, "BoundTaskHandler")
