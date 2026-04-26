"""Tests for TaskHandler, HandlerRegistry, and associated Protocols (issue #3)."""
import pytest
from pyfm.domain import task_registry
from pyfm.domain.task_registry import TaskHandler
from pyfm.domain.protocols import (
    InputBuilderProtocol,
    OutfileCatalogProtocol,
    AggregatorProtocol,
    TaskHandlerProtocol,
)
from pyfm.domain.conftypes import ConfigBase


# ---------------------------------------------------------------------------
# Minimal ConfigBase subclass for testing (avoids pydantic frozen-field errors)
# ---------------------------------------------------------------------------

class FakeConfig:
    """Lightweight stand-in for ConfigBase in tests."""


class AnotherConfig:
    pass


# ---------------------------------------------------------------------------
# Stub callables
# ---------------------------------------------------------------------------

def build_input_params_stub(config):
    return {"input": True}


def create_outfile_catalog_stub(config):
    return ["file1.h5"]


def build_aggregator_params_stub(config):
    return {"aggregate": True}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_registry():
    saved = dict(task_registry._handlers)
    task_registry.clear()
    yield
    task_registry.clear()
    task_registry._handlers.update(saved)


# ---------------------------------------------------------------------------
# TaskHandler dataclass
# ---------------------------------------------------------------------------

class TestTaskHandler:
    def test_requires_config_type(self):
        handler = TaskHandler(config_type=FakeConfig)
        assert handler.config_type is FakeConfig

    def test_callable_fields_default_to_none_internally(self):
        # Direct object.__getattribute__ access bypasses our guard — None stored
        handler = TaskHandler(config_type=FakeConfig)
        assert object.__getattribute__(handler, "build_input_params") is None
        assert object.__getattribute__(handler, "create_outfile_catalog") is None
        assert object.__getattribute__(handler, "build_aggregator_params") is None

    def test_set_callable_accessible(self):
        handler = TaskHandler(
            config_type=FakeConfig,
            build_input_params=build_input_params_stub,
        )
        assert handler.build_input_params is build_input_params_stub

    def test_unset_callable_raises_attribute_error(self):
        handler = TaskHandler(config_type=FakeConfig)
        with pytest.raises(AttributeError):
            _ = handler.build_input_params

    def test_unset_callable_not_accessible_via_hasattr(self):
        handler = TaskHandler(config_type=FakeConfig)
        assert not hasattr(handler, "build_input_params")
        assert not hasattr(handler, "create_outfile_catalog")
        assert not hasattr(handler, "build_aggregator_params")

    def test_set_callable_accessible_via_hasattr(self):
        handler = TaskHandler(
            config_type=FakeConfig,
            build_input_params=build_input_params_stub,
        )
        assert hasattr(handler, "build_input_params")

    def test_is_frozen(self):
        handler = TaskHandler(config_type=FakeConfig)
        with pytest.raises((AttributeError, TypeError)):
            handler.config_type = AnotherConfig  # type: ignore[misc]

    def test_all_callables_set(self):
        handler = TaskHandler(
            config_type=FakeConfig,
            build_input_params=build_input_params_stub,
            create_outfile_catalog=create_outfile_catalog_stub,
            build_aggregator_params=build_aggregator_params_stub,
        )
        assert handler.build_input_params is build_input_params_stub
        assert handler.create_outfile_catalog is create_outfile_catalog_stub
        assert handler.build_aggregator_params is build_aggregator_params_stub


# ---------------------------------------------------------------------------
# Protocol isinstance checks on TaskHandler
# ---------------------------------------------------------------------------

class TestProtocolSatisfaction:
    def test_no_callables_satisfies_no_task_protocols(self):
        handler = TaskHandler(config_type=FakeConfig)
        assert not isinstance(handler, InputBuilderProtocol)
        assert not isinstance(handler, OutfileCatalogProtocol)
        assert not isinstance(handler, AggregatorProtocol)
        assert not isinstance(handler, TaskHandlerProtocol)

    def test_build_input_params_satisfies_input_builder(self):
        handler = TaskHandler(
            config_type=FakeConfig,
            build_input_params=build_input_params_stub,
        )
        assert isinstance(handler, InputBuilderProtocol)
        assert not isinstance(handler, OutfileCatalogProtocol)
        assert not isinstance(handler, TaskHandlerProtocol)

    def test_create_outfile_catalog_satisfies_outfile_catalog(self):
        handler = TaskHandler(
            config_type=FakeConfig,
            create_outfile_catalog=create_outfile_catalog_stub,
        )
        assert isinstance(handler, OutfileCatalogProtocol)
        assert not isinstance(handler, InputBuilderProtocol)
        assert not isinstance(handler, TaskHandlerProtocol)

    def test_build_aggregator_params_satisfies_aggregator(self):
        handler = TaskHandler(
            config_type=FakeConfig,
            build_aggregator_params=build_aggregator_params_stub,
        )
        assert isinstance(handler, AggregatorProtocol)
        assert not isinstance(handler, TaskHandlerProtocol)

    def test_full_handler_satisfies_task_handler_protocol(self):
        handler = TaskHandler(
            config_type=FakeConfig,
            build_input_params=build_input_params_stub,
            create_outfile_catalog=create_outfile_catalog_stub,
            build_aggregator_params=build_aggregator_params_stub,
        )
        assert isinstance(handler, InputBuilderProtocol)
        assert isinstance(handler, OutfileCatalogProtocol)
        assert isinstance(handler, AggregatorProtocol)
        assert isinstance(handler, TaskHandlerProtocol)

    def test_input_and_outfile_only_satisfies_task_handler_protocol(self):
        """TaskHandlerProtocol requires only build_input_params + create_outfile_catalog."""
        handler = TaskHandler(
            config_type=FakeConfig,
            build_input_params=build_input_params_stub,
            create_outfile_catalog=create_outfile_catalog_stub,
        )
        assert isinstance(handler, TaskHandlerProtocol)
        assert not isinstance(handler, AggregatorProtocol)


# ---------------------------------------------------------------------------
# task_registry.register
# ---------------------------------------------------------------------------

class TestRegister:
    def test_register_minimal(self):
        task_registry.register("task_a", FakeConfig)
        handler = task_registry.get("task_a")
        assert handler.config_type is FakeConfig

    def test_register_with_all_callables(self):
        task_registry.register(
            "task_a",
            FakeConfig,
            build_input_params=build_input_params_stub,
            create_outfile_catalog=create_outfile_catalog_stub,
            build_aggregator_params=build_aggregator_params_stub,
        )
        handler = task_registry.get("task_a")
        assert handler.build_input_params is build_input_params_stub
        assert handler.create_outfile_catalog is create_outfile_catalog_stub
        assert handler.build_aggregator_params is build_aggregator_params_stub

    def test_register_duplicate_key_raises_value_error(self):
        task_registry.register("task_a", FakeConfig)
        with pytest.raises(ValueError, match="already registered"):
            task_registry.register("task_a", AnotherConfig)

    def test_register_unknown_callable_raises_type_error(self):
        with pytest.raises(TypeError, match="Unknown callable keyword"):
            task_registry.register("task_a", FakeConfig, invalid_fn=lambda c: None)

    def test_register_different_keys_independent(self):
        task_registry.register("task_a", FakeConfig, build_input_params=build_input_params_stub)
        task_registry.register("task_b", AnotherConfig, create_outfile_catalog=create_outfile_catalog_stub)

        ha = task_registry.get("task_a")
        hb = task_registry.get("task_b")

        assert ha.config_type is FakeConfig
        assert hb.config_type is AnotherConfig
        assert hasattr(ha, "build_input_params")
        assert not hasattr(ha, "create_outfile_catalog")
        assert hasattr(hb, "create_outfile_catalog")
        assert not hasattr(hb, "build_input_params")


# ---------------------------------------------------------------------------
# task_registry.get
# ---------------------------------------------------------------------------

class TestGet:
    def test_get_missing_key_raises_key_error(self):
        with pytest.raises(KeyError, match="no_such_key"):
            task_registry.get("no_such_key")

    def test_get_returns_task_handler(self):
        task_registry.register("task_a", FakeConfig)
        result = task_registry.get("task_a")
        assert isinstance(result, TaskHandler)

    def test_get_returns_correct_handler(self):
        task_registry.register("task_a", FakeConfig, build_input_params=build_input_params_stub)
        task_registry.register("task_b", AnotherConfig)

        assert task_registry.get("task_a").config_type is FakeConfig
        assert task_registry.get("task_b").config_type is AnotherConfig

    def test_get_then_isinstance_check(self):
        """Demonstrate Protocol satisfaction check at lookup time."""
        task_registry.register(
            "complete",
            FakeConfig,
            build_input_params=build_input_params_stub,
            create_outfile_catalog=create_outfile_catalog_stub,
        )
        task_registry.register("partial", FakeConfig)

        complete = task_registry.get("complete")
        partial = task_registry.get("partial")

        assert isinstance(complete, TaskHandlerProtocol)
        assert not isinstance(partial, TaskHandlerProtocol)


# ---------------------------------------------------------------------------
# task_registry.clear
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_removes_all_handlers(self):
        task_registry.register("task_a", FakeConfig)
        task_registry.register("task_b", AnotherConfig)
        task_registry.clear()

        with pytest.raises(KeyError):
            task_registry.get("task_a")
        with pytest.raises(KeyError):
            task_registry.get("task_b")

    def test_clear_allows_re_registration(self):
        task_registry.register("task_a", FakeConfig)
        task_registry.clear()
        # Should not raise after clear.
        task_registry.register("task_a", AnotherConfig)
        assert task_registry.get("task_a").config_type is AnotherConfig

    def test_clear_on_empty_registry_is_safe(self):
        task_registry.clear()  # already empty — must not raise


# ---------------------------------------------------------------------------
# No ConfigHandler / no inspect / no partial / no setattr machinery
# ---------------------------------------------------------------------------

class TestNoBannedMachinery:
    def test_config_handler_not_importable(self):
        import pyfm.domain.task_registry as reg_module
        assert not hasattr(reg_module, "ConfigHandler")

    def test_registry_module_does_not_import_inspect(self):
        import pyfm.domain.task_registry as reg_module
        import sys
        # inspect is not used in the new registry
        source_file = reg_module.__file__
        with open(source_file) as f:
            source = f.read()
        assert "import inspect" not in source
        assert "from functools import partial" not in source
        assert "setattr" not in source
