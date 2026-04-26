"""Tests for pyfm/tasks/register.py using the task_registry and build_hooks singletons."""
import pytest

from pyfm.domain import task_registry, build_hooks
from pyfm.domain.task_registry import TaskHandler
from pyfm.domain.protocols import TaskHandlerProtocol
from pyfm.tasks.register import (
    get_task_handler,
    get_task_key,
    list_registered_types,
    register_task,
)


# ---------------------------------------------------------------------------
# Minimal config stub with key ClassVar so register_task can derive the key
# ---------------------------------------------------------------------------

class FakeConfig:
    key = "fake_task"


class OtherConfig:
    key = "other_task"


class NoKeyConfig:
    pass


# ---------------------------------------------------------------------------
# Stub callables with the exact names used by the routing logic
# ---------------------------------------------------------------------------

def build_input_params(config):
    return {"input": True}


def create_outfile_catalog(config):
    return ["out.h5"]


def build_aggregator_params(config):
    return {"agg": True}


def preprocess_params(params):
    return params


def postprocess_config(config):
    return config


def validate_config(config):
    pass


# ---------------------------------------------------------------------------
# Fixtures — reset both registries before/after every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_registries():
    task_registry.clear()
    build_hooks.clear()
    yield
    task_registry.clear()
    build_hooks.clear()


# ---------------------------------------------------------------------------
# get_task_key
# ---------------------------------------------------------------------------

class TestGetTaskKey:
    def test_job_type_only(self):
        key = get_task_key(job_type="hadrons")
        assert key == "nanny_hadrons"

    def test_job_type_and_task_type(self):
        key = get_task_key(job_type="hadrons", task_type="lmi")
        assert key == "nanny_hadrons_lmi"

    def test_config_with_key_classvar(self):
        key = get_task_key(config=FakeConfig)
        assert key == "nanny_fake_task"

    def test_config_without_key_classvar_returns_none(self):
        key = get_task_key(config=NoKeyConfig)
        assert key is None

    def test_no_args_raises(self):
        with pytest.raises(ValueError):
            get_task_key()


# ---------------------------------------------------------------------------
# register_task — happy path
# ---------------------------------------------------------------------------

class TestRegisterTask:
    def test_register_minimal_no_funcs(self):
        register_task(FakeConfig)
        handler = task_registry.get("nanny_fake_task")
        assert handler.config_type is FakeConfig

    def test_register_with_all_three_positional_funcs(self):
        register_task(FakeConfig, build_input_params, create_outfile_catalog, build_aggregator_params)
        handler = task_registry.get("nanny_fake_task")
        assert handler.build_input_params is build_input_params
        assert handler.create_outfile_catalog is create_outfile_catalog
        assert handler.build_aggregator_params is build_aggregator_params

    def test_positional_order_does_not_matter(self):
        # Same as lmi.py: outfile first, then input, then aggregator
        register_task(FakeConfig, create_outfile_catalog, build_input_params, build_aggregator_params)
        handler = task_registry.get("nanny_fake_task")
        assert handler.build_input_params is build_input_params
        assert handler.create_outfile_catalog is create_outfile_catalog
        assert handler.build_aggregator_params is build_aggregator_params

    def test_register_with_preprocess_positional(self):
        register_task(FakeConfig, build_input_params, preprocess_params)
        hooks = build_hooks.get(FakeConfig)
        assert hooks is not None
        assert hooks.preprocess is preprocess_params

    def test_register_with_postprocess_positional(self):
        register_task(FakeConfig, build_input_params, postprocess_config)
        hooks = build_hooks.get(FakeConfig)
        assert hooks.postprocess is postprocess_config

    def test_register_with_validate_keyword(self):
        register_task(FakeConfig, build_input_params, validate=validate_config)
        hooks = build_hooks.get(FakeConfig)
        assert hooks.validate is validate_config

    def test_register_with_preprocess_keyword(self):
        register_task(FakeConfig, preprocess_params=preprocess_params)
        hooks = build_hooks.get(FakeConfig)
        assert hooks.preprocess is preprocess_params

    def test_register_with_all_keyword_hooks(self):
        register_task(
            FakeConfig,
            build_input_params,
            create_outfile_catalog,
            preprocess_params=preprocess_params,
            validate=validate_config,
        )
        hooks = build_hooks.get(FakeConfig)
        assert hooks.preprocess is preprocess_params
        assert hooks.validate is validate_config

    def test_default_preprocess_added_when_none_supplied(self):
        register_task(FakeConfig, build_input_params)
        hooks = build_hooks.get(FakeConfig)
        assert hooks is not None
        assert hooks.preprocess is not None
        # The default merges _preprocessor into params
        result = hooks.preprocess({"a": 1, "_preprocessor": {"b": 2}})
        assert result == {"a": 1, "b": 2}

    def test_explicit_preprocess_overrides_default(self):
        register_task(FakeConfig, preprocess_params)
        hooks = build_hooks.get(FakeConfig)
        assert hooks.preprocess is preprocess_params

    def test_register_config_without_key_is_silent_no_op(self):
        # Should not raise; silently skips
        register_task(NoKeyConfig)
        assert NoKeyConfig not in build_hooks._registry
        assert not any(
            True for k in task_registry._handlers if "NoKeyConfig" in k
        )

    def test_duplicate_registration_is_idempotent(self):
        # Calling register_task twice with the same config must not raise
        register_task(FakeConfig, build_input_params)
        register_task(FakeConfig, create_outfile_catalog)  # second call silently skipped
        handler = task_registry.get("nanny_fake_task")
        # First registration wins
        assert hasattr(handler, "build_input_params")
        assert not hasattr(handler, "create_outfile_catalog")

    def test_unknown_positional_func_name_is_ignored(self):
        def format_string(config, s):
            return s

        # nanny/core.py pattern — should not raise
        register_task(FakeConfig, format_string)
        handler = task_registry.get("nanny_fake_task")
        assert handler.config_type is FakeConfig


# ---------------------------------------------------------------------------
# get_task_handler
# ---------------------------------------------------------------------------

class TestGetTaskHandler:
    def test_returns_task_handler_for_full_registration(self):
        register_task(FakeConfig, build_input_params, create_outfile_catalog)
        handler = get_task_handler(job_type="fake", task_type="task")
        assert isinstance(handler, TaskHandler)
        assert handler.config_type is FakeConfig

    def test_returns_none_for_missing_key(self):
        handler = get_task_handler(job_type="no", task_type="such")
        assert handler is None

    def test_strict_mode_returns_none_for_incomplete_handler(self):
        # GaugeConfig-style: no task callables, not a full TaskHandlerProtocol
        register_task(FakeConfig)
        handler = get_task_handler(job_type="fake", task_type="task", strict=True)
        assert handler is None

    def test_non_strict_mode_returns_incomplete_handler(self):
        register_task(FakeConfig)
        handler = get_task_handler(job_type="fake", task_type="task", strict=False)
        assert handler is not None
        assert handler.config_type is FakeConfig

    def test_returns_handler_by_config(self):
        register_task(FakeConfig, build_input_params, create_outfile_catalog)
        handler = get_task_handler(config=FakeConfig, strict=False)
        assert handler is not None
        assert handler.config_type is FakeConfig

    def test_returns_none_for_config_without_key(self):
        handler = get_task_handler(config=NoKeyConfig)
        assert handler is None


# ---------------------------------------------------------------------------
# list_registered_types
# ---------------------------------------------------------------------------

class TestListRegisteredTypes:
    def test_empty_when_nothing_registered(self):
        assert list_registered_types() == []

    def test_lists_all_registered_keys(self):
        register_task(FakeConfig)
        register_task(OtherConfig)
        keys = list_registered_types()
        assert "nanny_fake_task" in keys
        assert "nanny_other_task" in keys

    def test_count_matches_number_of_registrations(self):
        register_task(FakeConfig)
        register_task(OtherConfig)
        assert len(list_registered_types()) == 2
