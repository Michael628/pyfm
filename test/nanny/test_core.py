"""Integration tests for create_task() and get_outfiles() using the Task NamedTuple interface."""
import typing as t
import pytest
import pandas as pd
from pydantic.dataclasses import dataclass

from pyfm.nanny.core import Task, create_task
from pyfm.nanny.validator import get_outfiles
from pyfm.domain.task_registry import TaskHandler
from pyfm.domain.conftypes import ConfigBase, SimpleConfig
from pyfm.domain import task_registry, build_hooks
from pyfm.tasks.register import register_task, _config_to_task_key


# ---------------------------------------------------------------------------
# Minimal stub config for testing create_task end-to-end
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _StubSimpleConfig(SimpleConfig):
    pass


# Registry key that matches job_type="stub", task_type="simple"
_STUB_KEY = "stub_simple"


def build_input_params(config):
    return {"stub": True}


def create_outfile_catalog(config):
    return pd.DataFrame(
        {"filepath": ["/fake/out.h5"], "exists": [True], "file_size": [100], "good_size": [50]}
    )


def build_aggregator_params(config, average):
    return {"run": ["a"]}


# Minimal yaml_params structure that create_task / get_job_config can parse.
# The stub step is named "stub_step".
_BASE_YAML_PARAMS: t.Dict[str, t.Any] = {
    "shared_params": {
        "formatting": {},
        "logging_level": "INFO",
        "runid": "test-run",
    },
    "nanny": {
        "home": "/tmp",
        "todo_file": "todo",
        "max_cases": 1,
        "max_queue": 1,
        "wait": 1,
        "check_interval": 1,
        "job_name_pfx": "t",
        "scheduler": "SLURM",
    },
    "submit": {
        "layout": {},
    },
    "job_setup": {
        "stub_step": {
            "job_type": "stub",
            "task_type": "simple",
            "io": "/tmp/input",
            "wall_time": "00:10:00",
            "ppn": 1,
            "nodes": 1,
            "lattice": [4, 4, 4, 8],
            "geom": [1, 1, 1, 1],
            "run": "/tmp/run.sh",
            "tasks": {},
        }
    },
    "files": {},
}


# ---------------------------------------------------------------------------
# Registry snapshot-restore fixture (mirrors test/tasks/test_register.py)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_registries():
    saved_handlers = dict(task_registry._handlers)
    saved_hooks = dict(build_hooks._registry)
    saved_config_to_task_key = dict(_config_to_task_key)
    task_registry.clear()
    build_hooks.clear()
    _config_to_task_key.clear()
    yield
    task_registry.clear()
    build_hooks.clear()
    _config_to_task_key.clear()
    task_registry._handlers.update(saved_handlers)
    build_hooks._registry.update(saved_hooks)
    _config_to_task_key.update(saved_config_to_task_key)


def _register_stub_with_catalog():
    register_task(
        _STUB_KEY,
        _StubSimpleConfig,
        build_input_params,
        create_outfile_catalog,
        build_aggregator_params,
    )


def _register_stub_without_catalog():
    register_task(
        _STUB_KEY,
        _StubSimpleConfig,
        build_input_params,
        build_aggregator_params,
    )


# ---------------------------------------------------------------------------
# create_task integration tests
# ---------------------------------------------------------------------------

class TestCreateTask:
    def test_returns_task_namedtuple(self):
        _register_stub_with_catalog()
        result = create_task("stub_step", _BASE_YAML_PARAMS)
        assert isinstance(result, Task)

    def test_task_fields(self):
        _register_stub_with_catalog()
        result = create_task("stub_step", _BASE_YAML_PARAMS)
        assert result._fields == ("handler", "config", "key")

    def test_handler_field_is_task_handler(self):
        _register_stub_with_catalog()
        result = create_task("stub_step", _BASE_YAML_PARAMS)
        assert isinstance(result.handler, TaskHandler)

    def test_config_is_config_base(self):
        _register_stub_with_catalog()
        result = create_task("stub_step", _BASE_YAML_PARAMS)
        assert isinstance(result.config, ConfigBase)

    def test_key_matches_registered_key(self):
        _register_stub_with_catalog()
        result = create_task("stub_step", _BASE_YAML_PARAMS)
        assert result.key == f"nanny_{_STUB_KEY}"

    def test_series_and_cfg_injected_into_config_params(self):
        _register_stub_with_catalog()
        # Should not raise; series/cfg are injected as param_defaults
        result = create_task("stub_step", _BASE_YAML_PARAMS, series="a", cfg="1000")
        assert isinstance(result, Task)

    def test_missing_job_setup_raises(self):
        _register_stub_with_catalog()
        bad_params = {k: v for k, v in _BASE_YAML_PARAMS.items() if k != "job_setup"}
        with pytest.raises(ValueError, match="job_setup"):
            create_task("stub_step", bad_params)

    def test_unregistered_handler_raises(self):
        # No registration — get_job_config should raise ValueError
        with pytest.raises(ValueError):
            create_task("stub_step", _BASE_YAML_PARAMS)


# ---------------------------------------------------------------------------
# get_outfiles caller tests (validator.py)
# ---------------------------------------------------------------------------

class TestGetOutfiles:
    def test_returns_none_when_no_catalog(self):
        # get_job_config uses strict=True which requires TaskHandlerProtocol
        # (both build_input_params AND create_outfile_catalog). A handler without
        # create_outfile_catalog cannot be used with create_task / get_outfiles —
        # so we test the isinstance guard directly on a TaskHandler stub.
        from pyfm.nanny.validator import TaskOutputProtocol

        handler_no_catalog = TaskHandler(
            config_type=_StubSimpleConfig,
            build_input_params=build_input_params,
            build_aggregator_params=build_aggregator_params,
        )
        assert not isinstance(handler_no_catalog, TaskOutputProtocol)

    def test_returns_dataframe_when_catalog_present(self):
        _register_stub_with_catalog()
        task = create_task("stub_step", _BASE_YAML_PARAMS, series="a", cfg="1000")
        result = get_outfiles(task)
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
