"""Tests for JobConfig, JobBundle, and build_job_configs."""
import typing as t
import pytest
import pandas as pd
from pydantic.dataclasses import dataclass

from pyfm.nanny.config import (
    JobConfig,
    JobBundle,
    build_job_configs,
    get_job_config,
    normalize_resources,
    validate_job,
)
from pyfm.domain import SimpleConfig, task_registry, build_hooks
from pyfm.tasks.register import register_task, _config_to_task_key


@dataclass(frozen=True)
class _StubConfig(SimpleConfig):
    pass


def build_input_params(config):
    return {"stub": True}


def create_outfile_catalog(config):
    return pd.DataFrame()


def build_aggregator_params(config, average):
    return {}


_BASE_YAML: t.Dict[str, t.Any] = {
    "shared_params": {"formatting": {}, "logging_level": "INFO", "runid": "test"},
    "submit": {"layout": {"ppn": 4, "stub_step": {"nodes": 1, "geom": [1, 1, 1, 1]}}},
    "job_setup": {
        "stub_step": {
            "job_type": "stub",
            "task_type": "simple",
            "io": "/tmp/in",
            "wall_time": "00:10:00",
            "ppn": 4,
            "nodes": 1,
            "lattice": [4, 4, 4, 8],
            "geom": [1, 1, 1, 1],
            "run": "/tmp/run.sh",
            "tasks": {},
        }
    },
    "files": {},
}


@pytest.fixture(autouse=True)
def _reset_registries():
    saved_handlers = dict(task_registry._handlers)
    saved_hooks = dict(build_hooks._registry)
    saved_cfg = dict(_config_to_task_key)
    task_registry.clear()
    build_hooks.clear()
    _config_to_task_key.clear()
    register_task(
        "stub_simple",
        _StubConfig,
        build_input_params,
        create_outfile_catalog,
        build_aggregator_params,
    )
    build_hooks.register(
        JobConfig, normalize=normalize_resources, validate=validate_job
    )
    yield
    task_registry.clear()
    build_hooks.clear()
    _config_to_task_key.clear()
    task_registry._handlers.update(saved_handlers)
    build_hooks._registry.update(saved_hooks)
    _config_to_task_key.update(saved_cfg)


class TestBuildJobConfigs:
    def test_builds_all_steps(self):
        configs = build_job_configs(_BASE_YAML)
        assert set(configs) == {"stub_step"}
        assert isinstance(configs["stub_step"], JobConfig)

    def test_max_cases_defaults_to_one(self):
        configs = build_job_configs(_BASE_YAML)
        assert configs["stub_step"].max_cases == 1


class TestJobConfigMaxCases:
    def _make(self, **kw):
        return JobConfig(
            run="x",
            job_type="stub",
            step="stub_step",
            io="i",
            wall_time="0:1",
            ppn=1,
            nodes=1,
            lattice=[1, 1, 1, 1],
            geom=[1, 1, 1, 1],
            params={},
            formatting={},
            logging_level="INFO",
            runid="r",
            **kw,
        )

    def test_default_max_cases(self):
        assert self._make().max_cases == 1

    def test_explicit_max_cases(self):
        assert self._make(max_cases=5).max_cases == 5


class TestJobBundle:
    def test_ncases_property(self):
        cfg = JobConfig(
            run="x",
            job_type="stub",
            step="s",
            io="i",
            wall_time="0:1",
            ppn=1,
            nodes=1,
            lattice=[1, 1, 1, 1],
            geom=[1, 1, 1, 1],
            params={},
            formatting={},
            logging_level="INFO",
            runid="r",
        )
        bundle = JobBundle(job_config=cfg, cfgno_steps=[["a.60", 1], ["a.100", 1]])
        assert bundle.ncases == 2


def _job_config(**kw):
    return JobConfig(
        run="x",
        job_type="stub",
        step="stub_step",
        io="i",
        wall_time="0:1",
        ppn=1,
        nodes=1,
        lattice=[1, 1, 1, 1],
        geom=[1, 1, 1, 1],
        params={},
        formatting={},
        logging_level="INFO",
        runid="r",
        **kw,
    )


_RESOURCES_YAML: t.Dict[str, t.Any] = {
    "shared_params": {"formatting": {}, "logging_level": "INFO", "runid": "test"},
    "submit": {
        "resources": {
            "ppn": 4,
            "max_cases": 2,
            "stub_step": {"nodes": 2, "geom": [1, 1, 1, 2]},
        }
    },
    "job_setup": {
        "stub_step": {
            "job_type": "stub",
            "task_type": "simple",
            "io": "/tmp/in",
            "wall_time": "00:10:00",
            "lattice": [4, 4, 4, 8],
            "run": "/tmp/run.sh",
            "tasks": {},
        }
    },
    "files": {},
}


class TestNormalizeResources:
    def test_renames_legacy_layout(self):
        out = normalize_resources({"step": "s", "layout": {"ppn": 8}})
        assert "layout" not in out
        assert out["ppn"] == 8

    def test_merges_global_and_per_step(self):
        out = normalize_resources(
            {
                "step": "stub_step",
                "resources": {
                    "ppn": 4,
                    "max_cases": 2,
                    "stub_step": {"nodes": 2, "geom": [1, 1, 1, 2]},
                },
            }
        )
        assert out["ppn"] == 4
        assert out["max_cases"] == 2
        assert out["nodes"] == 2
        assert out["geom"] == [1, 1, 1, 2]

    def test_per_step_overrides_global(self):
        out = normalize_resources(
            {"step": "s", "resources": {"max_cases": 1, "s": {"max_cases": 5}}}
        )
        assert out["max_cases"] == 5

    def test_get_job_config_uses_resources_merge(self):
        # ppn/nodes/geom/max_cases come from `resources`, NOT job_setup
        jc = get_job_config("stub_step", _RESOURCES_YAML)
        assert jc.ppn == 4
        assert jc.nodes == 2
        assert jc.geom == [1, 1, 1, 2]
        assert jc.max_cases == 2


class TestValidateJob:
    def test_valid_max_cases(self):
        validate_job(_job_config(max_cases=3))  # no raise

    def test_zero_max_cases_raises(self):
        with pytest.raises(ValueError, match="max_cases"):
            validate_job(_job_config(max_cases=0))
