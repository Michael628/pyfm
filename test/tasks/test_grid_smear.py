"""Tests for the standalone grid_smear task (grid_milc_to_ildg driver)."""

import pytest

from pyfm.domain import Outfile
from pyfm.tasks.grid.smear import (
    GridSmearConfig,
    build_input_params,
    create_outfile_catalog,
    normalize_params,
)


def _outfile(stem: str, good_size: int = 100) -> Outfile:
    return Outfile(filestem=stem, ext="", good_size=good_size)


def _config() -> GridSmearConfig:
    return GridSmearConfig(
        formatting={},
        logging_level="info",
        runid="test",
        ildg_links=_outfile("out/gauge.20"),
        long_links=_outfile("out/long.20"),
        fat_links=_outfile("out/fat.20"),
        v5_links=_outfile("in/milc.20", good_size=0),
    )


class TestRegistration:
    def test_resolves_via_job_and_task_type(self):
        from pyfm.tasks.register import get_task_handler, get_task_key

        assert get_task_key(job_type="grid", task_type="smear") == "nanny_grid_smear"
        handler = get_task_handler(job_type="grid", task_type="smear", strict=False)
        assert handler is not None
        assert handler.config_type is GridSmearConfig

    def test_listed_as_registered(self):
        from pyfm.tasks.register import list_registered_types

        assert "nanny_grid_smear" in list_registered_types()


class TestNormalize:
    def test_merges_preprocessor_slice(self):
        out = normalize_params({"a": 1, "_preprocessor": {"b": 2}})
        assert out == {"a": 1, "b": 2}
        assert "_preprocessor" not in out

    def test_no_preprocessor_is_noop(self):
        assert normalize_params({"a": 1}) == {"a": 1}


class TestBuildInputParams:
    def test_maps_v5_input_and_ildg_outputs(self):
        params = build_input_params(_config())
        assert params["milcFile"] == "in/milc.20"
        assert params["gaugeStem"] == "out/gauge.20"
        assert params["gaugeFatStem"] == "out/fat.20"
        assert params["gaugeLongStem"] == "out/long.20"

    def test_defaults_milcv5_save_behaviour(self):
        # The executable always reads MILC v5 and always writes all three ILDG
        # fields (save_smear=true). All three output stems are always present.
        params = build_input_params(_config())
        assert params["boundary"] == "1 1 1 -1"
        assert params["exitOnChecksumMismatch"] == "false"
        for stem in ("gaugeStem", "gaugeFatStem", "gaugeLongStem"):
            assert stem in params

    def test_trajectory_not_set_here(self):
        # trajectory is supplied by the grid XML wrapper (from cfg).
        assert "trajectory" not in build_input_params(_config())


class TestCreateOutfileCatalog:
    def test_covers_three_ildg_outputs(self):
        cat = create_outfile_catalog(_config())
        assert sorted(cat["filepath"]) == [
            "out/fat.20",
            "out/gauge.20",
            "out/long.20",
        ]

    def test_excludes_v5_input(self):
        cat = create_outfile_catalog(_config())
        assert all("in/milc" not in p for p in cat["filepath"])

    def test_carries_good_size(self):
        cat = create_outfile_catalog(_config())
        assert list(cat["good_size"]) == [100, 100, 100]
