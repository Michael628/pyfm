"""Tests for nbias random wall-source sampling on HighModeConfig."""

import copy

import pytest

from pyfm.domain import MassDict, OpList, Outfile
from pyfm.tasks.hadrons.types import HighModeConfig
# Final strategy import block (phase 3): build_input_params/
# create_outfile_catalog landed with phase 2, build_aggregator_params with
# phase 3.
from pyfm.tasks.hadrons.highmode.strategy import (
    build_aggregator_params,
    build_input_params,
    create_outfile_catalog,
    route_params,
    validate_config,
)


def make_config(**overrides):
    kwargs = dict(
        formatting={},
        logging_level="INFO",
        runid="test",
        mass=MassDict.from_dict({"l": 0.002426}),
        action_name="action_{mass}",
        solver_name="solver_{solver}_{mass}",
        low_modes_name="low_modes_{mass}",
        operations=OpList.from_dict({"pion_local": {"mass": ["l"]}}),
        high_modes=Outfile(filestem="corr/corr_{tsource}", ext=".20.h5", good_size=1),
        tstart=0,
        tstop=3,
        dt=1,
        noise=1,
        time=4,
        skip_cg=True,
        shift_gauge_name="shift_gauge",
    )
    kwargs.update(overrides)
    return HighModeConfig(**kwargs)


class TestSourcePlacement:
    def test_dt_mode_defaults_unchanged(self):
        config = make_config()
        assert config.tsource_range == [0, 1, 2, 3]
        assert config.source_axis == ["0", "1", "2", "3"]
        assert config.source_labels == ["t0", "t1", "t2", "t3"]
        assert [r.t0 for r in config.source_refs] == [0, 1, 2, 3]

    def test_bias_mode_labels_and_axis(self):
        config = make_config(nbias=8, bias_seed="seed_a_20")
        assert len(config.tsource_range) == 8
        assert config.source_labels == [f"n{i}" for i in range(8)]
        assert config.source_axis == config.source_labels
        assert all(0 <= t < config.time for t in config.tsource_range)

    def test_sampling_is_deterministic(self):
        a = make_config(nbias=8, bias_seed="seed_a_20")
        b = make_config(nbias=8, bias_seed="seed_a_20")
        assert a.tsource_range == b.tsource_range

    def test_seed_variation_changes_draws(self):
        a = make_config(nbias=8, bias_seed="seed_a_20").tsource_range
        b = make_config(nbias=8, bias_seed="seed_a_21").tsource_range
        assert a != b

    def test_with_replacement_duplicates_survive(self):
        # nbias > time guarantees duplicate time slices by pigeonhole; the
        # labels/axis must stay unique regardless.
        config = make_config(nbias=6, bias_seed="pigeonhole", time=4)
        assert len(config.tsource_range) == 6
        assert len(set(config.tsource_range)) < 6
        assert len(set(config.source_labels)) == 6
        assert len(set(config.source_axis)) == 6

    def test_bias_replace_defaults_to_true(self):
        config = make_config(nbias=4, bias_seed="seed_a_20")
        assert config.bias_replace is True

    def test_without_replacement_draws_unique_times(self):
        config = make_config(nbias=6, bias_seed="uniq", time=32, bias_replace=False)
        draws = config.tsource_range
        assert len(draws) == 6
        assert len(set(draws)) == 6
        assert all(0 <= t < 32 for t in draws)

    def test_without_replacement_is_deterministic(self):
        a = make_config(nbias=6, bias_seed="uniq", time=32, bias_replace=False)
        b = make_config(nbias=6, bias_seed="uniq", time=32, bias_replace=False)
        assert a.tsource_range == b.tsource_range

    def test_without_replacement_seed_variation_changes_draws(self):
        a = make_config(nbias=6, bias_seed="uniq_a", time=32, bias_replace=False).tsource_range
        b = make_config(nbias=6, bias_seed="uniq_b", time=32, bias_replace=False).tsource_range
        assert a != b

    def test_without_replacement_nbias_exceeding_time_refused(self):
        config = make_config(nbias=6, bias_seed="uniq", time=4, bias_replace=False)
        with pytest.raises(ValueError, match="without-replacement"):
            config.tsource_range

    def test_unseeded_draw_refused(self):
        config = make_config(nbias=4)
        with pytest.raises(ValueError, match="bias_seed"):
            config.tsource_range


class TestSeedComposition:
    @staticmethod
    def routed(tasks_slice, series=None, cfg=None):
        params = {"mass": {"l": 0.01}, "_preprocessor": copy.deepcopy(tasks_slice)}
        if series is not None:
            params["series"] = series
        if cfg is not None:
            params["cfg"] = cfg
        return route_params(params)

    def test_route_composes_series_cfg_seed(self):
        routed = self.routed({"nbias": 4, "bias_seed": "mybase"}, series="a", cfg="20")
        assert routed["nbias"] == 4
        assert routed["bias_seed"] == "mybase_a_20"

    def test_route_leaves_seed_untouched_without_nbias(self):
        routed = self.routed({"bias_seed": "mybase", "pion_local": {"mass": ["l"]}})
        assert routed["bias_seed"] == "mybase"

    def test_route_without_series_cfg_composes_empty_suffixes(self):
        routed = self.routed({"nbias": 4, "bias_seed": "mybase"})
        assert routed["bias_seed"] == "mybase__"

    def test_route_missing_seed_stays_none(self):
        routed = self.routed({"nbias": 4}, series="a", cfg="20")
        assert "bias_seed" not in routed

    def test_nbias_bias_seed_route_to_fields_not_operations(self):
        routed = self.routed(
            {"nbias": 4, "bias_seed": "b", "pion_local": {"mass": ["l"]}}
        )
        assert routed["operations"] == {"pion_local": {"mass": ["l"]}}
        assert "nbias" not in routed["operations"]
        assert "bias_seed" not in routed["operations"]

    def test_bias_replace_routes_to_field_not_operations(self):
        routed = self.routed(
            {"nbias": 4, "bias_seed": "b", "bias_replace": False, "pion_local": {"mass": ["l"]}}
        )
        assert routed["bias_replace"] is False
        assert "bias_replace" not in routed["operations"]


class TestBiasValidation:
    def test_nbias_requires_seed(self):
        config = make_config(nbias=4)
        with pytest.raises(ValueError, match="bias_seed"):
            validate_config(config)

    def test_nbias_must_be_positive(self):
        config = make_config(nbias=0, bias_seed="seed_a_20")
        with pytest.raises(ValueError, match="nbias"):
            validate_config(config)

    def test_valid_bias_config_passes(self):
        config = make_config(nbias=4, bias_seed="seed_a_20")
        validate_config(config)

    def test_without_replacement_nbias_exceeds_time_rejected(self):
        config = make_config(nbias=6, bias_seed="seed_a_20", time=4, bias_replace=False)
        with pytest.raises(ValueError, match="without-replacement"):
            validate_config(config)

    def test_valid_without_replacement_config_passes(self):
        config = make_config(nbias=4, bias_seed="seed_a_20", time=4, bias_replace=False)
        validate_config(config)


class TestBiasEmission:
    @staticmethod
    def make_bias_config(**overrides):
        return make_config(
            nbias=6, bias_seed="pigeonhole", time=4, skip_cg=False, **overrides
        )

    def test_noise_modules_use_block_labels_with_sampled_t0(self):
        config = self.make_bias_config(overwrite=True)
        result = build_input_params(config)

        for i, t0 in enumerate(config.tsource_range):
            name = f"noise_n{i}"
            assert name in result.modules
            assert result.modules[name]["options"]["t0"] == str(t0)

    def test_quark_and_contraction_names_use_block_labels(self):
        config = self.make_bias_config(overwrite=True)
        result = build_input_params(config)

        assert "quark_ama_pion_local_mass_l_n0" in result.modules
        assert "corr_ama_pion_local_mass_l_n0" in result.modules
        assert not any(n.startswith("noise_t") for n in result.modules)

    def test_outputs_use_block_axis_values(self):
        config = self.make_bias_config(overwrite=True)
        result = build_input_params(config)

        corr = next(n for n in result.modules if n.startswith("corr_"))
        assert "_n0" in result.modules[corr]["options"]["output"]

    def test_duplicate_times_yield_distinct_modules(self):
        config = self.make_bias_config(overwrite=True)
        result = build_input_params(config)
        times = config.tsource_range
        assert len({t for t in times if times.count(t) > 1}) >= 1  # precondition
        for i in range(config.nbias):
            assert f"noise_n{i}" in result.modules

    def test_dt_mode_module_names_unchanged(self):
        config = make_config(overwrite=True, skip_cg=False)
        result = build_input_params(config)
        for t in [0, 1, 2, 3]:
            assert f"noise_t{t}" in result.modules
        assert "noise_n0" not in result.modules


class TestBiasCatalogAxis:
    def test_catalog_tsource_axis_uses_block_labels(self):
        config = make_config(nbias=6, bias_seed="pigeonhole", time=4)
        df = create_outfile_catalog(config)
        assert sorted(df["tsource"].unique()) == [f"n{i}" for i in range(6)]

    def test_catalog_dt_axis_unchanged(self):
        config = make_config()
        df = create_outfile_catalog(config)
        assert sorted(df["tsource"].unique()) == ["0", "1", "2", "3"]


class TestBiasAggregatorAxis:
    @staticmethod
    def make_bias_config(**overrides):
        return make_config(nbias=6, bias_seed="pigeonhole", time=4, **overrides)

    def test_aggregator_replacements_use_block_axis(self):
        config = self.make_bias_config()
        params = build_aggregator_params(config, average=False)
        first = params[params["run"][0]]
        assert sorted(first["load_files"]["replacements"]["tsource"]) == [
            f"n{i}" for i in range(6)
        ]

    def test_run_prefix_namespaces_keys(self):
        config = self.make_bias_config()
        params = build_aggregator_params(config, average=False, run_prefix="bias_")
        plain = build_aggregator_params(config, average=False)
        assert params["run"] == [f"bias_{k}" for k in plain["run"]]
        for key in plain["run"]:
            assert params[f"bias_{key}"]["load_files"] == plain[key]["load_files"]

    def test_dt_aggregator_unchanged_without_prefix(self):
        config = make_config()
        params = build_aggregator_params(config, average=False)
        first = params[params["run"][0]]
        assert sorted(first["load_files"]["replacements"]["tsource"]) == [
            "0",
            "1",
            "2",
            "3",
        ]

    def test_average_action_averages_over_axis(self):
        config = self.make_bias_config()
        params = build_aggregator_params(config, average=True)
        first = params[params["run"][0]]
        assert first["actions"]["average"] == ["tsource"]
