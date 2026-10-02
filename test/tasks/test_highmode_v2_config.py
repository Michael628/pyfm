"""Tests for highmode_v2/config.py — entry sub-blocks (sources, low modes,
cg, output).

Builder-driven: each test drives ``build_config`` with a ``_preprocessor``
slice the way the entry's route hook will hand it down, pinning the
routing, enum coercion, Outfile label resolution, partial seed formatting
and validation rules of every sub-block."""

import pytest

from pyfm.core.builder import build_config
from pyfm.domain import MassDict, Outfile
from pyfm.tasks.hadrons.highmode_v2.config import (
    LMAHighModeConfig,
    BiasedSourceConfig,
    CacheMode,
    CgConfig,
    GridSourceConfig,
    LowModeMethod,
    LowModesConfig,
    MesonFieldConfig,
    OutputConfig,
    PreconMode,
    SourcesConfig,
)
from pyfm.tasks.hadrons.types import SolveCrossTerms

BASE = {"formatting": {}, "logging_level": "INFO", "runid": "test", "home": "/w"}

FILES = {
    "meson_field_cache": {
        "filestem": "lma-meson/m{mass}/mf_{series}",
        "good_size": 1286000,
    },
    "meson_nomass": {"filestem": "lma-meson/mf_{series}", "good_size": 1286000},
    "high_modes": {"filestem": "corr/corr_{tsource}", "good_size": 3900},
}


def make_params(**extra):
    return BASE | extra


class TestGridSourceConfig:
    def test_built_from_slice(self):
        config = build_config(
            GridSourceConfig,
            make_params(_preprocessor={"tstart": 0, "tstop": 3, "dt": 1}),
        )
        assert config.tsources == [0, 1, 2, 3]

    def test_inherits_shared_window_params(self):
        config = build_config(
            GridSourceConfig,
            make_params(tstart=0, tstop=5, dt=2, _preprocessor={}),
        )
        assert config.tsources == [0, 2, 4]


class TestBiasedSourceConfig:
    def test_fields_absorbed_from_slice(self):
        config = build_config(
            BiasedSourceConfig,
            make_params(_preprocessor={"n": 4, "seed": "seed", "replace": False}),
        )
        assert config.n == 4
        assert config.seed == "seed"
        assert config.replace is False

    def test_seed_template_resolved_at_build(self):
        config = build_config(
            BiasedSourceConfig,
            make_params(
                series="a", cfg="20", _preprocessor={"n": 2, "seed": "s_{series}_{cfg}"}
            ),
        )
        assert config.seed == "s_a_20"

    def test_seed_template_stays_partial_without_series_cfg(self):
        # Aggregation builds carry no series/cfg: the unresolved template
        # survives, and only label-bearing consumers (n{i}) read it.
        config = build_config(
            BiasedSourceConfig,
            make_params(_preprocessor={"n": 2, "seed": "s_{series}_{cfg}"}),
        )
        assert config.seed == "s_{series}_{cfg}"

    def test_replace_defaults_true(self):
        config = build_config(
            BiasedSourceConfig, make_params(_preprocessor={"n": 2, "seed": "s"})
        )
        assert config.replace is True


class TestSourcesConfig:
    # Fresh dict per call: route hooks pop keys out of the _preprocessor
    # slice, so a shared mutable class attribute would be consumed by the
    # first test that routes it.
    @staticmethod
    def _grid():
        return {"grid": {"tstart": 0, "tstop": 3, "dt": 1}}

    def test_grid_routed_to_subconfig(self):
        config = build_config(
            SourcesConfig, make_params(time=4, _preprocessor=self._grid())
        )
        assert config.grid_config is not None
        assert config.grid_config.tsources == [0, 1, 2, 3]
        assert config.biased_config is None

    def test_biased_routed_to_subconfig(self):
        config = build_config(
            SourcesConfig,
            make_params(
                time=4,
                _preprocessor={"biased": {"n": 2, "seed": "s"}},
            ),
        )
        assert config.biased_config is not None
        assert config.biased_config.n == 2
        assert config.grid_config is None

    def test_slice_overrides_shared_time_noise(self):
        config = build_config(
            SourcesConfig,
            make_params(time=8, noise=3, _preprocessor=self._grid() | {"time": 4}),
        )
        assert config.time == 4
        assert config.noise == 3

    def test_grid_true_uses_shared_window(self):
        config = build_config(
            SourcesConfig,
            make_params(time=4, tstart=1, tstop=2, dt=1, _preprocessor={"grid": True}),
        )
        assert config.grid_config.tsources == [1, 2]

    def test_grid_false_counts_as_absent(self):
        with pytest.raises(ValueError, match="exactly one"):
            build_config(
                SourcesConfig,
                make_params(time=4, _preprocessor={"grid": False}),
            )

    def test_neither_mode_rejected(self):
        with pytest.raises(ValueError, match="exactly one"):
            build_config(SourcesConfig, make_params(time=4, _preprocessor={}))

    def test_both_modes_rejected(self):
        with pytest.raises(ValueError, match="exactly one"):
            build_config(
                SourcesConfig,
                make_params(
                    time=4,
                    _preprocessor=self._grid() | {"biased": {"n": 2, "seed": "s"}},
                ),
            )

    def test_grid_source_enumeration(self):
        config = build_config(
            SourcesConfig,
            make_params(time=4, _preprocessor={"grid": {"tstart": 0, "tstop": 2, "dt": 2}}),
        )
        assert config.tsource_range == [0, 2]
        assert config.source_labels == ["t0", "t2"]
        assert config.source_axis == ["0", "2"]
        assert [(r.label, r.axis, r.t0) for r in config.source_refs] == [
            ("t0", "0", 0),
            ("t2", "2", 2),
        ]

    def test_biased_source_enumeration(self):
        config = build_config(
            SourcesConfig,
            make_params(time=4, _preprocessor={"biased": {"n": 3, "seed": "s"}}),
        )
        assert config.source_labels == ["n0", "n1", "n2"]
        assert config.source_axis == ["n0", "n1", "n2"]
        assert len(config.source_refs) == 3
        assert len({r.t0 for r in config.source_refs}) >= 1  # draws may repeat
        # Draws are a pure function of the stored seed.
        again = build_config(
            SourcesConfig,
            make_params(time=4, _preprocessor={"biased": {"n": 3, "seed": "s"}}),
        )
        assert [r.t0 for r in again.source_refs] == [r.t0 for r in config.source_refs]

    def test_biased_without_replacement_samples_distinct(self):
        config = build_config(
            SourcesConfig,
            make_params(
                time=4, _preprocessor={"biased": {"n": 3, "seed": "s", "replace": False}}
            ),
        )
        assert len(set(config.tsource_range)) == 3

    def test_biased_n_exceeds_time_rejected(self):
        with pytest.raises(ValueError, match="n <= time"):
            build_config(
                SourcesConfig,
                make_params(
                    time=4,
                    _preprocessor={"biased": {"n": 5, "seed": "s", "replace": False}},
                ),
            )

    def test_biased_requires_positive_n(self):
        with pytest.raises(ValueError, match="positive"):
            build_config(
                SourcesConfig,
                make_params(time=4, _preprocessor={"biased": {"n": 0, "seed": "s"}}),
            )

    def test_biased_requires_seed(self):
        with pytest.raises(ValueError, match="seed"):
            build_config(
                SourcesConfig,
                make_params(time=4, _preprocessor={"biased": {"n": 2, "seed": ""}}),
            )


class TestMesonFieldConfig:
    def test_file_label_resolved_via_files(self):
        config = build_config(
            MesonFieldConfig,
            make_params(_preprocessor={"file": "meson_field_cache"}),
            file_params=FILES,
        )
        assert config.file == Outfile(
            filestem="/w/lma-meson/m{mass}/mf_{series}",
            ext=".{cfg}/{gamma}_0_0_0.h5",
            good_size=1286000,
        )

    def test_cache_enum_coerced_from_string(self):
        config = build_config(
            MesonFieldConfig,
            make_params(
                _preprocessor={"file": "meson_field_cache", "cache": "build_and_load"}
            ),
            file_params=FILES,
        )
        assert config.cache is CacheMode.BUILD_AND_LOAD

    def test_blocksize_default_matches_old_config(self):
        config = build_config(
            MesonFieldConfig,
            make_params(_preprocessor={"file": "meson_field_cache"}),
            file_params=FILES,
        )
        assert config.blocksize == 12
        assert config.cache is CacheMode.LOAD


class TestLowModesConfig:
    def test_scalar_method_from_wrapped_slice(self):
        config = build_config(
            LowModesConfig, make_params(_preprocessor={"method": "solve"})
        )
        assert config.method is LowModeMethod.SOLVE
        assert config.meson_field_config is None

    def test_none_method(self):
        config = build_config(
            LowModesConfig, make_params(_preprocessor={"method": "none"})
        )
        assert config.method is LowModeMethod.NONE

    def test_meson_field_routes_and_pins_method(self):
        config = build_config(
            LowModesConfig,
            make_params(
                _preprocessor={
                    "meson_field": {
                        "file": "meson_field_cache",
                        "cache": "build_only",
                        "blocksize": 8,
                    }
                }
            ),
            file_params=FILES,
        )
        assert config.method is LowModeMethod.MESON_FIELD
        assert config.meson_field_config is not None
        assert config.meson_field_config.cache is CacheMode.BUILD_ONLY
        assert config.meson_field_config.blocksize == 8

    def test_meson_field_method_without_block_rejected(self):
        with pytest.raises(ValueError, match="meson_field block"):
            build_config(
                LowModesConfig, make_params(_preprocessor={"method": "meson_field"})
            )

    def test_solve_with_meson_field_block_rejected(self):
        # Direct-construction guard: route pins method from the block, so a
        # block plus a non-meson_field method only arises by hand.
        config = LowModesConfig(
            formatting={},
            logging_level="INFO",
            runid="test",
            method=LowModeMethod.SOLVE,
            meson_field_config=MesonFieldConfig(
                formatting={},
                logging_level="INFO",
                runid="test",
                file=Outfile(filestem="mf_{mass}", ext=".{cfg}/{gamma}_0_0_0.h5", good_size=1),
            ),
        )
        from pyfm.tasks.hadrons.highmode_v2.config import validate_low_modes

        with pytest.raises(ValueError, match="only valid with the meson_field method"):
            validate_low_modes(config)

    def test_filestem_mass_token_required(self):
        with pytest.raises(ValueError, match=r"\{mass\}"):
            build_config(
                LowModesConfig,
                make_params(_preprocessor={"meson_field": {"file": "meson_nomass"}}),
                file_params=FILES,
            )


class TestCgConfig:
    def test_defaults_match_old_config(self):
        config = build_config(CgConfig, make_params(_preprocessor={}))
        assert config.solver == "mpcg"
        assert config.residual == [1e-8]
        assert config.precon is PreconMode.CHAIN

    def test_precon_enum_coerced_from_string(self):
        config = build_config(
            CgConfig, make_params(_preprocessor={"precon": "each", "residual": [1e-4, 1e-8]})
        )
        assert config.precon is PreconMode.EACH

    def test_residual_scalar_wrapped_to_list(self):
        config = build_config(CgConfig, make_params(_preprocessor={"residual": 1e-6}))
        assert config.residual == [1e-6]

    def test_unknown_solver_rejected(self):
        with pytest.raises(ValueError, match="cg.solver"):
            build_config(CgConfig, make_params(_preprocessor={"solver": "bogus"}))

    def test_empty_residual_rejected(self):
        with pytest.raises(ValueError, match="at least one residual"):
            build_config(CgConfig, make_params(_preprocessor={"residual": []}))


class TestOutputConfig:
    def test_file_label_resolved_and_defaults(self):
        config = build_config(
            OutputConfig,
            make_params(_preprocessor={"file": "high_modes"}),
            file_params=FILES,
        )
        assert config.file.filestem == "/w/corr/corr_{tsource}"
        assert config.overwrite is False
        assert config.solve_cross_terms is SolveCrossTerms.DIAGONAL

    def test_solve_cross_terms_coerced_from_string(self):
        config = build_config(
            OutputConfig,
            make_params(
                _preprocessor={"file": "high_modes", "solve_cross_terms": "tiered"}
            ),
            file_params=FILES,
        )
        assert config.solve_cross_terms is SolveCrossTerms.TIERED


MASS = MassDict.from_dict({"l": 0.002426, "u": 0.001524})


def entry_slice(**overrides):
    slice_ = {
        "operations": {"gamma": ["pion_local"], "mass": ["l"]},
        "sources": {"time": 4, "grid": {"tstart": 0, "tstop": 3, "dt": 1}},
        "low_modes": "solve",
        "cg": {"solver": "mpcg", "residual": [1e-8]},
        "output": {"file": "high_modes"},
    }
    slice_.update(overrides)
    return slice_


def build_entry(label=None, **overrides):
    params = make_params(mass=MASS, _preprocessor=entry_slice(**overrides))
    if label is not None:
        params = params | {"label": label}
    return build_config(LMAHighModeConfig, params, file_params=FILES)


class TestLMAHighModeConfigRouting:
    def test_full_entry_build(self):
        config = build_entry()
        assert config.op_list[0].gamma.name == "PION_LOCAL"
        assert config.action_name == "stag_mass_{mass}"
        assert config.solver_name == "stag_{solver}_mass_{mass}"
        assert config.low_modes_name == "evecs_mass_{mass}"
        assert config.shift_gauge_name == "gauge_apbc"
        assert config.sources_config.grid_config.tsources == [0, 1, 2, 3]
        assert config.low_modes_config.method is LowModeMethod.SOLVE
        assert config.cg_config.solver == "mpcg"
        assert config.output_config.file.filestem == "/w/corr/corr_{tsource}"

    def test_entry_overrides_name_defaults(self):
        config = build_entry(action_name="act_{mass}", solver_name="s_{solver}_{mass}")
        assert config.action_name == "act_{mass}"
        assert config.solver_name == "s_{solver}_{mass}"

    def test_absent_low_modes_defaults_to_solve(self):
        config = build_entry(low_modes=None)
        assert config.low_modes_config.method is LowModeMethod.SOLVE

    def test_absent_cg_is_none(self):
        config = build_entry(cg=None)
        assert config.cg_config is None

    def test_cg_true_builds_defaults(self):
        config = build_entry(cg=True)
        assert config.cg_config is not None
        assert config.cg_config.solver == "mpcg"

    def test_label_flows_as_param_and_format_token(self):
        config = build_entry(label="sloppy", action_name="act_{label}_{mass}")
        assert config.label == "sloppy"
        assert config.action_name == "act_sloppy_{mass}"
        assert config.module_name("noise_fv") == "sloppy_noise_fv"

    def test_unkeyed_label_default_empty(self):
        config = build_entry()
        assert config.label == ""
        assert config.module_name("noise_fv") == "noise_fv"

    def test_split_grid_partial_strips_and_warns(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            config = build_entry(split_mpi_layout="1.1.1.2")
        assert config.split_mpi_layout is None
        assert config.subgrid_ranks is None
        assert "split_mpi_layout" in caplog.text

    def test_split_grid_both_kept(self):
        config = build_entry(split_mpi_layout="1.1.1.2", subgrid_ranks=2)
        assert config.split_mpi_layout == "1.1.1.2"
        assert config.subgrid_ranks == 2


class TestLMAHighModeConfigSolverLabels:
    def test_diagonal_default(self):
        config = build_entry()
        assert config.get_solver_labels() == ["ranLL", "ama"]

    def test_all_mode_cross_terms(self):
        config = build_entry(output={"file": "high_modes", "solve_cross_terms": "all"})
        assert config.get_solver_labels() == [
            "ranLL",
            "ama",
            "ranLL_ama",
            "ama_ranLL",
        ]

    def test_tiered_mode(self):
        config = build_entry(
            output={"file": "high_modes", "solve_cross_terms": "tiered"}
        )
        assert config.get_solver_labels() == ["ranLL", "ranLL_ama"]

    def test_none_low_modes_drops_ranll(self):
        # precon must be 'none' with low_modes 'none' (entry validation rule).
        config = build_entry(low_modes="none", cg={"precon": "none"})
        assert config.get_solver_labels() == ["ama"]
        assert config.effective_solve_cross_terms is SolveCrossTerms.DIAGONAL

    def test_multi_residual_labels(self):
        config = build_entry(cg={"residual": [1e-4, 1e-8]})
        assert config.get_solver_labels(skip_cross=True) == [
            "ranLL",
            "ama_0.0001",
            "ama_1e-08",
        ]

    def test_mass_cross_terms_labels(self):
        config = build_entry(
            operations={"gamma": ["pion_local"], "mass": ["l", "u"]},
            mass_cross_terms=True,
            output={"file": "high_modes"},
        )
        op = config.op_list[0]
        assert config.get_mass_labels(op) == [
            "002426",
            "001524",
            "002426_m001524",
        ]


class TestLMAHighModeConfigOperations:
    def test_list_form(self):
        config = build_entry(operations={"gamma": ["pion_local", "vec_local"], "mass": ["l"]})
        assert [op.gamma.name for op in config.op_list] == ["PION_LOCAL", "VEC_LOCAL"]
        assert config.masses == ["l"]

    def test_per_gamma_form(self):
        config = build_entry(operations={"pion_local": {"mass": ["l", "u"]}})
        assert config.op_list[0].gamma.name == "PION_LOCAL"
        assert list(config.op_list[0].mass) == ["l", "u"]

    def test_non_mapping_rejected(self):
        with pytest.raises(TypeError, match="operations must be a mapping"):
            build_entry(operations=["pion_local"])

    def test_bare_gamma_rejected(self):
        with pytest.raises(ValueError, match=r"\['gamma'\] must be nested under `operations:`"):
            build_entry(gamma=["pion_local"])

    def test_bare_mass_rejected(self):
        with pytest.raises(ValueError, match=r"\['mass'\] must be nested under `operations:`"):
            build_entry(mass=["l"])

    def test_typo_key_rejected(self):
        with pytest.raises(ValueError, match=r"Unknown high_modes entry keys: \['opertions'\]"):
            build_entry(opertions={"gamma": ["pion_local"], "mass": ["l"]})

    def test_stale_low_mode_method_rejected(self):
        with pytest.raises(ValueError, match="low_mode_method"):
            build_entry(low_mode_method="load")

    def test_raw_sub_config_name_rejected(self):
        with pytest.raises(ValueError, match="cg_config"):
            build_entry(cg_config={"solver": "mpcg"})

    def test_entry_fields_still_accepted(self):
        config = build_entry(mass_cross_terms=True, subgrid_ranks=2, split_mpi_layout="1.1.1.2")
        assert config.mass_cross_terms is True
        assert config.subgrid_ranks == 2


class TestLMAHighModeConfigValidation:
    def test_empty_operations_rejected(self):
        with pytest.raises(ValueError, match="no operations"):
            build_entry(operations={})

    def test_missing_operations_rejected(self):
        slice_ = entry_slice()
        del slice_["operations"]
        params = make_params(mass=MASS, _preprocessor=slice_)
        with pytest.raises(ValueError, match="no operations"):
            build_config(LMAHighModeConfig, params, file_params=FILES)

    def test_build_only_empty_operations_rejected(self):
        with pytest.raises(ValueError, match="no operations"):
            build_entry(
                operations={},
                low_modes={"meson_field": {"file": "meson_field_cache", "cache": "build_only"}},
                cg=None,
                output=None,
            )

    def test_none_low_modes_without_cg_rejected(self):
        with pytest.raises(ValueError, match="nothing to solve"):
            build_entry(low_modes="none", cg=None)

    def test_precon_with_none_low_modes_rejected(self):
        with pytest.raises(ValueError, match="precon"):
            build_entry(low_modes="none", cg={"precon": "chain"})

    def test_precon_none_with_none_low_modes_ok(self):
        config = build_entry(low_modes="none", cg={"precon": "none"})
        assert config.cg_config.precon is PreconMode.NONE

    def test_build_only_with_cg_rejected(self):
        with pytest.raises(ValueError, match="build_only"):
            build_entry(
                low_modes={"meson_field": {"file": "meson_field_cache", "cache": "build_only"}},
                output=None,
            )

    def test_build_only_with_output_rejected(self):
        with pytest.raises(ValueError, match="build_only"):
            build_entry(
                low_modes={"meson_field": {"file": "meson_field_cache", "cache": "build_only"}},
                cg=None,
            )

    def test_build_only_alone_ok(self):
        config = build_entry(
            low_modes={"meson_field": {"file": "meson_field_cache", "cache": "build_only"}},
            cg=None,
            output=None,
        )
        assert config.output_config is None

    def test_missing_output_rejected(self):
        with pytest.raises(ValueError, match="output"):
            build_entry(output=None)

    def test_meson_field_biased_replace_true_rejected(self):
        with pytest.raises(ValueError, match="replace"):
            build_entry(
                sources={"time": 4, "biased": {"n": 2, "seed": "s"}},
                low_modes={"meson_field": {"file": "meson_field_cache"}},
            )

    def test_meson_field_biased_replace_false_ok(self):
        config = build_entry(
            sources={"time": 4, "biased": {"n": 2, "seed": "s", "replace": False}},
            low_modes={"meson_field": {"file": "meson_field_cache"}},
        )
        assert config.use_meson_field is True

    def test_nonlocal_ops_require_shift_gauge(self):
        with pytest.raises(ValueError, match="shift_gauge_name"):
            build_entry(
                operations={"gamma": ["vec_onelink"], "mass": ["l"]},
                shift_gauge_name=None,
            )

    def test_nonpositive_subgrid_ranks_rejected(self):
        # Both split keys present so the both-or-neither strip doesn't eat
        # subgrid_ranks before validation sees it.
        with pytest.raises(ValueError, match="subgrid_ranks"):
            build_entry(split_mpi_layout="1.1.1.2", subgrid_ranks=0)
