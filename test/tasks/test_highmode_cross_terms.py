"""Unit tests for the split cross-term controls on HighModeConfig.

Covers the label contract (get_solver_labels / get_mass_labels per solve
mode), the shared pair-membership helper, the contraction-emission
bijection, and the legacy `cross_terms` translation.
"""

import pytest

from pyfm.domain import Gamma, MassDict, OpList, Outfile
from pyfm.nanny.taskbuilder import create_task
from pyfm.tasks.hadrons.highmode.strategy import (
    build_aggregator_params,
    build_input_params,
    create_outfile_catalog,
    normalize_params,
    route_params,
    sort_schedule,
)
from pyfm.tasks.hadrons.highmode.twopoint import contraction_gen
from pyfm.tasks.hadrons.types import HighModeConfig, SolveCrossTerms


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
        shift_gauge_name="shift_gauge",
    )
    kwargs.update(overrides)
    return HighModeConfig(**kwargs)


def make_two_mass_config(**overrides):
    return make_config(
        mass=MassDict.from_dict({"l": 0.002426, "u": 0.001524}),
        operations=OpList([OpList.Op(gamma=Gamma.PION_LOCAL, mass=("l", "u"))]),
        **overrides,
    )


class TestSolverLabels:
    def test_diagonal_default_matches_legacy_none(self):
        assert make_config().get_solver_labels() == ["ranLL", "ama"]

    def test_skip_cross_returns_base_regardless_of_mode(self):
        config = make_config(solve_cross_terms=SolveCrossTerms.ALL)
        assert config.get_solver_labels(skip_cross=True) == ["ranLL", "ama"]

    def test_all_appends_both_orientations(self):
        config = make_config(solve_cross_terms=SolveCrossTerms.ALL)
        assert config.get_solver_labels() == [
            "ranLL",
            "ama",
            "ranLL_ama",
            "ama_ranLL",
        ]

    def test_tiered_keeps_ll_and_l_quark_orientation_only(self):
        config = make_config(solve_cross_terms=SolveCrossTerms.TIERED)
        assert config.get_solver_labels() == ["ranLL", "ranLL_ama"]

    def test_mode_ignored_without_low_modes(self):
        config = make_config(
            skip_low_modes=True, solve_cross_terms=SolveCrossTerms.TIERED
        )
        assert config.get_solver_labels() == ["ama"]

    def test_mode_ignored_without_cg(self):
        config = make_config(skip_cg=True, solve_cross_terms=SolveCrossTerms.TIERED)
        assert config.get_solver_labels() == ["ranLL"]

    def test_multi_residual_all_emits_per_residual_lh_pairs(self):
        config = make_config(
            residual=[1e-6, 1e-8], solve_cross_terms=SolveCrossTerms.ALL
        )
        assert config.get_solver_labels() == [
            "ranLL",
            "ama_1e-06",
            "ama_1e-08",
            "ranLL_ama_1e-06",
            "ranLL_ama_1e-08",
            "ama_1e-06_ranLL",
            "ama_1e-08_ranLL",
        ]

    def test_multi_residual_never_crosses_cg_solvers(self):
        config = make_config(
            residual=[1e-6, 1e-8], solve_cross_terms=SolveCrossTerms.ALL
        )
        labels = config.get_solver_labels()
        assert "ama_1e-06_ama_1e-08" not in labels
        assert "ama_1e-08_ama_1e-06" not in labels

    def test_tiered_multi_residual_drops_all_hh(self):
        config = make_config(
            residual=[1e-6, 1e-8], solve_cross_terms=SolveCrossTerms.TIERED
        )
        assert config.get_solver_labels() == [
            "ranLL",
            "ranLL_ama_1e-06",
            "ranLL_ama_1e-08",
        ]


class TestMassLabels:
    def test_off_by_default(self):
        config = make_two_mass_config()
        op = config.op_list[0]
        assert config.get_mass_labels(op) == ["002426", "001524"]

    def test_toggle_appends_upper_triangular_cross_labels(self):
        config = make_two_mass_config(mass_cross_terms=True)
        op = config.op_list[0]
        assert config.get_mass_labels(op) == [
            "002426",
            "001524",
            "002426_m001524",
        ]

    def test_skip_cross_suppresses_mass_cross(self):
        config = make_two_mass_config(mass_cross_terms=True)
        op = config.op_list[0]
        assert config.get_mass_labels(op, skip_cross=True) == ["002426", "001524"]


class TestAdmitsSolvePair:
    @pytest.mark.parametrize(
        "mode,quark,antiquark,admitted",
        [
            (SolveCrossTerms.DIAGONAL, "ranLL", "ranLL", True),
            (SolveCrossTerms.DIAGONAL, "ama", "ama", True),
            (SolveCrossTerms.DIAGONAL, "ranLL", "ama", False),
            (SolveCrossTerms.DIAGONAL, "ama", "ranLL", False),
            (SolveCrossTerms.ALL, "ranLL", "ama", True),
            (SolveCrossTerms.ALL, "ama", "ranLL", True),
            (SolveCrossTerms.ALL, "ama_1e-06", "ama_1e-08", False),
            (SolveCrossTerms.TIERED, "ranLL", "ranLL", True),
            (SolveCrossTerms.TIERED, "ama", "ama", False),
            (SolveCrossTerms.TIERED, "ranLL", "ama", True),
            (SolveCrossTerms.TIERED, "ama", "ranLL", False),
        ],
    )
    def test_pair_table(self, mode, quark, antiquark, admitted):
        config = make_config(solve_cross_terms=mode)
        assert config.admits_solve_pair(quark, antiquark) is admitted

    def test_tiered_ignored_without_low_modes(self):
        config = make_config(
            skip_low_modes=True, solve_cross_terms=SolveCrossTerms.TIERED
        )
        assert config.admits_solve_pair("ama", "ama") is True


class TestContractionBijection:
    @pytest.mark.parametrize(
        "overrides,expected_dsets",
        [
            ({}, {"ranLL", "ama"}),
            (
                {"solve_cross_terms": SolveCrossTerms.ALL},
                {"ranLL", "ama", "ranLL_ama", "ama_ranLL"},
            ),
            ({"solve_cross_terms": SolveCrossTerms.TIERED}, {"ranLL", "ranLL_ama"}),
            (
                {"skip_low_modes": True, "solve_cross_terms": SolveCrossTerms.TIERED},
                {"ama"},
            ),
        ],
    )
    def test_emitted_dsets_match_label_list(self, overrides, expected_dsets):
        config = make_config(**overrides)
        emitted = {con.solver_label for _, con in contraction_gen(config)}
        assert emitted == expected_dsets
        assert emitted == set(config.get_solver_labels())

    def test_tiered_orientation_quark_carries_low_mode(self):
        config = make_config(solve_cross_terms=SolveCrossTerms.TIERED)
        crosses = [
            con
            for _, con in contraction_gen(config)
            if con.quark.solver != con.antiquark.solver
        ]
        assert {con.solver_label for con in crosses} == {"ranLL_ama"}
        for con in crosses:
            assert con.quark.solver == "ranLL"
            assert con.antiquark.solver == "ama"

    def test_mass_toggle_emits_cross_mass_contractions(self):
        config = make_two_mass_config(mass_cross_terms=True)
        emitted = {con.mass_label(config.mass) for _, con in contraction_gen(config)}
        assert "002426_m001524" in emitted
        assert "002426" in emitted

    def test_no_mass_cross_by_default(self):
        config = make_two_mass_config()
        emitted = {con.mass_label(config.mass) for _, con in contraction_gen(config)}
        assert emitted == {"002426", "001524"}


class TestLegacyTranslation:
    @pytest.mark.parametrize(
        "legacy,mass_cross,solve_cross",
        [
            ("none", False, SolveCrossTerms.DIAGONAL),
            ("mass", True, SolveCrossTerms.DIAGONAL),
            ("solve", False, SolveCrossTerms.ALL),
            ("all", True, SolveCrossTerms.ALL),
            ("SOLVE", False, SolveCrossTerms.ALL),
            (2, False, SolveCrossTerms.ALL),
            (3, True, SolveCrossTerms.ALL),
        ],
    )
    def test_translation_table(self, legacy, mass_cross, solve_cross):
        routed = normalize_params({"_preprocessor": {"cross_terms": legacy}})
        assert routed["_preprocessor"]["mass_cross_terms"] is mass_cross
        assert routed["_preprocessor"]["solve_cross_terms"] is solve_cross

    def test_canonical_keys_win(self):
        routed = normalize_params(
            {
                "_preprocessor": {
                    "cross_terms": "solve",
                    "solve_cross_terms": "tiered",
                    "mass_cross_terms": True,
                }
            }
        )
        slice_ = routed["_preprocessor"]
        assert slice_["solve_cross_terms"] == "tiered"
        assert slice_["mass_cross_terms"] is True

    def test_unknown_value_raises(self):
        with pytest.raises(ValueError, match="cross_terms"):
            normalize_params({"_preprocessor": {"cross_terms": "bogus"}})

    def test_legacy_key_routes_to_fields_not_operations(self):
        params = normalize_params(
            {
                "mass": {"l": 0.01},
                "_preprocessor": {
                    "cross_terms": "mass",
                    "pion_local": {"mass": ["l"]},
                },
            }
        )
        routed = route_params(params)
        assert routed["operations"] == {"pion_local": {"mass": ["l"]}}
        assert routed["mass_cross_terms"] is True
        assert "cross_terms" not in routed

    def test_no_legacy_key_is_noop(self):
        params = {"mass": {"l": 0.01}, "_preprocessor": {"pion_local": {"mass": ["l"]}}}
        assert normalize_params(params) is params


class TestScheduleRanking:
    def test_tiered_ranks_ranll_before_ama_before_cross(self):
        config = make_config(solve_cross_terms=SolveCrossTerms.TIERED)
        modules = [
            "corr_ranLL_ama_pion_local_mass_l_t0",
            "quark_ama_pion_local_mass_l_t0",
            "quark_ranLL_pion_local_mass_l_t0",
        ]
        assert sort_schedule(config, modules) == [
            "quark_ranLL_pion_local_mass_l_t0",
            "quark_ama_pion_local_mass_l_t0",
            "corr_ranLL_ama_pion_local_mass_l_t0",
        ]

    def test_diagonal_ranking_unchanged(self):
        config = make_config()  # DIAGONAL default
        modules = [
            "quark_ama_pion_local_mass_l_t0",
            "quark_ranLL_pion_local_mass_l_t0",
        ]
        assert sort_schedule(config, modules) == [
            "quark_ranLL_pion_local_mass_l_t0",
            "quark_ama_pion_local_mass_l_t0",
        ]

    def test_all_crosses_rank_after_base_modules(self):
        config = make_config(solve_cross_terms=SolveCrossTerms.ALL)
        modules = [
            "corr_ama_ranLL_pion_local_mass_l_t0",
            "corr_ranLL_ama_pion_local_mass_l_t0",
            "quark_ama_pion_local_mass_l_t0",
            "quark_ranLL_pion_local_mass_l_t0",
        ]
        ordered = sort_schedule(config, modules)
        assert ordered.index("quark_ranLL_pion_local_mass_l_t0") == 0
        assert ordered.index("quark_ama_pion_local_mass_l_t0") == 1
        # crosses last, in cross-label order (ranLL_ama before ama_ranLL)
        assert ordered[-2:] == [
            "corr_ranLL_ama_pion_local_mass_l_t0",
            "corr_ama_ranLL_pion_local_mass_l_t0",
        ]

    def test_tiered_build_orders_precon_chain(self):
        config = make_config(
            solve_cross_terms=SolveCrossTerms.TIERED, overwrite=True
        )
        schedule = build_input_params(config).schedule
        assert "quark_ranLL_pion_local_mass_l_t0" in schedule
        assert "quark_ama_pion_local_mass_l_t0" in schedule
        assert "corr_ranLL_ama_pion_local_mass_l_t0" in schedule
        assert not any(n.startswith("corr_ama_") for n in schedule)
        assert (
            schedule.index("quark_ranLL_pion_local_mass_l_t0")
            < schedule.index("quark_ama_pion_local_mass_l_t0")
            < schedule.index("corr_ranLL_ama_pion_local_mass_l_t0")
        )


class TestGridLmaValidation:
    def test_grid_lma_rejects_tiered(self, grid_params):
        grid_params["job_setup"]["lma"]["tasks"]["high_modes"][
            "solve_cross_terms"
        ] = "tiered"
        with pytest.raises(ValueError, match="grid_lma"):
            create_task("lma", grid_params, "a", "20")

    def test_grid_lma_rejects_legacy_solve_via_translation(self, grid_params):
        grid_params["job_setup"]["lma"]["tasks"]["high_modes"]["cross_terms"] = "solve"
        with pytest.raises(ValueError, match="grid_lma"):
            create_task("lma", grid_params, "a", "20")

    def test_grid_lma_accepts_diagonal_with_mass_cross(self, grid_params):
        grid_params["job_setup"]["lma"]["tasks"]["high_modes"][
            "mass_cross_terms"
        ] = True
        task = create_task("lma", grid_params, "a", "20")
        assert task.config.high_modes_config.mass_cross_terms is True


class TestAggregatorAxes:
    @staticmethod
    def _axis(params, key):
        return {
            v["load_files"]["replacements"][key]
            for v in params.values()
            if isinstance(v, dict) and "load_files" in v
        }

    def test_tiered_run_list_uses_tiered_dsets(self):
        config = make_config(solve_cross_terms=SolveCrossTerms.TIERED)
        params = build_aggregator_params(config, average=False)
        assert self._axis(params, "dset") == {"ranLL", "ranLL_ama"}

    def test_diagonal_run_list_unchanged(self):
        config = make_config()
        params = build_aggregator_params(config, average=False)
        assert self._axis(params, "dset") == {"ranLL", "ama"}

    def test_all_run_list_includes_both_orientations(self):
        config = make_config(solve_cross_terms=SolveCrossTerms.ALL)
        params = build_aggregator_params(config, average=False)
        assert self._axis(params, "dset") == {"ranLL", "ama", "ranLL_ama", "ama_ranLL"}

    def test_mass_cross_dsets_join_run_list(self):
        config = make_two_mass_config(mass_cross_terms=True)
        params = build_aggregator_params(config, average=False)
        assert self._axis(params, "mass") == {"002426", "001524", "002426_m001524"}

    def test_mass_axis_matches_catalog_axis(self):
        # catalog_files filters replacement keys to filestem placeholders
        # (utils/io.py), so the catalog needs {mass} in the filestem to
        # expose its mass axis here.
        config = make_two_mass_config(
            mass_cross_terms=True,
            high_modes=Outfile(
                filestem="corr/corr_{tsource}_{mass}_{dset}",
                ext=".20.h5",
                good_size=1,
            ),
        )
        params = build_aggregator_params(config, average=False)
        catalog = create_outfile_catalog(config)
        assert self._axis(params, "mass") == set(catalog["mass"].unique())

    def test_run_keys_and_entries_pair_up(self):
        config = make_config(solve_cross_terms=SolveCrossTerms.TIERED)
        params = build_aggregator_params(config, average=False)
        for key in params["run"]:
            assert key in params
        assert len(params["run"]) == len({k for k in params if k != "run"})
