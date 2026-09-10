"""Unit tests for the split cross-term controls on HighModeConfig.

Covers the label contract (get_solver_labels / get_mass_labels per solve
mode), the shared pair-membership helper, the contraction-emission
bijection, and the legacy `cross_terms` translation.
"""

import logging

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
    validate_config,
)
from pyfm.tasks.hadrons.highmode.twopoint import contraction_gen, quark_gen
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
    kwargs = dict(
        mass=MassDict.from_dict({"l": 0.002426, "u": 0.001524}),
        operations=OpList([OpList.Op(gamma=Gamma.PION_LOCAL, mass=("l", "u"))]),
    )
    kwargs.update(overrides)
    return make_config(**kwargs)


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

    def test_cross_labels_key_canonical_regardless_of_listing_order(self):
        ascending = make_two_mass_config(mass_cross_terms=True)  # mass=("l","u")
        descending = make_two_mass_config(
            mass_cross_terms=True,
            operations=OpList([OpList.Op(gamma=Gamma.PION_LOCAL, mass=("u", "l"))]),
        )
        # Diagonals follow listing order; the cross label is key-canonical
        # (raw key "l" < "u" -> l's value string first) in both cases.
        assert ascending.get_mass_labels(ascending.op_list[0]) == [
            "002426", "001524", "002426_m001524",
        ]
        assert descending.get_mass_labels(descending.op_list[0]) == [
            "001524", "002426", "002426_m001524",
        ]


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


class TestEffectiveSolveCrossTerms:
    def test_passthrough_when_both_solver_classes_present(self):
        config = make_config(solve_cross_terms=SolveCrossTerms.TIERED)
        assert config.effective_solve_cross_terms is SolveCrossTerms.TIERED

    @pytest.mark.parametrize(
        "flags",
        [{"skip_low_modes": True}, {"skip_cg": True}],
    )
    def test_skip_flags_collapse_to_diagonal(self, flags):
        config = make_config(solve_cross_terms=SolveCrossTerms.ALL, **flags)
        assert config.effective_solve_cross_terms is SolveCrossTerms.DIAGONAL


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

    @pytest.mark.parametrize("mass_order", [("l", "u"), ("u", "l")])
    def test_mass_label_bijection_holds_for_any_listing_order(self, mass_order):
        # I1 regression: the catalog/aggregation axis (get_mass_labels) and
        # the emitted filenames (TwoPointOp.mass_label via contraction_gen)
        # must be the same set regardless of op.mass listing order.
        config = make_config(
            mass=MassDict.from_dict({"l": 0.002426, "u": 0.001524}),
            operations=OpList([OpList.Op(gamma=Gamma.PION_LOCAL, mass=mass_order)]),
            mass_cross_terms=True,
        )
        op = config.op_list[0]
        emitted = {con.mass_label(config.mass) for _, con in contraction_gen(config)}
        assert emitted == set(config.get_mass_labels(op))


class TestDemandDrivenPropagators:
    @staticmethod
    def _props(config):
        return {(op.solver, op.gamma, op.mass) for op in quark_gen(config)}

    @staticmethod
    def _precons(config):
        return {
            (op.solver, op.gamma, op.mass): op.precon for op in quark_gen(config)
        }

    def test_tiered_skips_op_gamma_cg_solves(self):
        config = make_config(
            solve_cross_terms=SolveCrossTerms.TIERED,
            operations=OpList.from_dict({"vec_local": {"mass": ["l"]}}),
        )
        assert self._props(config) == {
            ("ranLL", Gamma.VEC_LOCAL, "l"),
            ("ranLL", Gamma.PION_LOCAL, "l"),
            ("ama", Gamma.PION_LOCAL, "l"),
        }

    def test_tiered_preserves_precon_chain(self):
        config = make_config(
            solve_cross_terms=SolveCrossTerms.TIERED,
            operations=OpList.from_dict({"vec_local": {"mass": ["l"]}}),
        )
        precons = self._precons(config)
        assert precons[("ama", Gamma.PION_LOCAL, "l")] == "ranLL"
        assert precons[("ranLL", Gamma.VEC_LOCAL, "l")] is None
        assert precons[("ranLL", Gamma.PION_LOCAL, "l")] is None

    def test_tiered_multi_residual_chains_remaining_solvers(self):
        config = make_config(
            solve_cross_terms=SolveCrossTerms.TIERED,
            residual=[1e-6, 1e-8],
            operations=OpList.from_dict({"vec_local": {"mass": ["l"]}}),
        )
        precons = self._precons(config)
        assert precons[("ama_1e-06", Gamma.PION_LOCAL, "l")] == "ranLL"
        assert precons[("ama_1e-08", Gamma.PION_LOCAL, "l")] == "ama_1e-06"
        assert ("ama_1e-06", Gamma.VEC_LOCAL, "l") not in precons
        assert ("ama_1e-08", Gamma.VEC_LOCAL, "l") not in precons

    @pytest.mark.parametrize(
        "overrides",
        [{}, {"solve_cross_terms": SolveCrossTerms.ALL}],
    )
    def test_diagonal_and_all_emit_every_base_propagator(self, overrides):
        # Regression pair: modes that admit HH keep the full set — identical
        # to the previous label-list-driven generator.
        config = make_config(
            operations=OpList.from_dict({"vec_local": {"mass": ["l"]}}),
            **overrides,
        )
        assert self._props(config) == {
            ("ranLL", Gamma.VEC_LOCAL, "l"),
            ("ranLL", Gamma.PION_LOCAL, "l"),
            ("ama", Gamma.VEC_LOCAL, "l"),
            ("ama", Gamma.PION_LOCAL, "l"),
        }

    def test_axial_ops_share_the_nonaxial_solve(self):
        config = make_config(
            solve_cross_terms=SolveCrossTerms.TIERED,
            operations=OpList.from_dict(
                {"vec_local": {"mass": ["l"]}, "axial_vec_local": {"mass": ["l"]}}
            ),
        )
        props = self._props(config)
        assert ("ranLL", Gamma.VEC_LOCAL, "l") in props  # shared by vec + axial ops
        assert ("ranLL", Gamma.IDENTITY, "l") in props
        assert ("ama", Gamma.IDENTITY, "l") in props
        assert ("ama", Gamma.VEC_LOCAL, "l") not in props

    def test_every_precon_names_an_emitted_module(self):
        config = make_config(
            solve_cross_terms=SolveCrossTerms.TIERED,
            residual=[1e-6, 1e-8],
            operations=OpList.from_dict({"vec_local": {"mass": ["l"]}}),
            overwrite=True,
        )
        result = build_input_params(config)
        for ref in config.source_refs:
            for op in quark_gen(config):
                glabel = op.gamma.name.lower()
                name = f"quark_{op.solver}_{glabel}_mass_{op.mass}_{ref.label}"
                assert name in result.modules
                if op.precon is not None:
                    guess = f"quark_{op.precon}_{glabel}_mass_{op.mass}_{ref.label}"
                    assert guess in result.modules, f"dangling precon {guess}"


class TestLegacyTranslation:
    @pytest.mark.parametrize(
        "legacy,mass_cross,solve_cross",
        [
            ("none", False, SolveCrossTerms.DIAGONAL),
            ("mass", True, SolveCrossTerms.DIAGONAL),
            ("solve", False, SolveCrossTerms.ALL),
            ("all", True, SolveCrossTerms.ALL),
            ("SOLVE", False, SolveCrossTerms.ALL),
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

    @pytest.mark.parametrize("legacy", [0, 1, 2, 3, "0", "3"])
    def test_numeric_values_rejected(self, legacy):
        # Numeric values were never accepted by the legacy enum
        # (SerializableEnum.from_dict matches member names); dropping the
        # aliases restores that historical strictness.
        with pytest.raises(ValueError, match="cross_terms"):
            normalize_params({"_preprocessor": {"cross_terms": legacy}})

    def test_root_level_legacy_key_translates(self):
        routed = normalize_params({"cross_terms": "mass", "_preprocessor": {}})
        assert routed["mass_cross_terms"] is True
        assert routed["solve_cross_terms"] is SolveCrossTerms.DIAGONAL
        assert "cross_terms" not in routed
        assert routed["_preprocessor"] == {}

    def test_translation_does_not_mutate_input(self):
        params = {
            "_preprocessor": {"cross_terms": "solve", "pion_local": {"mass": ["l"]}}
        }
        snapshot = {"_preprocessor": dict(params["_preprocessor"])}
        routed = normalize_params(params)
        assert params == snapshot  # caller's dict untouched
        assert routed["_preprocessor"]["solve_cross_terms"] is SolveCrossTerms.ALL
        assert "cross_terms" not in routed["_preprocessor"]

    def test_noop_returns_identity_without_preprocessor(self):
        bare = {"mass": {"l": 0.01}}
        assert normalize_params(bare) is bare

    def test_translation_logged_at_debug(self, caplog):
        with caplog.at_level("DEBUG"):
            normalize_params({"_preprocessor": {"cross_terms": "mass"}})
        assert any(
            "cross_terms" in r.message and "mass_cross_terms" in r.message
            for r in caplog.records
        )


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


class TestDowngradeWarning:
    def test_downgrade_warns_once_naming_modes_and_trigger(self, caplog):
        config = make_config(
            skip_low_modes=True, solve_cross_terms=SolveCrossTerms.TIERED
        )
        with caplog.at_level(logging.WARNING):
            validate_config(config)
        downgrade_records = [r for r in caplog.records if "downgraded" in r.message]
        assert len(downgrade_records) == 1
        assert "TIERED" in downgrade_records[0].message
        assert "DIAGONAL" in downgrade_records[0].message
        assert "skip_low_modes" in downgrade_records[0].message

    def test_both_flags_named_when_both_set(self, caplog):
        config = make_config(
            skip_low_modes=True, skip_cg=True, solve_cross_terms=SolveCrossTerms.ALL
        )
        with caplog.at_level(logging.WARNING):
            validate_config(config)
        message = next(r.message for r in caplog.records if "downgraded" in r.message)
        assert "skip_low_modes" in message and "skip_cg" in message

    def test_no_warning_without_downgrade(self, caplog):
        config = make_config(solve_cross_terms=SolveCrossTerms.TIERED)
        with caplog.at_level(logging.WARNING):
            validate_config(config)
        assert not any("downgraded" in r.message for r in caplog.records)


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

    def test_grid_lma_accepts_tiered_collapsed_to_diagonal(self, grid_params):
        # TIERED + no epack => skip_low_modes=True => effective DIAGONAL:
        # a valid Grid workload (previously over-rejected by the raw check).
        tasks = grid_params["job_setup"]["lma"]["tasks"]
        tasks["high_modes"]["solve_cross_terms"] = "tiered"
        tasks.pop("epack", None)
        tasks.pop("meson", None)  # meson requires epack (lmi validator)
        task = create_task("lma", grid_params, "a", "20")  # must not raise
        hm = task.config.high_modes_config
        assert hm.solve_cross_terms == SolveCrossTerms.TIERED
        assert hm.effective_solve_cross_terms is SolveCrossTerms.DIAGONAL
        # The epack pop also flips skip_epack; the default-built epack must
        # pass validation — pin the flag so the dependency is explicit.
        assert task.config.skip_epack is True
        # End-to-end: the accepted (effective-DIAGONAL) config must build a
        # Grid workload — exercising solver_map resolution without KeyError.
        task.handler.build_input_params(task.config)

    def test_grid_lma_rejects_multi_residual(self, grid_params):
        grid_params["job_setup"]["lma"]["tasks"]["high_modes"]["residual"] = [
            1e-6,
            1e-8,
        ]
        with pytest.raises(ValueError, match="one high-mode residual"):
            create_task("lma", grid_params, "a", "20")


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

    def test_tiered_average_run_list_and_actions(self):
        config = make_config(solve_cross_terms=SolveCrossTerms.TIERED)
        params = build_aggregator_params(config, average=True)
        first = params[params["run"][0]]
        assert first["actions"]["average"] == ["tsource"]
        assert first["actions"]["real"] is True
        assert first["actions"]["index"] == ["series_cfg", "gamma", "t"]
        assert self._axis(params, "dset") == {"ranLL", "ranLL_ama"}

    def test_mass_cross_average_aggregates_cross_mass(self):
        config = make_two_mass_config(mass_cross_terms=True)
        params = build_aggregator_params(config, average=True)
        assert self._axis(params, "mass") == {"002426", "001524", "002426_m001524"}
        for key in params["run"]:
            assert params[key]["actions"]["average"] == ["tsource"]

    def test_average_outfile_carries_avg_suffix(self):
        # The _avg suffix lands via get_processed_filename, which only
        # rewrites stems containing "correlators" (production shape — see
        # example/params_files); the sibling catalog-axis test overrides
        # high_modes the same way.
        config = make_config(
            solve_cross_terms=SolveCrossTerms.TIERED,
            high_modes=Outfile(
                filestem=(
                    "dt{dt}/correlators/m{mass}/{gamma_label}/{dset}/"
                    "corr_{dset}_m{mass}_t{tsource}_{series}"
                ),
                ext=".20.h5",
                good_size=1,
            ),
        )
        params = build_aggregator_params(config, average=True)
        first = params[params["run"][0]]
        assert "_avg" in first["out_files"]["filestem"]
