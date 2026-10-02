"""Tests for the chain_cg_solves toggle on HighModeConfig.

Field-default and task-slice routing coverage; the guess-wiring behavior
tests (emission, TIERED demand-driven cases, schedule order) live with the
generator rewiring.
"""

from pyfm.domain import MassDict, OpList, Outfile
from pyfm.tasks.hadrons.highmode.strategy import build_input_params, route_params
from pyfm.tasks.hadrons.highmode.twopoint import build_quarks, quark_gen
from pyfm.tasks.hadrons.types import HighModeConfig, SolveCrossTerms, SourceRef


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
        tstop=0,
        dt=1,
        noise=1,
        time=4,
        shift_gauge_name="shift_gauge",
    )
    kwargs.update(overrides)
    return HighModeConfig(**kwargs)


class TestChainDefault:
    def test_chain_cg_solves_defaults_to_true(self):
        assert make_config().chain_cg_solves is True


class TestRouting:
    def test_chain_cg_solves_routes_to_field_not_operations(self):
        routed = route_params(
            {
                "mass": {"l": 0.01},
                "_preprocessor": {
                    "chain_cg_solves": False,
                    "pion_local": {"mass": ["l"]},
                },
            }
        )
        assert routed["chain_cg_solves"] is False
        assert "chain_cg_solves" not in routed["operations"]


RUN_REF = SourceRef(label="t0", axis="0", t0=0)


class TestQuarkEmission:
    @staticmethod
    def modules(config):
        return build_quarks(config, [RUN_REF]).modules

    def test_chained_multi_residual_guesses_chain_through_modules(self):
        modules = self.modules(make_config(residual=[1e-6, 1e-8]))
        assert modules["quark_ranLL_pion_local_mass_l_t0"]["options"]["guess"] == ""
        assert (
            modules["quark_ama_1e-06_pion_local_mass_l_t0"]["options"]["guess"]
            == "quark_ranLL_pion_local_mass_l_t0"
        )
        assert (
            modules["quark_ama_1e-08_pion_local_mass_l_t0"]["options"]["guess"]
            == "quark_ama_1e-06_pion_local_mass_l_t0"
        )

    def test_independent_multi_residual_guesses_ranll_through_modules(self):
        modules = self.modules(
            make_config(residual=[1e-6, 1e-8], chain_cg_solves=False)
        )
        for solver in ("ama_1e-06", "ama_1e-08"):
            assert (
                modules[f"quark_{solver}_pion_local_mass_l_t0"]["options"]["guess"]
                == "quark_ranLL_pion_local_mass_l_t0"
            )

    def test_independent_without_low_modes_leaves_guess_empty(self):
        modules = self.modules(
            make_config(
                residual=[1e-6, 1e-8], skip_low_modes=True, chain_cg_solves=False
            )
        )
        assert modules["quark_ama_1e-06_pion_local_mass_l_t0"]["options"]["guess"] == ""
        assert modules["quark_ama_1e-08_pion_local_mass_l_t0"]["options"]["guess"] == ""

    def test_tiered_multi_residual_chains_remaining_solvers(self):
        # TIERED emits only the contract-gamma CG solves (demand-driven);
        # among those, the chain still runs ranLL -> ama_1e-06 -> ama_1e-08.
        modules = self.modules(
            make_config(residual=[1e-6, 1e-8], solve_cross_terms=SolveCrossTerms.TIERED)
        )
        assert not any(n.startswith("quark_ama_") for n in modules if "_vec_" in n)
        assert (
            modules["quark_ama_1e-08_pion_local_mass_l_t0"]["options"]["guess"]
            == "quark_ama_1e-06_pion_local_mass_l_t0"
        )

    def test_tiered_independent_guesses_ranll_for_every_cg_solve(self):
        modules = self.modules(
            make_config(
                residual=[1e-6, 1e-8],
                solve_cross_terms=SolveCrossTerms.TIERED,
                chain_cg_solves=False,
            )
        )
        for solver in ("ama_1e-06", "ama_1e-08"):
            assert (
                modules[f"quark_{solver}_pion_local_mass_l_t0"]["options"]["guess"]
                == "quark_ranLL_pion_local_mass_l_t0"
            )


class TestSolverSet:
    def test_cross_labels_never_become_solvers(self):
        # Successor of the Step-8 cross-label hardening: at the generator
        # level, only base solver labels ever appear as Op.solver.
        config = make_config(
            residual=[1e-6, 1e-8], solve_cross_terms=SolveCrossTerms.ALL
        )
        assert {op.solver for op in quark_gen(config)} == {
            "ranLL",
            "ama_1e-06",
            "ama_1e-08",
        }


class TestScheduleOrder:
    def test_ranll_precedes_ama_solves_in_both_modes(self):
        for chain in (True, False):
            config = make_config(
                residual=[1e-6, 1e-8], overwrite=True, chain_cg_solves=chain
            )
            schedule = build_input_params(config).schedule
            assert (
                schedule.index("quark_ranLL_pion_local_mass_l_t0")
                < schedule.index("quark_ama_1e-06_pion_local_mass_l_t0")
                < schedule.index("quark_ama_1e-08_pion_local_mass_l_t0")
            )
