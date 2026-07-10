"""Unit tests for performance benchmark parsing and scoring helpers."""

from types import SimpleNamespace
from unittest.mock import patch

from pyfm.performance import (
    ModuleObservation,
    benchmark_lmi_performance,
    build_component_score,
    classify_benchmark_component,
    count_planned_components,
    extract_grid_benchmark_observations,
    derive_node_count,
    extract_module_step_observations,
)


def test_derive_node_count_requires_valid_world_and_node_sizes():
    assert derive_node_count({"world_size": 160, "node_size": 8}) == 20
    assert derive_node_count({"world_size": None, "node_size": 8}) is None
    assert derive_node_count({"world_size": 160, "node_size": None}) is None
    assert derive_node_count({"world_size": 160, "node_size": 0}) is None
    assert derive_node_count({"world_size": 161, "node_size": 8}) is None


def test_classify_benchmark_component_maps_v1_lmi_modules():
    assert classify_benchmark_component("epack") == "epack"
    assert classify_benchmark_component("quark_ranLL_pion_local_mass_l_t0") == "ranLL"
    assert classify_benchmark_component("quark_lma_m000569_t0") == "ranLL"
    assert classify_benchmark_component("quark_ama_pion_local_mass_l_t0") == "ama"
    assert classify_benchmark_component("mf_pion_local_mass_l") == "meson_field_local"
    assert classify_benchmark_component("mf_vec_onelink_mass_l") == "meson_field_onelink"
    assert classify_benchmark_component("gauge") is None


class PlannedHadronsInput:
    schedule = [
        "epack",
        "quark_lma_m000569_t0",
        "quark_ama_pion_local_mass_l_t0",
        "mf_pion_local_mass_l",
        "mf_vec_onelink_mass_l",
    ]


def test_count_planned_components_preserves_hadrons_schedule_counting():
    counts = count_planned_components("nanny_hadrons_lmi", PlannedHadronsInput())

    assert counts == {
        "epack": 1,
        "ranLL": 1,
        "ama": 1,
        "meson_field_local": 1,
        "meson_field_onelink": 1,
    }


def test_count_planned_components_counts_grid_dict_sections():
    planned_input = {
        "epack": {"operation": "load"},
        "corr": {
            "elem": [
                {"quarkSolver": "lma", "antiquarkSolver": "lma"},
                {"quarkSolver": "mpcg", "antiquarkSolver": "mpcg"},
                {"quarkSolver": "lma", "antiquarkSolver": "mpcg"},
            ]
        },
        "a2a": {
            "elem": [
                {"spinTaste": {"gammas": "G5_G5 GX_GX GY_GY GZ_GZ"}},
                {"spinTaste": {"gammas": "GX_G1 GY_G1 GZ_G1"}},
                {"spinTaste": {"gammas": "G5_G5 GX_G1"}},
            ]
        },
    }

    counts = count_planned_components("nanny_grid", planned_input)

    assert counts == {
        "epack": 1,
        "ranLL": 3,
        "ama": 3,
        "meson_field_local": 2,
        "meson_field_onelink": 2,
    }


def test_build_component_score_uses_null_scores_for_not_started_components():
    score = build_component_score(
        planned_count=4,
        observed_modules=[],
        node_count=20,
    )

    assert score["planned_count"] == 4
    assert score["observed_count"] == 0
    assert score["progress"] == 0.0
    assert score["elapsed_seconds"] is None
    assert score["observed_node_seconds"] is None
    assert score["normalized_node_seconds"] is None


def test_build_component_score_normalizes_node_seconds_by_progress():
    score = build_component_score(
        planned_count=4,
        observed_modules=[
            ModuleObservation("quark_ama_pion_local_mass_l_t0", elapsed_seconds=5.0),
            ModuleObservation("quark_ama_pion_local_mass_l_t1", elapsed_seconds=7.0),
        ],
        node_count=10,
    )

    assert score["observed_count"] == 2
    assert score["progress"] == 0.5
    assert score["elapsed_seconds"] == 12.0
    assert score["observed_node_seconds"] == 120.0
    assert score["normalized_node_seconds"] == 240.0


def test_extract_module_step_observations_preserves_order_and_in_progress_tail(tmp_path):
    log_file = tmp_path / "hadrons.out"
    log_file.write_text(
        "\n".join(
            [
                "Hadrons : Message  : 1.000000 s : -------- Measurement step 1/3 (module 'epack') --------",
                "Hadrons : Message  : 4.000000 s : -------- Measurement step 2/3 (module 'quark_ama_pion_local_mass_l_t0') --------",
                "Hadrons : Message  : 9.000000 s : -------- Measurement step 3/3 (module 'mf_pion_local_mass_l') --------",
            ]
        )
    )

    observations, is_incomplete = extract_module_step_observations(str(log_file))

    assert is_incomplete is True
    assert [o.module_name for o in observations] == [
        "epack",
        "quark_ama_pion_local_mass_l_t0",
        "mf_pion_local_mass_l",
    ]
    assert observations[0].elapsed_seconds == 3.0
    assert observations[1].elapsed_seconds == 5.0
    assert observations[2].elapsed_seconds is None


def test_extract_grid_benchmark_observations_parses_solve_and_correlators(tmp_path):
    log_file = tmp_path / "grid-solve.out"
    log_file.write_text(
        "\n".join(
            [
                "Grid : Message : 9.247028 s : MODULE: MSolver::StagFermionIRL",
                "Grid : Message : 9.546174 s : Running IRL eigensolver...",
                "Grid : Message : 1538.546117 s : Converged 2030 eigenvectors",
                "Grid : Message : 1539.171070 s : Setting up meson contraction",
                "Grid : Message : 1539.171510 s : Correlator: quarkAction='l' (lma), antiquarkAction='l' (lma)",
                "Grid : Message : 1546.700793 s : Saving correlator to /tmp/ranLL/corr",
                "Grid : Message : 1546.859268 s : Setting up meson contraction",
                "Grid : Message : 1546.859297 s : Correlator: quarkAction='l' (mpcg), antiquarkAction='l' (mpcg)",
                "Grid : Message : 1558.811880 s : Saving correlator to /tmp/ama/corr",
                "Grid : Message : 1608.548000 s : ******* Grid Finalize                ******",
            ]
        )
    )

    observations, is_incomplete, epack_stats = extract_grid_benchmark_observations(
        str(log_file)
    )

    assert is_incomplete is False
    assert epack_stats["nconv"] == 2030
    assert [o.module_name for o in observations] == [
        "epack",
        "quark_lma_grid_corr",
        "quark_ama_grid_corr",
    ]
    assert observations[0].start_time == 9.546174
    assert observations[0].end_time == 1538.546117
    assert observations[1].elapsed_seconds == 1546.700793 - 1539.171070
    assert observations[2].elapsed_seconds == 1558.811880 - 1546.859268


def test_extract_grid_benchmark_observations_parses_epack_load_progress_and_a2a(
    tmp_path,
):
    log_file = tmp_path / "grid-load-a2a.out"
    log_file.write_text(
        "\n".join(
            [
                "Grid : Message : 1.827785 s : Loading eigenpack from /tmp/eig",
                "Grid : Message : 2.460900 s : Reading eigenvector 0",
                "Grid : Message : 2.786860 s : Reading eigenvector 1",
                "Grid : Message : 31.431866 s : Low mode projector setup complete",
                "Grid : Message : 272.424944 s : Computing all-to-all meson fields",
                "Grid : Message : 272.424959 s : Spin bilinears:",
                "Grid : Message : 272.424960 s :   G5_G5",
                "Grid : Message : 272.424962 s :   GX_GX",
                "Grid : Message : 272.424965 s : Meson field size: 48*2000*2000",
                "Grid : Message : 351.000000 s : All-to-all meson field construction complete (0)",
                "Grid : Message : 351.557453 s : Computing all-to-all meson fields",
                "Grid : Message : 351.557468 s : Spin bilinears:",
                "Grid : Message : 351.557469 s :   GX_G1",
                "Grid : Message : 351.557471 s :   GY_G1",
                "Grid : Message : 351.557473 s : Meson field size: 48*2000*2000",
                "Grid : Message : 402.000000 s : All-to-all meson field construction complete (1)",
                "Grid : Message : 403.000000 s : ******* Grid Finalize                ******",
            ]
        )
    )

    observations, is_incomplete, epack_stats = extract_grid_benchmark_observations(
        str(log_file)
    )

    assert is_incomplete is False
    assert epack_stats["eigenvectors_read"] == 2
    assert epack_stats["epack_load_observed"] is True
    assert [o.module_name for o in observations] == [
        "epack",
        "mf_grid_local",
        "mf_grid_onelink",
    ]
    assert observations[0].elapsed_seconds == 31.431866 - 1.827785


def test_extract_grid_benchmark_observations_reports_partial_epack_load(tmp_path):
    log_file = tmp_path / "grid-partial-load.out"
    log_file.write_text(
        "\n".join(
            [
                "Grid : Message : 1.827785 s : Loading eigenpack from /tmp/eig",
                "Grid : Message : 2.460900 s : Reading eigenvector 0",
                "Grid : Message : 2.786860 s : Reading eigenvector 1",
            ]
        )
    )

    observations, is_incomplete, epack_stats = extract_grid_benchmark_observations(
        str(log_file)
    )

    assert is_incomplete is True
    assert epack_stats["eigenvectors_read"] == 2
    assert epack_stats["epack_load_observed"] is True
    assert observations == [
        ModuleObservation("epack", start_time=1.827785, end_time=None, elapsed_seconds=None)
    ]


def test_extract_grid_benchmark_observations_counts_mixed_a2a_as_both(tmp_path):
    log_file = tmp_path / "grid-mixed-a2a.out"
    log_file.write_text(
        "\n".join(
            [
                "Grid : Message : 1.000000 s : Computing all-to-all meson fields",
                "Grid : Message : 1.100000 s : Spin bilinears:",
                "Grid : Message : 1.200000 s :   G5_G5",
                "Grid : Message : 1.300000 s :   GX_G1",
                "Grid : Message : 1.400000 s : Meson field size: 48*2000*2000",
                "Grid : Message : 5.000000 s : All-to-all meson field construction complete (0)",
                "Grid : Message : 6.000000 s : ******* Grid Finalize                ******",
            ]
        )
    )

    observations, _, _ = extract_grid_benchmark_observations(str(log_file))

    assert [o.module_name for o in observations] == [
        "mf_grid_local",
        "mf_grid_onelink",
    ]


def test_benchmark_lmi_performance_rejects_unsupported_task_key(tmp_path):
    log_file = tmp_path / "hadrons.out"
    log_file.write_text("dummy")
    task = SimpleNamespace(key="nanny_other")

    with patch("pyfm.nanny.taskbuilder.create_task", return_value=task):
        try:
            benchmark_lmi_performance("other", str(log_file), {})
        except ValueError as e:
            message = str(e)
        else:
            raise AssertionError("Expected unsupported task key to raise")

    assert "Hadrons LMI and Grid LMI" in message
    assert "nanny_other" in message


def test_benchmark_lmi_performance_uses_grid_parser_and_preserves_json_shape(
    tmp_path,
):
    log_file = tmp_path / "grid.out"
    log_file.write_text(
        "\n".join(
            [
                "srun grid_lma --grid 32.32.32.48",
                "SharedMemoryMpi:  World communicator of size 2",
                "SharedMemoryMpi:  Node  communicator of size 2",
                "Grid : Message : 1.000000 s : Loading eigenpack from /tmp/eig",
                "Grid : Message : 2.000000 s : Reading eigenvector 0",
                "Grid : Message : 3.000000 s : Low mode projector setup complete",
                "Grid : Message : 4.000000 s : Setting up meson contraction",
                "Grid : Message : 4.100000 s : Correlator: quarkAction='l' (lma), antiquarkAction='l' (lma)",
                "Grid : Message : 6.000000 s : Saving correlator to /tmp/ranLL/corr",
                "Grid : Message : 7.000000 s : ******* Grid Finalize                ******",
            ]
        )
    )
    planned_input = {
        "epack": {"operation": "load"},
        "corr": {"elem": [{"quarkSolver": "lma", "antiquarkSolver": "lma"}]},
    }
    config = SimpleNamespace(
        epack_config=SimpleNamespace(load=True, eigs=4, lanczos=None)
    )
    handler = SimpleNamespace(build_input_params=lambda config: planned_input)
    task = SimpleNamespace(key="nanny_grid", handler=handler, config=config)

    with patch("pyfm.nanny.taskbuilder.create_task", return_value=task):
        result = benchmark_lmi_performance("grid", str(log_file), {})

    assert result["schema_version"] == 1
    assert result["task_key"] == "nanny_grid"
    assert result["metadata"]["lattice_grid"] == "32.32.32.48"
    assert result["metadata"]["node_count"] == 1
    assert set(result["components"]) == {
        "epack",
        "ranLL",
        "ama",
        "meson_field_local",
        "meson_field_onelink",
    }
    assert result["components"]["epack"]["planned_count"] == 1
    assert result["components"]["epack"]["observed_count"] == 1
    assert result["components"]["epack"]["progress"] == 0.25
    assert result["components"]["epack"]["metadata"]["eigenvectors_read"] == 1
    assert result["components"]["ranLL"]["planned_count"] == 2
    assert result["components"]["ranLL"]["observed_count"] == 1
