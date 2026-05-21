"""Unit tests for performance benchmark parsing and scoring helpers."""

from pyfm.performance import (
    ModuleObservation,
    build_component_score,
    classify_benchmark_component,
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
