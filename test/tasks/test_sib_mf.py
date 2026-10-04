"""Tests for sib_mf.py — SIBMFConfig composite, own hooks, the batch
contract, the demand-driven resume gate, the flattened-stem block table,
and the split-noise per-world emission (with a shared-mode regression)."""

import copy

import pytest

from pyfm.domain import Gamma
from pyfm.nanny import write_input_file
from pyfm.nanny.taskbuilder import create_task
from pyfm.tasks.hadrons.sib_mf import (
    SIBMFConfig,
    _block_filepath,
    _needed_blocks,
    _pair_keys,
    _sib_outfile_catalog,
    build_aggregator_params,
    build_input_params,
    compare_outputs,
)
from pyfm.tasks.register import get_task_handler, get_task_key, list_registered_types


def _sib_tasks(params):
    return params["job_setup"]["sib_mf"]["tasks"]


def _write_file(path, size):
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"0" * int(size))


def _gated_params(params, tmp_path):
    """Shared-home, non-overwrite copy of the fixture for gate tests."""
    params = copy.deepcopy(params)
    params["shared_params"]["home"] = str(tmp_path)
    params["shared_params"]["overwrite"] = False
    return params


def _shared_params(params):
    """Fixture copy with the shared (split_noise: false) chain."""
    params = copy.deepcopy(params)
    _sib_tasks(params)["sib"]["output"]["split_noise"] = False
    return params


def _blocks(result):
    return {
        name: mod["options"]
        for name, mod in result.modules.items()
        if name.startswith(("mf_s_", "mf_vl_", "mf_vo_"))
    }


class TestRegistration:
    def test_hadrons_sib_mf_is_registered(self):
        assert "nanny_hadrons_sib_mf" in list_registered_types()

    def test_task_key_resolves_to_sib_mf_handler(self):
        handler = get_task_handler(job_type="hadrons", task_type="sib_mf")
        assert handler is not None
        assert handler.config_type is SIBMFConfig

    def test_sibling_registration_untouched(self):
        from pyfm.tasks.hadrons.lma_new import LMANewConfig

        assert get_task_key(config=LMANewConfig) == "nanny_hadrons_lma_new"
        assert get_task_key(config=SIBMFConfig) == "nanny_hadrons_sib_mf"

    def test_leaf_configs_registered(self):
        registered = list_registered_types()
        assert "nanny_hadrons_sib_mf_batch" in registered
        assert "nanny_hadrons_sib_mf_solver" in registered
        assert "nanny_hadrons_sib_mf_output" in registered


class TestBatchContract:
    """Negatives mirror HadronsMILC's sib-hvp-neg-* XMLs as YAML rejections,
    plus the construction-pinned contract the upstream refuses to check."""

    def test_batch_window_beyond_time_extent_rejected(self, hadrons_params):
        _sib_tasks(hadrons_params)["sib"]["batch"]["n_slices"] = 50  # tB=49 > time 4
        with pytest.raises(ValueError, match="time extent"):
            create_task("sib_mf", hadrons_params, "a", "20")

    @pytest.mark.parametrize("field", ["n_slices", "noise", "t_step"])
    def test_batch_nonpositive_rejected(self, hadrons_params, field):
        _sib_tasks(hadrons_params)["sib"]["batch"][field] = 0
        with pytest.raises(ValueError, match=field):
            create_task("sib_mf", hadrons_params, "a", "20")

    def test_solver_unknown_rejected(self, hadrons_params):
        _sib_tasks(hadrons_params)["sib"]["solver"]["solver"] = "bogus"
        with pytest.raises(ValueError, match="must be one of"):
            create_task("sib_mf", hadrons_params, "a", "20")

    def test_defl_mass_not_in_massdict_rejected(self, hadrons_params):
        _sib_tasks(hadrons_params)["sib"]["defl_mass"] = "x"
        with pytest.raises(ValueError, match="defl_mass"):
            create_task("sib_mf", hadrons_params, "a", "20")

    def test_operations_mass_not_in_massdict_rejected(self, hadrons_params):
        _sib_tasks(hadrons_params)["sib"]["operations"]["mass"] = ["x"]
        with pytest.raises(ValueError, match="mass.*not present"):
            create_task("sib_mf", hadrons_params, "a", "20")

    def test_non_sib_gamma_rejected(self, hadrons_params):
        _sib_tasks(hadrons_params)["sib"]["operations"]["gamma"] = ["pion_local"]
        with pytest.raises(ValueError, match="not SIB families"):
            create_task("sib_mf", hadrons_params, "a", "20")

    def test_op_without_mass_rejected(self, hadrons_params):
        _sib_tasks(hadrons_params)["sib"]["operations"]["mass"] = []
        with pytest.raises(ValueError, match="no mass"):
            create_task("sib_mf", hadrons_params, "a", "20")

    def test_unknown_sib_key_rejected(self, hadrons_params):
        _sib_tasks(hadrons_params)["sib"]["bogus"] = 1
        with pytest.raises(ValueError, match="Unknown sib section keys"):
            create_task("sib_mf", hadrons_params, "a", "20")

    def test_unknown_tasks_key_rejected(self, hadrons_params):
        _sib_tasks(hadrons_params)["bogus"] = {}
        with pytest.raises(ValueError, match="Unknown tasks keys"):
            create_task("sib_mf", hadrons_params, "a", "20")

    def test_precon_pins_batch_contract(self, hadrons_params):
        task = create_task("sib_mf", hadrons_params, "a", "20")
        result = build_input_params(task.config)
        p = result.modules["precon_n0"]["options"]
        assert (
            p["a2a_batch"],
            p["labels"],
            p["tA"],
            p["tB"],
            p["tStep"],
            p["noiseIndex"],
            p["nNoise"],
            p["noise"],
            p["mesonField"],
        ) == (
            "true",
            "G1_G1",
            "0",
            "3",
            "1",
            "0",
            "1",
            "noise_fv_n0_vec",
            "mfload_tab_n0",
        )

    def test_job_noise_flows_to_batch_and_stem(self, hadrons_params):
        task = create_task("sib_mf", hadrons_params, "a", "20")
        assert task.config.batch_config.noise == 2  # absorbed from job params
        assert task.config.output_config.split_noise is True
        result = build_input_params(task.config)
        assert result.modules["noise_fv_n0"]["options"]["nsrc"] == "1"
        assert result.modules["mf_s_ll_t0"]["options"]["output"].endswith(
            "e100n2/sib/ll_a"
        )


class TestModuleTopology:
    def test_schedule_dependency_order(self, hadrons_params):
        schedule = build_input_params(
            create_task("sib_mf", hadrons_params, "a", "20").config
        ).schedule
        assert (
            schedule.index("noise_fv_n0")
            < schedule.index("mf_tab_n0")
            < schedule.index("mfload_tab_n0")
            < schedule.index("precon_n0")
            < schedule.index("noise_t0_n0")
            < schedule.index("quark_h_n0_mass_l_t0")
            < schedule.index("mf_s_lh_n0_mass_l_t0")
        )

    def test_block_table_legs(self, hadrons_params):
        result = build_input_params(
            create_task("sib_mf", hadrons_params, "a", "20").config
        )
        blocks = _blocks(result)
        assert len(blocks) == 45
        assert blocks["mf_s_ll_t0"]["left"] == ""
        assert blocks["mf_s_ll_t0"]["right"] == ""
        assert blocks["mf_s_ll_t0"]["lowModes"] == "evecs_mass_l"
        assert blocks["mf_s_ll_t0"]["cbPairsLeft"] == "cbpairs_l_mass_l"
        assert blocks["mf_vo_np_n0_n1_t0"]["left"] == "noise_fv_n0_vec"
        assert blocks["mf_vo_np_n0_n1_t0"]["right"] == "precon_n1"
        assert blocks["mf_vo_np_n0_n1_t0"]["lowModes"] == ""
        assert "cbPairsLeft" not in blocks["mf_vo_np_n0_n1_t0"]
        assert blocks["mf_vl_lh_n0_mass_l_t0"]["right"] == "quark_h_n0_mass_l_t0"
        assert blocks["mf_vl_nh_n1_n0_mass_l_t0"]["left"] == "noise_fv_n1_vec"
        assert blocks["mf_vl_nh_n1_n0_mass_l_t0"]["right"] == "quark_h_n0_mass_l_t0"
        assert blocks["mf_vl_nh_n1_n0_mass_l_t0"]["lowModes"] == ""
        assert "cbPairsLeft" not in blocks["mf_vl_nh_n1_n0_mass_l_t0"]

    def test_stems_are_leg_pair_only(self, hadrons_params):
        result = build_input_params(
            create_task("sib_mf", hadrons_params, "a", "20").config
        )
        assert result.modules["mf_s_ll_t0"]["options"]["output"].endswith(
            "e100n2/sib/ll_a"
        )
        assert result.modules["mf_vl_ll_t0"]["options"]["output"].endswith(
            "e100n2/sib/ll_a"
        )
        assert (
            result.modules["mf_s_lh_n0_mass_l_t0"]["options"]["output"].endswith(
                "e100n2/sib/lh_ml_a_n0"
            )
        )
        assert result.modules["mf_tab_n0"]["options"]["output"].endswith(
            "e100n2/sib/tab_a_n0"
        )

    def test_onelink_binds_shift_gauge(self, hadrons_params):
        result = build_input_params(
            create_task("sib_mf", hadrons_params, "a", "20").config
        )
        assert result.modules["spintaste_vec_onelink"]["options"]["spinTaste"][
            "gauge"
        ] == "gauge_apbc"
        assert (
            result.modules["spintaste_vec_local"]["options"]["spinTaste"]["gauge"]
            == ""
        )


class TestResumeGate:
    def test_needed_blocks_axes(self, hadrons_params):
        task = create_task("sib_mf", hadrons_params, "a", "20")
        ref, h = _needed_blocks(
            task.config,
            {
                _block_filepath(task.config, "ll", "", "GX_GX"),
                _block_filepath(task.config, "nh", "_ml", "G1_G1", n_index=1, hp_index=0),
            },
        )
        assert ref == {Gamma.VEC_LOCAL: {("ll", None, None)}}
        assert h == {(Gamma.SCALAR_LOCAL, "nh", "l", 1, 0)}

    def test_all_outputs_complete_emits_infra_only(
        self, tmp_path, monkeypatch, hadrons_params
    ):
        monkeypatch.chdir(tmp_path)
        params = _gated_params(hadrons_params, tmp_path)
        task = create_task("sib_mf", params, "a", "20")
        for _, row in _sib_outfile_catalog(task.config).iterrows():
            _write_file(row["filepath"], row["good_size"])

        result = build_input_params(task.config)
        assert "precon_n0" not in result.modules
        assert "mf_tab_n0" not in result.modules
        assert not any(
            name.startswith(("mf_s_", "mf_vl_", "mf_vo_")) for name in result.modules
        )

    def test_missing_vec_local_ll_narrows_to_that_block(
        self, tmp_path, monkeypatch, hadrons_params
    ):
        monkeypatch.chdir(tmp_path)
        params = _gated_params(hadrons_params, tmp_path)
        task = create_task("sib_mf", params, "a", "20")
        catalog = _sib_outfile_catalog(task.config)
        for _, row in catalog.iterrows():
            if row["gamma"] in ("GX_GX", "GY_GY", "GZ_GZ") and row["leg_pair"] == "ll":
                continue
            _write_file(row["filepath"], row["good_size"])

        result = build_input_params(task.config)
        assert "mf_vl_ll_t0" in result.modules
        assert "mf_s_ll_t0" not in result.modules
        assert "mf_vo_ll_t0" not in result.modules
        # ll is low-side only: no world modules, no noise, no precon, no
        # tab, no h, no scalar SpinTaste — just the CB pairs and the
        # family SpinTaste.
        for absent in (
            "noise_fv_n0",
            "noise_t0_n0",
            "mf_tab_n0",
            "mfload_tab_n0",
            "precon_n0",
            "quark_h_n0_mass_l_t0",
            "sib_solver_mass_l",
            "spintaste_scalar_local",
        ):
            assert absent not in result.modules, absent
        assert "cbpairs_l_mass_l" in result.modules
        assert "spintaste_vec_local" in result.modules

    def test_overwrite_skips_the_gate(self, tmp_path, monkeypatch, hadrons_params):
        monkeypatch.chdir(tmp_path)
        # Fixture default: shared overwrite true -> full emission despite
        # (nonexistent) home files.
        task = create_task("sib_mf", hadrons_params, "a", "20")
        result = build_input_params(task.config)
        assert "precon_n0" in result.modules
        assert (
            sum(n.startswith(("mf_s_", "mf_vl_", "mf_vo_")) for n in result.schedule)
            == 45
        )


class TestCatalogAndCompare:
    def test_catalog_covers_outputs(self, hadrons_params):
        task = create_task("sib_mf", hadrons_params, "a", "20")
        df = _sib_outfile_catalog(task.config)
        nblocks = sum(len(op.gamma.gamma_list) * 15 for op in task.config.op_list)
        ntab = 2 * len(Gamma.SCALAR_LOCAL.gamma_list)
        assert len(df) == nblocks + ntab
        assert {"leg_pair", "mass", "gamma", "n_index", "hp_index"} <= set(df.columns)
        assert df["filepath"].str.endswith("e100n2/sib/ll_a.20/GX_GX_0_0_0.h5").any()
        assert df["filepath"].str.endswith(
            "e100n2/sib/np_a_n0_n1.20/GX_GX_0_0_0.h5"
        ).any()
        assert df["filepath"].str.endswith(
            "e100n2/sib/tab_a_n1.20/G1_G1_0_0_0.h5"
        ).any()

    def test_aggregator_empty(self, hadrons_params):
        task = create_task("sib_mf", hadrons_params, "a", "20")
        assert build_aggregator_params(task.config, True) == {}
        assert build_aggregator_params(task.config, False) == {}

    def test_compare_missing_file_report(self, hadrons_params):
        task = create_task("sib_mf", hadrons_params, "a", "20")
        report = compare_outputs(task.config, task.config)
        nblocks = sum(len(op.gamma.gamma_list) * 15 for op in task.config.op_list)
        ntab = 2 * len(Gamma.SCALAR_LOCAL.gamma_list)
        assert len(report) == nblocks + ntab
        assert (report["status"] == "missing_file").all()
        assert list(report.columns) == [
            "leg_pair",
            "mass",
            "gamma",
            "n_index",
            "hp_index",
            "filepath_a",
            "filepath_b",
            "max_abs_diff",
            "max_rel_diff",
            "within_tolerance",
            "status",
        ]

    def test_compare_pairs_present_files_per_world(
        self, tmp_path, monkeypatch, hadrons_params
    ):
        monkeypatch.chdir(tmp_path)
        params = _gated_params(hadrons_params, tmp_path)
        task = create_task("sib_mf", params, "a", "20")
        catalog = _sib_outfile_catalog(task.config)

        import h5py
        import numpy as np

        def _write_h5(path, gamma):
            from pathlib import Path

            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(path, "w") as f:
                grp = f.create_group(f"{gamma}_0_0_0")
                dt = np.dtype([("re", np.float64), ("im", np.float64)])
                grp.create_dataset("a2aMatrix", data=np.zeros(4, dtype=dt))

        # Write valid HDF5 only for the unsuffixed ll rows (they compare);
        # every noise-carrying leg pair stays missing, including the np
        # world combinations.
        for _, row in catalog.iterrows():
            if row["leg_pair"] == "ll":
                _write_h5(row["filepath"], row["gamma"])

        report = compare_outputs(task.config, task.config)
        compared = report[report["status"] == "compared"]
        missing = report[report["status"] == "missing_file"]
        assert (compared["leg_pair"] == "ll").all()
        assert len(compared) > 0
        # np rows carry their world indices in the report.
        np_missing = missing[missing["leg_pair"] == "np"]
        assert (
            (np_missing["n_index"] == "_n0") & (np_missing["hp_index"] == "_n1")
        ).any()
        assert ((np_missing["n_index"] == "") | (np_missing["n_index"].str.startswith("_n"))).all()


class TestSharedModeRegression:
    """``split_noise: false`` restores the Phase 2-4 shared-chain contract."""

    def test_shared_chain_end_to_end(self, hadrons_params):
        params = _shared_params(hadrons_params)
        task = create_task("sib_mf", params, "a", "20")
        result = build_input_params(task.config)

        assert task.config.output_config.split_noise is False
        blocks = _blocks(result)
        assert len(blocks) == 18

        p = result.modules["precon_t0"]["options"]
        assert p["nNoise"] == "2"
        assert p["noise"] == "noise_fv_vec"
        assert p["mesonField"] == "mfload_tab"

        assert result.modules["noise_t0"]["options"]["nSrc"] == "2"

        assert len(_sib_outfile_catalog(task.config)) == 43
        assert _pair_keys(task.config) == ["leg_pair", "mass", "gamma"]

        # Schedule order identical to the Phase 2 AVs.
        schedule = result.schedule
        assert (
            schedule.index("noise_fv")
            < schedule.index("noise_t0")
            < schedule.index("mf_tab")
            < schedule.index("mfload_tab")
            < schedule.index("precon_t0")
            < schedule.index("quark_h_mass_l_t0")
        )

    def test_shared_mode_split_modules_absent(self, hadrons_params):
        params = _shared_params(hadrons_params)
        result = build_input_params(create_task("sib_mf", params, "a", "20").config)
        for absent in (
            "noise_fv_n0",
            "noise_t0_n0",
            "mf_tab_n0",
            "mfload_tab_n0",
            "precon_n0",
            "quark_h_n0_mass_l_t0",
        ):
            assert absent not in result.modules, absent


def test_generate_sib_mf_input_end_to_end(tmp_path, monkeypatch, hadrons_params):
    """Full write_input_file dispatch for job_type=hadrons, task_type=sib_mf
    (the byte-level golden lives in test_generate_input.py's HADRONS_CASES)."""
    monkeypatch.chdir(tmp_path)

    write_input_file("sib_mf", hadrons_params, "a", "20")

    xml = (tmp_path / "in" / "sib-mf-a.20.xml").read_text()
    assert "<type>MSource::StagRandomWall</type>" in xml
    assert "<type>MFermion::SpinTaste</type>" in xml
    assert "<type>MContraction::StagA2AMesonField</type>" in xml
    assert "<type>MIO::LoadMesonField</type>" in xml
    assert "<type>MFermion::StagLMAMesonFieldProp</type>" in xml
    assert "<a2a_batch>true</a2a_batch>" in xml
    assert "<labels>G1_G1</labels>" in xml
    assert "<type>MFermion::StagGaugeProp</type>" in xml
    # Per-world external-noise wall wiring: noise_t0_n0 reads noise_fv_n0.
    assert "<noise>noise_fv_n0</noise>" in xml
    # v2-only: no Legacy modules anywhere in the SIB chain
    assert "Legacy</type>" not in xml
    sched = (tmp_path / "schedules" / "sib-mf-a.20.sched").read_text()
    assert sched.splitlines()[0] == "74"
    assert sched.splitlines()[0] == str(len(sched.splitlines()) - 1)
