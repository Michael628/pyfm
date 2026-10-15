"""Tests for qed_mf.py — QEDMFConfig composite, own hooks, the momentum
expansion, the EM seed convention, the demand-driven emission, the
catalog/resume gate, and compare_outputs."""

import copy
import dataclasses

import h5py
import numpy as np
import pytest
from pathlib import Path

from pyfm.domain import Gamma
from pyfm.nanny.taskbuilder import create_task
from pyfm.tasks.hadrons.qed_mf import (
    EM_FAMILY_JA,
    EM_FAMILY_J1L,
    QEDMFConfig,
    _expand_momenta,
    _needed_blocks,
    _qed_outfile_catalog,
    build_aggregator_params,
    build_input_params,
    compare_outputs,
    create_outfile_catalog,
)
from pyfm.tasks.register import get_task_handler, get_task_key, list_registered_types


def _qed_tasks(params):
    return params["job_setup"]["qed_mf"]["tasks"]


def _write_file(path, size):
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"0" * int(size))


class TestRegistration:
    def test_hadrons_qed_mf_is_registered(self):
        assert "nanny_hadrons_qed_mf" in list_registered_types()

    def test_task_key_resolves_to_qed_mf_handler(self):
        handler = get_task_handler(job_type="hadrons", task_type="qed_mf")
        assert handler is not None
        assert handler.config_type is QEDMFConfig

    def test_sibling_registrations_untouched(self):
        from pyfm.tasks.hadrons.lma_new import LMANewConfig
        from pyfm.tasks.hadrons.sib_mf import SIBMFConfig

        assert get_task_key(config=LMANewConfig) == "nanny_hadrons_lma_new"
        assert get_task_key(config=SIBMFConfig) == "nanny_hadrons_sib_mf"
        assert get_task_key(config=QEDMFConfig) == "nanny_hadrons_qed_mf"

    def test_leaf_configs_registered(self):
        registered = list_registered_types()
        for leaf in ("batch", "solver", "output", "em", "family"):
            assert f"nanny_hadrons_qed_mf_{leaf}" in registered


class TestMomentumExpansion:
    def test_conjugate_pairs_in_order(self):
        assert _expand_momenta([(1, 0, 0), (0, 1, 0)]) == [
            "1 0 0", "-1 0 0", "0 1 0", "0 -1 0",
        ]

    def test_zero_momentum_deduplicates(self):
        assert _expand_momenta([(0, 0, 0)]) == ["0 0 0"]
        assert _expand_momenta([(0, 0, 0), (1, 0, 0)]) == [
            "0 0 0", "1 0 0", "-1 0 0",
        ]

    def test_config_property_expands_fixture(self, hadrons_params):
        task = create_task("qed_mf", hadrons_params, "a", "20")
        assert task.config.mom_elems == ["1 0 0", "-1 0 0"]

    def test_empty_momentum_defaults_to_zero(self, hadrons_params):
        params = copy.deepcopy(hadrons_params)
        _qed_tasks(params)["qed"]["momentum"] = []
        task = create_task("qed_mf", params, "a", "20")
        assert task.config.mom_elems == ["0 0 0"]


class TestSeedConvention:
    def test_seeds_shared_across_families_per_sample(self, hadrons_params):
        task = create_task("qed_mf", hadrons_params, "a", "20")
        cfg = task.config
        # default seed base "qed_{series}_{cfg}" resolves to "qed_a_20"
        assert cfg.em_seeds(EM_FAMILY_JA) == ["qed_a_20_n0", "qed_a_20_n1"]
        assert cfg.em_seeds(EM_FAMILY_J1L) == cfg.em_seeds(EM_FAMILY_JA)
        assert cfg.n_em(EM_FAMILY_JA) == 2

    def test_family_n_em_override(self, hadrons_params):
        params = copy.deepcopy(hadrons_params)
        _qed_tasks(params)["qed"]["onelink"] = {"n_em": 3}
        task = create_task("qed_mf", params, "a", "20")
        assert task.config.n_em(EM_FAMILY_J1L) == 3
        assert task.config.n_em(EM_FAMILY_JA) == 2


class TestRouting:
    def test_aslash_bool_and_block_forms(self, hadrons_params):
        params = copy.deepcopy(hadrons_params)
        _qed_tasks(params)["qed"]["aslash"] = False
        _qed_tasks(params)["qed"]["onelink"] = True
        task = create_task("qed_mf", params, "a", "20")
        assert task.config.aslash_config is None
        assert task.config.onelink_config is not None
        assert task.config.em_families() == [EM_FAMILY_J1L]

    def test_unknown_qed_key_rejected(self, hadrons_params):
        _qed_tasks(hadrons_params)["qed"]["bogus"] = 1
        with pytest.raises(ValueError, match="Unknown qed section keys"):
            create_task("qed_mf", hadrons_params, "a", "20")

    def test_unknown_tasks_key_rejected(self, hadrons_params):
        hadrons_params["job_setup"]["qed_mf"]["tasks"]["bogus"] = {}
        with pytest.raises(ValueError, match="Unknown tasks keys"):
            create_task("qed_mf", hadrons_params, "a", "20")


class TestValidation:
    def test_batch_nonpositive_rejected(self, hadrons_params):
        _qed_tasks(hadrons_params)["qed"]["batch"] = {"noise": 0}
        with pytest.raises(ValueError, match="noise"):
            create_task("qed_mf", hadrons_params, "a", "20")

    def test_solver_unknown_rejected(self, hadrons_params):
        _qed_tasks(hadrons_params)["qed"]["solver"]["solver"] = "bogus"
        with pytest.raises(ValueError, match="must be one of"):
            create_task("qed_mf", hadrons_params, "a", "20")

    def test_defl_mass_not_in_massdict_rejected(self, hadrons_params):
        _qed_tasks(hadrons_params)["qed"]["defl_mass"] = "x"
        with pytest.raises(ValueError, match="defl_mass"):
            create_task("qed_mf", hadrons_params, "a", "20")

    def test_operations_mass_not_in_massdict_rejected(self, hadrons_params):
        _qed_tasks(hadrons_params)["qed"]["operations"]["mass"] = ["x"]
        with pytest.raises(ValueError, match="mass.*not present"):
            create_task("qed_mf", hadrons_params, "a", "20")

    def test_non_qed_gamma_rejected(self, hadrons_params):
        _qed_tasks(hadrons_params)["qed"]["operations"]["gamma"] = ["pion_local"]
        with pytest.raises(ValueError, match="not QED families"):
            create_task("qed_mf", hadrons_params, "a", "20")

    def test_em_required_when_families_enabled(self, hadrons_params):
        del _qed_tasks(hadrons_params)["qed"]["em"]
        with pytest.raises(ValueError, match="require the qed.em block"):
            create_task("qed_mf", hadrons_params, "a", "20")

    def test_em_rejected_without_families(self, hadrons_params):
        _qed_tasks(hadrons_params)["qed"]["aslash"] = False
        _qed_tasks(hadrons_params)["qed"]["onelink"] = False
        with pytest.raises(ValueError, match="no EM family is enabled"):
            create_task("qed_mf", hadrons_params, "a", "20")

    def test_bad_momentum_arity_rejected(self, hadrons_params):
        _qed_tasks(hadrons_params)["qed"]["momentum"] = [[1, 0]]
        with pytest.raises(ValueError, match="3 components"):
            create_task("qed_mf", hadrons_params, "a", "20")

    def test_misrouted_file_label_rejected(self, hadrons_params):
        _qed_tasks(hadrons_params)["qed"]["output"]["file"] = "qed_meson_blocks"
        with pytest.raises(ValueError, match="cfg_qed_h5"):
            create_task("qed_mf", hadrons_params, "a", "20")

    def test_misrouted_tab_label_rejected(self, hadrons_params):
        _qed_tasks(hadrons_params)["qed"]["output"]["tab"] = "qed_tab_only"
        with pytest.raises(ValueError, match="cfg_gamma_h5"):
            create_task("qed_mf", hadrons_params, "a", "20")


class TestInfraEmission:
    def test_infra_modules_present(self, hadrons_params):
        task = create_task("qed_mf", hadrons_params, "a", "20")
        result = build_input_params(task.config)
        assert {
            "gauge", "gauge_smear_fat", "gauge_smear_long", "gauge_apbc",
            "epack", "eval_save", "evecs_mass_l",
            "stag_mass_l", "istag_mass_l",
        } <= set(result.modules)

    def test_aggregator_empty(self, hadrons_params):
        task = create_task("qed_mf", hadrons_params, "a", "20")
        assert build_aggregator_params(task.config, True) == {}
        assert build_aggregator_params(task.config, False) == {}


class TestCatalog:
    def test_row_count_and_axes(self, hadrons_params):
        # vec_local: 4 leg pairs × 3 gammas × 2 momenta = 24; ja: 4 × 2
        # worlds × 1 channel = 8; j1l: 4 × 2 × 4 components = 32; tab 1.
        task = create_task("qed_mf", hadrons_params, "a", "20")
        df = _qed_outfile_catalog(task.config)
        assert len(df) == 65
        assert {
            "family", "leg_pair", "mass", "em_index", "file_key",
            "filepath", "good_size", "exists", "file_size",
        } <= set(df.columns)

    def test_vec_filepath_grammar(self, hadrons_params):
        task = create_task("qed_mf", hadrons_params, "a", "20")
        df = _qed_outfile_catalog(task.config)
        paths = set(df["filepath"])
        assert any(p.endswith("qed/vl_ll_a.20/GX_GX_1_0_0.h5") for p in paths)
        assert any(p.endswith("qed/vl_ll_a.20/GX_GX_-1_0_0.h5") for p in paths)
        assert any(p.endswith("qed/vl_nh_ml_a.20/GZ_GZ_-1_0_0.h5") for p in paths)

    def test_em_filepath_grammar(self, hadrons_params):
        task = create_task("qed_mf", hadrons_params, "a", "20")
        df = _qed_outfile_catalog(task.config)
        paths = set(df["filepath"])
        assert any(
            p.endswith("qed/ja_ll_em0_a.20/qed_a_20_n0_0_0_0_0.h5") for p in paths
        )
        assert any(
            p.endswith("qed/j1l_nl_em1_a.20/qed_a_20_n1_0_2_0_0_0.h5") for p in paths
        )

    def test_tab_row(self, hadrons_params):
        task = create_task("qed_mf", hadrons_params, "a", "20")
        df = _qed_outfile_catalog(task.config)
        assert any(
            p.endswith("qed/tab_a.20/G1_G1_0_0_0.h5") for p in set(df["filepath"])
        )

    def test_handler_catalog_includes_qed_rows(self, hadrons_params):
        task = create_task("qed_mf", hadrons_params, "a", "20")
        assert len(create_outfile_catalog(task.config)) >= 65


class TestNeededBlocks:
    def test_vec_narrows_to_group(self, hadrons_params):
        task = create_task("qed_mf", hadrons_params, "a", "20")
        df = _qed_outfile_catalog(task.config)
        bad = {
            f for f in df["filepath"]
            if f.endswith("qed/vl_ll_a.20/GX_GX_1_0_0.h5")
        }
        ref, h, em = _needed_blocks(task.config, bad)
        assert ref == {Gamma.VEC_LOCAL: {"ll"}}
        assert h == set() and em == set()

    def test_em_narrows_to_world(self, hadrons_params):
        task = create_task("qed_mf", hadrons_params, "a", "20")
        df = _qed_outfile_catalog(task.config)
        bad = {
            f for f in df["filepath"]
            if f.endswith("qed/j1l_nl_em1_a.20/qed_a_20_n1_0_2_0_0_0.h5")
        }
        ref, h, em = _needed_blocks(task.config, bad)
        assert em == {(EM_FAMILY_J1L, "nl", None, 1)}
        assert ref == {} and h == set()

    def test_vec_h_pair_demand(self, hadrons_params):
        task = create_task("qed_mf", hadrons_params, "a", "20")
        df = _qed_outfile_catalog(task.config)
        bad = {
            f for f in df["filepath"]
            if f.endswith("qed/vl_lh_ml_a.20/GX_GX_-1_0_0.h5")
        }
        ref, h, em = _needed_blocks(task.config, bad)
        assert h == {(Gamma.VEC_LOCAL, "lh", "l")}
        assert ref == {} and em == set()


class TestChainEmission:
    def test_chain_present_and_ordered(self, hadrons_params):
        task = create_task("qed_mf", hadrons_params, "a", "20")
        s = build_input_params(task.config).schedule
        for earlier, later in [
            ("cbpairs_l_mass_l", "em"),
            ("em", "qed_solver_mass_l"),
            ("noise_fv", "noise_batch"),
            ("noise_batch", "mf_tab"),
            ("mf_tab", "mfload_tab"),
            ("mfload_tab", "precon_batch"),
            ("precon_batch", "quark_h_mass_l"),
        ]:
            assert s.index(earlier) < s.index(later), (earlier, later)

    def test_em_module_options(self, hadrons_params):
        task = create_task("qed_mf", hadrons_params, "a", "20")
        em = build_input_params(task.config).modules["em"]
        assert em["id"]["type"] == "MGauge::StochEmFunc"
        assert em["options"] == {
            "gauge": "feynman", "zmScheme": "qedL", "improvement": "",
        }

    def test_precon_batch_contract(self, hadrons_params):
        task = create_task("qed_mf", hadrons_params, "a", "20")
        p = build_input_params(task.config).modules["precon_batch"]["options"]
        assert (
            p["a2a_batch"], p["labels"], p["tA"], p["tB"], p["tStep"],
            p["nNoise"], p["noise"], p["mesonField"], p["gammas"],
        ) == ("true", "G1_G1", "0", "3", "1", "2", "noise_fv_vec",
              "mfload_tab", "spintaste_scalar")

    def test_h_solve_wiring(self, hadrons_params):
        task = create_task("qed_mf", hadrons_params, "a", "20")
        h = build_input_params(task.config).modules["quark_h_mass_l"]
        assert h["id"]["type"] == "MFermion::StagGaugeProp"
        assert (h["options"]["source"], h["options"]["guess"],
                h["options"]["solver"], h["options"]["gammas"]) == (
            "noise_batch_vec", "precon_batch", "qed_solver_mass_l",
            "spintaste_scalar",
        )

    def test_spintaste_roles(self, hadrons_params):
        result = build_input_params(
            create_task("qed_mf", hadrons_params, "a", "20").config
        )
        mods = result.modules
        st = mods["spintaste_ja"]["options"]["spinTaste"]
        assert st["gammas"] == "(GX G1) (GY G1) (GZ G1) (GT G1)"
        assert st["gauge"] == "gauge"
        # per-family modules (cascade amendment): the fixture's vec_local
        # yields spintaste_vl only; s (scalar_local) and vo (vec_onelink)
        # are not selected
        assert mods["spintaste_vl"]["options"]["spinTaste"]["gammas"] == (
            "(GX GX) (GY GY) (GZ GZ)"
        )
        assert "spintaste_s" not in mods
        assert "spintaste_vo" not in mods

    def test_vec_onelink_binds_shift_gauge(self, hadrons_params):
        params = copy.deepcopy(hadrons_params)
        _qed_tasks(params)["qed"]["operations"]["gamma"] = ["vec_onelink"]
        result = build_input_params(create_task("qed_mf", params, "a", "20").config)
        assert (
            result.modules["spintaste_vo"]["options"]["spinTaste"]["gauge"]
            == "gauge_apbc"
        )


class TestChainGate:
    def _gated(self, params, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        params = copy.deepcopy(params)
        params["shared_params"]["home"] = str(tmp_path)
        params["shared_params"]["overwrite"] = False
        return create_task("qed_mf", params, "a", "20")

    def test_gated_chain_matches_overwrite_when_nothing_on_disk(
        self, hadrons_params, tmp_path, monkeypatch
    ):
        task = self._gated(hadrons_params, tmp_path, monkeypatch)
        gated = build_input_params(task.config)
        overwrite_cfg = dataclasses.replace(
            task.config,
            output_config=dataclasses.replace(
                task.config.output_config, overwrite=True
            ),
        )
        assert set(gated.modules) == set(build_input_params(overwrite_cfg).modules)

    def test_gate_all_complete_emits_no_chain(
        self, hadrons_params, tmp_path, monkeypatch
    ):
        task = self._gated(hadrons_params, tmp_path, monkeypatch)
        for _, row in _qed_outfile_catalog(task.config).iterrows():
            _write_file(row["filepath"], int(row["good_size"]) + 1)
        result = build_input_params(task.config)
        chain = {
            "em", "noise_fv", "noise_batch", "mf_tab", "mfload_tab",
            "precon_batch", "quark_h_mass_l", "qed_solver_mass_l",
            "cbpairs_l_mass_l", "cbpairs_r_mass_l",
            "spintaste_scalar", "spintaste_vl", "spintaste_ja",
        }
        assert not (chain & set(result.modules))


class TestBlockEmission:
    def test_block_inventory_and_uniqueness(self, hadrons_params):
        result = build_input_params(
            create_task("qed_mf", hadrons_params, "a", "20").config
        )
        blocks = sorted(
            n for n in result.modules if n.startswith("mf_") and n != "mf_tab"
        )
        assert blocks == sorted([
            # vec_local (4 leg pairs)
            "mf_vl_ll", "mf_vl_nl", "mf_vl_lh_mass_l", "mf_vl_nh_mass_l",
            # ja (4 leg pairs × 2 worlds)
            *[f"mf_ja_{lp}_em{i}" for lp in ("ll", "nl") for i in (0, 1)],
            *[f"mf_ja_{lp}_mass_l_em{i}" for lp in ("lh", "nh") for i in (0, 1)],
            # j1l (4 leg pairs × 2 worlds)
            *[f"mf_j1l_{lp}_em{i}" for lp in ("ll", "nl") for i in (0, 1)],
            *[f"mf_j1l_{lp}_mass_l_em{i}" for lp in ("lh", "nh") for i in (0, 1)],
        ])
        assert len(result.schedule) == len(set(result.schedule))

    def test_vec_block_wiring(self, hadrons_params):
        result = build_input_params(
            create_task("qed_mf", hadrons_params, "a", "20").config
        )
        ll = result.modules["mf_vl_ll"]["options"]
        assert ll["gammas"] == "spintaste_vl"
        assert ll["lowModes"] == "evecs_mass_l"
        assert ll["cbPairsLeft"] == "cbpairs_l_mass_l"
        assert ll["cbPairsRight"] == "cbpairs_r_mass_l"
        assert ll["left"] == "" and ll["right"] == ""
        assert ll["mom"] == {"elem": ["1 0 0", "-1 0 0"]}
        assert ll["output"].endswith("e100n2/qed/vl_ll_a")
        nh = result.modules["mf_vl_nh_mass_l"]["options"]
        assert nh["lowModes"] == "" and "cbPairsLeft" not in nh
        assert nh["left"] == "noise_fv_vec"
        assert nh["right"] == "quark_h_mass_l"
        assert nh["output"].endswith("e100n2/qed/vl_nh_ml_a")

    def test_em_block_wiring(self, hadrons_params):
        result = build_input_params(
            create_task("qed_mf", hadrons_params, "a", "20").config
        )
        ja = result.modules["mf_ja_ll_em0"]
        assert ja["id"]["type"] == "MContraction::StagA2AASlashMesonField"
        opts = ja["options"]
        assert opts["gammas"] == "spintaste_ja"
        assert opts["EmFunc"] == "em"
        assert opts["nEmFields"] == "1"
        assert opts["EmSeedString"] == "qed_a_20_n0"
        assert opts["mom"] == {"elem": ["0 0 0"]}
        assert opts["lowModes"] == "evecs_mass_l"
        assert opts["cbPairsLeft"] == "cbpairs_l_mass_l"
        assert opts["output"].endswith("e100n2/qed/ja_ll_em0_a")
        j1l = result.modules["mf_j1l_nh_mass_l_em1"]
        assert j1l["id"]["type"] == "MContraction::StagA2AASlashOneLinkMesonField"
        o = j1l["options"]
        assert o["gauge"] == "gauge"
        assert o["EmSeedString"] == "qed_a_20_n1"
        assert o["lowModes"] == "" and "cbPairsLeft" not in o
        assert o["left"] == "noise_fv_vec" and o["right"] == "quark_h_mass_l"
        assert o["output"].endswith("e100n2/qed/j1l_nh_ml_em1_a")

    def test_seed_shared_across_families_same_world(self, hadrons_params):
        result = build_input_params(
            create_task("qed_mf", hadrons_params, "a", "20").config
        )
        for world in (0, 1):
            assert (
                result.modules[f"mf_ja_ll_em{world}"]["options"]["EmSeedString"]
                == result.modules[f"mf_j1l_ll_em{world}"]["options"]["EmSeedString"]
            )

    def test_schedule_order_invariants(self, hadrons_params):
        s = build_input_params(
            create_task("qed_mf", hadrons_params, "a", "20").config
        ).schedule
        assert s.index("em") < s.index("mf_ja_ll_em0")
        assert s.index("mf_vl_ll") < s.index("noise_fv")  # cheap banking
        assert s.index("quark_h_mass_l") < s.index("mf_vl_lh_mass_l")
        assert s.index("noise_fv") < s.index("mf_vl_nl")
        assert s.index("quark_h_mass_l") < s.index("mf_j1l_nh_mass_l_em1")


class TestBlockGate:
    def _gated(self, params, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        params = copy.deepcopy(params)
        params["shared_params"]["home"] = str(tmp_path)
        params["shared_params"]["overwrite"] = False
        return create_task("qed_mf", params, "a", "20")

    def test_gate_narrows_to_world_and_chain(
        self, hadrons_params, tmp_path, monkeypatch
    ):
        task = self._gated(hadrons_params, tmp_path, monkeypatch)
        for _, row in _qed_outfile_catalog(task.config).iterrows():
            if (
                row["family"] == "j1l"
                and row["leg_pair"] == "nl"
                and row["em_index"] == "_em1"
            ):
                continue
            _write_file(row["filepath"], int(row["good_size"]) + 1)
        result = build_input_params(task.config)
        assert "mf_j1l_nl_em1" in result.modules
        assert "mf_j1l_nl_em0" not in result.modules
        assert "mf_vl_ll" not in result.modules
        # nl needs the noise leg + CB pairs + em, but not the h chain:
        assert "noise_fv" in result.modules
        assert "cbpairs_l_mass_l" in result.modules
        assert "em" in result.modules
        assert "precon_batch" not in result.modules
        assert "quark_h_mass_l" not in result.modules


class TestCompareOutputs:
    def test_compare_registered_on_handler(self):
        handler = get_task_handler(job_type="hadrons", task_type="qed_mf")
        assert handler.compare_outputs.__name__ == "compare_outputs"

    def test_absent_outputs_report_missing(self, hadrons_params):
        task = create_task("qed_mf", hadrons_params, "a", "20")
        df = compare_outputs(task.config, task.config)
        assert len(df) == 65
        assert (df["status"] == "missing_file").all()
        assert not df["within_tolerance"].any()

    def test_present_file_compares_dataset(
        self, hadrons_params, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        params = copy.deepcopy(hadrons_params)
        params["shared_params"]["home"] = str(tmp_path)
        task = create_task("qed_mf", params, "a", "20")
        df = _qed_outfile_catalog(task.config)
        row = df[
            df["filepath"].str.endswith(
                "qed/ja_ll_em0_a.20/qed_a_20_n0_0_0_0_0.h5"
            )
        ].iloc[0]
        path = Path(row["filepath"])
        path.parent.mkdir(parents=True, exist_ok=True)
        data = np.zeros((4, 2, 2), dtype=[("re", "<f8"), ("im", "<f8")])
        data["re"] = 1.0
        with h5py.File(path, "w") as f:
            f.create_group(str(row["file_key"])).create_dataset(
                "a2aMatrix", data=data
            )
        report = compare_outputs(task.config, task.config)
        hit = report[report["filepath_a"] == row["filepath"]]
        assert len(hit) == 1
        assert hit.iloc[0]["status"] == "compared"
        assert bool(hit.iloc[0]["within_tolerance"])
        assert hit.iloc[0]["max_abs_diff"] == 0.0
        assert (report["status"] == "missing_file").sum() == 64

    def test_seed_mismatch_raises(self, hadrons_params):
        task = create_task("qed_mf", hadrons_params, "a", "20")
        other = dataclasses.replace(
            task.config,
            em_config=dataclasses.replace(
                task.config.em_config, seed="other_{series}_{cfg}"
            ),
        )
        with pytest.raises(ValueError, match="qed.em.seed"):
            compare_outputs(task.config, other)
