"""Tests for low_mode_method='load': file-driven StagLMAMesonFieldProp low-mode producers."""

import copy
import logging
from pathlib import Path

import pytest

from pyfm.domain import MassDict, OpList, Outfile
from pyfm.tasks.hadrons.types import HighModeConfig
from pyfm.tasks.hadrons.highmode.strategy import (
    build_input_params,
    create_outfile_catalog,
    route_params,
    validate_config,
)
import pyfm.tasks.hadrons.modules as hadmods


MESON_STOCH_PROJ = Outfile(
    filestem="mesonfield/mf_{mass}", ext=".{cfg}/{gamma}_0_0_0.h5", good_size=1
)


def make_config(**overrides):
    kwargs = dict(
        formatting={},
        logging_level="INFO",
        runid="test",
        mass=MassDict.from_dict({"l": 0.002426, "u": 0.001524}),
        action_name="stag_mass_{mass}",
        solver_name="stag_{solver}_mass_{mass}",
        low_modes_name="evecs_mass_{mass}",
        operations=OpList.from_dict({"pion_local": {"mass": ["l", "u"]}}),
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


class TestLowModeMethodRouting:
    @staticmethod
    def routed(tasks_slice):
        params = {"mass": {"l": 0.01}, "_preprocessor": copy.deepcopy(tasks_slice)}
        return route_params(params)

    def test_low_mode_method_routes_to_field_not_operations(self):
        routed = self.routed({"low_mode_method": "load", "pion_local": {"mass": ["l"]}})
        assert routed["low_mode_method"] == "load"
        assert "low_mode_method" not in routed["operations"]

    def test_absent_key_leaves_dataclass_default(self):
        routed = self.routed({"pion_local": {"mass": ["l"]}})
        assert "low_mode_method" not in routed


class TestLowModeMethodValidation:
    def test_compute_default_passes(self):
        validate_config(make_config())

    def test_load_mode_passes_with_outfile_and_noise_one(self):
        validate_config(
            make_config(low_mode_method="load", meson_stoch_proj=MESON_STOCH_PROJ)
        )

    def test_invalid_value_rejected(self):
        config = make_config(low_mode_method="meson_field")
        with pytest.raises(ValueError, match="low_mode_method"):
            validate_config(config)

    def test_load_requires_noise_one(self):
        config = make_config(
            low_mode_method="load", noise=2, meson_stoch_proj=MESON_STOCH_PROJ
        )
        with pytest.raises(ValueError, match="noise"):
            validate_config(config)

    def test_load_rejects_nbias(self):
        config = make_config(
            low_mode_method="load",
            nbias=4,
            bias_seed="seed_a_20",
            meson_stoch_proj=MESON_STOCH_PROJ,
        )
        with pytest.raises(ValueError, match="nbias"):
            validate_config(config)

    def test_load_requires_meson_stoch_proj(self):
        config = make_config(low_mode_method="load")
        with pytest.raises(ValueError, match="meson_stoch_proj"):
            validate_config(config)

    def test_load_inert_under_skip_low_modes_warns(self, caplog):
        # skip_low_modes=True: no ranLL solver is emitted, so the knob and
        # its Outfile requirement are inert — loudly warned.
        with caplog.at_level(logging.WARNING):
            validate_config(make_config(low_mode_method="load", skip_low_modes=True))
        assert "inert" in caplog.text

    def test_load_requires_mass_token_in_filestem(self):
        config = make_config(
            low_mode_method="load",
            meson_stoch_proj=Outfile(
                filestem="mesonfield/shared",
                ext=".{cfg}/{gamma}_0_0_0.h5",
                good_size=1,
            ),
        )
        with pytest.raises(ValueError, match=r"\{mass\}"):
            validate_config(config)

    def test_blocksize_defaults_to_12(self):
        assert make_config().blocksize == 12


class TestModuleWrappers:
    def test_meson_field_without_cb_pairs_is_unchanged(self):
        module = hadmods.meson_field(
            name="mf", action="", block="200", gammas="(G1 G1)", gauge="gauge",
            low_modes="evecs_mass_l", left="", right="noise_fv_vec",
            output="mfout", apply_g5="false",
        )
        assert module["id"]["type"] == "MContraction::StagA2AMesonField"
        assert "cbPairsLeft" not in module["options"]
        assert "cbPairsRight" not in module["options"]

    def test_meson_field_with_cb_pairs_emits_both(self):
        module = hadmods.meson_field(
            name="mf", action="", block="200", gammas="(G1 G1)", gauge="gauge",
            low_modes="evecs_mass_l", left="", right="noise_fv_vec",
            output="mfout", apply_g5="false",
            cb_pairs_left="cbpairs_l_mass_l", cb_pairs_right="cbpairs_r_mass_l",
        )
        assert module["options"]["cbPairsLeft"] == "cbpairs_l_mass_l"
        assert module["options"]["cbPairsRight"] == "cbpairs_r_mass_l"

    def test_meson_field_rejects_one_sided_cb_pairs(self):
        with pytest.raises(ValueError, match="together"):
            hadmods.meson_field(
                name="mf", action="", block="200", gammas="(G1 G1)",
                gauge="gauge", low_modes="evecs_mass_l", left="",
                right="noise_fv_vec", output="mfout", apply_g5="false",
                cb_pairs_left="cbpairs_l_mass_l",
            )

    def test_eigen_pack_cb_pairs(self):
        module = hadmods.eigen_pack_cb_pairs(
            name="cbpairs_l_mass_l", eigen_pack="evecs_mass_l", action="stag_mass_l"
        )
        assert module["id"] == {
            "name": "cbpairs_l_mass_l",
            "type": "MUtilities::EigenPackCBPairs",
        }
        assert module["options"] == {
            "action": "stag_mass_l",
            "eigenPack": "evecs_mass_l",
        }

    def test_load_meson_field(self):
        module = hadmods.load_meson_field(
            name="mfload_mass_l",
            file="mesonfield/mf_l.@traj@/G1_G1_0_0_0.h5",
            dataset="G1_G1_0_0_0",
        )
        assert module["id"]["type"] == "MIO::LoadMesonField"
        assert module["options"]["file"].endswith("@traj@/G1_G1_0_0_0.h5")
        assert module["options"]["dataset"] == "G1_G1_0_0_0"
        assert module["options"]["side"] == ""
        assert module["options"]["lowModes"] == ""

    def test_lma_meson_field_prop(self):
        module = hadmods.lma_meson_field_prop(
            name="quark_ranLL_pion_local_mass_l",
            action="stag_mass_l",
            low_modes="evecs_mass_l",
            meson_field="mfload_mass_l_G1_G1",
            ta="0",
            tb="3",
            tstep="1",
            gammas="(G5 G5)",
            apply_g5="true",
            noise="noise_fv_vec",
        )
        assert module["id"] == {
            "name": "quark_ranLL_pion_local_mass_l",
            "type": "MFermion::StagLMAMesonFieldProp",
        }
        assert module["options"] == {
            "action": "stag_mass_l",
            "lowModes": "evecs_mass_l",
            "mesonField": "mfload_mass_l_G1_G1",
            "spinTaste": {"gammas": "(G5 G5)", "gauge": "", "applyG5": "true"},
            "noiseIndex": "0",
            "tA": "0",
            "tB": "3",
            "tStep": "1",
            "eigStart": "0",
            "nEigs": "-1",
            "negFirst": "",
            "pairScale": "",
            "noise": "noise_fv_vec",
        }


class TestLoadModeEmission:
    @staticmethod
    def load_config(**overrides):
        return make_config(
            low_mode_method="load",
            meson_stoch_proj=MESON_STOCH_PROJ,
            overwrite=True,
            skip_cg=False,
            **overrides,
        )

    def test_chain_modules_emitted_per_mass(self):
        result = build_input_params(self.load_config())
        for m in ("l", "u"):
            for prefix in ("cbpairs_l_", "cbpairs_r_", "mfwrite_"):
                assert f"{prefix}mass_{m}" in result.modules
            assert f"mfload_mass_{m}_G1_G1" in result.modules
            producer = result.modules[f"quark_ranLL_pion_local_mass_{m}"]
            assert producer["id"]["type"] == "MFermion::StagLMAMesonFieldProp"
            assert producer["options"]["noiseIndex"] == "0"
            assert producer["options"]["tA"] == "0"
            assert producer["options"]["tB"] == "3"
            assert producer["options"]["tStep"] == "1"

    def test_no_stag_lma_in_load_mode(self):
        result = build_input_params(self.load_config())
        assert all(
            mod["id"]["type"] != "MSolver::StagLMA"
            for mod in result.modules.values()
        )
        # the dead file-driven solver family is gone entirely
        assert not any(
            mod["id"]["type"] == "MSolver::StagLMAMesonField"
            for mod in result.modules.values()
        )

    def test_shared_fv_noise_feeds_random_walls(self):
        result = build_input_params(self.load_config())
        fv = result.modules["noise_fv"]
        assert fv["id"]["type"] == "MNoise::StagFullVolumeSpinColorDiagonal"
        assert fv["options"]["nsrc"] == "1"
        for t in range(4):
            assert result.modules[f"noise_t{t}"]["options"]["noise"] == "noise_fv"
        assert result.modules["mfwrite_mass_l"]["options"]["right"] == "noise_fv_vec"
        assert (
            result.modules["quark_ranLL_pion_local_mass_l"]["options"]["noise"]
            == "noise_fv_vec"
        )

    def test_writer_chain_options(self):
        result = build_input_params(self.load_config())
        w = result.modules["mfwrite_mass_l"]["options"]
        assert w["cbPairsLeft"] == "cbpairs_l_mass_l"
        assert w["cbPairsRight"] == "cbpairs_r_mass_l"
        assert w["lowModes"] == "evecs_mass_l"
        assert w["left"] == ""
        assert w["action"] == ""
        assert w["spinTaste"]["gammas"] == "(G5 G5)"
        assert w["spinTaste"]["gauge"] == "gauge"
        assert w["spinTaste"]["applyG5"] == "true"
        assert w["mom"] == {"elem": "0 0 0"}
        assert w["output"] == "mesonfield/mf_l"

    def test_loader_and_producer_reference_chain(self):
        result = build_input_params(self.load_config())
        loader = result.modules["mfload_mass_l_G1_G1"]["options"]
        assert loader["file"] == "mesonfield/mf_l.@traj@/G1_G1_0_0_0.h5"
        assert loader["dataset"] == "G1_G1_0_0_0"
        producer = result.modules["quark_ranLL_pion_local_mass_l"]["options"]
        assert producer["mesonField"] == "mfload_mass_l_G1_G1"
        assert producer["lowModes"] == "evecs_mass_l"

    def test_ranll_quark_props_replaced_by_producer_outputs(self):
        result = build_input_params(self.load_config())
        # No ranLL GaugeProp middlemen: the producer's outputs ARE the
        # quark propagators (quark_ranLL_{glabel}_mass_{m}_t{t}).
        assert not any(
            n.startswith("quark_ranLL_")
            and n.endswith(tuple(f"_t{t}" for t in range(4)))
            for n in result.modules
        )
        ama = result.modules["quark_ama_pion_local_mass_l_t0"]["options"]
        assert ama["solver"] == "stag_ama_mass_l"
        # The ama guess base names a producer output exactly (single
        # gamma: bare <name>_t<t>, GaugeProp gammaKey "" — no suffix).
        assert ama["guess"] == "quark_ranLL_pion_local_mass_l_t0"

    def test_writer_precedes_loader_in_schedule(self):
        result = build_input_params(self.load_config())
        for m in ("l", "u"):
            assert (
                result.schedule.index(f"mfwrite_mass_{m}")
                < result.schedule.index(f"mfload_mass_{m}_G1_G1")
                < result.schedule.index(f"quark_ranLL_pion_local_mass_{m}")
            )

    def test_producer_precedes_ama_quark_props(self):
        result = build_input_params(self.load_config())
        assert result.schedule.index("quark_ranLL_pion_local_mass_l") < (
            result.schedule.index("quark_ama_pion_local_mass_l_t0")
        )

    def test_compute_mode_emission_unchanged(self):
        result = build_input_params(make_config(overwrite=True, skip_cg=False))
        assert "noise_fv" not in result.modules
        assert not any(
            n.startswith(("cbpairs_", "mfwrite_", "mfload_")) for n in result.modules
        )
        assert (
            result.modules["quark_ranLL_pion_local_mass_l_t0"]["options"]["solver"]
            == "stag_ranLL_mass_l"
        )
        assert result.modules["stag_ranLL_mass_l"]["id"]["type"] == "MSolver::StagLMA"

    def test_solver_labels_and_catalog_mode_independent(self):
        compute = make_config(overwrite=True, skip_cg=False)
        load = self.load_config()
        assert compute.get_solver_labels() == load.get_solver_labels()
        assert create_outfile_catalog(compute).equals(create_outfile_catalog(load))

    def test_multi_gamma_ops_emit_union_writer_and_conjugated_loaders(self):
        config = self.load_config(
            operations=OpList.from_dict(
                {"pion_local": {"mass": ["l"]}, "vec_local": {"mass": ["l"]}}
            ),
            mass=MassDict.from_dict({"l": 0.002426}),
        )
        result = build_input_params(config)
        w = result.modules["mfwrite_mass_l"]["options"]["spinTaste"]
        # Union of REQUESTED strings with applyG5=true — the files stay
        # keyed by the conjugated names (pion -> G1_G1_0_0_0.h5).
        assert w["gammas"] == "(G5 G5) (GX GX) (GY GY) (GZ GZ)"
        assert w["applyG5"] == "true"
        for conj in ("G1_G1", "G5X_G5X", "G5Y_G5Y", "G5Z_G5Z"):
            assert f"mfload_mass_l_{conj}" in result.modules
        # One producer per op family; vec_local's outputs carry the
        # multi-gamma suffixes (..._t{t}_GX_GX etc., GaugeProp grammar).
        vec = result.modules["quark_ranLL_vec_local_mass_l"]["options"]
        assert vec["spinTaste"]["gammas"] == "(GX GX) (GY GY) (GZ GZ)"
        assert vec["spinTaste"]["applyG5"] == "true"
        assert vec["mesonField"] == (
            "mfload_mass_l_G5X_G5X mfload_mass_l_G5Y_G5Y mfload_mass_l_G5Z_G5Z"
        )


def _write_file(path, size):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"0" * int(size))


class TestLoadModeSkipIfComplete:
    # Ext mirrors what the config builder produces in real builds: `{cfg}`
    # pre-formatted to the concrete trajectory (here "20") — the catalog
    # helpers treat any leftover brace key as an error.
    STOCH = Outfile(
        filestem="mf/mf_{mass}", ext=".20/{gamma}_0_0_0.h5", good_size=10
    )

    def test_complete_file_skips_writer_but_keeps_loader(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = make_config(
            low_mode_method="load", meson_stoch_proj=self.STOCH, overwrite=False
        )
        for mass in ("l", "u"):
            _write_file(f"mf/mf_{mass}.20/G1_G1_0_0_0.h5", 10)

        result = build_input_params(config)
        for m in ("l", "u"):
            assert f"mfwrite_mass_{m}" not in result.modules
            assert f"cbpairs_l_mass_{m}" not in result.modules
            assert f"mfload_mass_{m}_G1_G1" in result.modules
            assert f"quark_ranLL_pion_local_mass_{m}" in result.modules

    def test_partial_completion_skips_only_complete_masses(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = make_config(
            low_mode_method="load", meson_stoch_proj=self.STOCH, overwrite=False
        )
        _write_file("mf/mf_l.20/G1_G1_0_0_0.h5", 4)  # undersized → bad
        _write_file("mf/mf_u.20/G1_G1_0_0_0.h5", 10)  # complete → skip

        result = build_input_params(config)
        assert "mfwrite_mass_l" in result.modules
        assert "mfwrite_mass_u" not in result.modules
        assert "mfload_mass_l_G1_G1" in result.modules
        assert "mfload_mass_u_G1_G1" in result.modules

    def test_no_pending_sources_omits_random_walls_not_chain(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = make_config(
            low_mode_method="load", meson_stoch_proj=self.STOCH, overwrite=False
        )
        # Complete correlator outputs for every source → run_refs empty.
        for t in range(4):
            _write_file(f"corr/corr_{t}.20.h5", 1)

        result = build_input_params(config)
        assert not any(n.startswith("noise_t") for n in result.modules)
        # noise_fv stays: the always-emitted loader/producer chain
        # references noise_fv_vec (producer self-check) — no dangling
        # references.
        assert "noise_fv" in result.modules
        # Writers still gated on meson-field completeness (nothing written):
        assert "mfwrite_mass_l" in result.modules
        assert "mfload_mass_l_G1_G1" in result.modules
        assert "quark_ranLL_pion_local_mass_l" in result.modules
