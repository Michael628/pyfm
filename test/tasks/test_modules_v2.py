"""Tests for the canonical SpinTaste-module-driven wrappers in modules.py
(spin_taste extension + quark_prop_v2/prop_contract_v2/meson_field_v2/
lma_meson_field_prop_v2)."""

import pytest

import pyfm.tasks.hadrons.modules as hadmods


class TestSpinTaste:
    def test_default_no_labels(self):
        module = hadmods.spin_taste(
            "spintaste_vec_local", gammas="(GX GX)(GY GY)(GZ GZ)", gauge="", apply_g5="true"
        )
        assert module["id"] == {"name": "spintaste_vec_local", "type": "MFermion::SpinTaste"}
        assert module["options"] == {
            "spinTaste": {
                "gammas": "(GX GX)(GY GY)(GZ GZ)",
                "gauge": "",
                "applyG5": "true",
            },
            "labels": "",
        }

    def test_labels_override(self):
        module = hadmods.spin_taste(
            "spintaste_axial_vec_local",
            gammas="(G5X G5X)(G5Y G5Y)(G5Z G5Z)",
            gauge="",
            apply_g5="true",
            labels="GX_GX GY_GY GZ_GZ",
        )
        assert module["options"]["labels"] == "GX_GX GY_GY GZ_GZ"
        assert module["options"]["spinTaste"]["gammas"] == "(G5X G5X)(G5Y G5Y)(G5Z G5Z)"

    def test_displacing_gamma_carries_gauge(self):
        module = hadmods.spin_taste(
            "spintaste_vec_onelink",
            gammas="(GX G1)(GY G1)(GZ G1)",
            gauge="gauge_apbc",
            apply_g5="true",
        )
        assert module["options"]["spinTaste"]["gauge"] == "gauge_apbc"


class TestQuarkPropV2:
    def test_basic_shape(self):
        module = hadmods.quark_prop_v2(
            name="quark_ranLL_vec_local_mass_l_t0",
            source="noise_t0",
            solver="stag_ranLL_mass_l",
            guess="",
            gammas="spintaste_vec_local",
        )
        assert module["id"] == {
            "name": "quark_ranLL_vec_local_mass_l_t0",
            "type": "MFermion::StagGaugeProp",
        }
        assert module["options"] == {
            "source": "noise_t0",
            "gammas": "spintaste_vec_local",
            "solver": "stag_ranLL_mass_l",
            "guess": "",
            "sourceLabel": "",
        }
        assert "subgrid" not in module

    def test_source_label_and_subgrid(self):
        module = hadmods.quark_prop_v2(
            name="n",
            source="mfload_map",
            solver="s",
            guess="g",
            gammas="spintaste_vec_local",
            source_label="GX_GX",
            subgrid=2,
        )
        assert module["options"]["sourceLabel"] == "GX_GX"
        assert module["subgrid"] == 2


class TestPropContractV2:
    def test_basic_shape(self):
        module = hadmods.prop_contract_v2(
            name="corr_ranLL_vec_local_mass_l_t0",
            source="quark_ranLL_vec_local_mass_l_t0",
            sink="quark_ranLL_pion_local_mass_l_t0",
            sink_fn="sink",
            source_shift="noise_t0_shift",
            sink_gammas="spintaste_vec_local",
            output="corr_out",
        )
        assert module["id"] == {
            "name": "corr_ranLL_vec_local_mass_l_t0",
            "type": "MContraction::StagMeson",
        }
        assert module["options"] == {
            "source": "quark_ranLL_vec_local_mass_l_t0",
            "sink": "quark_ranLL_pion_local_mass_l_t0",
            "sinkFunc": "sink",
            "sourceShift": "noise_t0_shift",
            "sinkGammas": "spintaste_vec_local",
            "output": "corr_out",
        }
        assert "sourceGammas" not in module["options"]
        assert "sinkSpinTaste" not in module["options"]

    def test_subgrid_tag(self):
        module = hadmods.prop_contract_v2(
            name="n", source="s", sink="sk", sink_fn="sf", source_shift="ss",
            sink_gammas="spintaste_axial_vec_local", output="o", subgrid=1,
        )
        assert module["subgrid"] == 1


class TestMesonFieldV2:
    def test_without_cb_pairs(self):
        module = hadmods.meson_field_v2(
            name="mf_vec_local_mass_l",
            block="200",
            gammas="spintaste_vec_local",
            low_modes="evecs_mass_l",
            left="",
            right="noise_fv_vec",
            output="mfout",
        )
        assert module["id"] == {
            "name": "mf_vec_local_mass_l",
            "type": "MContraction::StagA2AMesonField",
        }
        assert module["options"] == {
            "block": "200",
            "mom": {"elem": "0 0 0"},
            "gammas": "spintaste_vec_local",
            "lowModes": "evecs_mass_l",
            "left": "",
            "right": "noise_fv_vec",
            "output": "mfout",
        }
        assert "action" not in module["options"]
        assert "spinTaste" not in module["options"]

    def test_with_cb_pairs_emits_both(self):
        module = hadmods.meson_field_v2(
            name="n", block="200", gammas="spintaste_pion_local",
            low_modes="lm", left="", right="noise_fv_vec", output="o",
            cb_pairs_left="cbpairs_l_mass_l", cb_pairs_right="cbpairs_r_mass_l",
        )
        assert module["options"]["cbPairsLeft"] == "cbpairs_l_mass_l"
        assert module["options"]["cbPairsRight"] == "cbpairs_r_mass_l"

    def test_rejects_one_sided_cb_pairs(self):
        with pytest.raises(ValueError, match="together"):
            hadmods.meson_field_v2(
                name="n", block="200", gammas="spintaste_pion_local",
                low_modes="lm", left="", right="noise_fv_vec", output="o",
                cb_pairs_left="cbpairs_l_mass_l",
            )


class TestLmaMesonFieldPropV2:
    def test_basic_shape(self):
        module = hadmods.lma_meson_field_prop_v2(
            name="quark_ranLL_pion_local_mass_l",
            action="stag_mass_l",
            low_modes="evecs_mass_l",
            meson_field="mfload_mass_l_G1_G1",
            gammas="spintaste_pion_local",
            labels="G1_G1",
            ta="0",
            tb="3",
            tstep="1",
            noise="noise_fv_vec",
            n_noise="2",
        )
        assert module["id"] == {
            "name": "quark_ranLL_pion_local_mass_l",
            "type": "MFermion::StagLMAMesonFieldProp",
        }
        assert module["options"] == {
            "action": "stag_mass_l",
            "lowModes": "evecs_mass_l",
            "mesonField": "mfload_mass_l_G1_G1",
            "gammas": "spintaste_pion_local",
            "labels": "G1_G1",
            "noiseIndex": "0",
            "nNoise": "2",
            "tA": "0",
            "tB": "3",
            "tStep": "1",
            "eigStart": "0",
            "nEigs": "-1",
            "negFirst": "",
            "pairScale": "",
            "noise": "noise_fv_vec",
        }
