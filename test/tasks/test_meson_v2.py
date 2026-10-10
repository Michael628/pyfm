"""Tests for meson_v2.py — the canonical-schema counterpart of meson.py's
direct A2A meson-field build_input_params."""

from pyfm.domain import MassDict, OpList, Outfile
from pyfm.tasks.hadrons.meson import MesonConfig
from pyfm.tasks.hadrons import meson_v2


def make_config(**overrides):
    kwargs = dict(
        formatting={},
        logging_level="INFO",
        runid="test",
        action_name="stag_mass_{mass}",
        low_modes_name="evecs_mass_{mass}",
        mass=MassDict.from_dict({"l": 0.002426}),
        blocksize=200,
        operations=OpList.from_dict({"pion_local": {"mass": ["l"]}}),
        meson=Outfile(filestem="mf/mf_{mass}", ext=".h5", good_size=1),
        overwrite=True,
        shift_gauge_name="gauge_apbc",
    )
    kwargs.update(overrides)
    return MesonConfig(**kwargs)


class TestBuildInputParamsV2:
    def test_emits_spintaste_module_before_meson_field(self):
        result = meson_v2.build_input_params(make_config())
        assert "spintaste_mf_local_mass_l" in result.modules
        assert "mf_local_mass_l" in result.modules
        assert result.schedule.index("spintaste_mf_local_mass_l") < result.schedule.index(
            "mf_local_mass_l"
        )

    def test_spintaste_module_carries_the_gamma_string(self):
        result = meson_v2.build_input_params(make_config())
        spintaste = result.modules["spintaste_mf_local_mass_l"]
        assert spintaste["id"]["type"] == "MFermion::SpinTaste"
        assert spintaste["options"]["spinTaste"]["gammas"] == "(G5 G5)"
        assert spintaste["options"]["spinTaste"]["gauge"] == ""  # pion_local is local

    def test_meson_field_v2_references_spintaste_by_name(self):
        result = meson_v2.build_input_params(make_config())
        mf = result.modules["mf_local_mass_l"]
        assert mf["id"]["type"] == "MContraction::StagA2AMesonField"
        assert mf["options"]["gammas"] == "spintaste_mf_local_mass_l"
        assert "action" not in mf["options"]
        assert "cbPairsLeft" not in mf["options"]

    def test_nonlocal_gamma_uses_shift_gauge_name(self):
        config = make_config(operations=OpList.from_dict({"vec_onelink": {"mass": ["l"]}}))
        result = meson_v2.build_input_params(config)
        spintaste = result.modules["spintaste_mf_onelink_mass_l"]
        assert spintaste["options"]["spinTaste"]["gauge"] == "gauge_apbc"

    def test_missing_shift_gauge_name_raises(self):
        config = make_config(
            operations=OpList.from_dict({"vec_onelink": {"mass": ["l"]}}),
            shift_gauge_name=None,
        )
        try:
            meson_v2.build_input_params(config)
        except ValueError as e:
            assert "shift_gauge_name" in str(e)
        else:
            raise AssertionError("expected ValueError")

    def test_multi_mass_emits_one_module_pair_per_mass(self):
        config = make_config(
            mass=MassDict.from_dict({"l": 0.002426, "u": 0.001524}),
            operations=OpList.from_dict({"pion_local": {"mass": ["l", "u"]}}),
        )
        result = meson_v2.build_input_params(config)
        for m in ("l", "u"):
            assert f"spintaste_mf_local_mass_{m}" in result.modules
            assert f"mf_local_mass_{m}" in result.modules

    def test_left_right_default_to_empty_string(self):
        result = meson_v2.build_input_params(make_config())
        mf = result.modules["mf_local_mass_l"]
        assert mf["options"]["left"] == ""
        assert mf["options"]["right"] == ""

    def test_high_left_right_names_are_threaded_through(self):
        config = make_config(high_left_name="w_vec", high_right_name="noise_fv_vec")
        result = meson_v2.build_input_params(config)
        mf = result.modules["mf_local_mass_l"]
        assert mf["options"]["left"] == "w_vec"
        assert mf["options"]["right"] == "noise_fv_vec"
