"""Unit tests for the GaugeConfig action_type refactor.

Covers ``ActionType.from_dict``, the legacy ``free`` -> ``action_type``
normalize hook (including through the full build_config pipeline), the
``build_base_gauge`` / ``build_sp_gauge`` / ``build_action_modules`` branches
for FREE / LOAD / SMEAR (double + single precision), and the ``save_smear``
flag that writes the HISQ-smeared fat/long links to disk via SaveIldg.
"""
import dataclasses

import pytest

from pyfm.domain import MassDict, Outfile
from pyfm.tasks.hadrons.gauge import (
    ActionType,
    GaugeConfig,
    GaugeFileFormat,
    build_action_modules,
    build_base_gauge,
    build_sp_gauge,
    normalize_params,
)


def _outfile(label: str) -> Outfile:
    return Outfile(filestem=f"lat/{label}", ext=".{cfg}", good_size=100)


def _gauge_config(
    action_type: ActionType = ActionType.LOAD,
    save_smear: bool = False,
    format: GaugeFileFormat = GaugeFileFormat.ILDG,
) -> GaugeConfig:
    return GaugeConfig(
        formatting={},
        logging_level="info",
        runid="test",
        mass=MassDict.from_dict({"l": 0.1}),
        gauge_links=_outfile("gauge_links"),
        long_links=_outfile("long_links"),
        fat_links=_outfile("fat_links"),
        action_type=action_type,
        action_name="stag_mass_{mass}",
        save_smear=save_smear,
        format=format,
    )


class TestActionType:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("free", ActionType.FREE),
            ("FREE", ActionType.FREE),
            ("load", ActionType.LOAD),
            ("LOAD", ActionType.LOAD),
            ("smear", ActionType.SMEAR),
            ("Smear", ActionType.SMEAR),
            ("SMEAR", ActionType.SMEAR),
        ],
    )
    def test_from_dict_resolves_members(self, raw, expected):
        assert ActionType.from_dict(raw) is expected

    def test_from_dict_rejects_unknown(self):
        with pytest.raises(ValueError):
            ActionType.from_dict("nope")

    def test_legacy_member_names_no_longer_resolve(self):
        # improved/hisq were renamed to load/smear.
        for legacy in ("improved", "hisq"):
            with pytest.raises(ValueError):
                ActionType.from_dict(legacy)

    def test_default_is_load(self):
        assert _gauge_config().action_type is ActionType.LOAD

    def test_free_field_replaced_by_action_type(self):
        names = {f.name for f in dataclasses.fields(GaugeConfig)}
        assert "free" not in names
        assert "action_type" in names


class TestGaugeFileFormat:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("ildg", GaugeFileFormat.ILDG),
            ("ILDG", GaugeFileFormat.ILDG),
            ("milcv5", GaugeFileFormat.MILCV5),
            ("MILCV5", GaugeFileFormat.MILCV5),
            ("milc_v5", GaugeFileFormat.MILCV5),
            ("MILC_V5", GaugeFileFormat.MILCV5),
        ],
    )
    def test_from_dict_resolves_members(self, raw, expected):
        assert GaugeFileFormat.from_dict(raw) is expected

    def test_from_dict_rejects_unknown(self):
        with pytest.raises(ValueError):
            GaugeFileFormat.from_dict("nersc")

    def test_default_is_ildg(self):
        assert _gauge_config().format is GaugeFileFormat.ILDG

    def test_field_exists(self):
        names = {f.name for f in dataclasses.fields(GaugeConfig)}
        assert "format" in names


class TestNormalize:
    def test_free_true_maps_to_free(self):
        out = normalize_params({"free": True})
        assert out["action_type"] == "free"
        assert "free" not in out

    def test_free_false_maps_to_load(self):
        out = normalize_params({"free": False})
        assert out["action_type"] == "load"
        assert "free" not in out

    def test_string_truthiness(self):
        assert normalize_params({"free": "true"})["action_type"] == "free"
        assert normalize_params({"free": "false"})["action_type"] == "load"

    def test_explicit_action_type_wins(self):
        out = normalize_params({"free": True, "action_type": "smear"})
        assert out["action_type"] == "smear"

    def test_no_free_is_noop(self):
        out = normalize_params({"action_type": "smear", "mass": {"l": 0.1}})
        assert out["action_type"] == "smear"
        assert out["mass"] == {"l": 0.1}

    def test_free_under_preprocessor_is_seen(self):
        out = normalize_params({"_preprocessor": {"free": True}})
        assert out["action_type"] == "free"
        assert "_preprocessor" not in out


class TestBuildBaseGauge:
    def test_load_loads_all_links(self):
        out = build_base_gauge(_gauge_config(ActionType.LOAD))
        assert out.schedule == [
            "gauge",
            "gauge_smear_fat",
            "gauge_smear_long",
            "gauge_apbc",
        ]
        for name in ("gauge", "gauge_smear_fat", "gauge_smear_long"):
            assert out.modules[name]["id"]["type"] == "MIO::LoadIldg"
        assert out.modules["gauge_apbc"]["id"]["type"] == "MGauge::APBCGauge"

    def test_load_loads_correct_filestems(self):
        out = build_base_gauge(_gauge_config(ActionType.LOAD))
        assert out.modules["gauge"]["options"]["file"] == "lat/gauge_links"
        assert out.modules["gauge_smear_fat"]["options"]["file"] == "lat/fat_links"
        assert out.modules["gauge_smear_long"]["options"]["file"] == "lat/long_links"

    def test_free_smears_unit_gauge(self):
        out = build_base_gauge(_gauge_config(ActionType.FREE))
        # FREE uses a unit thin gauge but still routes it through HISQSmear so
        # that the KS phases and boundary are baked into the fat/long links.
        assert out.schedule == ["gauge", "gauge_smear", "gauge_apbc"]
        assert out.modules["gauge"]["id"]["type"] == "MGauge::Unit"
        smear = out.modules["gauge_smear"]
        assert smear["id"]["type"] == "MGauge::HISQSmear"
        assert smear["options"]["gauge"] == "gauge"
        # The fat/long links are outputs of the smear module, not top-level.
        assert "gauge_smear_fat" not in out.modules
        assert "gauge_smear_long" not in out.modules

    def test_smear_smears_thin_gauge(self):
        out = build_base_gauge(_gauge_config(ActionType.SMEAR))
        assert out.schedule == ["gauge", "gauge_smear", "gauge_apbc"]
        # The fat/long links are outputs of the smear module, not separate
        # top-level modules (they are referenced downstream as
        # gauge_smear_fat/gauge_smear_long).
        assert "gauge_smear_fat" not in out.modules
        assert "gauge_smear_long" not in out.modules
        assert out.modules["gauge"]["id"]["type"] == "MIO::LoadIldg"
        assert out.modules["gauge"]["options"]["file"] == "lat/gauge_links"
        smear = out.modules["gauge_smear"]
        assert smear["id"]["type"] == "MGauge::HISQSmear"
        assert smear["options"]["gauge"] == "gauge"
        assert out.modules["gauge_apbc"]["id"]["type"] == "MGauge::APBCGauge"

    def test_smear_save_smear_off_by_default(self):
        out = build_base_gauge(_gauge_config(ActionType.SMEAR))
        assert "save_fat" not in out.modules
        assert "save_long" not in out.modules

    def test_smear_save_smear_writes_fat_and_long(self):
        out = build_base_gauge(_gauge_config(ActionType.SMEAR, save_smear=True))
        assert out.schedule == [
            "gauge",
            "gauge_smear",
            "save_fat",
            "save_long",
            "gauge_apbc",
        ]
        fat = out.modules["save_fat"]
        long = out.modules["save_long"]
        assert fat["id"]["type"] == "MIO::SaveIldg"
        assert long["id"]["type"] == "MIO::SaveIldg"
        # Save the HISQSmear outputs (gauge_smear_fat/_long) to the outfiles.
        assert fat["options"]["gauge"] == "gauge_smear_fat"
        assert long["options"]["gauge"] == "gauge_smear_long"
        assert fat["options"]["fileStem"] == "lat/fat_links"
        assert long["options"]["fileStem"] == "lat/long_links"

    def test_save_smear_ignored_without_smear_action(self):
        # save_smear only makes sense when smearing; LOAD has nothing to save.
        out = build_base_gauge(_gauge_config(ActionType.LOAD, save_smear=True))
        assert "save_fat" not in out.modules
        assert "save_long" not in out.modules


class TestGaugeFileFormatRouting:
    """``format`` selects the reader module for every loaded gauge field."""

    def test_default_ildg_loads_via_loadildg(self):
        out = build_base_gauge(_gauge_config(ActionType.LOAD))
        for name in ("gauge", "gauge_smear_fat", "gauge_smear_long"):
            assert out.modules[name]["id"]["type"] == "MIO::LoadIldg"

    def test_milcv5_load_routes_all_links_to_loadmilc(self):
        out = build_base_gauge(
            _gauge_config(ActionType.LOAD, format=GaugeFileFormat.MILCV5)
        )
        for name in ("gauge", "gauge_smear_fat", "gauge_smear_long"):
            assert out.modules[name]["id"]["type"] == "MIO::LoadMilc"
        # Namestems are unchanged; only the reader module differs.
        assert out.modules["gauge"]["options"]["file"] == "lat/gauge_links"
        assert out.modules["gauge_smear_fat"]["options"]["file"] == "lat/fat_links"
        assert out.modules["gauge_smear_long"]["options"]["file"] == "lat/long_links"

    def test_milcv5_smear_routes_only_thin_gauge_to_loadmilc(self):
        out = build_base_gauge(
            _gauge_config(ActionType.SMEAR, format=GaugeFileFormat.MILCV5)
        )
        # The thin gauge is loaded with the MILC reader...
        assert out.modules["gauge"]["id"]["type"] == "MIO::LoadMilc"
        assert out.modules["gauge"]["options"]["file"] == "lat/gauge_links"
        # ...while fat/long are derived on the fly via HISQSmear (not loaded).
        assert "gauge_smear_fat" not in out.modules
        assert "gauge_smear_long" not in out.modules
        assert out.modules["gauge_smear"]["id"]["type"] == "MGauge::HISQSmear"

    def test_milcv5_free_uses_unit_gauge(self):
        # FREE never loads, so format is irrelevant.
        out = build_base_gauge(
            _gauge_config(ActionType.FREE, format=GaugeFileFormat.MILCV5)
        )
        assert out.modules["gauge"]["id"]["type"] == "MGauge::Unit"
        assert "gauge_smear_fat" not in out.modules


class TestBuildSpGauge:
    def test_load_casts_fat_and_long(self):
        out = build_sp_gauge(_gauge_config(ActionType.LOAD))
        assert out.schedule == ["gauge_smear_fatf", "gauge_smear_longf"]
        assert out.modules["gauge_smear_fatf"]["options"]["field"] == "gauge_smear_fat"
        assert out.modules["gauge_smear_longf"]["options"]["field"] == "gauge_smear_long"

    def test_smear_casts_smear_outputs(self):
        out = build_sp_gauge(_gauge_config(ActionType.SMEAR))
        assert out.schedule == ["gauge_smear_fatf", "gauge_smear_longf"]
        assert out.modules["gauge_smear_fatf"]["options"]["field"] == "gauge_smear_fat"
        assert out.modules["gauge_smear_longf"]["options"]["field"] == "gauge_smear_long"


class TestBuildActionModules:
    def test_load_dp_uses_fat_long_action(self):
        out = build_action_modules(_gauge_config(ActionType.LOAD), dp_masses=["l"])
        name = "stag_mass_l"
        assert out.schedule == [name]
        assert out.modules[name]["id"]["type"] == "MAction::ImprovedStaggeredMILC"
        assert out.modules[name]["options"]["gaugefat"] == "gauge_smear_fat"
        assert out.modules[name]["options"]["gaugelong"] == "gauge_smear_long"
        assert out.modules[name]["options"]["mass"] == "0.1"

    def test_load_sp_uses_fat_long_float(self):
        out = build_action_modules(_gauge_config(ActionType.LOAD), sp_masses=["l"])
        iname = "istag_mass_l"
        assert out.modules[iname]["id"]["type"] == "MAction::ImprovedStaggeredMILCF"
        assert out.modules[iname]["options"]["gaugefat"] == "gauge_smear_fatf"
        assert out.modules[iname]["options"]["gaugelong"] == "gauge_smear_longf"

    def test_smear_dp_uses_smear_outputs(self):
        out = build_action_modules(_gauge_config(ActionType.SMEAR), dp_masses=["l"])
        name = "stag_mass_l"
        # SMEAR now feeds ImprovedStaggered the HISQ-smeared fat/long links.
        assert out.modules[name]["id"]["type"] == "MAction::ImprovedStaggeredMILC"
        opts = out.modules[name]["options"]
        assert opts["gaugefat"] == "gauge_smear_fat"
        assert opts["gaugelong"] == "gauge_smear_long"
        assert opts["mass"] == "0.1"

    def test_smear_sp_uses_fat_long_float(self):
        out = build_action_modules(_gauge_config(ActionType.SMEAR), sp_masses=["l"])
        iname = "istag_mass_l"
        assert out.modules[iname]["id"]["type"] == "MAction::ImprovedStaggeredMILCF"
        assert out.modules[iname]["options"]["gaugefat"] == "gauge_smear_fatf"
        assert out.modules[iname]["options"]["gaugelong"] == "gauge_smear_longf"


class TestNormalizeIntegration:
    @staticmethod
    def _file_params():
        return {
            "home": "/tmp",
            "gauge_links": {"filestem": "lat/g", "good_size": 100},
            "fat_links": {"filestem": "lat/f", "good_size": 100},
            "long_links": {"filestem": "lat/l", "good_size": 100},
        }

    def test_legacy_free_true_builds_as_free(self):
        from pyfm.core.builder import build_config

        params = {
            "formatting": {},
            "logging_level": "info",
            "runid": "t",
            "mass": {"l": 0.1},
            "action_name": "stag_mass_{mass}",
            "_preprocessor": {"free": True},
        }
        cfg = build_config(GaugeConfig, params, self._file_params())
        assert cfg.action_type is ActionType.FREE

    def test_explicit_action_type_smear_builds(self):
        from pyfm.core.builder import build_config

        params = {
            "formatting": {},
            "logging_level": "info",
            "runid": "t",
            "mass": {"l": 0.1},
            "action_name": "stag_mass_{mass}",
            "_preprocessor": {"action_type": "smear"},
        }
        cfg = build_config(GaugeConfig, params, self._file_params())
        assert cfg.action_type is ActionType.SMEAR

    def test_format_string_coerced_via_from_dict(self):
        from pyfm.core.builder import build_config

        params = {
            "formatting": {},
            "logging_level": "info",
            "runid": "t",
            "mass": {"l": 0.1},
            "action_name": "stag_mass_{mass}",
            "format": "milcv5",
        }
        cfg = build_config(GaugeConfig, params, self._file_params())
        assert cfg.format is GaugeFileFormat.MILCV5

    def test_format_defaults_to_ildg_when_omitted(self):
        from pyfm.core.builder import build_config

        params = {
            "formatting": {},
            "logging_level": "info",
            "runid": "t",
            "mass": {"l": 0.1},
            "action_name": "stag_mass_{mass}",
        }
        cfg = build_config(GaugeConfig, params, self._file_params())
        assert cfg.format is GaugeFileFormat.ILDG
