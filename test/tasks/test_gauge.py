"""Unit tests for the GaugeConfig action_type refactor.

Covers ``ActionType.from_dict``, the legacy ``free`` -> ``action_type``
normalize hook (including through the full build_config pipeline), and the
``build_base_gauge`` / ``build_sp_gauge`` / ``build_action_modules`` branches
for FREE / IMPROVED / HISQ (double + single precision).
"""
import dataclasses

import pytest

from pyfm.domain import MassDict, Outfile
from pyfm.tasks.hadrons.gauge import (
    ActionType,
    GaugeConfig,
    build_action_modules,
    build_base_gauge,
    build_sp_gauge,
    normalize_params,
)


def _outfile(label: str) -> Outfile:
    return Outfile(filestem=f"lat/{label}", ext=".{cfg}", good_size=100)


def _gauge_config(action_type: ActionType = ActionType.IMPROVED) -> GaugeConfig:
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
    )


class TestActionType:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("free", ActionType.FREE),
            ("FREE", ActionType.FREE),
            ("improved", ActionType.IMPROVED),
            ("IMPROVED", ActionType.IMPROVED),
            ("hisq", ActionType.HISQ),
            ("Hisq", ActionType.HISQ),
            ("HISQ", ActionType.HISQ),
        ],
    )
    def test_from_dict_resolves_members(self, raw, expected):
        assert ActionType.from_dict(raw) is expected

    def test_from_dict_rejects_unknown(self):
        with pytest.raises(ValueError):
            ActionType.from_dict("nope")

    def test_default_is_improved(self):
        assert _gauge_config().action_type is ActionType.IMPROVED

    def test_free_field_replaced_by_action_type(self):
        names = {f.name for f in dataclasses.fields(GaugeConfig)}
        assert "free" not in names
        assert "action_type" in names


class TestNormalize:
    def test_free_true_maps_to_free(self):
        out = normalize_params({"free": True})
        assert out["action_type"] == "free"
        assert "free" not in out

    def test_free_false_maps_to_improved(self):
        out = normalize_params({"free": False})
        assert out["action_type"] == "improved"
        assert "free" not in out

    def test_string_truthiness(self):
        assert normalize_params({"free": "true"})["action_type"] == "free"
        assert normalize_params({"free": "false"})["action_type"] == "improved"

    def test_explicit_action_type_wins(self):
        out = normalize_params({"free": True, "action_type": "hisq"})
        assert out["action_type"] == "hisq"

    def test_no_free_is_noop(self):
        out = normalize_params({"action_type": "hisq", "mass": {"l": 0.1}})
        assert out["action_type"] == "hisq"
        assert out["mass"] == {"l": 0.1}

    def test_free_under_preprocessor_is_seen(self):
        out = normalize_params({"_preprocessor": {"free": True}})
        assert out["action_type"] == "free"
        assert "_preprocessor" not in out


class TestBuildBaseGauge:
    def test_improved_loads_all_links(self):
        out = build_base_gauge(_gauge_config(ActionType.IMPROVED))
        assert out.schedule == ["gauge", "gauge_fat", "gauge_long", "gauge_apbc"]
        for name in ("gauge", "gauge_fat", "gauge_long"):
            assert out.modules[name]["id"]["type"] == "MIO::LoadIldg"
        assert out.modules["gauge_apbc"]["id"]["type"] == "MGauge::APBCGauge"

    def test_improved_loads_correct_filestems(self):
        out = build_base_gauge(_gauge_config(ActionType.IMPROVED))
        assert out.modules["gauge"]["options"]["file"] == "lat/gauge_links"
        assert out.modules["gauge_fat"]["options"]["file"] == "lat/fat_links"
        assert out.modules["gauge_long"]["options"]["file"] == "lat/long_links"

    def test_free_uses_unit_gauge(self):
        out = build_base_gauge(_gauge_config(ActionType.FREE))
        assert out.schedule == ["gauge", "gauge_fat", "gauge_long", "gauge_apbc"]
        for name in ("gauge", "gauge_fat", "gauge_long"):
            assert out.modules[name]["id"]["type"] == "MGauge::Unit"

    def test_hisq_skips_fat_and_long(self):
        out = build_base_gauge(_gauge_config(ActionType.HISQ))
        assert out.schedule == ["gauge", "gauge_apbc"]
        assert "gauge_fat" not in out.modules
        assert "gauge_long" not in out.modules
        assert out.modules["gauge"]["id"]["type"] == "MIO::LoadIldg"
        assert out.modules["gauge"]["options"]["file"] == "lat/gauge_links"
        assert out.modules["gauge_apbc"]["id"]["type"] == "MGauge::APBCGauge"


class TestBuildSpGauge:
    def test_improved_casts_fat_and_long(self):
        out = build_sp_gauge(_gauge_config(ActionType.IMPROVED))
        assert out.schedule == ["gauge_fatf", "gauge_longf"]
        assert out.modules["gauge_fatf"]["options"]["field"] == "gauge_fat"
        assert out.modules["gauge_longf"]["options"]["field"] == "gauge_long"

    def test_hisq_casts_thin_gauge(self):
        out = build_sp_gauge(_gauge_config(ActionType.HISQ))
        assert out.schedule == ["gauge_f"]
        assert out.modules["gauge_f"]["options"]["field"] == "gauge"
        assert "gauge_fatf" not in out.modules
        assert "gauge_longf" not in out.modules


class TestBuildActionModules:
    def test_improved_dp_uses_fat_long_action(self):
        out = build_action_modules(_gauge_config(ActionType.IMPROVED), dp_masses=["l"])
        name = "stag_mass_l"
        assert out.schedule == [name]
        assert out.modules[name]["id"]["type"] == "MAction::ImprovedStaggeredMILC"
        assert out.modules[name]["options"]["gaugefat"] == "gauge_fat"
        assert out.modules[name]["options"]["gaugelong"] == "gauge_long"
        assert out.modules[name]["options"]["mass"] == "0.1"

    def test_improved_sp_uses_fat_long_float(self):
        out = build_action_modules(_gauge_config(ActionType.IMPROVED), sp_masses=["l"])
        iname = "istag_mass_l"
        assert out.modules[iname]["id"]["type"] == "MAction::ImprovedStaggeredMILCF"
        assert out.modules[iname]["options"]["gaugefat"] == "gauge_fatf"
        assert out.modules[iname]["options"]["gaugelong"] == "gauge_longf"

    def test_hisq_dp_uses_thin_gauge(self):
        out = build_action_modules(_gauge_config(ActionType.HISQ), dp_masses=["l"])
        name = "stag_mass_l"
        assert out.modules[name]["id"]["type"] == "MAction::HighlyImprovedStaggeredMILC"
        opts = out.modules[name]["options"]
        assert opts["gauge"] == "gauge"
        assert "gaugefat" not in opts
        assert "gaugelong" not in opts
        assert opts["boundary"] == "1 1 1 -1"
        assert opts["mass"] == "0.1"

    def test_hisq_sp_uses_thin_gauge_float(self):
        out = build_action_modules(_gauge_config(ActionType.HISQ), sp_masses=["l"])
        iname = "istag_mass_l"
        assert out.modules[iname]["id"]["type"] == "MAction::HighlyImprovedStaggeredMILCF"
        assert out.modules[iname]["options"]["gauge"] == "gauge_f"
        assert "gaugefat" not in out.modules[iname]["options"]


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

    def test_explicit_action_type_hisq_builds(self):
        from pyfm.core.builder import build_config

        params = {
            "formatting": {},
            "logging_level": "info",
            "runid": "t",
            "mass": {"l": 0.1},
            "action_name": "stag_mass_{mass}",
            "_preprocessor": {"action_type": "hisq"},
        }
        cfg = build_config(GaugeConfig, params, self._file_params())
        assert cfg.action_type is ActionType.HISQ
