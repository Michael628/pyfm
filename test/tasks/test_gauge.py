"""Unit tests for the GaugeConfig action_type refactor.

Covers ``ActionType.from_dict`` (including the ``SMEAR`` backward-compat
alias for ``SMEARILDG``), the legacy ``free`` -> ``action_type`` normalize
hook (including through the full build_config pipeline), the
``build_base_gauge`` / ``build_sp_gauge`` / ``build_action_modules`` branches
for FREE / LOAD / SMEARILDG / SMEARV5 (double + single precision), the
``save_ildg`` flag that writes the HISQ-smeared fat/long links to disk via
SaveIldg, and ``validate_config``'s required-Outfile checks.
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
    validate_config,
)


def _outfile(label: str) -> Outfile:
    return Outfile(filestem=f"lat/{label}", ext=".{cfg}", good_size=100)


_UNSET = object()


def _gauge_config(
    action_type: ActionType = ActionType.LOAD,
    save_ildg: bool = False,
    ildg_links=_UNSET,
    fat_links=_UNSET,
    long_links=_UNSET,
    v5_links=_UNSET,
) -> GaugeConfig:
    def _default_or(value, label):
        return _outfile(label) if value is _UNSET else value

    return GaugeConfig(
        formatting={},
        logging_level="info",
        runid="test",
        mass=MassDict.from_dict({"l": 0.1}),
        ildg_links=_default_or(ildg_links, "ildg_links"),
        long_links=_default_or(long_links, "long_links"),
        fat_links=_default_or(fat_links, "fat_links"),
        v5_links=_default_or(v5_links, "v5_links"),
        action_type=action_type,
        action_name="stag_mass_{mass}",
        save_ildg=save_ildg,
    )


class TestActionType:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("free", ActionType.FREE),
            ("FREE", ActionType.FREE),
            ("load", ActionType.LOAD),
            ("LOAD", ActionType.LOAD),
            ("smearildg", ActionType.SMEARILDG),
            ("SMEARILDG", ActionType.SMEARILDG),
            ("smearv5", ActionType.SMEARV5),
            ("SMEARV5", ActionType.SMEARV5),
            # Backward-compat alias: bare "smear" resolves to SMEARILDG.
            ("smear", ActionType.SMEARILDG),
            ("SMEAR", ActionType.SMEARILDG),
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

    def test_input_format_field_removed(self):
        # format is now implied by action_type (LOAD/SMEARILDG = ILDG,
        # SMEARV5 = MILC v5); there is no separate format field.
        names = {f.name for f in dataclasses.fields(GaugeConfig)}
        assert "input_format" not in names
        assert "format" not in names


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
        out = normalize_params({"free": True, "action_type": "smearildg"})
        assert out["action_type"] == "smearildg"

    def test_no_free_is_noop(self):
        out = normalize_params({"action_type": "smearildg", "mass": {"l": 0.1}})
        assert out["action_type"] == "smearildg"
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
        assert out.modules["gauge"]["options"]["file"] == "lat/ildg_links"
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

    def test_smearildg_smears_thin_gauge_loaded_as_ildg(self):
        out = build_base_gauge(_gauge_config(ActionType.SMEARILDG))
        assert out.schedule == ["gauge", "gauge_smear", "gauge_apbc"]
        # The fat/long links are outputs of the smear module, not separate
        # top-level modules (they are referenced downstream as
        # gauge_smear_fat/gauge_smear_long).
        assert "gauge_smear_fat" not in out.modules
        assert "gauge_smear_long" not in out.modules
        assert out.modules["gauge"]["id"]["type"] == "MIO::LoadIldg"
        assert out.modules["gauge"]["options"]["file"] == "lat/ildg_links"
        smear = out.modules["gauge_smear"]
        assert smear["id"]["type"] == "MGauge::HISQSmear"
        assert smear["options"]["gauge"] == "gauge"
        assert out.modules["gauge_apbc"]["id"]["type"] == "MGauge::APBCGauge"

    def test_smearv5_smears_thin_gauge_loaded_as_milc(self):
        out = build_base_gauge(_gauge_config(ActionType.SMEARV5))
        assert out.schedule == ["gauge", "gauge_smear", "gauge_apbc"]
        assert "gauge_smear_fat" not in out.modules
        assert "gauge_smear_long" not in out.modules
        assert out.modules["gauge"]["id"]["type"] == "MIO::LoadMilc"
        assert out.modules["gauge"]["options"]["file"] == "lat/v5_links"
        assert out.modules["gauge_smear"]["id"]["type"] == "MGauge::HISQSmear"

    def test_smear_save_ildg_off_by_default(self):
        out = build_base_gauge(_gauge_config(ActionType.SMEARILDG))
        assert "save_fat" not in out.modules
        assert "save_long" not in out.modules

    def test_smear_save_ildg_writes_fat_and_long(self):
        out = build_base_gauge(_gauge_config(ActionType.SMEARILDG, save_ildg=True))
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

    def test_save_ildg_ignored_without_smear_action(self):
        # save_ildg only makes sense when smearing; LOAD has nothing to save.
        out = build_base_gauge(_gauge_config(ActionType.LOAD, save_ildg=True))
        assert "save_fat" not in out.modules
        assert "save_long" not in out.modules

    def test_smearv5_save_ildg_also_saves_thin_gauge_as_ildg(self):
        # SMEARV5 reads the thin gauge as raw MILC v5; save_ildg must also
        # write it back out as ILDG (in addition to fat/long) so downstream
        # LOAD runs can consume it.
        out = build_base_gauge(_gauge_config(ActionType.SMEARV5, save_ildg=True))
        assert out.schedule == [
            "gauge",
            "gauge_smear",
            "save_gauge",
            "save_fat",
            "save_long",
            "gauge_apbc",
        ]
        save_gauge = out.modules["save_gauge"]
        assert save_gauge["id"]["type"] == "MIO::SaveIldg"
        assert save_gauge["options"]["gauge"] == "gauge"
        assert save_gauge["options"]["fileStem"] == "lat/ildg_links"

    def test_smearildg_save_ildg_does_not_resave_thin_gauge(self):
        # SMEARILDG already read the thin gauge as ILDG, so there is nothing
        # new to write for it.
        out = build_base_gauge(_gauge_config(ActionType.SMEARILDG, save_ildg=True))
        assert "save_gauge" not in out.modules


class TestBuildSpGauge:
    def test_load_casts_fat_and_long(self):
        out = build_sp_gauge(_gauge_config(ActionType.LOAD))
        assert out.schedule == ["gauge_smear_fatf", "gauge_smear_longf"]
        assert out.modules["gauge_smear_fatf"]["options"]["field"] == "gauge_smear_fat"
        assert out.modules["gauge_smear_longf"]["options"]["field"] == "gauge_smear_long"

    def test_smear_casts_smear_outputs(self):
        out = build_sp_gauge(_gauge_config(ActionType.SMEARILDG))
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
        out = build_action_modules(_gauge_config(ActionType.SMEARILDG), dp_masses=["l"])
        name = "stag_mass_l"
        # SMEAR* feeds ImprovedStaggered the HISQ-smeared fat/long links.
        assert out.modules[name]["id"]["type"] == "MAction::ImprovedStaggeredMILC"
        opts = out.modules[name]["options"]
        assert opts["gaugefat"] == "gauge_smear_fat"
        assert opts["gaugelong"] == "gauge_smear_long"
        assert opts["mass"] == "0.1"

    def test_smear_sp_uses_fat_long_float(self):
        out = build_action_modules(_gauge_config(ActionType.SMEARILDG), sp_masses=["l"])
        iname = "istag_mass_l"
        assert out.modules[iname]["id"]["type"] == "MAction::ImprovedStaggeredMILCF"
        assert out.modules[iname]["options"]["gaugefat"] == "gauge_smear_fatf"
        assert out.modules[iname]["options"]["gaugelong"] == "gauge_smear_longf"


class TestValidateConfig:
    def test_load_requires_all_three_links(self):
        for missing_field in ("ildg_links", "fat_links", "long_links"):
            cfg = _gauge_config(ActionType.LOAD, **{missing_field: None})
            with pytest.raises(ValueError, match=missing_field):
                validate_config(cfg)

    def test_load_with_all_links_passes(self):
        validate_config(_gauge_config(ActionType.LOAD))

    def test_smearildg_requires_ildg_links_only(self):
        cfg = _gauge_config(
            ActionType.SMEARILDG, ildg_links=None, fat_links=None, long_links=None
        )
        with pytest.raises(ValueError, match="ildg_links"):
            validate_config(cfg)

    def test_smearildg_with_ildg_links_passes(self):
        validate_config(
            _gauge_config(ActionType.SMEARILDG, fat_links=None, long_links=None)
        )

    def test_smearv5_requires_v5_links_only(self):
        cfg = _gauge_config(
            ActionType.SMEARV5, v5_links=None, fat_links=None, long_links=None
        )
        with pytest.raises(ValueError, match="v5_links"):
            validate_config(cfg)

    def test_smearv5_with_v5_links_passes(self):
        validate_config(
            _gauge_config(ActionType.SMEARV5, fat_links=None, long_links=None)
        )

    def test_free_requires_no_links(self):
        validate_config(
            _gauge_config(
                ActionType.FREE,
                ildg_links=None,
                fat_links=None,
                long_links=None,
                v5_links=None,
            )
        )

    def test_save_ildg_requires_fat_and_long_links(self):
        cfg = _gauge_config(
            ActionType.FREE, save_ildg=True, fat_links=None, long_links=None
        )
        with pytest.raises(ValueError, match="fat_links"):
            validate_config(cfg)

    def test_save_ildg_with_fat_and_long_passes(self):
        validate_config(_gauge_config(ActionType.FREE, save_ildg=True))

    def test_smearv5_save_ildg_also_requires_ildg_links(self):
        # save_ildg on SMEARV5 additionally writes the thin gauge back out as
        # ILDG, so ildg_links is required alongside fat_links/long_links.
        cfg = _gauge_config(ActionType.SMEARV5, save_ildg=True, ildg_links=None)
        with pytest.raises(ValueError, match="ildg_links"):
            validate_config(cfg)

    def test_smearv5_save_ildg_with_ildg_links_passes(self):
        validate_config(_gauge_config(ActionType.SMEARV5, save_ildg=True))

    def test_smearildg_save_ildg_does_not_require_ildg_links_extra(self):
        # SMEARILDG already required ildg_links to read the thin gauge; save_ildg
        # doesn't add a further requirement on it (no thin-gauge resave needed).
        validate_config(
            _gauge_config(ActionType.SMEARILDG, save_ildg=True)
        )


class TestNormalizeIntegration:
    @staticmethod
    def _file_params():
        return {
            "home": "/tmp",
            "ildg_links": {"filestem": "lat/g", "good_size": 100},
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

    def test_explicit_action_type_smearildg_builds(self):
        from pyfm.core.builder import build_config

        params = {
            "formatting": {},
            "logging_level": "info",
            "runid": "t",
            "mass": {"l": 0.1},
            "action_name": "stag_mass_{mass}",
            "_preprocessor": {"action_type": "smearildg"},
        }
        cfg = build_config(GaugeConfig, params, self._file_params())
        assert cfg.action_type is ActionType.SMEARILDG
