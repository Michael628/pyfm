"""Tests for lma_new.py — LMANewConfig composite + registry wiring."""

import dataclasses

import pytest

from pyfm.domain import MassDict, OpList, Outfile
from pyfm.nanny import write_input_file
from pyfm.tasks.hadrons.gauge import GaugeConfig, ActionType
from pyfm.tasks.hadrons.epack import EpackConfig
from pyfm.tasks.hadrons.meson import MesonConfig
from pyfm.tasks.hadrons.types import HighModeConfig
from pyfm.tasks.hadrons.lma_new import (
    LMANewConfig,
    build_input_params,
    normalize_params,
    postprocess_config,
    validate_config,
)
from pyfm.tasks.hadrons import lmi
from pyfm.tasks.register import get_task_handler, get_task_key, list_registered_types

MASS = MassDict.from_dict({"l": 0.002426})


def make_config(**hm_overrides):
    gauge_config = GaugeConfig(
        formatting={},
        logging_level="INFO",
        runid="test",
        mass=MASS,
        ildg_links=Outfile(filestem="g", ext=".ildg", good_size=1),
        fat_links=Outfile(filestem="f", ext=".ildg", good_size=1),
        long_links=Outfile(filestem="n", ext=".ildg", good_size=1),
        action_type=ActionType.LOAD,
        action_name="stag_mass_{mass}",
    )
    epack_config = EpackConfig(
        formatting={},
        logging_level="INFO",
        runid="test",
        mass=MASS,
        eigs=10,
        eig=Outfile(filestem="eig", ext=".h5", good_size=1),
        eigdir=Outfile(filestem="eigdir", ext=".h5", good_size=1),
        eval=Outfile(filestem="eval", ext=".h5", good_size=1),
        load=True,
        low_modes_name="evecs_mass_{mass}",
    )
    meson_config = MesonConfig(
        formatting={},
        logging_level="INFO",
        runid="test",
        action_name="stag_mass_{mass}",
        low_modes_name="evecs_mass_{mass}",
        mass=MASS,
        blocksize=200,
        operations=OpList.from_dict({"pion_local": {"mass": ["l"]}}),
        meson=Outfile(filestem="mf/mf_{mass}", ext=".h5", good_size=1),
    )
    hm_kwargs = dict(
        formatting={},
        logging_level="INFO",
        runid="test",
        mass=MASS,
        action_name="stag_mass_{mass}",
        solver_name="stag_{solver}_mass_{mass}",
        low_modes_name="evecs_mass_{mass}",
        operations=OpList.from_dict({"pion_local": {"mass": ["l"]}}),
        high_modes=Outfile(filestem="corr/corr_{tsource}", ext=".20.h5", good_size=1),
        tstart=0,
        tstop=0,
        dt=1,
        noise=1,
        time=4,
        skip_cg=True,
        overwrite=True,
    )
    hm_kwargs.update(hm_overrides)
    hm = HighModeConfig(**hm_kwargs)
    return LMANewConfig(
        formatting={},
        logging_level="INFO",
        runid="test",
        gauge_config=gauge_config,
        epack_config=epack_config,
        meson_config=[meson_config],
        high_modes_config=[hm],
        skip_epack=True,
    )


class TestRegistration:
    def test_hadrons_lma_new_is_registered(self):
        assert "nanny_hadrons_lma_new" in list_registered_types()

    def test_task_key_resolves_to_lma_new_handler(self):
        handler = get_task_handler(job_type="hadrons", task_type="lma_new")
        assert handler is not None
        assert handler.config_type is LMANewConfig

    def test_lmi_registration_untouched(self):
        # Confirms LMANewConfig didn't steal hadrons_lmi's registration via
        # the _config_to_task_key reverse-lookup guard.
        from pyfm.tasks.hadrons.lmi import LMIConfig

        assert get_task_key(config=LMIConfig) == "nanny_hadrons_lmi"
        assert get_task_key(config=LMANewConfig) == "nanny_hadrons_lma_new"


class TestBuildInputParams:
    def test_end_to_end_canonical_modules(self):
        config = make_config()
        result = build_input_params(config)
        assert result.modules["quark_ranLL_pion_local_mass_l_t0"]["id"]["type"] == (
            "MFermion::StagGaugeProp"
        )
        assert result.modules["corr_ranLL_pion_local_mass_l_t0"]["id"]["type"] == (
            "MContraction::StagMeson"
        )
        assert result.modules["mf_local_mass_l"]["id"]["type"] == (
            "MContraction::StagA2AMesonField"
        )
        assert any(
            mod["id"]["type"] == "MFermion::SpinTaste"
            for mod in result.modules.values()
        )
        # No Legacy types anywhere.
        assert not any(
            mod["id"]["type"].endswith("Legacy") for mod in result.modules.values()
        )

    def test_load_mode_is_permitted(self):
        # The exact scenario lmi.validate_config now rejects — LMANewConfig
        # must NOT reject it (validate_shared_config, not lmi.validate_config).
        stoch = Outfile(
            filestem="mesonfield/mf_{mass}", ext=".{cfg}/{gamma}_0_0_0.h5", good_size=1
        )
        config = make_config(low_mode_method="load", meson_stoch_proj=stoch)
        result = build_input_params(config)
        assert result.modules["quark_ranLL_pion_local_mass_l"]["id"]["type"] == (
            "MFermion::StagLMAMesonFieldProp"
        )


class TestBuildLhCache:
    def test_postprocess_config_synthesizes_meson_entry(self):
        stoch = Outfile(
            filestem="mesonfield/mf_{mass}", ext=".{cfg}/{gamma}_0_0_0.h5", good_size=1
        )
        config = make_config(low_mode_method="load", meson_stoch_proj=stoch)
        config = dataclasses.replace(config, build_lh_cache=True)

        result = postprocess_config(config)

        assert len(result.meson_config) == 2
        assert result.skip_meson is False
        synthesized = result.meson_config[-1]
        assert synthesized.meson is stoch
        assert synthesized.apply_g5 is True
        assert synthesized.high_right_name == "noise_fv_vec"
        assert synthesized.high_left_name == ""
        assert synthesized.operations is config.high_modes_config[0].operations

    def test_postprocess_config_noop_when_build_lh_cache_false(self):
        config = make_config()
        result = postprocess_config(config)
        assert result is config

    def test_postprocess_config_noop_when_no_load_mode_entries(self):
        config = make_config()
        config = dataclasses.replace(config, build_lh_cache=True)
        result = postprocess_config(config)
        assert result is config

    def test_validate_passes_when_build_lh_cache_synthesized(self):
        stoch = Outfile(
            filestem="mesonfield/mf_{mass}", ext=".{cfg}/{gamma}_0_0_0.h5", good_size=1
        )
        config = make_config(low_mode_method="load", meson_stoch_proj=stoch)
        config = dataclasses.replace(config, build_lh_cache=True, skip_epack=False)
        config = postprocess_config(config)
        validate_config(config)  # must not raise

    def test_validate_passes_for_load_only_stage(self):
        # The two-stage build-then-load workflow this feature exists for:
        # a prior job already wrote the load-cache files (build_lh_cache=True
        # there), this job just loads them (build_lh_cache=False,
        # skip_meson=True — no tasks.meson block at all). validate_config
        # must not treat "no meson entries in *this* job" as a
        # misconfiguration.
        stoch = Outfile(
            filestem="mesonfield/mf_{mass}", ext=".{cfg}/{gamma}_0_0_0.h5", good_size=1
        )
        config = make_config(low_mode_method="load", meson_stoch_proj=stoch)
        config = dataclasses.replace(
            config, meson_config=[], skip_meson=True, skip_epack=False
        )
        validate_config(config)  # must not raise

    def test_validate_rejects_missing_meson_stoch_proj_regardless_of_build_lh_cache(
        self,
    ):
        # The one invariant that's always wrong: a load-mode entry with no
        # files entry to read the (this-job- or prior-job-produced) cache
        # from. Checked unconditionally, not just under build_lh_cache=True.
        config = make_config(low_mode_method="load")
        config = dataclasses.replace(
            config, meson_config=[], skip_meson=True, skip_epack=False
        )
        with pytest.raises(ValueError, match="meson_stoch_proj"):
            validate_config(config)


class TestCacheOnlyPathway:
    STOCH = Outfile(
        filestem="mesonfield/mf_{mass}", ext=".{cfg}/{gamma}_0_0_0.h5", good_size=1
    )

    def test_postprocess_config_sets_cache_only_when_skip_high_modes(self):
        config = make_config(low_mode_method="load", meson_stoch_proj=self.STOCH)
        config = dataclasses.replace(config, build_lh_cache=True, skip_high_modes=True)
        result = postprocess_config(config)
        assert result.high_modes_config[0].cache_only is True

    def test_postprocess_config_leaves_cache_only_false_without_skip_high_modes(self):
        config = make_config(low_mode_method="load", meson_stoch_proj=self.STOCH)
        config = dataclasses.replace(config, build_lh_cache=True)
        result = postprocess_config(config)
        assert result.high_modes_config[0].cache_only is False

    def test_build_input_params_emits_noise_and_writer_without_quarks(self):
        config = make_config(low_mode_method="load", meson_stoch_proj=self.STOCH)
        config = dataclasses.replace(config, build_lh_cache=True, skip_high_modes=True)
        config = postprocess_config(config)
        result = build_input_params(config)
        assert "noise_fv" in result.modules
        assert "mf_local_mass_l" in result.modules
        assert not any(name.startswith("quark_") for name in result.modules)
        assert not any(
            mod["id"]["type"] == "MContraction::StagMeson"
            for mod in result.modules.values()
        )

    def test_validate_rejects_hand_set_cache_only_on_compute_mode_entry(self):
        # cache_only routes straight from YAML like any other HighModeConfig
        # field (lmi.route_params's high_modes_defaults | entry layering) —
        # validate_config must reject a hand-set cache_only=True that
        # doesn't match postprocess_config's own _needs_cache predicate.
        config = make_config(cache_only=True)
        config = dataclasses.replace(config, skip_epack=False)
        with pytest.raises(ValueError, match="cache_only"):
            validate_config(config)

    def test_validate_rejects_hand_set_cache_only_with_skip_low_modes(self):
        config = make_config(
            low_mode_method="load", meson_stoch_proj=self.STOCH,
            skip_low_modes=True, cache_only=True,
        )
        config = dataclasses.replace(config, skip_epack=False)
        with pytest.raises(ValueError, match="cache_only"):
            validate_config(config)

    def test_validate_passes_cache_only_from_postprocess_config(self):
        config = make_config(low_mode_method="load", meson_stoch_proj=self.STOCH)
        config = dataclasses.replace(
            config, build_lh_cache=True, skip_high_modes=True, skip_epack=False
        )
        config = postprocess_config(config)
        validate_config(config)  # must not raise

    def test_entries_not_needing_cache_stay_fully_skipped(self):
        config = make_config()
        config = dataclasses.replace(config, skip_high_modes=True)
        result = build_input_params(config)
        assert "sink" not in result.modules
        assert "stag_ranLL_mass_l" not in result.modules

    def test_aggregator_empty_for_cache_only_job(self):
        # lma_new reuses lmi.build_aggregator_params verbatim (register_task,
        # lma_new.py:322) — skip_high_modes=True (set alongside
        # cache_only by postprocess_config) means no correlator files are
        # ever scheduled, so aggregation must produce nothing rather than
        # describing files that don't exist.
        config = make_config(low_mode_method="load", meson_stoch_proj=self.STOCH)
        config = dataclasses.replace(config, build_lh_cache=True, skip_high_modes=True)
        config = postprocess_config(config)

        assert lmi.build_aggregator_params(config, average=False) == {}


class TestNormalizeParamsCacheOnly:
    def test_preserves_skip_high_modes_when_build_lh_cache_set(self):
        raw = {
            "build_lh_cache": True,
            "skip_high_modes": True,
            "_preprocessor": {"high_modes": {"gamma": ["pion_local"], "mass": ["l"]}},
        }
        result = normalize_params(raw)
        assert result["skip_high_modes"] is True
        assert result["_preprocessor"]["high_modes"]

    def test_pass_through_without_build_lh_cache(self):
        raw = {
            "skip_high_modes": True,
            "_preprocessor": {"high_modes": {"gamma": ["pion_local"], "mass": ["l"]}},
        }
        result = normalize_params(raw)
        assert result == lmi.normalize_params(raw)
        assert result["skip_high_modes"] is False

    def test_pass_through_without_wanting_skip_high_modes(self):
        raw = {
            "build_lh_cache": True,
            "_preprocessor": {"high_modes": {"gamma": ["pion_local"], "mass": ["l"]}},
        }
        result = normalize_params(raw)
        assert result == lmi.normalize_params(raw)
        assert result["skip_high_modes"] is False


class TestSplitMpiLayout:
    def test_none_when_unset(self):
        config = make_config()
        assert config.split_mpi_layout is None

    def test_single_entry_layout(self):
        config = make_config(split_mpi_layout="1.1.1.2")
        assert config.split_mpi_layout == "1.1.1.2"


def test_generate_lma_new_input_end_to_end(tmp_path, monkeypatch, hadrons_params):
    """Full write_input_file dispatch for job_type=hadrons, task_type=lma_new."""
    monkeypatch.chdir(tmp_path)
    hadrons_params["job_setup"]["lma_new_test"] = dict(hadrons_params["job_setup"]["lma"])
    hadrons_params["job_setup"]["lma_new_test"]["task_type"] = "lma_new"
    hadrons_params["job_setup"]["lma_new_test"]["io"] = "full-lma-new"
    hadrons_params["submit"]["resources"]["lma_new_test"] = hadrons_params["submit"][
        "resources"
    ]["lma"]

    write_input_file("lma_new_test", hadrons_params, "a", "20")

    xml = (tmp_path / "in" / "full-lma-new-a.20.xml").read_text()
    assert "<type>MFermion::SpinTaste</type>" in xml
    assert "<type>MFermion::StagGaugeProp</type>" in xml
    assert "<type>MFermion::StagGaugePropLegacy</type>" not in xml
    assert "<type>MContraction::StagMeson</type>" in xml
    assert "<type>MContraction::StagMesonLegacy</type>" not in xml


def test_cache_only_reachable_from_yaml_job_config(hadrons_params):
    """build_lh_cache=True + skip_high_modes=True must survive routing even
    with a real (non-empty) tasks.high_modes block — the reachability gap
    lmi.normalize_params has for this composite-only flag combination."""
    from pyfm.nanny.taskbuilder import create_task

    hadrons_params["job_setup"]["lma_new_test"] = dict(
        hadrons_params["job_setup"]["lma"]
    )
    hadrons_params["job_setup"]["lma_new_test"]["task_type"] = "lma_new"
    hadrons_params["job_setup"]["lma_new_test"]["params"] = dict(
        hadrons_params["job_setup"]["lma"].get("params", {}),
        build_lh_cache=True,
        skip_high_modes=True,
    )
    hadrons_params["submit"]["resources"]["lma_new_test"] = hadrons_params["submit"][
        "resources"
    ]["lma"]

    task = create_task("lma_new_test", hadrons_params, "a", "20")

    assert task.config.skip_high_modes is True
    assert len(task.config.high_modes_config) == 1
