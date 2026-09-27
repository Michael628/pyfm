"""Tests for lma_new.py — LMANewConfig composite + registry wiring."""

import pytest

from pyfm.domain import MassDict, OpList, Outfile
from pyfm.nanny import write_input_file
from pyfm.tasks.hadrons.gauge import GaugeConfig, ActionType
from pyfm.tasks.hadrons.epack import EpackConfig
from pyfm.tasks.hadrons.meson import MesonConfig
from pyfm.tasks.hadrons.types import HighModeConfig
from pyfm.tasks.hadrons.lma_new import LMANewConfig, build_input_params
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
        meson_config=meson_config,
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
