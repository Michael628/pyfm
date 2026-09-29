"""Tests for highmode_v2/strategy.py — canonical-schema load-mode chain and
top-level dispatch."""

import pytest

from pyfm.domain import Gamma, MassDict, OpList, Outfile
from pyfm.tasks.hadrons.types import CorrelatorStrategy, HighModeConfig
from pyfm.tasks.hadrons.highmode_v2 import strategy, twopoint

MESON_STOCH_PROJ = Outfile(
    filestem="mesonfield/mf_{mass}", ext=".{cfg}/{gamma}_0_0_0.h5", good_size=1
)


def make_config(**overrides):
    kwargs = dict(
        formatting={},
        logging_level="INFO",
        runid="test",
        mass=MassDict.from_dict({"l": 0.002426}),
        action_name="stag_mass_{mass}",
        solver_name="stag_{solver}_mass_{mass}",
        low_modes_name="evecs_mass_{mass}",
        operations=OpList.from_dict({"pion_local": {"mass": ["l"]}}),
        high_modes=Outfile(filestem="corr/corr_{tsource}", ext=".20.h5", good_size=1),
        tstart=0,
        tstop=3,
        dt=1,
        noise=1,
        time=4,
        skip_cg=True,
        shift_gauge_name="gauge_apbc",
        meson_stoch_proj=MESON_STOCH_PROJ,
    )
    kwargs.update(overrides)
    return HighModeConfig(**kwargs)


class TestBuildLmaMesonFieldChain:
    def test_producer_references_shared_spintaste_with_required_labels(self):
        config = make_config()
        _, names = twopoint.build_spintaste_modules(config)
        result = strategy.build_lma_meson_field_chain(
            config,
            mass_label="l",
            action="stag_mass_l",
            low_modes="evecs_mass_l",
            gammas=[Gamma.PION_LOCAL],
            spintaste_names=names,
        )
        producer = result.modules["quark_ranLL_pion_local_mass_l"]
        assert producer["id"]["type"] == "MFermion::StagLMAMesonFieldProp"
        assert producer["options"]["gammas"] == names[Gamma.PION_LOCAL]
        assert producer["options"]["labels"] == "G5_G5"
        assert producer["options"]["mesonField"] == "mfload_mass_l_G1_G1"

    def test_no_writer_or_cbpairs_modules(self):
        # The writer (cbpairs + SpinTaste + meson_field_v2) moved entirely
        # out of highmode_v2 — this function only loads and produces.
        config = make_config()
        _, names = twopoint.build_spintaste_modules(config)
        result = strategy.build_lma_meson_field_chain(
            config,
            mass_label="l",
            action="stag_mass_l",
            low_modes="evecs_mass_l",
            gammas=[Gamma.PION_LOCAL],
            spintaste_names=names,
        )
        assert "mfwrite_mass_l" not in result.modules
        assert "cbpairs_l_mass_l" not in result.modules
        assert "cbpairs_r_mass_l" not in result.modules
        assert "spintaste_mfwrite_mass_l" not in result.modules
        assert "mfload_mass_l_G1_G1" in result.modules
        assert "quark_ranLL_pion_local_mass_l" in result.modules


class TestBuildInputParamsV2LoadMode:
    def test_end_to_end_load_mode_shape(self):
        config = make_config(
            low_mode_method="load",
            meson_stoch_proj=MESON_STOCH_PROJ,
            overwrite=True,
            skip_cg=False,
        )
        result = strategy.build_input_params(config)
        assert result.modules["quark_ranLL_pion_local_mass_l"]["id"]["type"] == (
            "MFermion::StagLMAMesonFieldProp"
        )
        assert all(
            mod["id"]["type"] != "MSolver::StagLMA"
            for mod in result.modules.values()
        )
        # Contractions reference the shared SpinTaste module, not an inline dict.
        for name, mod in result.modules.items():
            if mod["id"]["type"] == "MContraction::StagMeson":
                assert "sourceGammas" not in mod["options"]
                assert "sinkSpinTaste" not in mod["options"]

    def test_shared_spintaste_module_precedes_its_consumers(self):
        config = make_config(
            low_mode_method="load",
            meson_stoch_proj=MESON_STOCH_PROJ,
            overwrite=True,
            skip_cg=False,
        )
        result = strategy.build_input_params(config)
        base_idx = result.schedule.index("spintaste_pion_local")
        assert base_idx < result.schedule.index("quark_ranLL_pion_local_mass_l")
        assert base_idx < result.schedule.index("corr_ranLL_pion_local_mass_l_t0")

    def test_noise_module_name_is_config_driven(self):
        config = make_config(
            low_mode_method="load",
            meson_stoch_proj=MESON_STOCH_PROJ,
            overwrite=True,
            skip_cg=False,
            noise_name="custom_noise",
        )
        result = strategy.build_input_params(config)
        assert "custom_noise" in result.modules
        assert result.modules["custom_noise"]["id"]["type"] == (
            "MNoise::StagFullVolumeSpinColorDiagonal"
        )
        producer = result.modules["quark_ranLL_pion_local_mass_l"]
        assert producer["options"]["noise"] == "custom_noise_vec"


class TestBuildInputParamsV2ComputeMode:
    def test_compute_mode_unchanged_shape(self):
        config = make_config(overwrite=True, skip_cg=False)
        result = strategy.build_input_params(config)
        assert "noise_fv" not in result.modules
        assert (
            result.modules["stag_ranLL_mass_l"]["id"]["type"] == "MSolver::StagLMA"
        )
        quark = result.modules["quark_ranLL_pion_local_mass_l_t0"]
        assert quark["id"]["type"] == "MFermion::StagGaugeProp"
        assert quark["options"]["gammas"] == "spintaste_pion_local"


class TestStrategyDispatch:
    def test_sib_strategy_rejected(self):
        config = make_config(correlator_strategy=CorrelatorStrategy.SIB)
        _, names = twopoint.build_spintaste_modules(config)
        with pytest.raises(ValueError, match="TWOPOINT"):
            strategy.build_quark_strategy(config, config.source_refs, names)
        with pytest.raises(ValueError, match="TWOPOINT"):
            strategy.build_contract_strategy(config, config.source_refs, names)
