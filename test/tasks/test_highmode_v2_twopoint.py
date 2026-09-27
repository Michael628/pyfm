"""Tests for highmode_v2/twopoint.py — canonical-schema quark/contraction
module emission, including the SpinTaste-module dedup and axial-gamma
labels-override reuse trick."""

from pyfm.domain import Gamma, MassDict, OpList, Outfile
from pyfm.tasks.hadrons.types import HighModeConfig, SourceRef
from pyfm.tasks.hadrons.highmode_v2.twopoint import (
    build_spintaste_modules,
    build_quarks,
    build_contractions,
)

REF = SourceRef(label="t0", axis="0", t0=0)


def make_config(**overrides):
    kwargs = dict(
        formatting={},
        logging_level="INFO",
        runid="test",
        mass=MassDict.from_dict({"l": 0.01}),
        action_name="stag_mass_{mass}",
        solver_name="stag_{solver}_mass_{mass}",
        low_modes_name="evecs_mass_{mass}",
        operations=OpList.from_dict({"vec_local": {"mass": ["l"]}}),
        high_modes=Outfile(
            filestem="corr/corr_{gamma_label}_{dset}_m{mass}_t{tsource}", ext=".h5", good_size=1
        ),
        tstart=0,
        tstop=0,
        dt=1,
        noise=1,
        time=4,
        skip_cg=True,
        shift_gauge_name="gauge_apbc",
    )
    kwargs.update(overrides)
    return HighModeConfig(**kwargs)


class TestBuildSpintasteModules:
    def test_one_module_per_solved_gamma(self):
        config = make_config()
        modules, names = build_spintaste_modules(config)
        # vec_local (quark) + pion_local (antiquark, g5-hermiticity partner)
        assert set(names.keys()) == {Gamma.VEC_LOCAL, Gamma.PION_LOCAL}
        assert names[Gamma.VEC_LOCAL] == "spintaste_vec_local"
        assert modules["spintaste_vec_local"]["options"]["spinTaste"]["gammas"] == (
            "(GX GX) (GY GY) (GZ GZ)"
        )
        assert modules["spintaste_vec_local"]["options"]["labels"] == ""

    def test_axial_gamma_gets_own_module_with_labels_override(self):
        config = make_config(
            operations=OpList.from_dict(
                {"vec_local": {"mass": ["l"]}, "axial_vec_local": {"mass": ["l"]}}
            )
        )
        modules, names = build_spintaste_modules(config)
        assert Gamma.AXIAL_VEC_LOCAL in names
        axial_module = modules[names[Gamma.AXIAL_VEC_LOCAL]]
        # Own physics: its own raw gamma string, same apply_g5 convention.
        assert axial_module["options"]["spinTaste"]["gammas"] == (
            "(G5X G5X) (G5Y G5Y) (G5Z G5Z)"
        )
        assert axial_module["options"]["spinTaste"]["applyG5"] == "true"
        # Labels overridden to the SHARED quark-side module's own labels.
        assert axial_module["options"]["labels"] == "GX_GX GY_GY GZ_GZ"
        # The base (non-axial) module is untouched — no override.
        assert modules[names[Gamma.VEC_LOCAL]]["options"]["labels"] == ""

    def test_nonlocal_gamma_threads_shift_gauge_name(self):
        config = make_config(operations=OpList.from_dict({"vec_onelink": {"mass": ["l"]}}))
        modules, names = build_spintaste_modules(config)
        onelink_module = modules[names[Gamma.VEC_ONELINK]]
        assert onelink_module["options"]["spinTaste"]["gauge"] == "gauge_apbc"

    def test_dedup_is_idempotent_across_repeated_gammas(self):
        # pion_local and scalar_local (IDENTITY) both independently requested
        # alongside vec_local/axial_vec_local — same antiquark gammas as the
        # ones quark_gen would derive; the module set must not duplicate.
        config = make_config(
            operations=OpList.from_dict(
                {
                    "vec_local": {"mass": ["l"]},
                    "axial_vec_local": {"mass": ["l"]},
                    "pion_local": {"mass": ["l"]},
                }
            )
        )
        modules, names = build_spintaste_modules(config)
        assert len(modules) == len(names)  # one module per distinct Gamma key


class TestBuildQuarksV2:
    def test_shared_propagator_for_direct_and_axial(self):
        config = make_config(
            operations=OpList.from_dict(
                {"vec_local": {"mass": ["l"]}, "axial_vec_local": {"mass": ["l"]}}
            )
        )
        _, names = build_spintaste_modules(config)
        result = build_quarks(config, [REF], names)
        # Exactly one VEC_LOCAL solve — shared by both correlators.
        assert "quark_ranLL_vec_local_mass_l_t0" in result.modules
        vec_quark = result.modules["quark_ranLL_vec_local_mass_l_t0"]
        assert vec_quark["id"]["type"] == "MFermion::StagGaugeProp"
        assert vec_quark["options"]["gammas"] == "spintaste_vec_local"
        # No separate axial-labeled quark solve exists.
        assert not any("axial" in name for name in result.modules)

    def test_load_mode_skips_ranll_gauge_prop(self):
        config = make_config(low_mode_method="load")
        _, names = build_spintaste_modules(config)
        result = build_quarks(config, [REF], names)
        assert not any(n.startswith("quark_ranLL_") for n in result.modules)


class TestBuildContractionsV2:
    def test_direct_and_axial_reference_correct_spintaste_and_antiquark(self):
        config = make_config(
            operations=OpList.from_dict(
                {"vec_local": {"mass": ["l"]}, "axial_vec_local": {"mass": ["l"]}}
            )
        )
        _, names = build_spintaste_modules(config)
        result = build_contractions(config, [REF], names)

        direct = result.modules["corr_ranLL_vec_local_mass_l_t0"]
        assert direct["id"]["type"] == "MContraction::StagMeson"
        assert direct["options"]["sinkGammas"] == "spintaste_vec_local"
        assert direct["options"]["source"] == "quark_ranLL_vec_local_mass_l_t0"
        # sink is now a GammaMapElement bridge, not the raw antiquark
        # TGammaMap propagator — bypasses checkKeys for the sink side.
        assert direct["options"]["sink"] == "quark_ranLL_pion_local_mass_l_t0_elem"
        elem = result.modules["quark_ranLL_pion_local_mass_l_t0_elem"]
        assert elem["id"]["type"] == "MUtilities::GammaMapElement"
        assert elem["options"]["map"] == "quark_ranLL_pion_local_mass_l_t0"
        assert elem["options"]["label"] == "G5_G5"

        axial = result.modules["corr_ranLL_axial_vec_local_mass_l_t0"]
        assert axial["options"]["sinkGammas"] == "spintaste_axial_vec_local"
        # Same shared quark-side propagator, DIFFERENT antiquark (G1_G1 vs G5_G5).
        assert axial["options"]["source"] == "quark_ranLL_vec_local_mass_l_t0"
        assert axial["options"]["sink"] == "quark_ranLL_scalar_local_mass_l_t0_elem"
        axial_elem = result.modules["quark_ranLL_scalar_local_mass_l_t0_elem"]
        assert axial_elem["options"]["map"] == "quark_ranLL_scalar_local_mass_l_t0"
        assert axial_elem["options"]["label"] == "G1_G1"
        assert "sourceGammas" not in axial["options"]
        assert "sinkSpinTaste" not in axial["options"]

    def test_antiquark_element_deduped_across_ops_sharing_antiquark(self):
        config = make_config(
            operations=OpList.from_dict(
                {"vec_local": {"mass": ["l"]}, "fourvec_local": {"mass": ["l"]}}
            )
        )
        _, names = build_spintaste_modules(config)
        result = build_contractions(config, [REF], names)
        assert (
            len([n for n in result.modules if n.endswith("_elem")]) == 1
        )  # both share the pion_local antiquark
