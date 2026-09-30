"""Tests for highmode_v2/twopoint.py — own op algebra, precon-driven
quark_gen, label-prefixed module emission (D4/D6/D8)."""

import pytest

from pyfm.domain import Gamma, MassDict, OpList, Outfile
from pyfm.tasks.hadrons.types import SolveCrossTerms, SourceRef
from pyfm.tasks.hadrons.highmode_v2.config import (
    BiasedSourceConfig,
    CgConfig,
    GridSourceConfig,
    LMAHighModeConfig,
    LowModeMethod,
    LowModesConfig,
    MesonFieldConfig,
    OutputConfig,
    PreconMode,
    SourcesConfig,
)
from pyfm.tasks.hadrons.highmode_v2.twopoint import (
    TwoPointOp,
    build_spintaste_modules,
    build_quarks,
    build_contractions,
    quark_gen,
)
from pyfm.tasks.hadrons.highmode_v2.twopoint import contraction_gen

REF = SourceRef(label="t0", axis="0", t0=0)
BASE = {"formatting": {}, "logging_level": "INFO", "runid": "test"}


def sub_kwargs(**extra):
    return BASE | extra


def make_config(label="", **overrides):
    kwargs = dict(
        **BASE,
        label=label,
        mass=MassDict.from_dict({"l": 0.01}),
        action_name="stag_mass_{mass}",
        solver_name="stag_{solver}_mass_{mass}",
        low_modes_name="evecs_mass_{mass}",
        operations=OpList.from_dict({"vec_local": {"mass": ["l"]}}),
        sources_config=SourcesConfig(
            **sub_kwargs(
                time=4,
                grid_config=GridSourceConfig(**sub_kwargs(tstart=0, tstop=0, dt=1)),
            )
        ),
        low_modes_config=LowModesConfig(**sub_kwargs(method=LowModeMethod.SOLVE)),
        cg_config=CgConfig(**sub_kwargs()),
        output_config=OutputConfig(
            **sub_kwargs(
                file=Outfile(
                    filestem="corr/corr_{gamma_label}_{dset}_m{mass}_t{tsource}",
                    ext=".h5",
                    good_size=1,
                )
            )
        ),
        shift_gauge_name="gauge_apbc",
    )
    kwargs.update(overrides)
    return LMAHighModeConfig(**kwargs)


def meson_field_low_modes():
    return LowModesConfig(
        **sub_kwargs(
            method=LowModeMethod.MESON_FIELD,
            meson_field_config=MesonFieldConfig(
                **sub_kwargs(
                    file=Outfile(
                        filestem="mesonfield/mf_{mass}",
                        ext=".{cfg}/{gamma}_0_0_0.h5",
                        good_size=1,
                    )
                )
            ),
        )
    )


class TestContractionGen:
    def test_axial_pairs_with_identity_reusing_vec_solve(self):
        config = make_config(
            operations=OpList.from_dict(
                {"vec_local": {"mass": ["l"]}, "axial_vec_local": {"mass": ["l"]}}
            )
        )
        pairs = {op.gamma.name: con for op, con in contraction_gen(config)}
        assert pairs["AXIAL_VEC_LOCAL"].antiquark.gamma is Gamma.IDENTITY
        assert pairs["AXIAL_VEC_LOCAL"].quark.gamma is Gamma.VEC_LOCAL
        assert pairs["VEC_LOCAL"].antiquark.gamma is Gamma.PION_LOCAL


class TestQuarkGen:
    def test_chain_precon_loosest_guesses_ranll(self):
        config = make_config()
        ops = {op.solver: op for op in quark_gen(config)}
        assert ops["ranLL"].precon is None
        assert ops["ama"].precon == "ranLL"

    def test_chain_multi_residual_chains_previous_solve(self):
        config = make_config(cg_config=CgConfig(**sub_kwargs(residual=[1e-4, 1e-8])))
        ops = {op.solver: op for op in quark_gen(config)}
        assert ops["ama_0.0001"].precon == "ranLL"
        assert ops["ama_1e-08"].precon == "ama_0.0001"

    def test_each_mode_every_solve_guesses_ranll(self):
        config = make_config(
            cg_config=CgConfig(**sub_kwargs(precon=PreconMode.EACH, residual=[1e-4, 1e-8]))
        )
        ops = {op.solver: op for op in quark_gen(config)}
        assert ops["ama_0.0001"].precon == "ranLL"
        assert ops["ama_1e-08"].precon == "ranLL"

    def test_none_mode_no_guesses(self):
        config = make_config(cg_config=CgConfig(**sub_kwargs(precon=PreconMode.NONE)))
        ops = {op.solver: op for op in quark_gen(config)}
        assert ops["ama"].precon is None

    def test_tiered_drops_dead_cg_solves(self):
        config = make_config(
            operations=OpList.from_dict(
                {"pion_local": {"mass": ["l"]}, "vec_local": {"mass": ["l"]}}
            ),
            output_config=OutputConfig(
                **sub_kwargs(
                    file=Outfile(filestem="corr", ext=".h5", good_size=1),
                    solve_cross_terms=SolveCrossTerms.TIERED,
                )
            ),
        )
        solvers = {op.solver for op in quark_gen(config)}
        # op-gamma CG solve has no consumer under TIERED; contract-gamma CG persists
        assert "ama" in solvers
        assert {op.gamma for op in quark_gen(config) if op.solver == "ama"} == {
            Gamma.PION_LOCAL
        }
        # chain falls back to the nearest earlier EMITTED solver: ranLL
        ama = next(op for op in quark_gen(config) if op.solver == "ama")
        assert ama.precon == "ranLL"

    def test_without_low_modes_chain_gives_no_guess(self):
        config = make_config(
            low_modes_config=LowModesConfig(**sub_kwargs(method=LowModeMethod.NONE)),
            cg_config=CgConfig(**sub_kwargs(precon=PreconMode.NONE)),
        )
        ops = {op.solver: op for op in quark_gen(config)}
        assert set(ops) == {"ama"}
        assert ops["ama"].precon is None


class TestBuildSpintasteModules:
    def test_one_module_per_solved_gamma_legacy_names(self):
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
        assert axial_module["options"]["spinTaste"]["gammas"] == (
            "(G5X G5X) (G5Y G5Y) (G5Z G5Z)"
        )
        assert axial_module["options"]["spinTaste"]["applyG5"] == "true"
        assert axial_module["options"]["labels"] == "GX_GX GY_GY GZ_GZ"
        assert modules[names[Gamma.VEC_LOCAL]]["options"]["labels"] == ""

    def test_nonlocal_gamma_threads_shift_gauge_name(self):
        config = make_config(
            operations=OpList.from_dict({"vec_onelink": {"mass": ["l"]}})
        )
        modules, names = build_spintaste_modules(config)
        onelink_module = modules[names[Gamma.VEC_ONELINK]]
        assert onelink_module["options"]["spinTaste"]["gauge"] == "gauge_apbc"

    def test_dedup_is_idempotent_across_repeated_gammas(self):
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
        assert len(modules) == len(names)

    def test_labeled_entry_prefixes_spintaste_names(self):
        config = make_config(label="sloppy")
        modules, names = build_spintaste_modules(config)
        assert names[Gamma.VEC_LOCAL] == "sloppy_spintaste_vec_local"
        assert "sloppy_spintaste_vec_local" in modules
        assert "spintaste_vec_local" not in modules


class TestBuildQuarksV2:
    def test_shared_propagator_for_direct_and_axial(self):
        config = make_config(
            operations=OpList.from_dict(
                {"vec_local": {"mass": ["l"]}, "axial_vec_local": {"mass": ["l"]}}
            )
        )
        _, names = build_spintaste_modules(config)
        result = build_quarks(config, [REF], names)
        assert "quark_ranLL_vec_local_mass_l_t0" in result.modules
        vec_quark = result.modules["quark_ranLL_vec_local_mass_l_t0"]
        assert vec_quark["id"]["type"] == "MFermion::StagGaugeProp"
        assert vec_quark["options"]["gammas"] == "spintaste_vec_local"
        assert vec_quark["options"]["solver"] == "stag_ranLL_mass_l"
        assert not any("axial" in name for name in result.modules)

    def test_meson_field_mode_skips_ranll_gauge_prop(self):
        config = make_config(low_modes_config=meson_field_low_modes())
        _, names = build_spintaste_modules(config)
        result = build_quarks(config, [REF], names)
        assert not any(n.startswith("quark_ranLL_") for n in result.modules)
        # CG quarks still emitted
        assert "quark_ama_vec_local_mass_l_t0" in result.modules

    def test_chain_guess_reference_emitted(self):
        config = make_config(cg_config=CgConfig(**sub_kwargs(residual=[1e-4, 1e-8])))
        _, names = build_spintaste_modules(config)
        result = build_quarks(config, [REF], names)
        tight = result.modules["quark_ama_1e-08_vec_local_mass_l_t0"]
        assert tight["options"]["guess"] == "quark_ama_0.0001_vec_local_mass_l_t0"

    def test_labeled_entry_prefixes_quark_solver_noise_names(self):
        config = make_config(label="sloppy")
        _, names = build_spintaste_modules(config)
        result = build_quarks(config, [REF], names)
        quark = result.modules["sloppy_quark_ama_vec_local_mass_l_t0"]
        assert quark["options"]["source"] == "sloppy_noise_t0"
        # solver template filled then prefixed
        assert quark["options"]["solver"] == "sloppy_stag_ama_mass_l"
        assert quark["options"]["guess"] == "sloppy_quark_ranLL_vec_local_mass_l_t0"


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
        assert direct["options"]["sink"] == "quark_ranLL_pion_local_mass_l_t0_elem"
        elem = result.modules["quark_ranLL_pion_local_mass_l_t0_elem"]
        assert elem["id"]["type"] == "MUtilities::GammaMapElement"
        assert elem["options"]["label"] == "G5_G5"

        axial = result.modules["corr_ranLL_axial_vec_local_mass_l_t0"]
        assert axial["options"]["sinkGammas"] == "spintaste_axial_vec_local"
        assert axial["options"]["source"] == "quark_ranLL_vec_local_mass_l_t0"
        assert axial["options"]["sink"] == "quark_ranLL_scalar_local_mass_l_t0_elem"
        axial_elem = result.modules["quark_ranLL_scalar_local_mass_l_t0_elem"]
        assert axial_elem["options"]["label"] == "G1_G1"

    def test_antiquark_element_deduped_across_ops_sharing_antiquark(self):
        config = make_config(
            operations=OpList.from_dict(
                {"vec_local": {"mass": ["l"]}, "fourvec_local": {"mass": ["l"]}}
            )
        )
        _, names = build_spintaste_modules(config)
        result = build_contractions(config, [REF], names)
        # one deduped antiquark element per solver (ranLL + ama), each shared
        # across the vec_local and fourvec_local ops (4 contractions -> 2)
        assert sorted(n for n in result.modules if n.endswith("_elem")) == [
            "quark_ama_pion_local_mass_l_t0_elem",
            "quark_ranLL_pion_local_mass_l_t0_elem",
        ]

    def test_output_filestem_formatted_not_prefixed(self):
        config = make_config()
        _, names = build_spintaste_modules(config)
        result = build_contractions(config, [REF], names)
        corr = result.modules["corr_ranLL_vec_local_mass_l_t0"]
        assert corr["options"]["output"] == "corr/corr_vec_local_ranLL_m01_t0"

    def test_labeled_entry_prefixes_corr_modules_not_files(self):
        config = make_config(label="sloppy")
        _, names = build_spintaste_modules(config)
        result = build_contractions(config, [REF], names)
        corr = result.modules["sloppy_corr_ranLL_vec_local_mass_l_t0"]
        assert corr["options"]["source"] == "sloppy_quark_ranLL_vec_local_mass_l_t0"
        assert corr["options"]["sourceShift"] == "sloppy_noise_t0_shift"
        # user-owned filestem: unprefixed
        assert corr["options"]["output"] == "corr/corr_vec_local_ranLL_m01_t0"


class TestNameHelpers:
    def test_meson_field_producer_names(self):
        from pyfm.tasks.hadrons.highmode_v2.twopoint import (
            meson_field_output_name,
            meson_field_producer_name,
            quark_name,
        )

        config = make_config(low_modes_config=meson_field_low_modes())
        assert meson_field_producer_name(config, "vec_local", "l") == (
            "quark_ranLL_vec_local_mass_l"
        )
        assert meson_field_output_name(config, "vec_local", "l", REF) == (
            "quark_ranLL_vec_local_mass_l_t0"
        )
        assert quark_name(config, "ranLL", "vec_local", "l", REF) == (
            "quark_ranLL_vec_local_mass_l_t0"
        )
        assert quark_name(config, "ama", "vec_local", "l", REF) == (
            "quark_ama_vec_local_mass_l_t0"
        )

    def test_biased_meson_field_producer_names(self):
        from pyfm.tasks.hadrons.highmode_v2.twopoint import meson_field_producer_name

        config = make_config(
            low_modes_config=meson_field_low_modes(),
            sources_config=SourcesConfig(
                **sub_kwargs(
                    time=4,
                    biased_config=BiasedSourceConfig(**sub_kwargs(n=2, seed="s")),
                )
            ),
        )
        assert meson_field_producer_name(config, "vec_local", "l", REF) == (
            "quark_ranLL_vec_local_mass_l_t0"
        )
        with pytest.raises(ValueError, match="SourceRef"):
            meson_field_producer_name(config, "vec_local", "l", None)

    def test_solver_template_label_offered_and_prefixed(self):
        from pyfm.tasks.hadrons.highmode_v2.twopoint import solver_module_name

        config = make_config(label="sloppy", solver_name="s_{label}_{solver}_m_{mass}")
        assert solver_module_name(config, "ama", "l") == "sloppy_s_sloppy_ama_m_l"

    def test_sink_and_noise_names_prefixed(self):
        from pyfm.tasks.hadrons.highmode_v2.twopoint import noise_rw_name, sink_name

        config = make_config(label="sloppy")
        assert sink_name(config) == "sloppy_sink"
        assert noise_rw_name(config, REF) == "sloppy_noise_t0"
        assert sink_name(make_config()) == "sink"
