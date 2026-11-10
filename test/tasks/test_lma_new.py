"""Tests for lma_new.py — LMANewConfig composite, own hooks (no legacy
task imports), keyed/unkeyed high_modes entries, presence-driven
sections."""

import copy
import dataclasses
import re
from collections import Counter

import pytest

from pyfm.nanny import write_input_file
from pyfm.nanny.taskbuilder import create_task
from pyfm.tasks.hadrons.lma_new import (
    LMANewConfig,
    DEFAULT_HM_LABEL,
    _merge_modules,
    build_aggregator_params,
    build_input_params,
    compare_outputs,
)
from pyfm.tasks.hadrons.highmode_v2.config import LMAHighModeConfig
from pyfm.tasks.register import get_task_handler, get_task_key, list_registered_types


class TestRegistration:
    def test_hadrons_lma_new_is_registered(self):
        assert "nanny_hadrons_lma_new" in list_registered_types()

    def test_task_key_resolves_to_lma_new_handler(self):
        handler = get_task_handler(job_type="hadrons", task_type="lma_new")
        assert handler is not None
        assert handler.config_type is LMANewConfig

    def test_lmi_registration_untouched(self):
        from pyfm.tasks.hadrons.lmi import LMIConfig

        assert get_task_key(config=LMIConfig) == "nanny_hadrons_lmi"
        assert get_task_key(config=LMANewConfig) == "nanny_hadrons_lma_new"

    def test_no_skip_flags_on_lma_new_config(self):
        fields = {f.name for f in dataclasses.fields(LMANewConfig)}
        assert "skip_epack" not in fields
        assert "skip_meson" not in fields
        assert "skip_high_modes" not in fields
        assert "build_lh_cache" not in fields


class TestKeyedEntries:
    def test_entries_built_with_labels(self, hadrons_params):
        task = create_task("lma_new", hadrons_params, "a", "20")
        assert set(task.config.high_modes_config) == {"sloppy", "bias"}
        assert all(
            isinstance(hm, LMAHighModeConfig)
            for hm in task.config.high_modes_config.values()
        )
        assert task.config.high_modes_config["sloppy"].label == "sloppy"
        assert task.config.high_modes_config["bias"].label == "bias"

    def test_shared_defaults_layer_under_entries(self, hadrons_params):
        task = create_task("lma_new", hadrons_params, "a", "20")
        for hm in task.config.high_modes_config.values():
            assert hm.op_list[0].gamma.name == "PION_LOCAL"  # shared operations
            assert hm.masses == ["l"]

    def test_unkeyed_single_entry_gets_default_label(self, hadrons_params):
        params = copy.deepcopy(hadrons_params)
        tasks = params["job_setup"]["lma_new"]["tasks"]
        entry = tasks["high_modes"]["entries"]["sloppy"]
        tasks["high_modes"] = {
            "operations": {"gamma": ["pion_local"], "mass": ["l"]},
        } | entry
        task = create_task("lma_new", params, "a", "20")
        assert DEFAULT_HM_LABEL == "hm"
        assert set(task.config.high_modes_config) == {"hm"}
        assert task.config.high_modes_config["hm"].label == "hm"

    def test_entry_operations_replace_shared(self, hadrons_params):
        params = copy.deepcopy(hadrons_params)
        entries = params["job_setup"]["lma_new"]["tasks"]["high_modes"]["entries"]
        entries["bias"]["operations"] = {"vec_local": {"mass": ["l"]}}
        task = create_task("lma_new", params, "a", "20")
        bias = task.config.high_modes_config["bias"]
        sloppy = task.config.high_modes_config["sloppy"]
        assert [op.gamma.name for op in bias.op_list] == ["VEC_LOCAL"]  # no merge
        assert [op.gamma.name for op in sloppy.op_list] == ["PION_LOCAL"]

    @pytest.mark.parametrize("key, value", [("gamma", ["pion_local"]), ("mass", ["l"])])
    def test_bare_shared_op_key_rejected(self, hadrons_params, key, value):
        params = copy.deepcopy(hadrons_params)
        params["job_setup"]["lma_new"]["tasks"]["high_modes"][key] = value
        with pytest.raises(ValueError, match="must be nested under `operations:`"):
            create_task("lma_new", params, "a", "20")

    def test_unknown_entry_key_rejected(self, hadrons_params):
        params = copy.deepcopy(hadrons_params)
        entries = params["job_setup"]["lma_new"]["tasks"]["high_modes"]["entries"]
        entries["sloppy"]["low_mode_method"] = "load"
        with pytest.raises(ValueError, match="Unknown high_modes entry keys"):
            create_task("lma_new", params, "a", "20")

    def test_empty_entries_means_no_high_modes(self, hadrons_params):
        params = copy.deepcopy(hadrons_params)
        params["job_setup"]["lma_new"]["tasks"]["high_modes"]["entries"] = {}
        task = create_task("lma_new", params, "a", "20")
        assert task.config.high_modes_config == {}

    def test_absent_high_modes_block(self, hadrons_params):
        params = copy.deepcopy(hadrons_params)
        del params["job_setup"]["lma_new"]["tasks"]["high_modes"]
        task = create_task("lma_new", params, "a", "20")
        assert task.config.high_modes_config == {}


class TestModuleIdentity:
    def test_labeled_modules_disjoint(self, hadrons_params):
        task = create_task("lma_new", hadrons_params, "a", "20")
        result = build_input_params(task.config)
        assert "sloppy_noise_fv" in result.modules
        assert "bias_noise_n0" in result.modules
        assert "noise_fv" not in result.modules  # nothing unprefixed dangles
        assert "sloppy_mf_local_mass_l" in result.modules  # per-entry writer
        # CB pairs are shared per mass, never label-prefixed.
        assert "cbpairs_l_mass_l" in result.modules
        assert "cbpairs_r_mass_l" in result.modules
        assert "sloppy_cbpairs_l_mass_l" not in result.modules

    def test_unkeyed_modules_get_default_label_prefix(self, hadrons_params):
        params = copy.deepcopy(hadrons_params)
        tasks = params["job_setup"]["lma_new"]["tasks"]
        entry = tasks["high_modes"]["entries"]["sloppy"]
        tasks["high_modes"] = {
            "operations": {"gamma": ["pion_local"], "mass": ["l"]},
        } | entry
        task = create_task("lma_new", params, "a", "20")
        result = build_input_params(task.config)
        assert "hm_noise_fv" in result.modules
        assert "hm_mf_local_mass_l" in result.modules
        assert "hm_spintaste_mf_local_mass_l" in result.modules
        assert "hm_mfload_mass_l_G1_G1" in result.modules
        # one single-slice producer per source, named ..._t{t0}
        assert any(
            n.startswith("hm_quark_ranLL_pion_local_mass_l_t")
            and m["id"]["type"] == "MFermion::StagLMAMesonFieldProp"
            for n, m in result.modules.items()
        )
        # Nothing entry-owned is left unprefixed; CB pairs stay shared.
        for name in ("noise_fv", "mf_local_mass_l", "mfload_mass_l_G1_G1"):
            assert name not in result.modules
        assert "cbpairs_l_mass_l" in result.modules
        assert "hm_cbpairs_l_mass_l" not in result.modules
        assert not any(n.startswith("sloppy_") for n in result.modules)

    def test_epack_always_emitted_and_mass_shifts_cover_entries(self, hadrons_params):
        task = create_task("lma_new", hadrons_params, "a", "20")
        result = build_input_params(task.config)
        assert result.modules["epack"]["id"]["type"] == (
            "MIO::StagLoadFermionEigenPack"
        )
        assert "evecs_mass_l" in result.modules  # epack mass shift


class TestMesonStanzaWithCacheWriter:
    """A ``meson:`` stanza next to an unkeyed ``build_only`` entry: the
    entry's cache writer must not overwrite the stanza's modules."""

    @staticmethod
    def _params(hadrons_params, meson_extra=None):
        params = copy.deepcopy(hadrons_params)
        params["job_setup"]["lma_new"]["tasks"] = {
            "gauge": {"action_type": "load"},
            "epack": {"load": False, "save_evals": True, "save_eigs": True},
            "meson": {"gamma": ["pion_local", "vec_local"], "mass": ["l"]}
            | (meson_extra or {}),
            "high_modes": {
                "operations": {
                    "pion_local": {"mass": ["l"]},
                    "vec_local": {"mass": ["l"]},
                },
                "sources": {"grid": True},
                "low_modes": {
                    "meson_field": {
                        "file": "meson_stoch_proj",
                        "cache": "build_only",
                        "blocksize": 60,
                    }
                },
            },
        }
        return params

    def test_stanza_and_writer_modules_both_survive(self, hadrons_params):
        task = create_task("lma_new", self._params(hadrons_params), "a", "20")
        result = build_input_params(task.config)
        stanza = result.modules["mf_local_mass_l"]
        writer = result.modules["hm_mf_local_mass_l"]
        stanza_st = result.modules["spintaste_mf_local_mass_l"]["options"]
        writer_st = result.modules["hm_spintaste_mf_local_mass_l"]["options"]
        # Stanza keeps its own options: no CB pairs, unfolded spin-taste.
        assert "cbPairsLeft" not in stanza["options"]
        assert stanza_st["spinTaste"]["applyG5"] == "false"
        # Writer reads the entry's noise and the shared per-mass CB pairs.
        assert writer["options"]["right"] == "hm_noise_fv_vec"
        assert writer["options"]["cbPairsLeft"] == "cbpairs_l_mass_l"
        assert writer_st["spinTaste"]["applyG5"] == "true"
        assert stanza["options"]["output"] != writer["options"]["output"]
        for name in ("mf_local_mass_l", "hm_mf_local_mass_l"):
            assert result.schedule.count(name) == 1

    def test_stanza_with_cb_pairs_shares_writer_pairs(self, hadrons_params):
        params = self._params(hadrons_params, meson_extra={"cb_pairs": True})
        task = create_task("lma_new", params, "a", "20")
        result = build_input_params(task.config)
        assert result.modules["mf_local_mass_l"]["options"]["cbPairsLeft"] == (
            "cbpairs_l_mass_l"
        )
        assert [n for n in result.modules if "cbpairs" in n] == [
            "cbpairs_l_mass_l",
            "cbpairs_r_mass_l",
        ]
        assert result.schedule.count("cbpairs_l_mass_l") == 1


class TestMergeModules:
    def test_identical_duplicate_is_allowed(self):
        module = {"id": {"name": "stag_mass_l"}, "options": {"mass": "0.1"}}
        modules = {"stag_mass_l": dict(module)}
        _merge_modules(modules, {"stag_mass_l": dict(module)})
        assert modules == {"stag_mass_l": module}

    def test_conflicting_duplicate_raises(self):
        modules = {"mf_local_mass_l": {"options": {"applyG5": "false"}}}
        with pytest.raises(ValueError, match="'mf_local_mass_l'"):
            _merge_modules(
                modules, {"mf_local_mass_l": {"options": {"applyG5": "true"}}}
            )
        assert modules["mf_local_mass_l"]["options"]["applyG5"] == "false"


class TestTwoStageCacheWorkflow:
    def test_build_only_then_load_share_noise_identity(self, hadrons_params):
        build_params = copy.deepcopy(hadrons_params)
        load_params = copy.deepcopy(hadrons_params)
        build_entry = build_params["job_setup"]["lma_new"]["tasks"]["high_modes"][
            "entries"
        ]["sloppy"]
        build_entry["low_modes"]["meson_field"]["cache"] = "build_only"
        build_entry.pop("cg")  # build_only forbids cg/output blocks
        build_entry.pop("output")
        load_entry = load_params["job_setup"]["lma_new"]["tasks"]["high_modes"][
            "entries"
        ]["sloppy"]
        load_entry["low_modes"]["meson_field"]["cache"] = "load"

        # Drop the mpcg bias entry from the build job so `istag_mass_l`'s
        # absence is attributable to the build_only sloppy entry (action
        # names are mass-based and shared across entries).
        del build_params["job_setup"]["lma_new"]["tasks"]["high_modes"]["entries"][
            "bias"
        ]

        build_task = create_task("lma_new", build_params, "a", "20")
        load_task = create_task("lma_new", load_params, "a", "20")

        assert build_task.config.runid == load_task.config.runid
        build_hm = build_task.config.high_modes_config["sloppy"]
        load_hm = load_task.config.high_modes_config["sloppy"]
        # The noise module name is the cross-job identity (same runid +
        # same label → same Hadrons seed): identical in both jobs.
        assert build_hm.noise_name == load_hm.noise_name == "sloppy_noise_fv"

        build_result = build_input_params(build_task.config)
        load_result = build_input_params(load_task.config)
        assert "sloppy_noise_fv" in build_result.modules
        assert "sloppy_noise_fv" in load_result.modules
        # Per-source noise only exists past the build_only stop.
        assert not any(n.startswith("sloppy_noise_t") for n in build_result.modules)
        assert any(n.startswith("sloppy_noise_t") for n in load_result.modules)

        # build_only still emits the entry's dp action modules: the cache
        # writer's EigenPackCBPairs modules reference stag_mass_<mass>.
        assert build_result.modules["stag_mass_l"]["id"]["type"] == (
            "MAction::ImprovedStaggeredMILC"
        )
        assert "istag_mass_l" not in build_result.modules  # no cg → no sp action
        assert "stag_mass_l" in load_result.modules

        # build_only writes the cache and stops; load keeps the producers
        assert "sloppy_mf_local_mass_l" in build_result.modules
        assert not any(
            n.startswith("sloppy_quark_ranLL") for n in build_result.modules
        )
        assert any(
            n.startswith("sloppy_quark_ranLL_pion_local_mass_l_t")
            and m["id"]["type"] == "MFermion::StagLMAMesonFieldProp"
            for n, m in load_result.modules.items()
        )


class TestValidation:
    def test_split_mpi_layout_conflict_rejected(self, hadrons_params):
        params = copy.deepcopy(hadrons_params)
        entries = params["job_setup"]["lma_new"]["tasks"]["high_modes"]["entries"]
        entries["sloppy"]["split_mpi_layout"] = "1.1.1.2"
        entries["sloppy"]["subgrid_ranks"] = 2
        entries["bias"]["split_mpi_layout"] = "2.2.1.1"
        entries["bias"]["subgrid_ranks"] = 2
        with pytest.raises(ValueError, match="split_mpi_layout"):
            create_task("lma_new", params, "a", "20")

    def test_shared_output_filestem_warns(self, hadrons_params, caplog):
        import logging

        params = copy.deepcopy(hadrons_params)
        entries = params["job_setup"]["lma_new"]["tasks"]["high_modes"]["entries"]
        entries["bias"]["output"]["file"] = "high_modes"  # same as sloppy
        with caplog.at_level(logging.WARNING):
            create_task("lma_new", params, "a", "20")
        assert "filestem" in caplog.text


class TestProgramSubgridKnobs:
    """Program-level phasing knobs: routing, validation, entry rejection."""

    @staticmethod
    def _split_params(hadrons_params):
        params = copy.deepcopy(hadrons_params)
        entries = params["job_setup"]["lma_new"]["tasks"]["high_modes"]["entries"]
        for label in ("sloppy", "bias"):
            entries[label]["split_mpi_layout"] = "1 1 1 1"
            entries[label]["subgrid_ranks"] = 4
        return params

    def test_knobs_reach_config_fields(self, hadrons_params):
        params = self._split_params(hadrons_params)
        tasks = params["job_setup"]["lma_new"]["tasks"]
        tasks["subgrid_phase_size"] = 8
        tasks["subgrid_last_phase_size"] = 16
        cfg = create_task("lma_new", params, "a", "20").config
        assert cfg.subgrid_phase_size == 8
        assert cfg.subgrid_last_phase_size == 16
        assert cfg.subgrid_phases is None

    def test_count_knob_reaches_config(self, hadrons_params):
        params = self._split_params(hadrons_params)
        params["job_setup"]["lma_new"]["tasks"]["subgrid_phases"] = 3
        cfg = create_task("lma_new", params, "a", "20").config
        assert cfg.subgrid_phases == 3

    def test_exclusive_knobs_rejected(self, hadrons_params):
        params = self._split_params(hadrons_params)
        tasks = params["job_setup"]["lma_new"]["tasks"]
        tasks["subgrid_phases"] = 3
        tasks["subgrid_phase_size"] = 8
        with pytest.raises(ValueError, match="mutually exclusive"):
            create_task("lma_new", params, "a", "20")

    def test_last_without_size_rejected(self, hadrons_params):
        params = self._split_params(hadrons_params)
        params["job_setup"]["lma_new"]["tasks"]["subgrid_last_phase_size"] = 16
        with pytest.raises(ValueError, match="requires subgrid_phase_size"):
            create_task("lma_new", params, "a", "20")

    def test_non_multiple_size_rejected(self, hadrons_params):
        params = self._split_params(hadrons_params)
        params["job_setup"]["lma_new"]["tasks"]["subgrid_phase_size"] = 9
        with pytest.raises(ValueError, match="subgrid_phase_size"):
            create_task("lma_new", params, "a", "20")

    def test_default_last_size_is_omitted(self, hadrons_params):
        params = self._split_params(hadrons_params)
        params["job_setup"]["lma_new"]["tasks"]["subgrid_phase_size"] = 8
        cfg = create_task("lma_new", params, "a", "20").config
        # the default is applied by the planner (last = phase size), the
        # field itself stays None
        assert cfg.subgrid_last_phase_size is None

    def test_knobs_require_every_entry_split(self, hadrons_params):
        params = self._split_params(hadrons_params)
        params["job_setup"]["lma_new"]["tasks"]["subgrid_phase_size"] = 8
        entries = params["job_setup"]["lma_new"]["tasks"]["high_modes"]["entries"]
        entries["bias"].pop("split_mpi_layout")
        entries["bias"].pop("subgrid_ranks")
        with pytest.raises(ValueError, match="unsplit"):
            create_task("lma_new", params, "a", "20")

    def test_knobs_without_entries_rejected(self, hadrons_params):
        params = self._split_params(hadrons_params)
        tasks = params["job_setup"]["lma_new"]["tasks"]
        tasks["subgrid_phase_size"] = 8
        tasks["high_modes"]["entries"] = {}
        with pytest.raises(ValueError, match="require high_modes entries"):
            create_task("lma_new", params, "a", "20")

    def test_per_entry_subgrid_phases_rejected_with_pointer(self, hadrons_params):
        params = self._split_params(hadrons_params)
        entries = params["job_setup"]["lma_new"]["tasks"]["high_modes"]["entries"]
        entries["sloppy"]["subgrid_phases"] = 2
        with pytest.raises(ValueError, match="program-level"):
            create_task("lma_new", params, "a", "20")

    def test_subgrid_ranks_conflict_rejected(self, hadrons_params):
        params = self._split_params(hadrons_params)
        entries = params["job_setup"]["lma_new"]["tasks"]["high_modes"]["entries"]
        entries["bias"]["subgrid_ranks"] = 2  # layouts agree, ranks differ
        with pytest.raises(ValueError, match="Conflicting subgrid_ranks"):
            create_task("lma_new", params, "a", "20")

    def test_no_knobs_allows_mixed_entries(self, hadrons_params):
        # mixed split/unsplit stays legal without phasing knobs
        params = self._split_params(hadrons_params)
        entries = params["job_setup"]["lma_new"]["tasks"]["high_modes"]["entries"]
        entries["bias"].pop("split_mpi_layout")
        entries["bias"].pop("subgrid_ranks")
        result = build_input_params(create_task("lma_new", params, "a", "20").config)
        assert any(m.get("subgrid") is not None for m in result.modules.values())
        assert "bias_noise_n0" in result.modules

    def test_lmi_rejects_program_knobs(self, hadrons_params):
        from pyfm.tasks.hadrons import lmi

        params = copy.deepcopy(hadrons_params)
        block = params["job_setup"]["lma_new"]["tasks"]
        block["subgrid_phase_size"] = 8
        hm = block["high_modes"]
        entries = hm["entries"] if "entries" in hm else {"mass": hm}
        lmi_params = {
            "_preprocessor": {
                "gauge": {"action_type": "load"},
                "epack": {"load": True},
                "high_modes": [{"mass": {}} | {k: v for k, v in entries["sloppy"].items()
                                                if k not in ("cg", "output", "low_modes")}],
                "subgrid_phase_size": 8,
            }
        }
        with pytest.raises(ValueError, match="not supported by hadrons_lmi"):
            lmi.normalize_params(lmi_params)


class TestPooledSplitProgram:
    """lma_new-level pooled phasing: two entries, one phase sequence."""

    @staticmethod
    def _two_split_entries(hadrons_params):
        params = copy.deepcopy(hadrons_params)
        entries = params["job_setup"]["lma_new"]["tasks"]["high_modes"]["entries"]
        for label in ("sloppy", "bias"):
            entries[label]["split_mpi_layout"] = "1 1 1 1"
            entries[label]["subgrid_ranks"] = 4
        return params

    def test_two_entry_phase_sequence_monotonic(self, hadrons_params):
        # the retired D-phase-offset emitted 0,0,1,1,0,0,3,3 for two
        # N=2 entries; the pooled program emits each phase once
        params = self._two_split_entries(hadrons_params)
        params["job_setup"]["lma_new"]["tasks"]["subgrid_phases"] = 2
        result = build_input_params(create_task("lma_new", params, "a", "20").config)
        sched_phases = [
            result.modules[n].get("splitPhase", 0)
            for n in result.schedule
            if result.modules[n].get("subgrid") is not None
        ]
        assert sched_phases == sorted(sched_phases)
        assert set(sched_phases) == {0, 1}
        # phase 1's hoist carries both entries' walls before its tags
        first_p1 = min(
            i
            for i, n in enumerate(result.schedule)
            if result.modules[n].get("splitPhase") == 1
        )
        assert result.schedule.index("bias_noise_n0") < first_p1
        assert any(
            n.startswith("sloppy_noise_t") and result.schedule.index(n) < first_p1
            for n in result.modules
        )


class TestProductionSplitPrograms:
    """The handoff's done-when shapes, emitted through lma_new."""

    @staticmethod
    def _emit(hadrons_params, tasks):
        params = copy.deepcopy(hadrons_params)
        params["job_setup"]["lma_new"]["tasks"] = tasks
        task = create_task("lma_new", params, "a", "20")
        return build_input_params(task.config), task.config

    @staticmethod
    def _entry(**overrides):
        entry = {
            "operations": {"pion_local": {"mass": ["l"]}},
            "sources": {
                "time": 8,
                "noise": 2,
                "grid": {"tstart": 0, "tstop": 7, "dt": 1},
            },
            "low_modes": {"meson_field": {"file": "mf_meson", "cache": "load"}},
            "cg": {"solver": "mpcg", "residual": [1.0e-4]},
            "output": {"file": "corr_modes"},
            "split_mpi_layout": "1 1 1 1",
            "subgrid_ranks": 4,
        }
        entry.update(overrides)
        return entry

    @staticmethod
    def _tasks(**entries):
        return {
            "gauge": {"action_type": "load"},
            "epack": {"load": True, "save_evals": True, "save_eigs": False},
            "high_modes": {"entries": dict(entries)},
        }

    def test_ph_b8i_layout(self, hadrons_params):
        # card PH's ph-b8i shape: 16 full sources (8 slices x noise 2),
        # subgrid_phase_size 8 -> phases [8, 8]; tags t0 % 4, t0 // 4
        tasks = self._tasks(hh=self._entry())
        tasks["subgrid_phase_size"] = 8
        result, _ = self._emit(hadrons_params, tasks)
        mods = result.modules
        tagged = [(n, m) for n, m in mods.items() if m.get("subgrid") is not None]
        assert tagged
        for name, module in tagged:
            t0 = int(re.search(r"_t(\d+)", name).group(1))
            assert module["subgrid"] == t0 % 4, name
            assert module.get("splitPhase", 0) == t0 // 4, name
        assert {m.get("splitPhase", 0) for _, m in tagged} == {0, 1}

    def test_o_phased_tier_layout(self, hadrons_params):
        # O's phased tier: 6 sloppy refs (weight 1) + 1 chained bias ref
        # (weight 2) -> ONE phase; the bias source alone on its subgrid,
        # the sloppy refs two a subgrid on the others; crosses anchored
        hh = self._entry(
            sources={
                "time": 48,
                "noise": 1,
                "grid": {"tstart": 0, "tstop": 40, "dt": 8},
            },
            cg={"solver": "mpcg", "residual": [1.0e-4], "precon": "each"},
            output={"file": "corr_hh_modes", "solve_cross_terms": "all"},
        )
        hbias = self._entry(
            sources={
                "time": 48,
                "noise": 1,
                "biased": {"n": 1, "seed": "s", "replace": False},
            },
            cg={"solver": "mpcg", "residual": [1.0e-4, 1.0e-8]},
            output={"file": "corr_bias_modes", "solve_cross_terms": "all"},
        )
        tasks = self._tasks(hh=hh, hbias=hbias)
        tasks["subgrid_phase_size"] = 8
        result, _ = self._emit(hadrons_params, tasks)
        mods = result.modules
        tagged = [m for m in mods.values() if m.get("subgrid") is not None]
        assert tagged and all(m.get("splitPhase", 0) == 0 for m in tagged)
        bias_sg = mods["hbias_quark_ama_0.0001_pion_local_mass_l_n0"]["subgrid"]
        hh_sgs = [
            mods[f"hh_quark_ama_pion_local_mass_l_t{t}"]["subgrid"]
            for t in (0, 8, 16, 24, 32, 40)
        ]
        assert all(sg != bias_sg for sg in hh_sgs)
        assert sorted(Counter(hh_sgs).values()) == [2, 2, 2]
        # each cross contraction follows its anchor (O's reorder);
        # hh's single residual yields the bare "ama" solver label
        anchor = "hh_corr_ama_pion_local_mass_l_t0"
        cross = "hh_corr_ama_ranLL_pion_local_mass_l_t0"
        assert result.schedule.index(anchor) < result.schedule.index(cross)

    def test_fused_48_program(self, hadrons_params):
        # 20 slices x noise 2 + 8 chained bias draws x noise 1 = 48
        # sources at sizes 8/16 -> [8, 8, 8, 8, 16]; the bias refs all in
        # the last phase, whose hoist holds the eigenpack's last reader
        hh = self._entry(
            sources={
                "time": 48,
                "noise": 2,
                "grid": {"tstart": 0, "tstop": 38, "dt": 2},
            },
            output={"file": "corr_hh_modes"},
        )
        hbias = self._entry(
            sources={
                "time": 48,
                "noise": 1,
                "biased": {"n": 8, "seed": "s", "replace": False},
            },
            cg={"solver": "mpcg", "residual": [1.0e-4, 1.0e-8]},
            output={"file": "corr_bias_modes"},
        )
        tasks = self._tasks(hh=hh, hbias=hbias)
        tasks["subgrid_phase_size"] = 8
        tasks["subgrid_last_phase_size"] = 16
        result, _ = self._emit(hadrons_params, tasks)
        mods = result.modules
        sched = result.schedule
        phases = {
            m.get("splitPhase", 0)
            for m in mods.values()
            if m.get("subgrid") is not None
        }
        assert phases == {0, 1, 2, 3, 4}
        hbias_phases = {
            m.get("splitPhase", 0)
            for n, m in mods.items()
            if n.startswith("hbias_") and m.get("subgrid") is not None
        }
        assert hbias_phases == {4}
        tagged_idx = [
            i for i, n in enumerate(sched) if mods[n].get("subgrid") is not None
        ]
        last_first = min(
            i for i in tagged_idx if mods[sched[i]].get("splitPhase", 0) == 4
        )
        prev_last = max(
            i for i in tagged_idx if mods[sched[i]].get("splitPhase", 0) == 3
        )
        readers = [
            i
            for i, n in enumerate(sched)
            if "evecs_mass_l" in str(mods[n]["options"])
        ]
        assert readers
        assert prev_last < max(readers) < last_first


class TestAggregatorAndCompare:
    def test_aggregator_merges_prefixed_runs(self, hadrons_params):
        task = create_task("lma_new", hadrons_params, "a", "20")
        params = build_aggregator_params(task.config, average=False)
        run = params["run"]
        assert any(k.startswith("sloppy_") for k in run)
        assert any(k.startswith("bias_") for k in run)
        assert len(run) == len(set(run))  # label prefixes keep keys distinct
        for key in run:
            assert key in params

    def test_compare_pairs_by_label(self, hadrons_params):
        task_a = create_task("lma_new", hadrons_params, "a", "20")
        task_b = create_task("lma_new", hadrons_params, "a", "20")
        report = compare_outputs(task_a.config, task_b.config)
        # No files on disk: every expected row is a missing_file row, one
        # per (gamma, mass, tsource, dset) across BOTH entries.
        assert not report.empty
        assert (report["status"] == "missing_file").all()
        assert set(report["gamma_label"]) == {"pion_local"}
        assert list(report.columns) == [
            "gamma_label", "mass", "tsource", "dset", "filepath_a", "filepath_b",
            "max_abs_diff", "max_rel_diff", "within_tolerance", "status",
        ]

    def test_compare_rejects_mismatched_labels(self, hadrons_params):
        task_a = create_task("lma_new", hadrons_params, "a", "20")
        params = copy.deepcopy(hadrons_params)
        del params["job_setup"]["lma_new"]["tasks"]["high_modes"]["entries"]["bias"]
        task_b = create_task("lma_new", params, "a", "20")
        with pytest.raises(ValueError, match="labels"):
            compare_outputs(task_a.config, task_b.config)


def test_generate_lma_new_input_end_to_end(tmp_path, monkeypatch, hadrons_params):
    """Full write_input_file dispatch for job_type=hadrons, task_type=lma_new."""
    monkeypatch.chdir(tmp_path)

    write_input_file("lma_new", hadrons_params, "a", "20")

    xml = (tmp_path / "in" / "full-lma-new-a.20.xml").read_text()
    assert "<type>MFermion::SpinTaste</type>" in xml
    assert "<type>MFermion::StagGaugeProp</type>" in xml
    assert "<type>MFermion::StagLMAMesonFieldProp</type>" in xml
    assert "<type>MContraction::StagMeson</type>" in xml
    assert "<type>MContraction::StagA2AMesonField</type>" in xml  # per-entry writer
    assert "<type>MFermion::StagGaugePropLegacy</type>" not in xml
    assert "<type>MContraction::StagMesonLegacy</type>" not in xml
