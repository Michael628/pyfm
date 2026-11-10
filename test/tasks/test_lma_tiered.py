"""Tests for lma_tiered.py — the TIERED channel of hadrons_lma_new:
program-level `tasks.subgrid_gather` routing, split-only gating, and the
task's TIERED admission (D8)."""

import copy

import pytest

from pyfm.nanny.taskbuilder import create_task


def _tiered_params(hadrons_params):
    """Re-key the fixture's lma_new job block to lma_tiered and force
    TIERED solving entries — a valid lma_tiered config by construction
    (D8; the fixture ships DIAGONAL defaults)."""
    params = copy.deepcopy(hadrons_params)
    block = params["job_setup"].pop("lma_new")
    block["task_type"] = "lma_tiered"
    params["job_setup"]["lma_tiered"] = block
    # Per-step resources are keyed by step name — mirror lma_new's layout
    # so JobConfig builds (geom is required).
    resources = params.get("submit", {}).get("resources", {})
    if "lma_new" in resources and "lma_tiered" not in resources:
        resources["lma_tiered"] = copy.deepcopy(resources["lma_new"])
    for entry in block["tasks"]["high_modes"]["entries"].values():
        entry["output"]["solve_cross_terms"] = "tiered"
    return params


class TestSubgridGatherConfig:
    """Program-level `tasks.subgrid_gather` routing, split-only gating,
    and the task's TIERED admission (D8)."""

    @staticmethod
    def _split_entry(entry):
        entry["split_mpi_layout"] = "1 1 1 1"
        entry["subgrid_ranks"] = 4

    def test_tasks_level_bool_reaches_config_field(self, hadrons_params):
        params = _tiered_params(hadrons_params)
        entries = params["job_setup"]["lma_tiered"]["tasks"]["high_modes"]["entries"]
        for entry in entries.values():
            self._split_entry(entry)
        params["job_setup"]["lma_tiered"]["tasks"]["subgrid_gather"] = True
        task = create_task("lma_tiered", params, "a", "20")
        assert task.config.subgrid_gather is True

    def test_default_false_without_key(self, hadrons_params):
        task = create_task("lma_tiered", _tiered_params(hadrons_params), "a", "20")
        assert task.config.subgrid_gather is False

    def test_gather_without_split_rejected(self, hadrons_params):
        params = _tiered_params(hadrons_params)
        params["job_setup"]["lma_tiered"]["tasks"]["subgrid_gather"] = True
        with pytest.raises(ValueError, match="subgrid_gather requires split-grid"):
            create_task("lma_tiered", params, "a", "20")

    def test_non_bool_rejected(self, hadrons_params):
        params = _tiered_params(hadrons_params)
        params["job_setup"]["lma_tiered"]["tasks"]["subgrid_gather"] = "true"
        with pytest.raises(TypeError, match="subgrid_gather must be a boolean"):
            create_task("lma_tiered", params, "a", "20")

    def test_entry_level_key_rejected(self, hadrons_params):
        params = _tiered_params(hadrons_params)
        entries = params["job_setup"]["lma_tiered"]["tasks"]["high_modes"]["entries"]
        entries["sloppy"]["subgrid_gather"] = True
        with pytest.raises(ValueError, match="Unknown high_modes entry keys"):
            create_task("lma_tiered", params, "a", "20")

    def test_non_tiered_entry_rejected_with_pointer(self, hadrons_params):
        params = _tiered_params(hadrons_params)
        entries = params["job_setup"]["lma_tiered"]["tasks"]["high_modes"]["entries"]
        entries["sloppy"]["output"]["solve_cross_terms"] = "all"
        with pytest.raises(ValueError, match="TIERED programs only"):
            create_task("lma_tiered", params, "a", "20")

    def test_build_only_cache_entry_exempt_from_admission(self, hadrons_params):
        # is_cache_build_only entries solve nothing (no cg/output by the
        # config rules) — the MF-style cache composition stays legal
        params = _tiered_params(hadrons_params)
        entry = params["job_setup"]["lma_tiered"]["tasks"]["high_modes"][
            "entries"
        ]["sloppy"]
        entry["low_modes"]["meson_field"]["cache"] = "build_only"
        entry.pop("cg")
        entry.pop("output")
        create_task("lma_tiered", params, "a", "20")  # no raise


from pyfm.tasks.hadrons.lma_tiered import (
    LMATieredConfig,
    _validate_split_world_reads,
    build_input_params,
)
from pyfm.tasks.hadrons import highmode_v2, lma_new


class TestSubgridGatherWiring:
    """The program-level flag reaches the pooled split emission; the
    element guard rejects non-element world reads of tagged outputs.
    Post-split-phases (`6c21dfc`) bridges are tagged (Fault 1 landed), so
    these run unconditionally — no skip."""

    def _gather_task(self, hadrons_params):
        params = _tiered_params(hadrons_params)
        tasks = params["job_setup"]["lma_tiered"]["tasks"]
        entries = tasks["high_modes"]["entries"]
        for entry in entries.values():
            entry.update(
                operations={"pion_local": {"mass": ["l"]}, "vec_local": {"mass": ["l"]}},
                split_mpi_layout="1 1 1 2",
                subgrid_ranks=2,
            )
            # solve_cross_terms already forced TIERED by _tiered_params
        # program-level phasing knob — entry-level subgrid_phases is
        # rejected with a pointer post-split-phases
        tasks["subgrid_phases"] = 2
        tasks["subgrid_gather"] = True
        return create_task("lma_tiered", params, "a", "20")

    def test_flag_reaches_every_entry_emission(self, hadrons_params):
        task = self._gather_task(hadrons_params)
        result = build_input_params(task.config)
        sched = result.schedule
        phases = {}
        for i, n in enumerate(sched):
            m = result.modules[n]
            if m.get("subgrid") is not None:
                k = m.get("splitPhase", 0)
                phases[k] = max(phases.get(k, -1), i)
        # Per-entry source naming differs (grid sources suffix `_t{t0}`;
        # biased sources suffix `_n{noise}`) — derive each slice's quark
        # ama pion solve generically and mirror its name onto the vec
        # triplet and the pion LH.
        for label in task.config.high_modes_config:
            prefix = f"{label}_quark_ama_pion_local_mass_l_"
            for quark in (
                n for n in sched
                if n.startswith(prefix) and not n.endswith("_elem")
            ):
                tail = quark[len(prefix):]
                k = result.modules[quark].get("splitPhase", 0)
                trip = [
                    f"{label}_quark_ranLL_vec_local_mass_l_{tail}",
                    f"{label}_corr_ranLL_vec_local_mass_l_{tail}",
                    f"{label}_corr_ranLL_ama_vec_local_mass_l_{tail}",
                ]
                assert sched.index(quark) <= phases[k]
                for name in trip:
                    assert result.modules[name].get("subgrid") is None
                    assert sched.index(name) > phases[k]
                assert (
                    sched.index(trip[0]) < sched.index(trip[1]) < sched.index(trip[2])
                )
                # the pion LH of the same slice stays tagged (the contrast)
                lh_pion = result.modules[f"{label}_corr_ranLL_ama_pion_local_mass_l_{tail}"]
                assert lh_pion.get("subgrid") is not None

    def test_guard_rejects_non_element_world_read(self):
        modules = {
            "quark_ama_pion_local_mass_l_t0": {
                "id": {"name": "quark_ama_pion_local_mass_l_t0",
                       "type": "MFermion::StagGaugeProp"},
                "options": {"source": "noise_t0"}, "subgrid": 0,
            },
            "corr_bad_t0": {
                "id": {"name": "corr_bad_t0", "type": "MContraction::StagMeson"},
                "options": {"source": "quark_ama_pion_local_mass_l_t0"},
            },
        }
        with pytest.raises(ValueError, match="GammaMapElement"):
            _validate_split_world_reads(modules)

    def test_guard_accepts_element_world_read(self):
        modules = {
            "quark_ama_pion_local_mass_l_t0": {
                "id": {"name": "quark_ama_pion_local_mass_l_t0",
                       "type": "MFermion::StagGaugeProp"},
                "options": {"source": "noise_t0"}, "subgrid": 0,
            },
            "quark_ama_pion_local_mass_l_t0_elem": {
                "id": {"name": "quark_ama_pion_local_mass_l_t0_elem",
                       "type": "MUtilities::GammaMapElement"},
                "options": {"map": "quark_ama_pion_local_mass_l_t0"}, "subgrid": 0,
            },
            "corr_ranLL_ama_vec_local_mass_l_t0": {
                "id": {"name": "corr_ranLL_ama_vec_local_mass_l_t0",
                       "type": "MContraction::StagMeson"},
                "options": {"sink": "quark_ama_pion_local_mass_l_t0_elem"},
            },
        }
        _validate_split_world_reads(modules)  # no raise


# --- Slice 4: append — the bare tier through the lma_tiered channel
# --- (task-level pins: Scope's "catalog/aggregator consistency" and
# --- split-mode tagging/bridging of the pion ama chain, driving
# --- create_task("lma_tiered", ...) → build_input_params with the
# --- _tiered_params + split-entry shape above) ---
class TestTieredBareTierTask:
    """TIERED bare tier, task level: per source and rung the contract-gamma
    ama solve alone (no quark_ama_vec_* anywhere — the e350552 lesson),
    ranLL + ranLL_ama per operator through the pooled split emission,
    vec LHs sinking on the pion ama element, catalog/aggregator
    consistency (get_solver_labels-driven, [ranLL, ranLL_ama])."""

    @staticmethod
    def _task(hadrons_params):
        params = _tiered_params(hadrons_params)
        tasks = params["job_setup"]["lma_tiered"]["tasks"]
        entries = tasks["high_modes"]["entries"]
        for entry in entries.values():
            entry.update(
                operations={
                    "pion_local": {"mass": ["l"]},
                    "vec_local": {"mass": ["l"]},
                },
                split_mpi_layout="1 1 1 2",
                subgrid_ranks=2,
            )
        tasks["subgrid_phases"] = 1
        return create_task("lma_tiered", params, "a", "20")

    @staticmethod
    def _slices(task):
        """Per entry label, the emitted slice tails — the suffix after
        ``{label}_quark_ama_pion_local_mass_l_`` on each ama pion quark
        module (grid slices suffix ``_t{t0}``, biased ``_n{i}_t{t0}``).
        """
        result = build_input_params(task.config)
        slices = {}
        for label in task.config.high_modes_config:
            prefix = f"{label}_quark_ama_pion_local_mass_l_"
            slices[label] = sorted(
                n[len(prefix):]
                for n in result.modules
                if n.startswith(prefix) and not n.endswith("_elem")
            )
        return result, slices

    def test_no_dead_vec_ama_solves(self, hadrons_params):
        task = self._task(hadrons_params)
        result = build_input_params(task.config)
        ama = [
            n
            for n, m in result.modules.items()
            if m["id"]["type"] == "MFermion::StagGaugeProp" and "_ama_" in n
        ]
        assert ama, "no ama solves emitted at all"
        assert all("pion_local" in n for n in ama)
        assert not any("quark_ama_vec_" in n for n in result.modules)

    def test_ll_and_lh_per_operator_per_slice(self, hadrons_params):
        task = self._task(hadrons_params)
        result, slices = self._slices(task)
        for label, tails in slices.items():
            assert tails, f"no slices emitted for entry {label}"
            for tail in tails:
                for g in ("pion_local", "vec_local"):
                    for dset in ("ranLL", "ranLL_ama"):
                        name = f"{label}_corr_{dset}_{g}_mass_l_{tail}"
                        assert name in result.modules, name

    def test_vec_lh_sinks_on_tagged_pion_ama_element(self, hadrons_params):
        task = self._task(hadrons_params)
        result, slices = self._slices(task)
        for label, tails in slices.items():
            for tail in tails:
                quark = f"{label}_quark_ama_pion_local_mass_l_{tail}"
                elem = f"{quark}_elem"
                lh = result.modules[f"{label}_corr_ranLL_ama_vec_local_mass_l_{tail}"]
                assert lh["options"]["source"] == (
                    f"{label}_quark_ranLL_vec_local_mass_l_{tail}"
                )
                assert lh["options"]["sink"] == elem
                element = result.modules[elem]
                assert element["id"]["type"] == "MUtilities::GammaMapElement"
                assert element["options"]["map"] == quark
                # split-mode tagging/bridging: the bridge shares its
                # map's placement (same subgrid, same splitPhase)
                assert element.get("subgrid") == lh.get("subgrid")
                assert element.get("splitPhase") == lh.get("splitPhase")

    def test_catalog_and_aggregator_consistency(self, hadrons_params):
        task = self._task(hadrons_params)
        result, _ = self._slices(task)
        catalog = lma_new.create_outfile_catalog(task.config)
        all_labels: set = set()
        for label, hm in task.config.high_modes_config.items():
            labels = set(hm.get_solver_labels())
            all_labels |= labels
            # the emitted contraction dsets are exactly the TIERED pairs
            # of that label-driven list
            emitted = {
                n.split("_corr_")[1].split("_" + g)[0]
                for n in result.modules
                if n.startswith(f"{label}_corr_")
                for g in ("pion_local", "vec_local")
                if f"_{g}_mass_l_" in n
            }
            assert emitted == {"ranLL", "ranLL_ama"}
            assert emitted <= labels
        # the catalog's dset list IS get_solver_labels(), per entry —
        # nothing hardcoded, nothing missing (gauge/epack/meson rows
        # carry no dset; drop them)
        dsets = catalog["dset"].dropna().unique()
        assert set(dsets) == all_labels
        # aggregator params build without error off the same config
        params = lma_new.build_aggregator_params(task.config, average=True)
        assert params

    def test_guard_noop_without_tagged_modules(self):
        modules = {
            "corr_x": {"id": {"name": "corr_x", "type": "MContraction::StagMeson"},
                       "options": {"source": "quark_y"}},
        }
        _validate_split_world_reads(modules)  # no raise

    def test_guard_ignores_tagged_readers_of_tagged_outputs(self):
        modules = {
            "quark_ama_pion_local_mass_l_t0": {
                "id": {"name": "quark_ama_pion_local_mass_l_t0",
                       "type": "MFermion::StagGaugeProp"},
                "options": {"source": "noise_t0"}, "subgrid": 0,
            },
            "corr_tagged_t0": {
                "id": {"name": "corr_tagged_t0", "type": "MContraction::StagMeson"},
                "options": {"source": "quark_ama_pion_local_mass_l_t0"},
                "subgrid": 0,
            },
        }
        _validate_split_world_reads(modules)  # no raise
