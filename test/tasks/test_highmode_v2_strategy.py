"""Tests for highmode_v2/strategy.py — per-entry emission: meson-field
chain (grid + biased), cache writer, build_only stop, resume gate,
entry catalog."""

import pandas as pd
import pytest

from pyfm.domain import Gamma, MassDict, OpList, Outfile
from pyfm.tasks.hadrons.types import HadronsInput
from pyfm.tasks.hadrons.highmode_v2 import (
    build_aggregator_params,
    compare_outputs,
    strategy,
    twopoint,
)
from pyfm.tasks.hadrons.highmode_v2.config import (
    BiasedSourceConfig,
    CacheMode,
    CgConfig,
    GridSourceConfig,
    LMAHighModeConfig,
    LowModeMethod,
    LowModesConfig,
    MesonFieldConfig,
    OutputConfig,
    SourcesConfig,
)

BASE = {"formatting": {}, "logging_level": "INFO", "runid": "test"}

MF_FILE = Outfile(
    # No {cfg} token: the per-entry cache writer's catalog (meson.MesonConfig)
    # can only supply gamma/mass replacements; loaders build their own
    # traj-dir filenames independent of this ext.
    filestem="mesonfield/mf_{mass}",
    ext=".h5",
    good_size=1
)


def sub_kwargs(**extra):
    return BASE | extra


def make_sources(**overrides):
    kwargs = dict(
        time=4,
        grid_config=GridSourceConfig(**sub_kwargs(tstart=0, tstop=3, dt=1)),
    )
    kwargs.update(overrides)
    return SourcesConfig(**sub_kwargs(**kwargs))


def make_low_modes(method=LowModeMethod.SOLVE, **mf_overrides):
    if method is LowModeMethod.MESON_FIELD:
        return LowModesConfig(
            **sub_kwargs(
                method=method,
                meson_field_config=MesonFieldConfig(
                    **sub_kwargs(file=MF_FILE, **mf_overrides)
                ),
            )
        )
    return LowModesConfig(**sub_kwargs(method=method))


def make_output(**overrides):
    kwargs = dict(
        file=Outfile(
            filestem="corr/corr_{gamma_label}_{dset}_mass_{mass}_{tsource}",
            ext=".20.h5",
            good_size=1,
        ),
        overwrite=True,
    )
    kwargs.update(overrides)
    return OutputConfig(**sub_kwargs(**kwargs))


def make_config(label="", **overrides):
    kwargs = dict(
        **BASE,
        label=label,
        mass=MassDict.from_dict({"l": 0.002426}),
        action_name="stag_mass_{mass}",
        solver_name="stag_{solver}_mass_{mass}",
        low_modes_name="evecs_mass_{mass}",
        operations=OpList.from_dict({"pion_local": {"mass": ["l"]}}),
        sources_config=make_sources(),
        low_modes_config=make_low_modes(),
        cg_config=CgConfig(**sub_kwargs()),
        output_config=make_output(),
        shift_gauge_name="gauge_apbc",
    )
    kwargs.update(overrides)
    return LMAHighModeConfig(**kwargs)


class TestBuildLmaMesonFieldChain:
    def _config(self, label="", sources_config=None):
        return make_config(
            label=label,
            sources_config=sources_config or make_sources(),
            low_modes_config=make_low_modes(LowModeMethod.MESON_FIELD),
            cg_config=None,
            output_config=None,
        )

    def _split(self, config, refs=None):
        _, names = twopoint.build_spintaste_modules(config)
        return strategy.build_lma_meson_field_chain(
            config,
            mass_label="l",
            action="stag_mass_l",
            low_modes="evecs_mass_l",
            gammas=[Gamma.PION_LOCAL],
            spintaste_names=names,
            refs=config.sources_config.source_refs if refs is None else refs,
        )

    def _chain(self, config, refs=None):
        loaders, producers = self._split(config, refs)
        return HadronsInput(
            modules=loaders.modules | producers.modules,
            schedule=loaders.schedule + producers.schedule,
        )

    def test_grid_producer_references_shared_spintaste_with_required_labels(self):
        result = self._chain(self._config())
        producer = result.modules["quark_ranLL_pion_local_mass_l_t2"]
        assert producer["id"]["type"] == "MFermion::StagLMAMesonFieldProp"
        assert producer["options"]["gammas"] == "spintaste_pion_local"
        assert producer["options"]["labels"] == "G5_G5"
        assert producer["options"]["mesonField"] == "mfload_mass_l_G1_G1"
        assert producer["options"]["tA"] == "2"
        assert producer["options"]["tB"] == "2"
        assert producer["options"]["tStep"] == "1"
        assert producer["options"]["noise"] == "noise_fv_vec"

    def test_grid_emits_one_single_slice_producer_per_source(self):
        result = self._chain(self._config())
        producers = sorted(n for n in result.modules if n.startswith("quark_ranLL_"))
        assert producers == [f"quark_ranLL_pion_local_mass_l_t{t}" for t in range(4)]
        # no doubled time suffix: module name == published object
        assert not any("_t0_t0" in n for n in producers)

    def test_split_return_separates_loaders_from_producers(self):
        loaders, producers = self._split(self._config())
        assert loaders.schedule == ["mfload_mass_l_G1_G1"]
        assert set(loaders.modules) == {"mfload_mass_l_G1_G1"}
        assert producers.schedule == [
            f"quark_ranLL_pion_local_mass_l_t{t}" for t in range(4)
        ]

    def test_producers_follow_given_refs_only(self):
        config = self._config()
        refs = config.sources_config.source_refs
        _, producers = self._split(config, refs=[refs[1]])
        assert producers.schedule == ["quark_ranLL_pion_local_mass_l_t1"]

    def test_loader_file_mass_is_prefix_removed_value(self):
        result = self._chain(self._config())
        loader = result.modules["mfload_mass_l_G1_G1"]
        # {mass} in the cache-file filestem is filled with the prefix-removed
        # mass VALUE ("002426"), byte-matching the writer's output grammar
        # (meson_v2 / MesonField config behavior) — never the raw massdict
        # key ("l").
        assert loader["options"]["file"] == (
            "mesonfield/mf_002426.@traj@/G1_G1_0_0_0.h5"
        )

    def test_no_writer_or_cbpairs_modules(self):
        result = self._chain(self._config())
        assert "mfwrite_mass_l" not in result.modules
        assert "cbpairs_l_mass_l" not in result.modules
        assert "mfload_mass_l_G1_G1" in result.modules
        assert "quark_ranLL_pion_local_mass_l_t0" in result.modules

    def test_labeled_entry_prefixes_loaders_and_producers(self):
        result = self._chain(self._config(label="sloppy"))
        assert "sloppy_mfload_mass_l_G1_G1" in result.modules
        producer = result.modules["sloppy_quark_ranLL_pion_local_mass_l_t0"]
        assert producer["options"]["noise"] == "sloppy_noise_fv_vec"
        assert producer["options"]["mesonField"] == "sloppy_mfload_mass_l_G1_G1"

    def test_biased_emits_one_producer_per_slice_sharing_loaders(self):
        biased = make_sources(
            grid_config=None,
            biased_config=BiasedSourceConfig(**sub_kwargs(n=2, seed="s")),
        )
        config = self._config(sources_config=biased)
        refs = config.sources_config.source_refs
        result = self._chain(config)
        producers = sorted(n for n in result.modules if n.startswith("quark_ranLL_"))
        assert producers == [
            f"quark_ranLL_pion_local_mass_l_n0_t{refs[0].t0}",
            f"quark_ranLL_pion_local_mass_l_n1_t{refs[1].t0}",
        ]
        for name, ref in zip(producers, refs):
            # tA=tB=drawn t0: a single-time window per slice
            assert result.modules[name]["options"]["tA"] == str(ref.t0)
            assert result.modules[name]["options"]["tB"] == str(ref.t0)
            assert result.modules[name]["options"]["tStep"] == "1"
        # loaders shared: exactly one
        assert [n for n in result.modules if n.startswith("mfload_")] == [
            "mfload_mass_l_G1_G1"
        ]


class TestBuildInputParamsSolveMode:
    def test_compute_mode_unchanged_shape(self):
        config = make_config()
        result = strategy.build_input_params(config)
        assert "noise_fv" not in result.modules
        assert result.modules["stag_ranLL_mass_l"]["id"]["type"] == "MSolver::StagLMA"
        assert result.modules["quark_ranLL_pion_local_mass_l_t0"]["id"]["type"] == (
            "MFermion::StagGaugeProp"
        )
        assert result.modules["corr_ranLL_pion_local_mass_l_t0"]["id"]["type"] == (
            "MContraction::StagMeson"
        )

    def test_cg_solver_emission_matches_cg_block(self):
        config = make_config(
            cg_config=CgConfig(**sub_kwargs(solver="cg", residual=[1e-8]))
        )
        result = strategy.build_input_params(config)
        assert result.modules["stag_ama_mass_l"]["id"]["type"] == "MSolver::StagCGMILC"

    def test_no_cg_block_means_no_cg_modules(self):
        config = make_config(cg_config=None)
        result = strategy.build_input_params(config)
        assert not any("ama" in n for n in result.modules)

    def test_sink_and_sorted_schedule_precons_first(self):
        config = make_config()
        result = strategy.build_input_params(config)
        assert "sink" in result.modules
        sched = result.schedule
        assert sched.index("quark_ranLL_pion_local_mass_l_t0") < sched.index(
            "quark_ama_pion_local_mass_l_t0"
        )


class TestBuildInputParamsMesonField:
    def _mf_config(self, label="", cache=CacheMode.LOAD, **overrides):
        kwargs = dict(
            label=label,
            low_modes_config=make_low_modes(LowModeMethod.MESON_FIELD, cache=cache),
        )
        kwargs.update(overrides)
        return make_config(**kwargs)

    def test_end_to_end_load_mode_shape(self):
        config = self._mf_config()
        result = strategy.build_input_params(config)
        assert result.modules["quark_ranLL_pion_local_mass_l_t0"]["id"]["type"] == (
            "MFermion::StagLMAMesonFieldProp"
        )
        assert "quark_ranLL_pion_local_mass_l" not in result.modules
        assert all(
            mod["id"]["type"] != "MSolver::StagLMA"
            for mod in result.modules.values()
        )
        # contractions reference producer outputs (grid: _t{tsource})
        corr = result.modules["corr_ranLL_pion_local_mass_l_t0"]
        assert corr["options"]["source"] == "quark_ranLL_pion_local_mass_l_t0"

    def test_noise_precedes_producer_and_writer(self):
        config = self._mf_config(cache=CacheMode.BUILD_AND_LOAD)
        result = strategy.build_input_params(config)
        sched = result.schedule
        assert sched.index("noise_fv") < sched.index("quark_ranLL_pion_local_mass_l_t0")
        assert sched.index("noise_fv") < sched.index("mf_local_mass_l")

    def test_grid_producers_interleave_with_their_slice_consumers(self):
        config = self._mf_config()
        sched = strategy.build_input_params(config).schedule
        for t in range(4):
            producer = sched.index(f"quark_ranLL_pion_local_mass_l_t{t}")
            assert sched.index("mfload_mass_l_G1_G1") < producer
            assert producer < sched.index(f"corr_ranLL_pion_local_mass_l_t{t}")
            assert producer < sched.index(f"quark_ama_pion_local_mass_l_t{t}")
            if t > 0:
                assert sched.index(f"corr_ranLL_pion_local_mass_l_t{t - 1}") < producer

    def test_resume_gate_skips_producers_for_done_slices(self, monkeypatch):
        catalog = pd.DataFrame(
            {"tsource": ["0", "1", "2", "3"], "exists": [True, False, True, True]}
        )
        monkeypatch.setattr(strategy, "create_outfile_catalog", lambda config: catalog)
        config = self._mf_config(output_config=make_output(overwrite=False))
        result = strategy.build_input_params(config)
        producers = [
            n
            for n, m in result.modules.items()
            if m["id"]["type"] == "MFermion::StagLMAMesonFieldProp"
        ]
        assert producers == ["quark_ranLL_pion_local_mass_l_t1"]

    def test_build_and_load_emits_writer_after_noise(self):
        config = self._mf_config(cache=CacheMode.BUILD_AND_LOAD)
        result = strategy.build_input_params(config)
        writer = result.modules["mf_local_mass_l"]
        assert writer["id"]["type"] == "MContraction::StagA2AMesonField"
        assert writer["options"]["right"] == "noise_fv_vec"
        assert writer["options"]["lowModes"] == "evecs_mass_l"
        assert "spintaste_mf_local_mass_l" in result.modules

    def test_writer_uses_cb_pairs(self):
        config = self._mf_config(cache=CacheMode.BUILD_AND_LOAD)
        result = strategy.build_input_params(config)
        writer = result.modules["mf_local_mass_l"]
        assert writer["options"]["cbPairsLeft"] == "cbpairs_l_mass_l"
        assert writer["options"]["cbPairsRight"] == "cbpairs_r_mass_l"
        for name in ("cbpairs_l_mass_l", "cbpairs_r_mass_l"):
            module = result.modules[name]
            assert module["id"]["type"] == "MUtilities::EigenPackCBPairs"
            assert module["options"]["eigenPack"] == "evecs_mass_l"
            assert result.schedule.index(name) < result.schedule.index(
                "mf_local_mass_l"
            )

    def test_labeled_writer_names_prefixed(self):
        config = self._mf_config(label="sloppy", cache=CacheMode.BUILD_AND_LOAD)
        result = strategy.build_input_params(config)
        assert "sloppy_mf_local_mass_l" in result.modules
        assert "sloppy_spintaste_mf_local_mass_l" in result.modules
        writer = result.modules["sloppy_mf_local_mass_l"]
        assert writer["options"]["right"] == "sloppy_noise_fv_vec"

    def test_load_mode_emits_no_writer(self):
        config = self._mf_config()  # default cache=LOAD
        result = strategy.build_input_params(config)
        assert not any(
            n == "mf_local_mass_l" or n.startswith("spintaste_mf_")
            for n in result.modules
        )

    def test_build_only_stops_after_writer(self):
        config = self._mf_config(
            cache=CacheMode.BUILD_ONLY, cg_config=None, output_config=None
        )
        result = strategy.build_input_params(config)
        assert "mf_local_mass_l" in result.modules
        assert "noise_fv" in result.modules
        assert "sink" in result.modules
        assert not any(n.startswith("quark_") for n in result.modules)
        assert not any(n.startswith("corr_") for n in result.modules)
        assert "stag_ama_mass_l" not in result.modules


class TestBuildInputParamsBiasedMesonField:
    def _biased_config(self, label="", **overrides):
        biased = make_sources(
            grid_config=None,
            biased_config=BiasedSourceConfig(
                **sub_kwargs(n=2, seed="s", replace=False)
            ),
        )
        kwargs = dict(
            label=label,
            sources_config=biased,
            low_modes_config=make_low_modes(LowModeMethod.MESON_FIELD),
            cg_config=None,
        )
        kwargs.update(overrides)
        return make_config(**kwargs)

    def test_biased_end_to_end_producers_and_references(self):
        config = self._biased_config()
        result = strategy.build_input_params(config)
        refs = config.sources_config.source_refs
        assert "noise_n0" in result.modules
        assert result.modules["noise_n0"]["options"]["t0"] == str(refs[0].t0)
        producer = f"quark_ranLL_pion_local_mass_l_n0_t{refs[0].t0}"
        assert producer in result.modules
        assert "quark_ranLL_pion_local_mass_l_n0" not in result.modules
        corr = result.modules["corr_ranLL_pion_local_mass_l_n0"]
        # contraction resolves the producer's published output, never a
        # dangling quark_..._n0 GaugeProp name
        assert corr["options"]["source"] == producer

    def test_biased_producers_interleave_with_their_block_consumers(self):
        config = self._biased_config()
        refs = config.sources_config.source_refs
        sched = strategy.build_input_params(config).schedule
        p0 = sched.index(f"quark_ranLL_pion_local_mass_l_n0_t{refs[0].t0}")
        p1 = sched.index(f"quark_ranLL_pion_local_mass_l_n1_t{refs[1].t0}")
        c0 = sched.index("corr_ranLL_pion_local_mass_l_n0")
        c1 = sched.index("corr_ranLL_pion_local_mass_l_n1")
        assert p0 < c0 < p1 < c1

    def test_biased_ama_guess_resolves_producer_output(self):
        config = self._biased_config(cg_config=CgConfig(**sub_kwargs()))
        result = strategy.build_input_params(config)
        refs = config.sources_config.source_refs
        guess = result.modules["quark_ama_pion_local_mass_l_n0"]["options"]["guess"]
        assert guess == f"quark_ranLL_pion_local_mass_l_n0_t{refs[0].t0}"

    def test_labeled_biased_chain_disjoint_from_other_entries(self):
        config = self._biased_config(label="bias")
        result = strategy.build_input_params(config)
        refs = config.sources_config.source_refs
        assert "bias_noise_n0" in result.modules
        assert f"bias_quark_ranLL_pion_local_mass_l_n0_t{refs[0].t0}" in result.modules


class TestCreateOutfileCatalog:
    def test_grid_axis_enumerated(self):
        config = make_config()
        df = strategy.create_outfile_catalog(config)
        assert set(df["tsource"]) == {"0", "1", "2", "3"}
        assert set(df["dset"]) == {"ranLL", "ama"}

    def test_empty_operations_gives_empty_frame(self):
        config = make_config(operations=OpList.from_dict({}))
        assert strategy.create_outfile_catalog(config).empty

    def test_build_only_entry_catalogs_empty(self):
        config = make_config(
            low_modes_config=make_low_modes(
                LowModeMethod.MESON_FIELD, cache=CacheMode.BUILD_ONLY
            ),
            cg_config=None,
            output_config=None,
        )
        assert strategy.create_outfile_catalog(config).empty


class TestResumeGate:
    def test_overwrite_true_runs_all_sources(self):
        config = make_config()
        result = strategy.build_input_params(config)
        assert all(f"noise_t{t}" in result.modules for t in range(4))

    def test_incomplete_sources_only(self, monkeypatch):
        config = make_config(output_config=make_output(overwrite=False))
        fake = pd.DataFrame({"tsource": ["1", "2"], "exists": [False, False]})
        monkeypatch.setattr(strategy, "create_outfile_catalog", lambda c: fake)
        result = strategy.build_input_params(config)
        assert "noise_t1" in result.modules and "noise_t2" in result.modules
        assert "noise_t0" not in result.modules and "noise_t3" not in result.modules


class TestBuildAggregatorParams:
    def test_unlabeled_run_keys_unprefixed(self):
        config = make_config()
        params = build_aggregator_params(config, average=False)
        assert params["run"] == ["pion_local_002426_ranLL", "pion_local_002426_ama"]
        for key in params["run"]:
            assert key in params

    def test_labeled_run_keys_prefixed(self):
        config = make_config(label="sloppy")
        params = build_aggregator_params(config, average=False)
        assert params["run"] == [
            "sloppy_pion_local_002426_ranLL",
            "sloppy_pion_local_002426_ama",
        ]

    def test_build_only_aggregates_nothing(self):
        config = make_config(
            low_modes_config=make_low_modes(
                LowModeMethod.MESON_FIELD, cache=CacheMode.BUILD_ONLY
            ),
            cg_config=None,
            output_config=None,
        )
        assert build_aggregator_params(config, average=False) == {}

    def test_average_actions_and_outfile_suffix(self):
        # get_processed_filename only routes to processed/{format}_avg for
        # legacy-style "correlators/" filestems; test with one such stem.
        config = make_config(
            output_config=make_output(
                file=Outfile(
                    filestem="correlators/corr_{gamma_label}_{dset}_mass_{mass}_{tsource}",
                    ext=".20.h5",
                    good_size=1,
                )
            )
        )
        params = build_aggregator_params(config, average=True)
        entry = params["pion_local_002426_ranLL"]
        assert entry["actions"]["average"] == ["tsource"]
        assert entry["actions"]["real"] is True
        assert "_avg" in entry["out_files"]["filestem"]

    def test_time_axis_labels_from_sources(self):
        config = make_config()
        params = build_aggregator_params(config, average=False)
        entry = params["pion_local_002426_ranLL"]
        assert entry["load_files"]["labels"]["t"] == "0..3"


class TestCompareOutputs:
    def _entry_config(self, tmp_path, monkeypatch, name, values):
        import h5py
        import numpy as np

        # Filestem must carry every pairing key: catalog_files keeps only
        # replacement keys that appear in the filestem, and compare_outputs
        # pairs rows on (gamma_label, mass, tsource, dset).
        monkeypatch.chdir(tmp_path)
        out = OutputConfig(
            **sub_kwargs(
                file=Outfile(
                    filestem=(
                        f"cmp_{name}/"
                        "corr_{gamma_label}_{dset}_mass_{mass}_{tsource}"
                    ),
                    ext=".h5",
                    good_size=1,
                ),
                overwrite=True,
            )
        )
        config = make_config(cg_config=None, output_config=out)
        for dset in ("ranLL", "ama"):
            for t in ("0", "1", "2", "3"):
                path = (
                    tmp_path
                    / f"cmp_{name}"
                    / f"corr_pion_local_{dset}_mass_002426_{t}.h5"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                with h5py.File(path, "w") as f:
                    f.create_dataset(
                        "/meson/meson_0/corr",
                        data=np.asarray(values, dtype=np.float64),
                    )
        return config

    def test_identical_files_within_tolerance(self, tmp_path, monkeypatch):
        values = [1.0, 0.5, 0.25, 0.125]
        a = self._entry_config(tmp_path, monkeypatch, "a", values)
        b = self._entry_config(tmp_path, monkeypatch, "b", values)
        report = compare_outputs(a, b)
        rows = report[report["dset"] == "ranLL"]
        assert len(rows) == 4  # tsource 0..3
        assert (rows["status"] == "compared").all()
        assert rows["within_tolerance"].all()

    def test_differing_files_outside_tolerance(self, tmp_path, monkeypatch):
        a = self._entry_config(tmp_path, monkeypatch, "a", [1.0, 0.5, 0.25, 0.125])
        b = self._entry_config(tmp_path, monkeypatch, "b", [1.0, 0.5, 0.25, 0.5])
        report = compare_outputs(a, b)
        rows = report[report["dset"] == "ranLL"]
        assert (rows["status"] == "compared").all()
        assert (~rows["within_tolerance"]).all()

    def test_missing_files_reported(self, tmp_path, monkeypatch):
        a = self._entry_config(tmp_path, monkeypatch, "a", [1.0, 0.5, 0.25, 0.125])
        b = self._entry_config(tmp_path, monkeypatch, "b", [1.0, 0.5, 0.25, 0.125])
        for path in sorted((tmp_path / "cmp_b").glob("corr_*.h5")):
            path.unlink()
        report = compare_outputs(a, b)
        rows = report[report["dset"] == "ranLL"]
        assert (rows["status"] == "missing_file").all()
