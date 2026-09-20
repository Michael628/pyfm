import copy
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from pyfm.nanny.taskbuilder import create_task
from pyfm.nanny import write_input_file
from pyfm.tasks.hadrons import highmode, meson

HADRONS_CASES = [
    ("lma", "full-lma"),
    ("meson", "meson"),
    ("epack_load", "epack-load"),
    ("epack_solve", "epack-solve"),
    ("high_modes", "high-modes"),
]


@pytest.mark.parametrize("job_step,io_prefix", HADRONS_CASES)
def test_generate_hadrons_input(
    tmp_path,
    monkeypatch,
    tasks_data_dir,
    hadrons_params,
    assert_xml_equal,
    job_step,
    io_prefix,
):
    monkeypatch.chdir(tmp_path)

    write_input_file(job_step, hadrons_params, "a", "20")

    expected_sched = (
        (tasks_data_dir / "schedules" / f"test-{io_prefix}-a.20.sched")
        .read_text()
        .splitlines()
    )
    actual_sched = (
        (tmp_path / "schedules" / f"{io_prefix}-a.20.sched").read_text().splitlines()
    )
    assert actual_sched[0] == expected_sched[0], "Module count mismatch"
    assert set(actual_sched[1:]) == set(
        expected_sched[1:]
    ), "Schedule module set mismatch"

    assert_xml_equal(
        tmp_path / "in" / f"{io_prefix}-a.20.xml",
        tasks_data_dir / "in" / f"test-{io_prefix}-a.20.xml",
    )


def test_generate_high_modes_cg_input(tmp_path, monkeypatch, hadrons_params):
    monkeypatch.chdir(tmp_path)
    hadrons_params["shared_params"]["solver"] = "cg"

    write_input_file("high_modes", hadrons_params, "a", "20")

    xml = (tmp_path / "in" / "high-modes-a.20.xml").read_text()
    schedule = (tmp_path / "schedules" / "high-modes-a.20.sched").read_text()

    assert "MSolver::StagCGMILC" in xml
    assert "MSolver::StagMixedPrecisionCG" not in xml
    assert "MSolver::RBPrecCGMILC" not in xml
    assert "<guesser />" in xml or "<guesser></guesser>" in xml
    assert "gauge_smear_fatf" not in schedule
    assert "gauge_smear_longf" not in schedule


GRID_CASES = [
    ("lma", "grid-full-lma"),
]


@pytest.mark.parametrize("job_step,io_prefix", GRID_CASES)
def test_generate_grid_input(
    tmp_path,
    monkeypatch,
    tasks_data_dir,
    grid_params,
    assert_xml_equal,
    job_step,
    io_prefix,
):
    monkeypatch.chdir(tmp_path)

    write_input_file(job_step, grid_params, "a", "20")

    assert_xml_equal(
        tmp_path / "in" / f"{io_prefix}-a.20.xml",
        tasks_data_dir / "in" / f"test-{io_prefix}-a.20.xml",
    )


def _write_file(path, size):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"0" * int(size))


def test_generate_grid_input_skips_complete_high_mode_sources(
    tmp_path, monkeypatch, grid_params
):
    monkeypatch.chdir(tmp_path)
    params = copy.deepcopy(grid_params)
    params["shared_params"]["home"] = str(tmp_path)
    params["shared_params"]["overwrite"] = False

    task = create_task("lma", params, "a", "20")
    catalog = highmode.create_outfile_catalog(task.config.high_modes_config[0])
    for filepath in catalog[catalog["tsource"] == "0"]["filepath"]:
        _write_file(filepath, 1)

    write_input_file("lma", params, "a", "20")

    root = ET.parse(tmp_path / "in" / "grid-full-lma-a.20.xml").getroot()
    source_times = [elem.text for elem in root.findall(".//sources/elem/t0")]

    assert "0" not in source_times
    assert source_times == ["1", "2", "3"]
    assert root.find(".//corr") is not None


def test_generate_grid_input_omits_complete_meson_a2a(
    tmp_path, monkeypatch, grid_params
):
    monkeypatch.chdir(tmp_path)
    params = copy.deepcopy(grid_params)
    params["shared_params"]["home"] = str(tmp_path)
    params["shared_params"]["overwrite"] = False

    task = create_task("lma", params, "a", "20")
    catalog = meson.create_outfile_catalog(task.config.meson_config)
    for _, row in catalog.iterrows():
        _write_file(row["filepath"], row["good_size"])

    write_input_file("lma", params, "a", "20")

    root = ET.parse(tmp_path / "in" / "grid-full-lma-a.20.xml").getroot()

    assert root.find(".//a2a") is None


def test_generate_contract_input(
    tmp_path, monkeypatch, tasks_data_dir, contract_params
):
    monkeypatch.chdir(tmp_path)

    write_input_file("contract", contract_params, "a", "20")

    actual = yaml.safe_load((tmp_path / "in" / "contract-a.20.yaml").read_text())
    expected = yaml.safe_load(
        (tasks_data_dir / "in" / "test-contract-a.20.yaml").read_text()
    )
    assert actual == expected, "Contract YAML output mismatch"


class TestEmptyOpListGuards:
    """Empty op_list produces an empty catalog that lacks the 'tsource' column.

    Both get_high_mode_run_tsources and build_input_params must guard against
    df["tsource"] KeyError by short-circuiting to an empty run_tsources list.
    """

    @staticmethod
    def _empty_op_config(**overrides):
        from pyfm.domain import MassDict, OpList, Outfile
        from pyfm.tasks.hadrons.types import HighModeConfig

        return HighModeConfig(
            formatting={},
            logging_level="INFO",
            runid="test",
            mass=MassDict.from_dict({"l": 0.01}),
            action_name="action_{mass}",
            solver_name="solver_{solver}_{mass}",
            low_modes_name="low_modes",
            operations=OpList([]),
            high_modes=Outfile(
                filestem="corr/corr_{tsource}", ext=".h5", good_size=1
            ),
            tstart=0,
            tstop=3,
            dt=1,
            noise=1,
            time=64,
            skip_cg=True,
            shift_gauge_name="shift_gauge",
            **overrides,
        )

    def test_get_high_mode_run_tsources_empty_op_list(self):
        from pyfm.tasks.grid.lma import get_high_mode_run_tsources

        config = self._empty_op_config(overwrite=False)
        # Must not raise KeyError on df["tsource"]
        assert get_high_mode_run_tsources(config) == []

    def test_build_input_params_empty_op_list_no_crash(self):
        from pyfm.tasks.hadrons.highmode.strategy import build_input_params

        config = self._empty_op_config(overwrite=False)
        # Must not raise KeyError on df["tsource"]
        result = build_input_params(config)
        # No noise modules since run_tsources is empty
        assert all("noise_t" not in name for name in result.schedule)


def _set_split_grid(params, *, mpi_layout=None, subgrid_ranks=None, cross_terms=None):
    """Configure split-grid on the high_modes task slice (opt-in fields)."""
    hm = params["job_setup"]["high_modes"]["tasks"]["high_modes"]
    if mpi_layout is not None:
        hm["split_mpi_layout"] = mpi_layout
    if subgrid_ranks is not None:
        hm["subgrid_ranks"] = subgrid_ranks
    if cross_terms is not None:
        hm["cross_terms"] = cross_terms
    return params


def _modules_by_name(xml_path):
    """Parse the hadrons input XML -> {module_name: subgrid_text_or_None}."""
    root = ET.parse(xml_path).getroot()
    result = {}
    for module in root.findall(".//module"):
        name = module.find("id/name").text
        subgrid = module.find("subgrid")
        result[name] = subgrid.text if subgrid is not None else None
    return result


def test_generate_hadrons_split_grid_input(tmp_path, monkeypatch, hadrons_params):
    """Both split-grid fields set: global <split> present, CG quarks tagged, ranLL not."""
    monkeypatch.chdir(tmp_path)
    _set_split_grid(hadrons_params, mpi_layout="1 1 1 2", subgrid_ranks=2)

    write_input_file("high_modes", hadrons_params, "a", "20")

    xml_path = tmp_path / "in" / "high-modes-a.20.xml"
    root = ET.parse(xml_path).getroot()
    mpi_split = root.find(".//parameters/split/mpiSplit")
    assert mpi_split is not None and mpi_split.text == "1 1 1 2"

    modules = _modules_by_name(xml_path)
    ama = {n: s for n, s in modules.items() if n.startswith("quark_ama_")}
    ranll = {n: s for n, s in modules.items() if n.startswith("quark_ranLL_")}
    assert ama and ranll, "expected both CG (ama) and ranLL propagators"

    for name, subgrid in ama.items():
        tsource = int(name.rsplit("_t", 1)[1])
        assert subgrid == str(tsource % 2), f"{name}: subgrid {subgrid!r} != {tsource % 2}"
    for name, subgrid in ranll.items():
        assert subgrid is None, f"{name}: ranLL must not carry a subgrid"


def test_split_grid_tags_cross_term_contractions(tmp_path, monkeypatch, hadrons_params):
    """Cross-term corr_*ama*ranLL* contractions are tagged; pure-ranLL are not."""
    monkeypatch.chdir(tmp_path)
    _set_split_grid(
        hadrons_params, mpi_layout="1 1 1 2", subgrid_ranks=2, cross_terms="solve"
    )

    write_input_file("high_modes", hadrons_params, "a", "20")

    modules = _modules_by_name(tmp_path / "in" / "high-modes-a.20.xml")
    contractions = {n: s for n, s in modules.items() if n.startswith("corr_")}
    cross = {n: s for n, s in contractions.items() if "ama" in n and "ranLL" in n}
    ranll_only = {n: s for n, s in contractions.items() if "ama" not in n and "ranLL" in n}

    assert cross, "expected cross-term contractions (cross_terms=solve)"
    for name, subgrid in cross.items():
        assert subgrid is not None, f"cross-term {name} must carry a subgrid"
    for name, subgrid in ranll_only.items():
        assert subgrid is None, f"pure-ranLL {name} must not carry a subgrid"


def test_split_grid_partial_config_strips_and_warns(
    tmp_path, monkeypatch, caplog, hadrons_params
):
    """Partial config (only one field) strips both with a warning; no split/subgrid emitted."""
    monkeypatch.chdir(tmp_path)
    _set_split_grid(hadrons_params, mpi_layout="1 1 1 2")  # subgrid_ranks absent

    with caplog.at_level(logging.WARNING):
        write_input_file("high_modes", hadrons_params, "a", "20")

    assert "split_mpi_layout" in caplog.text and "subgrid_ranks" in caplog.text

    root = ET.parse(tmp_path / "in" / "high-modes-a.20.xml").getroot()
    assert root.find(".//parameters/split") is None
    assert root.find(".//subgrid") is None


def test_split_grid_rejects_nonpositive_subgrid_ranks(
    tmp_path, monkeypatch, hadrons_params
):
    """A non-positive subgrid_ranks is a hard misconfiguration; validate_config raises."""
    monkeypatch.chdir(tmp_path)
    _set_split_grid(hadrons_params, mpi_layout="1 1 1 2", subgrid_ranks=0)
    with pytest.raises(ValueError, match="subgrid_ranks"):
        write_input_file("high_modes", hadrons_params, "a", "20")


def test_split_grid_preserves_schedule_ordering(
    tmp_path, monkeypatch, hadrons_params
):
    """Module names are unchanged by the subgrid value, so the schedule order is invariant."""
    monkeypatch.chdir(tmp_path)

    split_params = copy.deepcopy(hadrons_params)
    _set_split_grid(split_params, mpi_layout="1 1 1 2", subgrid_ranks=2)

    split_task = create_task("high_modes", split_params, "a", "20")
    split_schedule = split_task.handler.build_input_params(split_task.config).schedule

    plain_task = create_task("high_modes", hadrons_params, "a", "20")
    plain_schedule = plain_task.handler.build_input_params(plain_task.config).schedule

    assert split_schedule == plain_schedule


# --- composite high_modes list integration ---
class TestHighModesListComposite:
    @staticmethod
    def _append_bias_entry(
        params, *, nbias=4, bias_seed="bias-seed", label="bias_modes",
        gamma=("pion_local",), mass=("l",), residual=None
    ):
        tasks = params["job_setup"]["lma"]["tasks"]
        main = tasks["high_modes"]
        bias = {
            "nbias": nbias,
            "bias_seed": bias_seed,
            "gamma": list(gamma),
            "mass": list(mass),
        }
        if label is not None:
            bias["high_modes"] = label
        if residual is not None:
            bias["residual"] = residual
        tasks["high_modes"] = [main, bias]
        return params

    def test_lma_with_bias_entry_emits_block_modules_with_sampled_t0(
        self, tmp_path, monkeypatch, hadrons_params
    ):
        monkeypatch.chdir(tmp_path)
        self._append_bias_entry(hadrons_params)

        task = create_task("lma", hadrons_params, "a", "20")
        write_input_file("lma", hadrons_params, "a", "20")

        root = ET.parse(tmp_path / "in" / "full-lma-a.20.xml").getroot()
        t0_by_name = {}
        for module in root.findall(".//module"):
            name = module.find("id/name").text
            if name.startswith("noise_n"):
                t0_by_name[name] = module.find("options/t0").text

        assert sorted(t0_by_name) == [f"noise_n{i}" for i in range(4)]
        for i, t0 in enumerate(task.config.high_modes_config[1].tsource_range):
            assert t0_by_name[f"noise_n{i}"] == str(t0)

    def test_bias_outputs_use_bias_modes_filestem(
        self, tmp_path, monkeypatch, hadrons_params
    ):
        monkeypatch.chdir(tmp_path)
        self._append_bias_entry(hadrons_params)

        write_input_file("lma", hadrons_params, "a", "20")

        xml = (tmp_path / "in" / "full-lma-a.20.xml").read_text()
        assert "correlators_bias" in xml
        assert "nb4" in xml  # {nbias} format key namespaces the bias outputs

    def test_bias_without_rebind_warns(
        self, tmp_path, monkeypatch, caplog, hadrons_params
    ):
        monkeypatch.chdir(tmp_path)
        self._append_bias_entry(hadrons_params, label=None)

        with caplog.at_level(logging.WARNING):
            write_input_file("lma", hadrons_params, "a", "20")

        # The generalized pairwise warning fires for the second entry (it
        # shares the default files label with the first); the remedy text
        # names the bias_modes rebind.
        assert "high_modes_config entry 1" in caplog.text

    def test_empty_second_entry_changes_nothing(
        self, tmp_path, monkeypatch, hadrons_params
    ):
        monkeypatch.chdir(tmp_path)

        plain_task = create_task("lma", hadrons_params, "a", "20")
        plain_sched = plain_task.handler.build_input_params(plain_task.config).schedule

        doubled_params = copy.deepcopy(hadrons_params)
        tasks = doubled_params["job_setup"]["lma"]["tasks"]
        tasks["high_modes"] = [tasks["high_modes"], {}]
        doubled_task = create_task("lma", doubled_params, "a", "20")
        doubled_sched = doubled_task.handler.build_input_params(
            doubled_task.config
        ).schedule

        assert len(doubled_task.config.high_modes_config) == 2
        assert plain_sched == doubled_sched

    def test_single_dict_coerced_to_list(self, hadrons_params):
        task = create_task("lma", hadrons_params, "a", "20")
        assert isinstance(task.config.high_modes_config, list)
        assert len(task.config.high_modes_config) == 1
        # Byte-identity of the single-entry XML is pinned by the golden tests.

    def test_absent_or_empty_high_modes_sets_skip_flag(self, hadrons_params):
        task = create_task("lma", hadrons_params)
        assert task.config.skip_high_modes is False  # fixture has high_modes

        absent = copy.deepcopy(hadrons_params)
        del absent["job_setup"]["lma"]["tasks"]["high_modes"]
        task = create_task("lma", absent)
        assert task.config.skip_high_modes is True
        assert task.config.high_modes_config == []

        empty = copy.deepcopy(hadrons_params)
        empty["job_setup"]["lma"]["tasks"]["high_modes"] = []
        task = create_task("lma", empty)
        assert task.config.skip_high_modes is True
        assert task.config.high_modes_config == []

    def test_tasks_bias_key_fails_loudly(self, hadrons_params):
        hadrons_params["job_setup"]["lma"]["tasks"]["bias"] = {"nbias": 4}
        with pytest.raises(KeyError, match="bias_config"):
            create_task("lma", hadrons_params)

    def test_aggregator_merges_bias_family(self, hadrons_params):
        self._append_bias_entry(hadrons_params)
        task = create_task("lma", hadrons_params)
        params = task.handler.build_aggregator_params(task.config, average=False)

        run = params["run"]
        assert any(k.startswith("hm1_") for k in run)
        assert any(not k.startswith("hm") for k in run)  # entry 0 unprefixed
        for key in run:
            assert key in params
        bias_key = next(k for k in run if k.startswith("hm1_"))
        assert "correlators_bias" in params[bias_key]["load_files"]["filestem"]

    def test_aggregation_path_catalog_enumerates_all_blocks(self, hadrons_params):
        # Aggregation builds configs without series/cfg (aggregator.py), so
        # the seed composes empty suffixes there; the catalog axis (a pure
        # function of nbias) must still enumerate every block — the design's
        # coherence invariant.
        self._append_bias_entry(hadrons_params)
        task = create_task("lma", hadrons_params)

        bias = task.config.high_modes_config[1]
        assert bias.bias_seed.endswith("__")
        assert bias.source_axis == [f"n{i}" for i in range(4)]

    def test_epack_mass_shifts_include_bias_masses(self, hadrons_params):
        self._append_bias_entry(hadrons_params, mass=("d",))
        task = create_task("lma", hadrons_params, "a", "20")
        result = task.handler.build_input_params(task.config)

        # epack mass-shift modules are named low_modes_name.format(mass=label)
        # on the epack config (shared route default); every bias mass label
        # must join the shifted set.
        epack_cfg = task.config.epack_config
        for mass_label in task.config.high_modes_config[1].masses:
            assert epack_cfg.low_modes_name.format(mass=mass_label) in result.modules


# --- grid list-shape routing safety ---
def test_grid_lma_accepts_single_dict_high_modes(tmp_path, monkeypatch, grid_params):
    """Single-dict tasks.high_modes still builds on Grid (coerced to [dict])."""
    monkeypatch.chdir(tmp_path)

    task = create_task("lma", grid_params, "a", "20")
    assert isinstance(task.config.high_modes_config, list)
    assert len(task.config.high_modes_config) == 1

    write_input_file("lma", grid_params, "a", "20")
    assert (tmp_path / "in" / "grid-full-lma-a.20.xml").exists()


def test_grid_lma_rejects_multiple_high_mode_entries(grid_params):
    tasks = grid_params["job_setup"]["lma"]["tasks"]
    tasks["high_modes"] = [tasks["high_modes"], dict(tasks["high_modes"])]
    with pytest.raises(ValueError, match="at most one"):
        create_task("lma", grid_params, "a", "20")


def test_grid_lma_tasks_bias_key_fails_loudly(grid_params):
    grid_params["job_setup"]["lma"]["tasks"]["bias"] = {"nbias": 4}
    with pytest.raises(KeyError, match="bias_config"):
        create_task("lma", grid_params, "a", "20")


# --- appended: TIERED end-to-end (phase 5) ---
def test_generate_high_modes_tiered_input(tmp_path, monkeypatch, hadrons_params):
    """TIERED: ranLL_ama cross contractions and dsets; no HH (ama) contraction
    modules; ama propagators persist (the LH contraction consumes them)."""
    monkeypatch.chdir(tmp_path)
    hadrons_params["job_setup"]["high_modes"]["tasks"]["high_modes"][
        "solve_cross_terms"
    ] = "tiered"

    write_input_file("high_modes", hadrons_params, "a", "20")

    modules = _modules_by_name(tmp_path / "in" / "high-modes-a.20.xml")
    contractions = {n for n in modules if n.startswith("corr_")}
    cross = {n for n in contractions if "ama" in n and "ranLL" in n}

    assert cross, "expected ranLL_ama cross contractions under TIERED"
    assert all(n.startswith("corr_ranLL_ama_") for n in cross)
    assert not any(n.startswith("corr_ama_") for n in contractions), "HH dropped"
    assert not any("ama_ranLL" in n for n in contractions), "reverse orientation dropped"
    assert any(n.startswith("quark_ama_") for n in modules), "ama propagators persist"

    xml = (tmp_path / "in" / "high-modes-a.20.xml").read_text()
    # module names and output filestems both name the ranLL_ama dset
    assert "ranLL_ama" in xml


def test_generate_high_modes_tiered_skips_dead_cg_solves(
    tmp_path, monkeypatch, hadrons_params
):
    """TIERED with a non-pion op (the fixture's high_modes step lists
    pion_local + vec_local): the op-gamma ama propagators have zero
    contraction consumers and are not emitted; the contract-gamma ama
    propagators persist (consumed by ranLL_ama), precon-chained to ranLL."""
    monkeypatch.chdir(tmp_path)
    params = copy.deepcopy(hadrons_params)  # sibling-test convention
    params["job_setup"]["high_modes"]["tasks"]["high_modes"][
        "solve_cross_terms"
    ] = "tiered"

    write_input_file("high_modes", params, "a", "20")

    modules = _modules_by_name(tmp_path / "in" / "high-modes-a.20.xml")

    # Dead solves gone: the vec_local op-gamma CG propagator has no consumer
    # under TIERED (the LH cross consumes only the PION_LOCAL antiquark side).
    assert not any(n.startswith("quark_ama_vec_local") for n in modules)
    # Live propagators persist.
    assert any(n.startswith("quark_ama_pion_local") for n in modules)
    assert any(n.startswith("corr_ranLL_ama_") for n in modules)

    # Precon chain: the ranLL contract-gamma solve precedes its ama consumer.
    # [1:] skips the module-count header line (golden-comparison convention).
    sched = (tmp_path / "schedules" / "high-modes-a.20.sched").read_text().splitlines()[1:]
    assert (
        sched.index("quark_ranLL_pion_local_mass_l_t0")
        < sched.index("quark_ama_pion_local_mass_l_t0")
    )


# --- appended: low_mode_method='load' end-to-end (file-driven LMA) ---
def test_generate_high_modes_load_method_input(tmp_path, monkeypatch, hadrons_params):
    """low_mode_method=load: fv noise, meson-field chain, _t{t} solver refs."""
    monkeypatch.chdir(tmp_path)
    hadrons_params["job_setup"]["high_modes"]["tasks"]["high_modes"][
        "low_mode_method"
    ] = "load"

    write_input_file("high_modes", hadrons_params, "a", "20")

    modules = _modules_by_name(tmp_path / "in" / "high-modes-a.20.xml")
    assert "noise_fv" in modules
    for m in ("l", "u"):
        assert f"cbpairs_l_mass_{m}" in modules
        assert f"cbpairs_r_mass_{m}" in modules
        assert f"mfwrite_mass_{m}" in modules
        assert f"mfload_mass_{m}" in modules

    xml = (tmp_path / "in" / "high-modes-a.20.xml").read_text()
    assert "<type>MSolver::StagLMAMesonField</type>" in xml
    assert "<type>MSolver::StagLMA</type>" not in xml
    assert "MIO::LoadMesonField" in xml
    assert "<noise>noise_fv</noise>" in xml  # RandomWalls reference the fv module
    assert "noise_fv_vec" in xml
    # quark props bind the per-timeslice solver family members
    assert "stag_ranLL_mass_l_t0" in xml
    assert "stag_ranLL_mass_l_t3" in xml

    sched = (
        tmp_path / "schedules" / "high-modes-a.20.sched"
    ).read_text().splitlines()[1:]
    for m in ("l", "u"):
        assert sched.index(f"mfwrite_mass_{m}") < sched.index(f"mfload_mass_{m}")
        assert sched.index(f"mfload_mass_{m}") < sched.index(f"stag_ranLL_mass_{m}")


def test_lmi_rejects_conflicting_load_method_on_shared_mass(hadrons_params):
    """Two high_modes list entries sharing a mass label with different
    low_mode_method values would collide on the shared solver module name —
    lmi.validate_config raises instead of silently last-wins."""
    tasks = hadrons_params["job_setup"]["lma"]["tasks"]
    main = tasks["high_modes"]
    conflict = dict(main)
    conflict["low_mode_method"] = "load"
    conflict["mass"] = ["l"]  # shares "l" with the main (compute) entry
    tasks["high_modes"] = [main, conflict]

    with pytest.raises(ValueError, match="low_mode_method"):
        create_task("lma", hadrons_params, "a", "20")


def test_grid_lma_rejects_load_method(grid_params):
    grid_params["job_setup"]["lma"]["tasks"]["high_modes"]["low_mode_method"] = "load"
    # Satisfy the shared HighModeConfig validate (files entry + noise==1) so
    # the GRID capability guard is the failure, not the child validation.
    grid_params["files"]["meson_stoch_proj"] = {
        "filestem": "lma-meson/m{mass}/mf_{series}",
        "good_size": 1286000,
    }
    with pytest.raises(ValueError, match="low_mode_method"):
        create_task("lma", grid_params, "a", "20")
