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
    catalog = highmode.create_outfile_catalog(task.config.high_modes_config)
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
