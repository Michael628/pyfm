import pytest
import yaml

from pyfm.nanny import write_input_file

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
    assert "gauge_fatf" not in schedule
    assert "gauge_longf" not in schedule


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
