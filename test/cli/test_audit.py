"""Unit tests for pyfm CLI audit subcommand dispatch logic."""

import json
from unittest.mock import patch

import pandas as pd

from pyfm.cli import cli


def test_audit_runtime_dispatches(tmp_path, runner):
    output_file = tmp_path / "hadrons.out"
    output_file.write_text("dummy log")

    with patch("pyfm.cli.audit.analyze_file") as mock_analyze:
        result = runner.invoke(cli, ["audit", "runtime", str(output_file)])

    assert result.exit_code == 0, result.output
    mock_analyze.assert_called_once_with(str(output_file))


def test_audit_runtime_requires_existing_file(runner):
    result = runner.invoke(cli, ["audit", "runtime", "missing.out"])

    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_audit_output_dispatches(runner):
    report = pd.DataFrame(
        [
            {
                "gamma_label": "vec_local",
                "mass": "01",
                "tsource": "0",
                "dset": "ranLL",
                "filepath_a": "a.h5",
                "filepath_b": "b.h5",
                "max_abs_diff": 0.0,
                "max_rel_diff": 0.0,
                "within_tolerance": True,
                "status": "compared",
            }
        ]
    )
    params = {"job_setup": {}}
    with (
        patch("pyfm.cli.audit.utils.io.load_param", return_value=params),
        patch("pyfm.cli.audit.compare_task_outputs", return_value=report) as mock_cmp,
    ):
        result = runner.invoke(
            cli, ["audit", "output", "-j", "a", "b", "-s", "e", "-n", "100"]
        )
    assert result.exit_code == 0, result.output
    mock_cmp.assert_called_once_with(params, "a", "b", "e", "100", rtol=1e-9, atol=1e-12)


def test_audit_output_exit_nonzero_on_mismatch(runner):
    report = pd.DataFrame(
        [
            {
                "gamma_label": "vec_local",
                "mass": "01",
                "tsource": "0",
                "dset": "ranLL",
                "filepath_a": "a.h5",
                "filepath_b": "b.h5",
                "max_abs_diff": 1.0,
                "max_rel_diff": 1.0,
                "within_tolerance": False,
                "status": "compared",
            }
        ]
    )
    with (
        patch("pyfm.cli.audit.utils.io.load_param", return_value={"job_setup": {}}),
        patch("pyfm.cli.audit.compare_task_outputs", return_value=report),
    ):
        result = runner.invoke(
            cli, ["audit", "output", "-j", "a", "b", "-s", "e", "-n", "100"]
        )
    assert result.exit_code == 1
    assert "OUTSIDE tolerance" in result.output


def test_audit_output_reports_hard_failure(runner):
    with (
        patch("pyfm.cli.audit.utils.io.load_param", return_value={"job_setup": {}}),
        patch(
            "pyfm.cli.audit.compare_task_outputs",
            side_effect=ValueError("No comparison protocol"),
        ),
    ):
        result = runner.invoke(
            cli, ["audit", "output", "-j", "a", "b", "-s", "e", "-n", "100"]
        )
    assert result.exit_code != 0
    assert "No comparison protocol" in result.output


def test_audit_output_requires_two_jobs(runner):
    result = runner.invoke(
        cli, ["audit", "output", "-j", "a", "-s", "e", "-n", "100"]
    )
    assert result.exit_code != 0


def test_audit_runtime_reports_no_timing_data(tmp_path, runner):
    output_file = tmp_path / "hadrons.out"
    output_file.write_text("dummy log")

    with patch(
        "pyfm.cli.audit.analyze_file",
        side_effect=ValueError("No timing data found in the file."),
    ):
        result = runner.invoke(cli, ["audit", "runtime", str(output_file)])

    assert result.exit_code != 0
    assert "No timing data found" in result.output


def test_audit_benchmark_dispatches_json(tmp_path, runner):
    log_file = tmp_path / "hadrons.out"
    param_file = tmp_path / "params.yaml"
    log_file.write_text("dummy log")
    param_file.write_text("job_setup: {}\n")

    backend_result = {"schema_version": 1, "components": {}}
    params = {"job_setup": {}}

    with (
        patch("pyfm.cli.audit.utils.io.load_param", return_value=params) as mock_load,
        patch(
            "pyfm.cli.audit.benchmark_lmi_performance",
            return_value=backend_result,
        ) as mock_benchmark,
    ):
        result = runner.invoke(
            cli,
            [
                "audit",
                "benchmark",
                "--job",
                "lmi",
                "--log",
                str(log_file),
                "-p",
                str(param_file),
            ],
        )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == backend_result
    mock_load.assert_called_once_with(str(param_file))
    mock_benchmark.assert_called_once_with("lmi", str(log_file), params)


def test_audit_benchmark_reports_backend_error(tmp_path, runner):
    log_file = tmp_path / "hadrons.out"
    param_file = tmp_path / "params.yaml"
    log_file.write_text("dummy log")
    param_file.write_text("job_setup: {}\n")

    with (
        patch("pyfm.cli.audit.utils.io.load_param", return_value={"job_setup": {}}),
        patch(
            "pyfm.cli.audit.benchmark_lmi_performance",
            side_effect=ValueError(
                "Performance benchmark only supports Hadrons LMI and Grid LMI jobs"
            ),
        ),
    ):
        result = runner.invoke(
            cli,
            [
                "audit",
                "benchmark",
                "--job",
                "other",
                "--log",
                str(log_file),
                "-p",
                str(param_file),
            ],
        )

    assert result.exit_code != 0
    assert "Performance benchmark only supports Hadrons LMI and Grid LMI jobs" in result.output


def test_audit_benchmark_requires_existing_log(runner):
    result = runner.invoke(
        cli,
        ["audit", "benchmark", "--job", "lmi", "--log", "missing.out"],
    )

    assert result.exit_code != 0
    assert "does not exist" in result.output
