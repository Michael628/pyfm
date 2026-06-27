"""Unit tests for pyfm CLI audit subcommand dispatch logic."""

import json
from unittest.mock import patch

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
