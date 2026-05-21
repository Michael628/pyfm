"""Unit tests for pyfm CLI performance subcommand dispatch logic."""

import json
from unittest.mock import patch

from pyfm.cli import cli


def test_performance_analyze_dispatches(tmp_path, runner):
    output_file = tmp_path / "hadrons.out"
    output_file.write_text("dummy log")

    with patch("pyfm.cli.performance.analyze_file") as mock_analyze:
        result = runner.invoke(cli, ["performance", "analyze", str(output_file)])

    assert result.exit_code == 0, result.output
    mock_analyze.assert_called_once_with(str(output_file))


def test_performance_analyze_requires_existing_file(runner):
    result = runner.invoke(cli, ["performance", "analyze", "missing.out"])

    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_performance_analyze_reports_no_timing_data(tmp_path, runner):
    output_file = tmp_path / "hadrons.out"
    output_file.write_text("dummy log")

    with patch(
        "pyfm.cli.performance.analyze_file",
        side_effect=ValueError("No timing data found in the file."),
    ):
        result = runner.invoke(cli, ["performance", "analyze", str(output_file)])

    assert result.exit_code != 0
    assert "No timing data found" in result.output


def test_performance_benchmark_dispatches_json(tmp_path, runner):
    log_file = tmp_path / "hadrons.out"
    param_file = tmp_path / "params.yaml"
    log_file.write_text("dummy log")
    param_file.write_text("job_setup: {}\n")

    backend_result = {"schema_version": 1, "components": {}}
    params = {"job_setup": {}}

    with (
        patch("pyfm.cli.performance.utils.io.load_param", return_value=params) as mock_load,
        patch(
            "pyfm.cli.performance.benchmark_lmi_performance",
            return_value=backend_result,
        ) as mock_benchmark,
    ):
        result = runner.invoke(
            cli,
            [
                "performance",
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


def test_performance_benchmark_reports_backend_error(tmp_path, runner):
    log_file = tmp_path / "hadrons.out"
    param_file = tmp_path / "params.yaml"
    log_file.write_text("dummy log")
    param_file.write_text("job_setup: {}\n")

    with (
        patch("pyfm.cli.performance.utils.io.load_param", return_value={"job_setup": {}}),
        patch(
            "pyfm.cli.performance.benchmark_lmi_performance",
            side_effect=ValueError("Performance benchmark only supports hadrons_lmi jobs"),
        ),
    ):
        result = runner.invoke(
            cli,
            [
                "performance",
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
    assert "Performance benchmark only supports hadrons_lmi jobs" in result.output


def test_performance_benchmark_requires_existing_log(runner):
    result = runner.invoke(
        cli,
        ["performance", "benchmark", "--job", "lmi", "--log", "missing.out"],
    )

    assert result.exit_code != 0
    assert "does not exist" in result.output
