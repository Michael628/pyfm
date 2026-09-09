"""Unit tests for pyfm CLI export corr subcommand dispatch logic."""
from unittest.mock import patch

from pyfm.cli import cli


FAKE_PARAMS = {"tasks": []}


def test_corr_dispatches_defaults(runner):
    with (
        patch("pyfm.utils.io.load_param", return_value=FAKE_PARAMS),
        patch("pyfm.nanny.aggregator.aggregate_task_data") as mock_agg,
        patch("pyfm.utils.set_logging_level"),
    ):
        result = runner.invoke(cli, ["export", "corr", "-j", "hadrons_lmi"])
        assert result.exit_code == 0, result.output
        mock_agg.assert_called_once_with(
            "hadrons_lmi",
            FAKE_PARAMS,
            format="csv",
            average=False,
            skip_existing=False,
            generate_manifest=False,
            max_workers=1,
        )


def test_corr_average_flag(runner):
    with (
        patch("pyfm.utils.io.load_param", return_value=FAKE_PARAMS),
        patch("pyfm.nanny.aggregator.aggregate_task_data") as mock_agg,
        patch("pyfm.utils.set_logging_level"),
    ):
        result = runner.invoke(cli, ["export", "corr", "-j", "hadrons_lmi", "--average"])
        assert result.exit_code == 0, result.output
        mock_agg.assert_called_once_with(
            "hadrons_lmi",
            FAKE_PARAMS,
            format="csv",
            average=True,
            skip_existing=False,
            generate_manifest=False,
            max_workers=1,
        )


def test_corr_skip_existing_flag(runner):
    with (
        patch("pyfm.utils.io.load_param", return_value=FAKE_PARAMS),
        patch("pyfm.nanny.aggregator.aggregate_task_data") as mock_agg,
        patch("pyfm.utils.set_logging_level"),
    ):
        result = runner.invoke(cli, ["export", "corr", "-j", "hadrons_lmi", "--skip-existing"])
        assert result.exit_code == 0, result.output
        mock_agg.assert_called_once_with(
            "hadrons_lmi",
            FAKE_PARAMS,
            format="csv",
            average=False,
            skip_existing=True,
            generate_manifest=False,
            max_workers=1,
        )


def test_corr_generate_manifest_flag(runner):
    """Positive --generate-manifest dispatch on the real export corr command
    (previously covered only via the deprecated task aggregate alias)."""
    with (
        patch("pyfm.utils.io.load_param", return_value=FAKE_PARAMS),
        patch("pyfm.nanny.aggregator.aggregate_task_data") as mock_agg,
        patch("pyfm.utils.set_logging_level"),
    ):
        result = runner.invoke(cli, ["export", "corr", "-j", "hadrons_lmi", "--generate-manifest"])
        assert result.exit_code == 0, result.output
        mock_agg.assert_called_once_with(
            "hadrons_lmi",
            FAKE_PARAMS,
            format="csv",
            average=False,
            skip_existing=False,
            generate_manifest=True,
            max_workers=1,
        )


def test_corr_generate_manifest_conflicts_with_skip_existing(runner):
    with (
        patch("pyfm.utils.io.load_param", return_value=FAKE_PARAMS),
        patch("pyfm.nanny.aggregator.aggregate_task_data") as mock_agg,
        patch("pyfm.utils.set_logging_level"),
    ):
        result = runner.invoke(
            cli,
            ["export", "corr", "-j", "hadrons_lmi", "--generate-manifest", "--skip-existing"],
        )
        assert result.exit_code != 0
        mock_agg.assert_not_called()


def test_corr_custom_format(runner):
    with (
        patch("pyfm.utils.io.load_param", return_value=FAKE_PARAMS),
        patch("pyfm.nanny.aggregator.aggregate_task_data") as mock_agg,
        patch("pyfm.utils.set_logging_level"),
    ):
        result = runner.invoke(cli, ["export", "corr", "-j", "hadrons_lmi", "-f", "hdf5"])
        assert result.exit_code == 0, result.output
        mock_agg.assert_called_once_with(
            "hadrons_lmi",
            FAKE_PARAMS,
            format="hdf5",
            average=False,
            skip_existing=False,
            generate_manifest=False,
            max_workers=1,
        )


def test_corr_rejects_invalid_format(runner):
    result = runner.invoke(cli, ["export", "corr", "-j", "hadrons_lmi", "-f", "xml"])
    assert result.exit_code != 0
    assert "is not one of" in result.output


def test_corr_missing_job_fails(runner):
    result = runner.invoke(cli, ["export", "corr"])
    assert result.exit_code != 0


def test_convert_dispatches_defaults(runner):
    with (
        patch("pyfm.utils.io.load_param", return_value=FAKE_PARAMS),
        patch("pyfm.nanny.aggregator.convert_task_data") as mock_conv,
        patch("pyfm.utils.set_logging_level"),
    ):
        result = runner.invoke(
            cli, ["export", "convert", "-j", "hadrons_lmi", "-f", "hdf5"]
        )
        assert result.exit_code == 0, result.output
        mock_conv.assert_called_once_with(
            "hadrons_lmi",
            FAKE_PARAMS,
            input_format="csv",
            output_format="hdf5",
            output=None,
            average=False,
        )


def test_convert_average_flag(runner):
    """--average permits identical input/output formats and forwards average=True."""
    with (
        patch("pyfm.utils.io.load_param", return_value=FAKE_PARAMS),
        patch("pyfm.nanny.aggregator.convert_task_data") as mock_conv,
        patch("pyfm.utils.set_logging_level"),
    ):
        result = runner.invoke(cli, ["export", "convert", "-j", "hadrons_lmi", "--average"])
        assert result.exit_code == 0, result.output
        mock_conv.assert_called_once_with(
            "hadrons_lmi",
            FAKE_PARAMS,
            input_format="csv",
            output_format="csv",
            output=None,
            average=True,
        )


def test_convert_rejects_same_formats_without_average(runner):
    with (
        patch("pyfm.utils.io.load_param", return_value=FAKE_PARAMS),
        patch("pyfm.nanny.aggregator.convert_task_data") as mock_conv,
        patch("pyfm.utils.set_logging_level"),
    ):
        result = runner.invoke(cli, ["export", "convert", "-j", "hadrons_lmi"])
        assert result.exit_code != 0
        assert "unless --average" in result.output
        mock_conv.assert_not_called()


def test_convert_input_and_output_format(runner):
    with (
        patch("pyfm.utils.io.load_param", return_value=FAKE_PARAMS),
        patch("pyfm.nanny.aggregator.convert_task_data") as mock_conv,
        patch("pyfm.utils.set_logging_level"),
    ):
        result = runner.invoke(
            cli,
            ["export", "convert", "-j", "hadrons_lmi", "--input-format", "hdf5", "-f", "parquet"],
        )
        assert result.exit_code == 0, result.output
        mock_conv.assert_called_once_with(
            "hadrons_lmi",
            FAKE_PARAMS,
            input_format="hdf5",
            output_format="parquet",
            output=None,
            average=False,
        )


def test_convert_output_flag(runner):
    with (
        patch("pyfm.utils.io.load_param", return_value=FAKE_PARAMS),
        patch("pyfm.nanny.aggregator.convert_task_data") as mock_conv,
        patch("pyfm.utils.set_logging_level"),
    ):
        result = runner.invoke(
            cli,
            [
                "export",
                "convert",
                "-j",
                "hadrons_lmi",
                "-f",
                "hdf5",
                "--output",
                "out/converted.csv",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_conv.assert_called_once_with(
            "hadrons_lmi",
            FAKE_PARAMS,
            input_format="csv",
            output_format="hdf5",
            output="out/converted.csv",
            average=False,
        )


def test_convert_rejects_invalid_input_format(runner):
    result = runner.invoke(cli, ["export", "convert", "-j", "hadrons_lmi", "--input-format", "xml"])
    assert result.exit_code != 0
    assert "is not one of" in result.output


def test_convert_rejects_invalid_output_format(runner):
    result = runner.invoke(cli, ["export", "convert", "-j", "hadrons_lmi", "-f", "xml"])
    assert result.exit_code != 0
    assert "is not one of" in result.output


def test_convert_missing_job_fails(runner):
    result = runner.invoke(cli, ["export", "convert"])
    assert result.exit_code != 0


def test_tar_dispatches_job(runner):
    with (
        patch("pyfm.utils.io.load_param", return_value=FAKE_PARAMS),
        patch("pyfm.nanny.aggregator.tar_task_data") as mock_tar,
        patch("pyfm.utils.set_logging_level"),
    ):
        result = runner.invoke(cli, ["export", "tar", "-j", "hadrons_lmi"])
        assert result.exit_code == 0, result.output
        mock_tar.assert_called_once_with(
            "hadrons_lmi", FAKE_PARAMS, include_dirs=(), output=None
        )


def test_tar_output_flag(runner):
    with (
        patch("pyfm.utils.io.load_param", return_value=FAKE_PARAMS),
        patch("pyfm.nanny.aggregator.tar_task_data") as mock_tar,
        patch("pyfm.utils.set_logging_level"),
    ):
        result = runner.invoke(
            cli, ["export", "tar", "-j", "hadrons_lmi", "-o", "myrun.tar"]
        )
        assert result.exit_code == 0, result.output
        mock_tar.assert_called_once_with(
            "hadrons_lmi", FAKE_PARAMS, include_dirs=(), output="myrun.tar"
        )


def test_tar_include_dirs(runner, tmp_path):
    d1 = tmp_path / "data1"
    d2 = tmp_path / "data2"
    d1.mkdir()
    d2.mkdir()
    with (
        patch("pyfm.utils.io.load_param", return_value=FAKE_PARAMS),
        patch("pyfm.nanny.aggregator.tar_task_data") as mock_tar,
        patch("pyfm.utils.set_logging_level"),
    ):
        result = runner.invoke(
            cli,
            ["export", "tar", "-j", "hadrons_lmi",
             "--include", str(d1), "--include", str(d2)],
        )
        assert result.exit_code == 0, result.output
        mock_tar.assert_called_once_with(
            "hadrons_lmi", FAKE_PARAMS, include_dirs=(str(d1), str(d2)), output=None
        )


def test_tar_include_only(runner, tmp_path):
    d = tmp_path / "extras"
    d.mkdir()
    with (
        patch("pyfm.utils.io.load_param", return_value=FAKE_PARAMS),
        patch("pyfm.nanny.aggregator.tar_task_data") as mock_tar,
        patch("pyfm.utils.set_logging_level"),
    ):
        result = runner.invoke(
            cli, ["export", "tar", "--include", str(d), "-o", "bundle.tar"]
        )
        assert result.exit_code == 0, result.output
        mock_tar.assert_called_once_with(
            None, FAKE_PARAMS, include_dirs=(str(d),), output="bundle.tar"
        )


def test_tar_missing_both_fails(runner):
    result = runner.invoke(cli, ["export", "tar"])
    assert result.exit_code != 0
    assert "Nothing to archive" in result.output
