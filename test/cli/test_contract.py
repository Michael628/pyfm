"""Unit tests for pyfm CLI contract subcommand dispatch logic."""
from unittest.mock import MagicMock, patch

import pytest

from pyfm.cli import cli


def _fake_config():
    cfg = MagicMock()
    cfg.diagrams = {"hvp_vl": MagicMock()}
    cfg.hardware = "cpu"
    cfg.logging_level = "INFO"
    cfg.comm_size = 1
    cfg.rank = 0
    cfg.overwrite = False
    return cfg


def test_contract_run_dispatches(runner):
    with (
        patch("pyfm.utils.io.load_param", return_value={}),
        patch("pyfm.utils.set_logging_level", return_value=MagicMock()),
        patch("pyfm.core.builder.build_config") as mock_bc,
        patch("pyfm.a2a.execute") as mock_execute,
    ):
        mock_bc.return_value = _fake_config()
        result = runner.invoke(cli, ["contract", "run", "-p", "params.yaml"])
        assert result.exit_code == 0, result.output
        mock_bc.assert_called_once()
        mock_execute.assert_not_called()


def test_contract_run_missing_param_file_fails(runner):
    result = runner.invoke(cli, ["contract", "run"])
    assert result.exit_code != 0


def test_contract_run_uses_param_file(runner):
    with (
        patch("pyfm.utils.io.load_param", return_value={}) as mock_load,
        patch("pyfm.utils.set_logging_level", return_value=MagicMock()),
        patch("pyfm.core.builder.build_config"),
        patch("pyfm.a2a.execute"),
    ):
        result = runner.invoke(cli, ["contract", "run", "-p", "my_params.yaml"])
        assert result.exit_code == 0, result.output
        mock_load.assert_called_once_with("my_params.yaml")


class TestSibDispatch:
    @pytest.fixture()
    def sib_params(self):
        return {
            "diagrams": {
                "hvp_vl": {
                    "contraction_type": "SIB",
                    "operations": {"vec_local": {"mass": ["l"]}},
                }
            },
            "time": 4,
            "logging_level": "INFO",
            "runid": "t",
        }

    def test_sniff_detects_sib(self, sib_params):
        from pyfm.cli.contract import _sniff_sib

        assert _sniff_sib(sib_params) is True
        assert _sniff_sib({"diagrams": {}}) is False
        assert _sniff_sib({}) is False
        assert (
            _sniff_sib(
                {
                    "diagrams": {
                        "pion": {"contraction_type": "TWOPOINT"},
                    }
                }
            )
            is False
        )

    def test_sib_params_build_sib_config(self, runner, sib_params):
        with (
            patch("pyfm.utils.io.load_param", return_value=sib_params),
            patch("pyfm.utils.set_logging_level", return_value=MagicMock()),
            patch("pyfm.core.builder.build_config") as mock_bc,
            patch("pyfm.a2a.execute") as mock_execute,
        ):
            cfg = _fake_config()
            cfg.time = 4
            cfg.diagrams = {"hvp_vl": MagicMock()}
            mock_bc.return_value = cfg
            mock_execute.return_value = {}
            result = runner.invoke(cli, ["contract", "run", "-p", "params.yaml"])
            assert result.exit_code == 0, result.output
            args, _kwargs = mock_bc.call_args
            from pyfm.a2a.types import SIBContractConfig

            assert args[0] is SIBContractConfig
            mock_execute.assert_called_once()

    def test_nonsib_still_builds_contract_config(self, runner):
        params = {
            "diagrams": {"pion": {"contraction_type": "TWOPOINT"}},
            "time": 4,
            "logging_level": "INFO",
            "runid": "t",
        }
        with (
            patch("pyfm.utils.io.load_param", return_value=params),
            patch("pyfm.utils.set_logging_level", return_value=MagicMock()),
            patch("pyfm.core.builder.build_config") as mock_bc,
            patch("pyfm.a2a.execute"),
        ):
            mock_bc.return_value = _fake_config()
            result = runner.invoke(cli, ["contract", "run", "-p", "params.yaml"])
            assert result.exit_code == 0, result.output
            args, _kwargs = mock_bc.call_args
            from pyfm.a2a.types import ContractConfig

            assert args[0] is ContractConfig

    def test_time_average_rejected_for_sib(self, runner, sib_params):
        with (
            patch("pyfm.utils.io.load_param", return_value=sib_params),
            patch("pyfm.utils.set_logging_level", return_value=MagicMock()),
            patch("pyfm.core.builder.build_config"),
        ):
            result = runner.invoke(
                cli, ["contract", "run", "-p", "params.yaml", "--time-average"]
            )
            assert result.exit_code != 0
            assert "not supported" in result.output
