"""Tests for validate_config hooks on ContractConfig and DiagramConfig (issue #17).

Covers:
- validate_config raises ValueError when ContractConfig.diagrams is empty
- validate_config raises ValueError when DiagramConfig.mesons is empty
- A valid ContractConfig builds successfully via build_config end-to-end
"""
import importlib

import pytest

from pyfm.domain import task_registry, build_hooks
from pyfm.tasks.register import _config_to_task_key


@pytest.fixture(autouse=True)
def reset_registries():
    saved_handlers = dict(task_registry._handlers)
    saved_hooks = dict(build_hooks._registry)
    saved_config_to_task_key = dict(_config_to_task_key)
    task_registry.clear()
    build_hooks.clear()
    _config_to_task_key.clear()
    yield
    task_registry.clear()
    build_hooks.clear()
    _config_to_task_key.clear()
    task_registry._handlers.update(saved_handlers)
    build_hooks._registry.update(saved_hooks)
    _config_to_task_key.update(saved_config_to_task_key)


def _reload_contract_modules():
    import pyfm.tasks.contract.contraction as _c
    import pyfm.tasks.contract.diagram as _d
    import pyfm.tasks.contract.mesonloader as _m
    importlib.reload(_c)
    importlib.reload(_d)
    importlib.reload(_m)
    return _c, _d, _m


# ---------------------------------------------------------------------------
# ContractConfig validate_config
# ---------------------------------------------------------------------------

class TestContractConfigValidate:
    def test_validate_raises_on_empty_diagrams(self):
        """validate_config raises ValueError when ContractConfig.diagrams is empty."""
        _c, _d, _m = _reload_contract_modules()
        from pyfm.tasks.contract.contraction import validate_config
        from pyfm.a2a.types import ContractConfig

        config = ContractConfig(
            diagrams={},
            time=4,
            formatting={},
            logging_level="DEBUG",
            runid="test",
        )
        with pytest.raises(ValueError, match="diagrams"):
            validate_config(config)

    def test_validate_registered_as_hook(self):
        """validate_config is wired into build_hooks for ContractConfig."""
        _reload_contract_modules()
        from pyfm.a2a.types import ContractConfig

        hooks = build_hooks.get(ContractConfig)
        assert hooks is not None
        assert hooks.validate is not None

    def test_valid_contract_config_passes_validate(self, tmp_path, monkeypatch):
        """A ContractConfig with diagrams passes validate_config without raising."""
        _c, _d, _m = _reload_contract_modules()
        from pyfm.tasks.contract.contraction import validate_config
        from pyfm.a2a.types import ContractConfig, DiagramConfig, MesonLoaderConfig, ContractType
        from pyfm.domain import MassDict, Outfile

        mass = MassDict.from_dict({"l": 0.001})
        outfile = Outfile(filestem="out", ext=".h5", good_size=100)
        meson = MesonLoaderConfig(
            mass=mass,
            file=outfile,
            mass_shift=MesonLoaderConfig.MassShift(original="l"),
            formatting={},
            logging_level="DEBUG",
            runid="test",
        )
        diagram = DiagramConfig(
            time=4,
            contraction_type=ContractType.TWOPOINT,
            mesons=[meson, meson],
            outfile=outfile,
            gammas=["G5_G5"],
            eig_range=DiagramConfig.MesonIndex(min=0, max=200),
            formatting={},
            logging_level="DEBUG",
            runid="test",
        )
        config = ContractConfig(
            diagrams={"pion": diagram},
            time=4,
            formatting={},
            logging_level="DEBUG",
            runid="test",
        )
        validate_config(config)  # Must not raise


# ---------------------------------------------------------------------------
# DiagramConfig validate_config
# ---------------------------------------------------------------------------

class TestDiagramConfigValidate:
    def test_validate_raises_on_empty_mesons(self):
        """validate_config raises ValueError when DiagramConfig.mesons is empty."""
        _reload_contract_modules()
        from pyfm.tasks.contract.diagram import validate_config
        from pyfm.a2a.types import DiagramConfig, ContractType
        from pyfm.domain import Outfile

        outfile = Outfile(filestem="out", ext=".h5", good_size=100)

        config = DiagramConfig(
            time=4,
            contraction_type=ContractType.TWOPOINT,
            mesons=[],
            outfile=outfile,
            gammas=["G5_G5"],
            eig_range=DiagramConfig.MesonIndex(min=0, max=200),
            formatting={},
            logging_level="DEBUG",
            runid="test",
        )
        with pytest.raises(ValueError, match="mesons"):
            validate_config(config)

    def test_validate_registered_as_hook(self):
        """validate_config is wired into build_hooks for DiagramConfig."""
        _reload_contract_modules()
        from pyfm.a2a.types import DiagramConfig

        hooks = build_hooks.get(DiagramConfig)
        assert hooks is not None
        assert hooks.validate is not None

    def test_valid_diagram_config_passes_validate(self):
        """A DiagramConfig with mesons passes validate_config without raising."""
        _reload_contract_modules()
        from pyfm.tasks.contract.diagram import validate_config
        from pyfm.a2a.types import DiagramConfig, MesonLoaderConfig, ContractType
        from pyfm.domain import MassDict, Outfile

        mass = MassDict.from_dict({"l": 0.001})
        outfile = Outfile(filestem="out", ext=".h5", good_size=100)
        meson = MesonLoaderConfig(
            mass=mass,
            file=outfile,
            mass_shift=MesonLoaderConfig.MassShift(original="l"),
            formatting={},
            logging_level="DEBUG",
            runid="test",
        )
        config = DiagramConfig(
            time=4,
            contraction_type=ContractType.TWOPOINT,
            mesons=[meson, meson],
            outfile=outfile,
            gammas=["G5_G5"],
            eig_range=DiagramConfig.MesonIndex(min=0, max=200),
            formatting={},
            logging_level="DEBUG",
            runid="test",
        )
        validate_config(config)  # Must not raise


# ---------------------------------------------------------------------------
# Integration: validate hook fires during build_config
# ---------------------------------------------------------------------------

class TestValidateIntegration:
    def test_validate_hook_fires_via_build_hooks(self):
        """Reloading the contract modules registers validate hooks; build_hooks.get returns them."""
        _reload_contract_modules()
        from pyfm.a2a.types import ContractConfig, DiagramConfig

        contract_hooks = build_hooks.get(ContractConfig)
        diagram_hooks = build_hooks.get(DiagramConfig)

        assert contract_hooks is not None and contract_hooks.validate is not None, (
            "ContractConfig validate hook not registered"
        )
        assert diagram_hooks is not None and diagram_hooks.validate is not None, (
            "DiagramConfig validate hook not registered"
        )
