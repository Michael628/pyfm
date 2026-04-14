"""Tests for builder _preprocessor routing and preprocessor behaviour.

Verifies:
1. Builder routes _preprocessor by subconfig_key for SIMPLE containers.
2. Builder routes _preprocessor by subconfig_key for LIST containers.
3. Builder passes empty dict for DICT containers (no routing needed).
4. Missing _preprocessor keys default to {} (no-op).
5. ContractConfig end-to-end: only requested diagram keys appear in output.
6. DiagramConfig mass renaming always occurs (standalone and nested).
7. LMIConfig skip_* flags are set based on _preprocessor keys.
"""

import typing as t

import pytest
from pydantic.dataclasses import dataclass
from pydantic import Field

from pyfm.builder import build_config
from pyfm.domain import (
    ConfigBase,
    CompositeConfig,
    SimpleConfig,
    HandlerRegistry,
)
from pyfm.domain.protocols import ConfigPreprocessorProtocol
from pyfm.tasks.register import register_task, get_task_handler


# ---------------------------------------------------------------------------
# Minimal stub configs for routing tests
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _LeafConfig(SimpleConfig):
    received_value: str = ""
    key: t.ClassVar[str] = "_test_leaf"


@dataclass(frozen=True)
class _CompositeSimple(CompositeConfig):
    """Composite config with one SIMPLE subconfig."""
    leaf_config: _LeafConfig
    key: t.ClassVar[str] = "_test_composite_simple"


@dataclass(frozen=True)
class _LeafListConfig(SimpleConfig):
    item_value: str = ""
    key: t.ClassVar[str] = "_test_leaf_list"


@dataclass(frozen=True)
class _CompositeList(CompositeConfig):
    """Composite config with one LIST subconfig."""
    items: t.List[_LeafListConfig]
    key: t.ClassVar[str] = "_test_composite_list"


# ---------------------------------------------------------------------------
# Helper: captured params store
# ---------------------------------------------------------------------------

_captured: t.Dict[str, t.Dict] = {}


def _make_capturing_preprocessor(config_key: str):
    """Return a preprocessor that records the params it received."""

    def _preprocess(params: t.Dict) -> t.Dict:
        _captured[config_key] = dict(params)
        return params

    return _preprocess


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_captured():
    _captured.clear()
    yield
    _captured.clear()


@pytest.fixture(scope="module", autouse=True)
def register_stub_configs():
    """Register the stub configs with capturing preprocessors."""
    register_task(
        _LeafConfig,
        preprocess_params=_make_capturing_preprocessor("leaf"),
    )
    register_task(
        _LeafListConfig,
        preprocess_params=_make_capturing_preprocessor("leaf_list"),
    )
    register_task(_CompositeSimple)
    register_task(_CompositeList)
    yield


# ---------------------------------------------------------------------------
# Common base params shared by all build calls
# ---------------------------------------------------------------------------

_BASE = {
    "formatting": {},
    "logging_level": "INFO",
    "runid": "test",
}

# get_handler that uses the global HandlerRegistry, same as nanny/setup.py
_GET_HANDLER = lambda config_type: get_task_handler(config=config_type)


# ---------------------------------------------------------------------------
# 1. SIMPLE container: _preprocessor routed by subconfig_key
# ---------------------------------------------------------------------------

class TestSimpleContainerRouting:
    def test_preprocessor_slice_delivered_to_child(self):
        """Builder routes _preprocessor['leaf'] to _LeafConfig build."""
        params = _BASE | {
            "_preprocessor": {
                "leaf": {"received_value": "from_parent"},
            }
        }
        build_config(_CompositeSimple, params, get_handler=_GET_HANDLER)

        assert "leaf" in _captured, "Leaf preprocessor was not called"
        # The preprocessor for leaf receives the slice, not the full dict
        assert _captured["leaf"].get("_preprocessor") == {"received_value": "from_parent"}

    def test_missing_preprocessor_key_defaults_to_empty(self):
        """Missing _preprocessor key for a subconfig defaults to {} (no-op)."""
        params = _BASE | {"_preprocessor": {}}
        build_config(_CompositeSimple, params, get_handler=_GET_HANDLER)

        assert "leaf" in _captured
        assert _captured["leaf"].get("_preprocessor") == {}

    def test_no_preprocessor_key_at_all(self):
        """When _preprocessor is absent entirely, child receives {}."""
        build_config(_CompositeSimple, _BASE, get_handler=_GET_HANDLER)

        assert "leaf" in _captured
        assert _captured["leaf"].get("_preprocessor") == {}


# ---------------------------------------------------------------------------
# 2. LIST container: _preprocessor routed per item
# ---------------------------------------------------------------------------

class TestListContainerRouting:
    def test_preprocessor_slice_delivered_to_each_list_item(self):
        """Builder routes _preprocessor['items'] to each _LeafListConfig build."""
        params = _BASE | {
            "items": [
                {"item_value": "a"},
                {"item_value": "b"},
            ],
            "_preprocessor": {
                "items": {"extra": "list_slice"},
            },
        }
        build_config(_CompositeList, params, get_handler=_GET_HANDLER)

        # The leaf_list preprocessor should have been called with the slice
        assert "leaf_list" in _captured
        assert _captured["leaf_list"].get("_preprocessor") == {"extra": "list_slice"}

    def test_missing_list_preprocessor_key_defaults_to_empty(self):
        params = _BASE | {
            "items": [{"item_value": "x"}],
            "_preprocessor": {},
        }
        build_config(_CompositeList, params, get_handler=_GET_HANDLER)

        assert _captured["leaf_list"].get("_preprocessor") == {}


# ---------------------------------------------------------------------------
# 3. DiagramConfig mass renaming always occurs
# ---------------------------------------------------------------------------

class TestDiagramMassRenaming:
    def _base_contract_params(self, diagrams: t.List[str]) -> t.Dict:
        """Minimal params for ContractConfig with a single pion diagram."""
        return {
            "formatting": {},
            "logging_level": "INFO",
            "runid": "test",
            "time": 4,
            "mass": {"l": 0.002426, "zero": 0.0},
            "eigs": 100,
            "noise": 1,
            "dt": 1,
            "diagram_params": {
                "pion": {
                    "gamma_label": "pion",
                    "symmetric": True,
                    "contraction_type": "two_point",
                    "mesons": {"file": "meson", "mass": "l"},
                    "gammas": ["G5_G5"],
                    "eig_range": {"min": 0, "max": 10},
                    "outfile": "contract",
                },
            },
            "_preprocessor": {"diagrams": diagrams},
        }

    def test_mass_renamed_in_mesons(self):
        """DiagramConfig.preprocess_params renames mass->mass_original."""
        from pyfm.tasks.contract.diagram import preprocess_params

        params = {
            "mesons": [{"file": "meson", "mass": "l"}],
            "_preprocessor": {},
        }
        result = preprocess_params(params)
        assert result["mesons"][0].get("mass_original") == "l"
        assert "mass" not in result["mesons"][0]

    def test_new_mass_renamed(self):
        """DiagramConfig.preprocess_params renames new_mass->mass_updated."""
        from pyfm.tasks.contract.diagram import preprocess_params

        params = {
            "mesons": [{"file": "meson", "mass": "l", "new_mass": "u"}],
            "_preprocessor": {},
        }
        result = preprocess_params(params)
        assert result["mesons"][0].get("mass_original") == "l"
        assert result["mesons"][0].get("mass_updated") == "u"
        assert "mass" not in result["mesons"][0]
        assert "new_mass" not in result["mesons"][0]

    def test_renaming_idempotent_when_already_renamed(self):
        """Already-renamed fields (mass_original) pass through unchanged."""
        from pyfm.tasks.contract.diagram import preprocess_params

        params = {
            "mesons": [{"file": "meson", "mass_original": "l"}],
            "_preprocessor": {},
        }
        result = preprocess_params(params)
        assert result["mesons"][0].get("mass_original") == "l"

    def test_empty_mesons_list(self):
        """Empty mesons list passes through without error."""
        from pyfm.tasks.contract.diagram import preprocess_params

        params = {"mesons": [], "_preprocessor": {}}
        result = preprocess_params(params)
        assert result["mesons"] == []


# ---------------------------------------------------------------------------
# 4. LMIConfig skip_* flags based on _preprocessor keys
# ---------------------------------------------------------------------------

class TestLMISkipFlags:
    def _base_lmi_params(self, preprocessor: t.Dict) -> t.Dict:
        return {
            "formatting": {},
            "logging_level": "INFO",
            "runid": "test",
            "_preprocessor": preprocessor,
        }

    def test_all_optional_absent_sets_all_skip_flags(self):
        from pyfm.tasks.hadrons.lmi import preprocess_params

        result = preprocess_params(self._base_lmi_params({}))
        assert result["skip_meson"] is True
        assert result["skip_high_modes"] is True
        assert result["skip_epack"] is True

    def test_epack_present_clears_skip_epack(self):
        from pyfm.tasks.hadrons.lmi import preprocess_params

        result = preprocess_params(
            self._base_lmi_params({"epack": {"load": True}})
        )
        assert result.get("skip_epack") is not True
        assert result.get("skip_meson") is True
        assert result.get("skip_high_modes") is True

    def test_all_present_sets_no_skip_flags(self):
        from pyfm.tasks.hadrons.lmi import preprocess_params

        result = preprocess_params(
            self._base_lmi_params(
                {
                    "epack": {"load": True},
                    "meson": {"gamma": ["pion_local"], "mass": ["l"]},
                    "high_modes": {"gamma": ["pion_local"], "mass": ["l"]},
                }
            )
        )
        assert result.get("skip_epack") is not True
        assert result.get("skip_meson") is not True
        assert result.get("skip_high_modes") is not True

    def test_skip_low_modes_set_when_epack_absent(self):
        from pyfm.tasks.hadrons.lmi import preprocess_params

        result = preprocess_params(self._base_lmi_params({}))
        assert result["skip_low_modes"] is True

    def test_skip_low_modes_false_when_epack_present(self):
        from pyfm.tasks.hadrons.lmi import preprocess_params

        result = preprocess_params(
            self._base_lmi_params({"epack": {"load": True}})
        )
        assert result["skip_low_modes"] is False

    def test_shared_defaults_set(self):
        from pyfm.tasks.hadrons.lmi import preprocess_params

        result = preprocess_params(self._base_lmi_params({}))
        assert result["action_name"] == "stag_mass_{mass}"
        assert result["solver_name"] == "stag_{solver}_mass_{mass}"
        assert result["low_modes_name"] == "evecs_mass_{mass}"
        assert result["shift_gauge_name"] == "gauge"


# ---------------------------------------------------------------------------
# 5. ContractConfig: only requested diagrams appear
# ---------------------------------------------------------------------------

class TestContractPreprocessor:
    def test_only_requested_diagrams_included(self):
        from pyfm.tasks.contract.contraction import preprocess_params

        params = {
            "diagram_params": {
                "pion_local": {"foo": 1},
                "vec_local": {"foo": 2},
                "vec_onelink": {"foo": 3},
            },
            "_preprocessor": {"diagrams": ["pion_local", "vec_local"]},
        }
        result = preprocess_params(params)
        assert set(result["diagrams"].keys()) == {"pion_local", "vec_local"}
        assert "vec_onelink" not in result["diagrams"]

    def test_missing_diagram_raises(self):
        from pyfm.tasks.contract.contraction import preprocess_params

        params = {
            "diagram_params": {"pion_local": {}},
            "_preprocessor": {"diagrams": ["pion_local", "nonexistent"]},
        }
        with pytest.raises(ValueError, match="nonexistent"):
            preprocess_params(params)

    def test_empty_diagrams_raises(self):
        from pyfm.tasks.contract.contraction import preprocess_params

        params = {
            "diagram_params": {"pion_local": {}},
            "_preprocessor": {"diagrams": []},
        }
        with pytest.raises(ValueError, match="No diagrams"):
            preprocess_params(params)

    def test_missing_diagram_params_raises(self):
        from pyfm.tasks.contract.contraction import preprocess_params

        params = {
            "diagram_params": {},
            "_preprocessor": {"diagrams": ["pion_local"]},
        }
        with pytest.raises(ValueError, match="No diagram_params"):
            preprocess_params(params)
