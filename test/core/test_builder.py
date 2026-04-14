"""Tests for build_config() in pyfm/core/builder.py (issue #4).

Coverage:
- Simple config building (no hooks, with hooks)
- Composite config building (no hooks, with hooks)
- Recursive hook application per config type
- Missing hooks are a no-op
- Hooks called in correct order: preprocess -> construct -> postprocess -> validate
"""
import pytest
from dataclasses import field
from typing import Any, Dict, List

from pydantic.dataclasses import dataclass

from pyfm.domain.conftypes import SimpleConfig, CompositeConfig
from pyfm.domain import build_hooks
from pyfm.core.builder import build_config


# ---------------------------------------------------------------------------
# Minimal config types for testing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PartConfig(SimpleConfig):
    """Leaf / sub-config used inside composite configs."""
    color: str = ""
    size: int = 0


@dataclass(frozen=True)
class PartListConfig(SimpleConfig):
    """Sub-config used in LIST container tests."""
    tag: str = ""


@dataclass(frozen=True)
class BoxConfig(CompositeConfig):
    """Composite config that embeds a single PartConfig."""
    part_config: PartConfig = None  # type: ignore[assignment]
    label: str = ""


@dataclass(frozen=True)
class MultiPartConfig(CompositeConfig):
    """Composite config that embeds a list of PartListConfigs."""
    part_list_config: List[PartListConfig] = field(default_factory=list)
    name: str = ""


@dataclass(frozen=True)
class DictPartsConfig(CompositeConfig):
    """Composite config with a DICT container of PartConfig children."""
    parts_config: Dict[str, PartConfig] = field(default_factory=dict)
    title: str = ""


# ---------------------------------------------------------------------------
# Base params helpers
# ---------------------------------------------------------------------------

BASE = {"formatting": {}, "logging_level": "INFO", "runid": "run0"}


def make_params(**extra):
    return BASE | extra


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_hooks():
    """Clear the build_hooks registry before and after every test."""
    build_hooks.clear()
    yield
    build_hooks.clear()


# ---------------------------------------------------------------------------
# Simple config — no hooks
# ---------------------------------------------------------------------------

class TestSimpleConfigNoHooks:
    def test_basic_construction(self):
        params = make_params(color="blue", size=3)
        config = build_config(PartConfig, params)
        assert isinstance(config, PartConfig)
        assert config.color == "blue"
        assert config.size == 3

    def test_base_fields_set(self):
        params = make_params(color="red")
        config = build_config(PartConfig, params)
        assert config.logging_level == "INFO"
        assert config.runid == "run0"

    def test_file_params_default_to_empty_dict(self):
        params = make_params(color="green")
        # Should not raise even when file_params is omitted
        config = build_config(PartConfig, params)
        assert config.color == "green"

    def test_explicit_empty_file_params(self):
        params = make_params(color="yellow")
        config = build_config(PartConfig, params, file_params={})
        assert config.color == "yellow"


# ---------------------------------------------------------------------------
# Simple config — with hooks
# ---------------------------------------------------------------------------

class TestSimpleConfigWithHooks:
    def test_preprocess_called(self):
        """Preprocess hook transforms params before construction."""
        def preprocess(p):
            return p | {"color": "overridden"}

        build_hooks.register(PartConfig, preprocess=preprocess)
        params = make_params(color="original")
        config = build_config(PartConfig, params)
        assert config.color == "overridden"

    def test_postprocess_called(self):
        """Postprocess hook receives the built config and its return value is used."""
        postprocessed = []

        def postprocess(config):
            postprocessed.append(config)
            return config

        build_hooks.register(PartConfig, postprocess=postprocess)
        params = make_params(color="blue")
        config = build_config(PartConfig, params)
        assert len(postprocessed) == 1
        assert postprocessed[0] is config

    def test_validate_called(self):
        """Validate hook is called after postprocessing."""
        validated = []

        def validate(config):
            validated.append(config.color)

        build_hooks.register(PartConfig, validate=validate)
        params = make_params(color="purple")
        build_config(PartConfig, params)
        assert validated == ["purple"]

    def test_validate_can_raise(self):
        """Validate hook can raise to reject invalid configs."""
        def validate(config):
            if config.color == "invalid":
                raise ValueError("color is invalid")

        build_hooks.register(PartConfig, validate=validate)
        with pytest.raises(ValueError, match="color is invalid"):
            build_config(PartConfig, make_params(color="invalid"))

    def test_hooks_called_in_order(self):
        """Hooks must fire in the order: preprocess -> construct -> postprocess -> validate."""
        call_order = []

        def preprocess(p):
            call_order.append("preprocess")
            return p

        def postprocess(config):
            call_order.append("postprocess")
            return config

        def validate(config):
            call_order.append("validate")

        build_hooks.register(
            PartConfig,
            preprocess=preprocess,
            postprocess=postprocess,
            validate=validate,
        )
        build_config(PartConfig, make_params())
        assert call_order == ["preprocess", "postprocess", "validate"]

    def test_postprocess_return_value_used(self):
        """The config returned by postprocess is used as the final result."""
        replacement = PartConfig(**make_params(color="replaced"))

        def postprocess(config):
            return replacement

        build_hooks.register(PartConfig, postprocess=postprocess)
        result = build_config(PartConfig, make_params(color="original"))
        assert result is replacement
        assert result.color == "replaced"

    def test_missing_hooks_are_noop(self):
        """A config type with no registered hooks builds normally."""
        params = make_params(color="noop")
        config = build_config(PartConfig, params)
        assert config.color == "noop"

    def test_partial_hooks_only_preprocess(self):
        """Only a preprocess hook — postprocess and validate should not be called."""
        called = []

        def preprocess(p):
            called.append("pre")
            return p | {"color": "pre-only"}

        build_hooks.register(PartConfig, preprocess=preprocess)
        config = build_config(PartConfig, make_params(color="original"))
        assert called == ["pre"]
        assert config.color == "pre-only"


# ---------------------------------------------------------------------------
# Composite config — no hooks
# ---------------------------------------------------------------------------

class TestCompositeConfigNoHooks:
    def test_basic_construction(self):
        params = make_params(color="cyan", size=7, label="mybox")
        config = build_config(BoxConfig, params)
        assert isinstance(config, BoxConfig)
        assert config.label == "mybox"
        assert isinstance(config.part_config, PartConfig)

    def test_subconfig_fields_populated(self):
        params = make_params(color="magenta", size=5)
        config = build_config(BoxConfig, params)
        assert config.part_config.color == "magenta"
        assert config.part_config.size == 5

    def test_subconfig_inherits_base_fields(self):
        params = make_params(color="teal")
        config = build_config(BoxConfig, params)
        assert config.part_config.logging_level == "INFO"
        assert config.part_config.runid == "run0"


# ---------------------------------------------------------------------------
# Composite config — per-type hooks applied recursively
# ---------------------------------------------------------------------------

class TestCompositeConfigWithHooks:
    def test_parent_preprocess_applied(self):
        """Parent preprocess hook transforms params before subconfig construction."""
        def box_preprocess(p):
            return p | {"label": "from-hook"}

        build_hooks.register(BoxConfig, preprocess=box_preprocess)
        params = make_params(color="orange", label="original")
        config = build_config(BoxConfig, params)
        assert config.label == "from-hook"

    def test_subconfig_preprocess_applied(self):
        """Subconfig's own preprocess hook is applied when it is recursively built."""
        def part_preprocess(p):
            return p | {"color": "sub-hook-color"}

        build_hooks.register(PartConfig, preprocess=part_preprocess)
        params = make_params(color="ignored", label="box")
        config = build_config(BoxConfig, params)
        assert config.part_config.color == "sub-hook-color"

    def test_parent_and_subconfig_hooks_independent(self):
        """Parent and subconfig hooks are each applied to the correct config type."""
        box_calls = []
        part_calls = []

        def box_pre(p):
            box_calls.append("box_pre")
            return p

        def part_pre(p):
            part_calls.append("part_pre")
            return p

        build_hooks.register(BoxConfig, preprocess=box_pre)
        build_hooks.register(PartConfig, preprocess=part_pre)

        build_config(BoxConfig, make_params(color="red"))
        assert box_calls == ["box_pre"]
        assert part_calls == ["part_pre"]

    def test_parent_postprocess_called_after_subconfig_built(self):
        """Parent postprocess fires after the composite (including subconfigs) is built."""
        post_received = []

        def box_post(config):
            post_received.append(config)
            return config

        build_hooks.register(BoxConfig, postprocess=box_post)
        config = build_config(BoxConfig, make_params(color="blue"))
        assert len(post_received) == 1
        assert isinstance(post_received[0], BoxConfig)
        # Subconfig must already be populated when postprocess fires
        assert isinstance(post_received[0].part_config, PartConfig)

    def test_subconfig_postprocess_called(self):
        """Subconfig's postprocess hook fires during its own build_config recursion."""
        part_post_called = []

        def part_post(config):
            part_post_called.append(config.color)
            return config

        build_hooks.register(PartConfig, postprocess=part_post)
        build_config(BoxConfig, make_params(color="teal"))
        assert part_post_called == ["teal"]

    def test_validate_called_for_subconfig(self):
        """Validate hook for a subconfig type is called during recursive construction."""
        validated_colors = []

        def part_validate(config):
            validated_colors.append(config.color)

        build_hooks.register(PartConfig, validate=part_validate)
        build_config(BoxConfig, make_params(color="gold"))
        assert validated_colors == ["gold"]

    def test_hook_order_for_composite(self):
        """Full order: parent-pre -> sub-pre -> sub-post -> sub-validate -> parent-post -> parent-validate."""
        order = []

        build_hooks.register(
            BoxConfig,
            preprocess=lambda p: (order.append("box_pre"), p)[1],
            postprocess=lambda c: (order.append("box_post"), c)[1],
            validate=lambda c: order.append("box_validate"),
        )
        build_hooks.register(
            PartConfig,
            preprocess=lambda p: (order.append("part_pre"), p)[1],
            postprocess=lambda c: (order.append("part_post"), c)[1],
            validate=lambda c: order.append("part_validate"),
        )
        build_config(BoxConfig, make_params(color="white"))
        assert order == [
            "box_pre",
            "part_pre",
            "part_post",
            "part_validate",
            "box_post",
            "box_validate",
        ]


# ---------------------------------------------------------------------------
# List container composite config
# ---------------------------------------------------------------------------

class TestListContainerComposite:
    def test_list_subconfig_empty_preprocessor_list(self):
        """LIST container with missing or empty _preprocessor slice yields []."""
        params = make_params(name="empty", _preprocessor={"part_list_config": []})
        config = build_config(MultiPartConfig, params)
        assert config.part_list_config == []

        params2 = make_params(name="missing")
        config2 = build_config(MultiPartConfig, params2)
        assert config2.part_list_config == []

    def test_list_subconfig_from_preprocessor_list(self):
        """LIST items are taken only from _preprocessor[field_name] as a list of dicts."""
        params = make_params(
            name="multi",
            _preprocessor={
                "part_list_config": [{"tag": "a"}, {"tag": "b"}, {"tag": "c"}],
            },
        )
        config = build_config(MultiPartConfig, params)
        assert len(config.part_list_config) == 3
        tags = [s.tag for s in config.part_list_config]
        assert tags == ["a", "b", "c"]

    def test_list_subconfig_rejects_non_list_preprocessor(self):
        with pytest.raises(TypeError, match="must be a list"):
            build_config(
                MultiPartConfig,
                make_params(
                    name="bad",
                    _preprocessor={"part_list_config": {"tag": "not-a-list"}},
                ),
            )

    def test_list_subconfig_hooks_applied_per_item(self):
        """Subconfig hooks are applied to each item in a list container."""
        part_pre_calls = []

        def part_pre(p):
            part_pre_calls.append(p.get("tag", ""))
            return p

        build_hooks.register(PartListConfig, preprocess=part_pre)
        params = make_params(
            _preprocessor={"part_list_config": [{"tag": "x"}, {"tag": "y"}]},
        )
        build_config(MultiPartConfig, params)
        assert len(part_pre_calls) == 2


# ---------------------------------------------------------------------------
# _preprocessor routing — SIMPLE container
# ---------------------------------------------------------------------------

class TestPreprocessorRoutingSimple:
    def test_subconfig_receives_its_slice(self):
        """Builder routes _preprocessor[subconfig_key] to the child's preprocess hook."""
        received = {}

        def part_pre(p):
            received["preprocessor"] = p.get("_preprocessor")
            return p

        build_hooks.register(PartConfig, preprocess=part_pre)
        params = make_params(
            _preprocessor={"part_config": {"color": "routed"}},
        )
        build_config(BoxConfig, params)
        assert received["preprocessor"] == {"color": "routed"}

    def test_missing_subconfig_key_defaults_to_empty(self):
        """When _preprocessor has no entry for a subconfig, child receives {}."""
        received = {}

        def part_pre(p):
            received["preprocessor"] = p.get("_preprocessor")
            return p

        build_hooks.register(PartConfig, preprocess=part_pre)
        params = make_params(_preprocessor={})
        build_config(BoxConfig, params)
        assert received["preprocessor"] == {}

    def test_absent_preprocessor_defaults_to_empty(self):
        """When _preprocessor is entirely absent, child receives {}."""
        received = {}

        def part_pre(p):
            received["preprocessor"] = p.get("_preprocessor")
            return p

        build_hooks.register(PartConfig, preprocess=part_pre)
        build_config(BoxConfig, make_params())
        assert received["preprocessor"] == {}

    def test_parent_preprocessor_consumes_and_routes(self):
        """Parent preprocess sets _preprocessor; builder routes the child's slice."""
        received = {}

        def box_pre(p):
            return p | {"_preprocessor": {"part_config": {"color": "injected"}}}

        def part_pre(p):
            received["preprocessor"] = p.get("_preprocessor")
            return p

        build_hooks.register(BoxConfig, preprocess=box_pre)
        build_hooks.register(PartConfig, preprocess=part_pre)
        build_config(BoxConfig, make_params())
        assert received["preprocessor"] == {"color": "injected"}

    def test_child_preprocessor_uses_routed_slice(self):
        """Child's preprocessor applies its routed _preprocessor slice to produce config fields."""
        def part_pre(p):
            return p | p.get("_preprocessor", {})

        build_hooks.register(PartConfig, preprocess=part_pre)
        params = make_params(
            color="original",
            _preprocessor={"part_config": {"color": "overridden"}},
        )
        config = build_config(BoxConfig, params)
        assert config.part_config.color == "overridden"

    def test_sibling_subconfig_slices_are_independent(self):
        """Each subconfig key receives only its own slice, not the full _preprocessor."""
        received = {}

        def part_pre(p):
            received["part_preprocessor"] = p.get("_preprocessor")
            return p

        build_hooks.register(PartConfig, preprocess=part_pre)
        params = make_params(
            _preprocessor={
                "part_config": {"color": "for-part"},
                "noise_config": {"color": "for-noise"},
            },
        )
        build_config(BoxConfig, params)
        assert received["part_preprocessor"] == {"color": "for-part"}


# ---------------------------------------------------------------------------
# _preprocessor routing — LIST container
# ---------------------------------------------------------------------------

class TestPreprocessorRoutingList:
    def test_list_items_merge_params_from_preprocessor_entries(self):
        """LIST children are built from merged params + each _preprocessor list entry."""
        received = []

        def part_list_pre(p):
            received.append(p.get("_preprocessor"))
            return p

        build_hooks.register(PartListConfig, preprocess=part_list_pre)
        params = make_params(
            _preprocessor={
                "part_list_config": [
                    {"tag": "a", "extra": "value"},
                    {"tag": "b", "extra": "value"},
                ],
            },
        )
        build_config(MultiPartConfig, params)
        assert len(received) == 2
        assert all(r == {} for r in received)

    def test_list_missing_slice_defaults_to_empty(self):
        """Each LIST child receives an empty _preprocessor routing dict."""
        received = []

        def part_list_pre(p):
            received.append(p.get("_preprocessor"))
            return p

        build_hooks.register(PartListConfig, preprocess=part_list_pre)
        params = make_params(
            _preprocessor={"part_list_config": [{"tag": "x"}, {"tag": "y"}]},
        )
        build_config(MultiPartConfig, params)
        assert all(r == {} for r in received)


# ---------------------------------------------------------------------------
# _preprocessor routing — DICT container
# ---------------------------------------------------------------------------


class TestDictContainerComposite:
    def test_dict_subconfig_empty_params_dict(self):
        params = make_params(title="t", parts_config={})
        config = build_config(DictPartsConfig, params)
        assert config.parts_config == {}

    def test_dict_subconfig_builds_one_child_per_key(self):
        params = make_params(
            title="crate",
            parts_config={"a": {"color": "red", "size": 1}, "b": {"color": "blue", "size": 2}},
        )
        config = build_config(DictPartsConfig, params)
        assert set(config.parts_config.keys()) == {"a", "b"}
        assert config.parts_config["a"].color == "red"
        assert config.parts_config["b"].size == 2

    def test_dict_per_key_preprocessor_slices(self):
        received: Dict[str, Any] = {}

        def part_pre(p):
            received[p.get("color", "")] = p.get("_preprocessor")
            return p

        build_hooks.register(PartConfig, preprocess=part_pre)
        params = make_params(
            title="x",
            parts_config={"left": {"color": "L"}, "right": {"color": "R"}},
            _preprocessor={
                "parts_config": {
                    "left": {"size": 9},
                    "right": {"size": 7},
                }
            },
        )
        build_config(DictPartsConfig, params)
        assert received["L"] == {"size": 9}
        assert received["R"] == {"size": 7}

    def test_dict_missing_preprocessor_key_per_entry_defaults_empty(self):
        received = {}

        def part_pre(p):
            received[p["color"]] = p.get("_preprocessor")
            return p

        build_hooks.register(PartConfig, preprocess=part_pre)
        params = make_params(
            title="y",
            parts_config={"only": {"color": "c"}},
            _preprocessor={"parts_config": {}},
        )
        build_config(DictPartsConfig, params)
        assert received["c"] == {}
