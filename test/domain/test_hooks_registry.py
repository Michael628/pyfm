"""Tests for BuildHooks and BuildHooksRegistry (pyfm/domain/hooks.py)."""
import pytest
from pyfm.domain.hooks import BuildHooks, BuildHooksRegistry


# ---------------------------------------------------------------------------
# Helper stubs
# ---------------------------------------------------------------------------

class FakeConfigA:
    pass


class FakeConfigB:
    pass


class FakeConfigC:
    pass


def preprocess_stub(params):
    return params


def postprocess_stub(config):
    return config


def validate_stub(config):
    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_registry():
    """Clear the registry before (and after) every test for isolation."""
    BuildHooksRegistry.clear()
    yield
    BuildHooksRegistry.clear()


# ---------------------------------------------------------------------------
# BuildHooks dataclass
# ---------------------------------------------------------------------------

class TestBuildHooks:
    def test_all_fields_default_to_none(self):
        hooks = BuildHooks()
        assert hooks.preprocess is None
        assert hooks.postprocess is None
        assert hooks.validate is None

    def test_fields_can_be_set(self):
        hooks = BuildHooks(
            preprocess=preprocess_stub,
            postprocess=postprocess_stub,
            validate=validate_stub,
        )
        assert hooks.preprocess is preprocess_stub
        assert hooks.postprocess is postprocess_stub
        assert hooks.validate is validate_stub

    def test_is_frozen(self):
        hooks = BuildHooks(preprocess=preprocess_stub)
        with pytest.raises((AttributeError, TypeError)):
            hooks.preprocess = None  # type: ignore[misc]

    def test_partial_fields(self):
        hooks = BuildHooks(validate=validate_stub)
        assert hooks.preprocess is None
        assert hooks.postprocess is None
        assert hooks.validate is validate_stub


# ---------------------------------------------------------------------------
# BuildHooksRegistry.register
# ---------------------------------------------------------------------------

class TestRegister:
    def test_register_stores_hooks(self):
        BuildHooksRegistry.register(FakeConfigA, preprocess=preprocess_stub)
        result = BuildHooksRegistry.get(FakeConfigA)
        assert result is not None
        assert result.preprocess is preprocess_stub

    def test_register_all_hooks(self):
        BuildHooksRegistry.register(
            FakeConfigA,
            preprocess=preprocess_stub,
            postprocess=postprocess_stub,
            validate=validate_stub,
        )
        hooks = BuildHooksRegistry.get(FakeConfigA)
        assert hooks.preprocess is preprocess_stub
        assert hooks.postprocess is postprocess_stub
        assert hooks.validate is validate_stub

    def test_register_no_hooks(self):
        """Registering with no hooks is valid and yields an empty BuildHooks."""
        BuildHooksRegistry.register(FakeConfigA)
        hooks = BuildHooksRegistry.get(FakeConfigA)
        assert hooks == BuildHooks()

    def test_register_duplicate_raises_value_error(self):
        BuildHooksRegistry.register(FakeConfigA, preprocess=preprocess_stub)
        with pytest.raises(ValueError, match="Hooks already registered"):
            BuildHooksRegistry.register(FakeConfigA, validate=validate_stub)

    def test_register_unknown_keyword_raises_type_error(self):
        with pytest.raises(TypeError, match="Unknown hook keyword"):
            BuildHooksRegistry.register(FakeConfigA, unknown_hook=lambda x: x)

    def test_register_different_types_independent(self):
        BuildHooksRegistry.register(FakeConfigA, preprocess=preprocess_stub)
        BuildHooksRegistry.register(FakeConfigB, validate=validate_stub)

        hooks_a = BuildHooksRegistry.get(FakeConfigA)
        hooks_b = BuildHooksRegistry.get(FakeConfigB)

        assert hooks_a.preprocess is preprocess_stub
        assert hooks_a.validate is None
        assert hooks_b.validate is validate_stub
        assert hooks_b.preprocess is None


# ---------------------------------------------------------------------------
# BuildHooksRegistry.get
# ---------------------------------------------------------------------------

class TestGet:
    def test_get_returns_none_for_unregistered_type(self):
        assert BuildHooksRegistry.get(FakeConfigA) is None

    def test_get_returns_correct_hooks_after_registration(self):
        BuildHooksRegistry.register(FakeConfigA, postprocess=postprocess_stub)
        hooks = BuildHooksRegistry.get(FakeConfigA)
        assert isinstance(hooks, BuildHooks)
        assert hooks.postprocess is postprocess_stub

    def test_get_does_not_affect_other_types(self):
        BuildHooksRegistry.register(FakeConfigA, validate=validate_stub)
        assert BuildHooksRegistry.get(FakeConfigB) is None


# ---------------------------------------------------------------------------
# BuildHooksRegistry.clear
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_removes_all_entries(self):
        BuildHooksRegistry.register(FakeConfigA, preprocess=preprocess_stub)
        BuildHooksRegistry.register(FakeConfigB, validate=validate_stub)
        BuildHooksRegistry.clear()
        assert BuildHooksRegistry.get(FakeConfigA) is None
        assert BuildHooksRegistry.get(FakeConfigB) is None

    def test_clear_allows_re_registration(self):
        BuildHooksRegistry.register(FakeConfigA, preprocess=preprocess_stub)
        BuildHooksRegistry.clear()
        # Should not raise now that the registry is empty.
        BuildHooksRegistry.register(FakeConfigA, validate=validate_stub)
        hooks = BuildHooksRegistry.get(FakeConfigA)
        assert hooks.validate is validate_stub

    def test_clear_on_empty_registry_is_safe(self):
        BuildHooksRegistry.clear()  # already empty — must not raise


# ---------------------------------------------------------------------------
# 1:1 enforcement edge cases
# ---------------------------------------------------------------------------

class TestOneToOneEnforcement:
    def test_second_register_same_type_always_raises(self):
        BuildHooksRegistry.register(FakeConfigA)
        with pytest.raises(ValueError):
            BuildHooksRegistry.register(FakeConfigA)

    def test_multiple_types_each_raise_on_second_register(self):
        BuildHooksRegistry.register(FakeConfigA, preprocess=preprocess_stub)
        BuildHooksRegistry.register(FakeConfigB, postprocess=postprocess_stub)

        with pytest.raises(ValueError):
            BuildHooksRegistry.register(FakeConfigA, validate=validate_stub)

        with pytest.raises(ValueError):
            BuildHooksRegistry.register(FakeConfigB, preprocess=preprocess_stub)

    def test_clear_then_register_three_types(self):
        for cfg in (FakeConfigA, FakeConfigB, FakeConfigC):
            BuildHooksRegistry.register(cfg, validate=validate_stub)

        BuildHooksRegistry.clear()

        for cfg in (FakeConfigA, FakeConfigB, FakeConfigC):
            BuildHooksRegistry.register(cfg, preprocess=preprocess_stub)
            assert BuildHooksRegistry.get(cfg).preprocess is preprocess_stub
