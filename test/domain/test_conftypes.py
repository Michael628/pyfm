"""Tests for SerializableEnum.from_dict — underscore-insensitive lookup."""

import pytest

from pyfm.domain.conftypes import SerializableEnum
from pyfm.tasks.hadrons.highmode_v2.config import CacheMode, LowModeMethod


class Plain(SerializableEnum):
    ALPHA = 0
    BETA = 1


class TestFromDict:
    def test_plain_members_case_insensitive(self):
        assert Plain.from_dict("alpha") is Plain.ALPHA
        assert Plain.from_dict("BETA") is Plain.BETA

    def test_underscore_members_resolved(self):
        assert CacheMode.from_dict("build_and_load") is CacheMode.BUILD_AND_LOAD
        assert CacheMode.from_dict("build_only") is CacheMode.BUILD_ONLY
        assert CacheMode.from_dict("load") is CacheMode.LOAD
        assert LowModeMethod.from_dict("meson_field") is LowModeMethod.MESON_FIELD

    def test_underscores_optional_in_input(self):
        assert CacheMode.from_dict("BUILDANDLOAD") is CacheMode.BUILD_AND_LOAD

    def test_unknown_member_raises(self):
        with pytest.raises(ValueError, match="Invalid serializable type"):
            Plain.from_dict("gamma")

    def test_non_string_rejected(self):
        with pytest.raises(ValueError, match="must be string"):
            Plain.from_dict(3)
