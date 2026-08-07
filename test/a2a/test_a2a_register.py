"""Regression tests for the register_a2a elimination (issue #7).

Verifies that:
- pyfm/a2a/register.py no longer exists
- pyfm.a2a imports cleanly without the deleted module
- The a2a configs are still reachable via nanny_* keys after importing
  the contract task modules that call register_task()
"""
from pathlib import Path

import importlib

import pytest

from pyfm.domain import task_registry, build_hooks
from pyfm.tasks.register import _config_to_task_key


REPO_ROOT = Path(__file__).parent.parent.parent


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


def test_register_py_does_not_exist():
    assert not (REPO_ROOT / "pyfm" / "a2a" / "register.py").exists()


def test_pyfm_a2a_imports_without_error():
    import pyfm.a2a  # noqa: F401


def test_contract_configs_registered_via_nanny_keys():
    # The registry is cleared by the autouse fixture before each test.
    # The contract submodules are already cached in sys.modules, so importing
    # them does not re-run their module bodies. Reload each to re-trigger
    # the register_task() calls and confirm the nanny_* keys are produced.
    import pyfm.tasks.contract.contraction as _c
    import pyfm.tasks.contract.diagram as _d
    import pyfm.tasks.contract.mesonloader as _m
    importlib.reload(_c)
    importlib.reload(_d)
    importlib.reload(_m)

    assert task_registry.get("nanny_contract") is not None
    assert task_registry.get("nanny_contract_diagram") is not None
    assert task_registry.get("nanny_contract_mesonloader") is not None


def test_no_a2a_scoped_keys_registered():
    import pyfm.a2a  # noqa: F401
    import pyfm.tasks.contract.contraction as _c
    import pyfm.tasks.contract.diagram as _d
    import pyfm.tasks.contract.mesonloader as _m
    importlib.reload(_c)
    importlib.reload(_d)
    importlib.reload(_m)

    a2a_keys = [k for k in task_registry._handlers if k.startswith("a2a_")]
    assert a2a_keys == [], f"Unexpected a2a_* keys: {a2a_keys}"
