"""Plugin-style task registry system with decorators.

This module provides a flexible plugin architecture for registering task configurations
using decorators. All task modules are imported at module initialization to ensure
complete registration.

IMPORTANT: This module should only be imported via:
    from pyfm.nanny import tasks

Direct imports may not work correctly due to initialization dependencies.
"""

import typing as t
from typing import Dict, Callable, Optional, List
import logging

from pyfm.nanny.registry import (
    register_task,
    get_task_registry,
    clear_registry,
)

logger = logging.getLogger(__name__)

# Import all task modules to trigger registration
# This ensures complete registry population when tasks module is imported
import pyfm.nanny.tasks.smear
import pyfm.nanny.tasks.contract
import pyfm.nanny.tasks.hadrons.lmi

logger.debug(
    f"Imported all task modules. Registry contains {len(get_task_registry())} job types."
)


def get_task_factory(
    job_type: str, task_type: Optional[str] = None
) -> Callable[..., t.Any]:
    task_registry = get_task_registry()
    try:
        task_class = task_registry[job_type][task_type]
        return task_class.from_dict
    except KeyError:
        available_jobs = list(task_registry.keys())
        if job_type not in task_registry:
            raise KeyError(
                f"Unknown job_type '{job_type}'. Available: {available_jobs}"
            )

        available_tasks = list(task_registry[job_type].keys())
        raise KeyError(
            f"Unknown task_type '{task_type}' for job_type '{job_type}'. Available: {available_tasks}"
        )


def list_available_tasks() -> Dict[str, List[Optional[str]]]:
    task_registry = get_task_registry()
    return {job_type: list(tasks.keys()) for job_type, tasks in task_registry.items()}


def get_registry_info() -> Dict[str, t.Any]:
    task_registry = get_task_registry()

    task_count = sum(len(tasks) for tasks in task_registry.values())

    return {
        "task_registry_size": task_count,
        "job_types": list(task_registry.keys()),
        "task_registry": {
            job_type: {task_type: cls.__name__ for task_type, cls in tasks.items()}
            for job_type, tasks in task_registry.items()
        },
    }


# Re-export decorators and utilities for convenience
__all__ = [
    "get_task_factory",
    "list_available_tasks",
    "get_registry_info",
    "register_task",
    "clear_registry",
]
