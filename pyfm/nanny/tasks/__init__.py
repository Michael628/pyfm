"""Plugin-style task registry system with decorators.

This module provides a flexible plugin architecture for registering task configurations
and submit configurations using decorators. All task modules are imported at module
initialization to ensure complete registration.

IMPORTANT: This module should only be imported via:
    from pyfm.nanny import tasks

Direct imports may not work correctly due to initialization dependencies.
"""

import typing as t
from typing import Dict, Type, Callable, Optional, List
import logging

from pyfm.nanny import SubmitConfig
from pyfm.nanny.registry import (
    register_task, 
    register_submit_config, 
    get_task_registry, 
    get_submit_registry,
    clear_registry
)

logger = logging.getLogger(__name__)

# Import all task modules to trigger registration
# This ensures complete registry population when tasks module is imported
import pyfm.nanny.tasks.smear
import pyfm.nanny.tasks.contract
import pyfm.nanny.tasks.hadrons.lmi
import pyfm.nanny.tasks.hadrons.seq_sib
import pyfm.nanny.tasks.hadrons.seq_dhop
import pyfm.nanny.tasks.hadrons.high_a2a_vectors
import pyfm.nanny.tasks.hadrons.a2a_sib
import pyfm.nanny.tasks.hadrons.test_a2a_vectors

logger.debug(f"Imported all task modules. Registry contains {len(get_task_registry())} job types.")


def get_task_factory(job_type: str, task_type: Optional[str] = None) -> Callable[..., t.Any]:
    """Get task factory function for the specified job and task type.
    
    Parameters
    ----------
    job_type : str
        Type of job (e.g., 'smear', 'contract', 'hadrons')
    task_type : Optional[str]
        Specific task type (e.g., 'lmi', 'seq_sib')
        
    Returns
    -------
    Callable[..., Any]
        Factory function that creates task configurations
        
    Raises
    ------
    KeyError
        If job_type or task_type is not registered
    """
    task_registry = get_task_registry()
    try:
        task_class = task_registry[job_type][task_type]
        return task_class.from_dict
    except KeyError:
        available_jobs = list(task_registry.keys())
        if job_type not in task_registry:
            raise KeyError(f"Unknown job_type '{job_type}'. Available: {available_jobs}")
        
        available_tasks = list(task_registry[job_type].keys())
        raise KeyError(f"Unknown task_type '{task_type}' for job_type '{job_type}'. Available: {available_tasks}")


def get_submit_factory(job_type: str) -> Callable[..., SubmitConfig]:
    """Get submit config factory for the specified job type.
    
    Parameters
    ----------
    job_type : str
        Type of job (e.g., 'smear', 'contract', 'hadrons')
        
    Returns
    -------
    Callable[..., SubmitConfig]
        Factory function that creates submit configurations
        
    Raises
    ------
    KeyError
        If job_type is not registered
    """
    submit_registry = get_submit_registry()
    try:
        submit_class = submit_registry[job_type]
        return submit_class.create
    except KeyError:
        available_jobs = list(submit_registry.keys())
        raise KeyError(f"Unknown job_type '{job_type}'. Available: {available_jobs}")


def list_available_tasks() -> Dict[str, List[Optional[str]]]:
    """List all available task types by job type.
    
    Returns
    -------
    Dict[str, List[Optional[str]]]
        Dictionary mapping job types to lists of available task types
    """
    task_registry = get_task_registry()
    return {job_type: list(tasks.keys()) for job_type, tasks in task_registry.items()}


def list_available_submit_configs() -> List[str]:
    """List all available submit config types.
    
    Returns
    -------
    List[str]
        List of available job types with submit configs
    """
    submit_registry = get_submit_registry()
    return list(submit_registry.keys())


def get_registry_info() -> Dict[str, t.Any]:
    """Get detailed information about the current registry state.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with registry statistics and contents
    """
    task_registry = get_task_registry()
    submit_registry = get_submit_registry()
    
    task_count = sum(len(tasks) for tasks in task_registry.values())
    submit_count = len(submit_registry)
    
    return {
        'task_registry_size': task_count,
        'submit_registry_size': submit_count,
        'job_types': list(task_registry.keys()),
        'task_registry': {
            job_type: {
                task_type: cls.__name__ 
                for task_type, cls in tasks.items()
            }
            for job_type, tasks in task_registry.items()
        },
        'submit_registry': {
            job_type: cls.__name__ 
            for job_type, cls in submit_registry.items()
        }
    }


# Re-export decorators and utilities for convenience
__all__ = [
    'get_task_factory',
    'get_submit_factory', 
    'list_available_tasks',
    'list_available_submit_configs',
    'get_registry_info',
    'register_task',
    'register_submit_config',
    'clear_registry'
]


if __name__ == "__main__":
    from pyfm import utils

    param = utils.load_param("params.yaml")

    jc = config.get_job_config(param, "SIB")
    sc = config.get_submit_config(param, jc, series="a", cfg="100")

    stuff = jc.input_params(sc)
