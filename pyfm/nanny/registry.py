"""Task registry system with decorators.

This module provides the core registry functionality for task and submit config
registration, separated from the main tasks module to avoid circular imports.
"""

import typing as t
from typing import Dict, Type, Optional, List
import logging

logger = logging.getLogger(__name__)

# Global registries populated by decorators
_TASK_REGISTRY: Dict[str, Dict[Optional[str], t.Any]] = {}
_SUBMIT_REGISTRY: Dict[str, t.Any] = {}


def register_task(job_type: str, task_type: Optional[str] = None):
    """Decorator to register task classes in the global registry.
    
    Parameters
    ----------
    job_type : str
        The job type (e.g., 'smear', 'contract', 'hadrons')
    task_type : Optional[str]
        The specific task type (e.g., 'lmi', 'seq_sib'). None for job types
        that don't have subtypes.
        
    Examples
    --------
    >>> @register_task("smear")
    ... class SmearTask(TaskBase):
    ...     pass
    
    >>> @register_task("hadrons", "lmi")
    ... class LMITask(TaskBase):
    ...     pass
    """
    def decorator(cls: t.Any) -> t.Any:
        if job_type not in _TASK_REGISTRY:
            _TASK_REGISTRY[job_type] = {}
        
        if task_type in _TASK_REGISTRY[job_type]:
            logger.warning(
                f"Overriding existing task registration: {job_type}.{task_type} "
                f"(was {_TASK_REGISTRY[job_type][task_type]}, now {cls})"
            )
        
        _TASK_REGISTRY[job_type][task_type] = cls
        logger.debug(f"Registered task: {job_type}.{task_type} -> {cls}")
        return cls
    return decorator


def register_submit_config(job_type: str):
    """Decorator to register submit config classes in the global registry.
    
    Parameters
    ----------
    job_type : str
        The job type (e.g., 'smear', 'contract', 'hadrons')
        
    Examples
    --------
    >>> @register_submit_config("smear")
    ... class SubmitSmearConfig(SubmitConfig):
    ...     pass
    """
    def decorator(cls: t.Any) -> t.Any:
        if job_type in _SUBMIT_REGISTRY:
            logger.warning(
                f"Overriding existing submit config registration: {job_type} "
                f"(was {_SUBMIT_REGISTRY[job_type]}, now {cls})"
            )
        
        _SUBMIT_REGISTRY[job_type] = cls
        logger.debug(f"Registered submit config: {job_type} -> {cls}")
        return cls
    return decorator


def get_task_registry() -> Dict[str, Dict[Optional[str], t.Any]]:
    """Get the current task registry."""
    return _TASK_REGISTRY


def get_submit_registry() -> Dict[str, t.Any]:
    """Get the current submit config registry."""
    return _SUBMIT_REGISTRY


def clear_registry():
    """Clear all registries. Useful for testing."""
    global _TASK_REGISTRY, _SUBMIT_REGISTRY
    _TASK_REGISTRY.clear()
    _SUBMIT_REGISTRY.clear()
    logger.debug("Cleared all registries")