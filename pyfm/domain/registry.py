from typing import List, Dict, Any, Callable, Optional
import inspect
from functools import partial
from .conftypes import ConfigBase


class TaskHandler:
    def __init__(self, task_key: str):
        self._config_type = ConfigBase
        self._config = None
        self.task_key = task_key

    @property
    def config(self):
        if self._config is None:
            raise ValueError(f"Config not set for {self.task_key}")
        return self._config

    @config.setter
    def config(self, value):
        self._config = value

    def get_config_type(self):
        return self._config_type

    def set_method(self, function: Callable, method_name: Optional[str] = None):
        name = method_name if method_name is not None else function.__name__

        # Get function signature to check for 'config' parameter
        sig = inspect.signature(function)

        if "config" in sig.parameters:
            if "config" != list(sig.parameters.keys())[0]:
                raise ValueError(
                    f"First parameter of function {function.__name__} must be 'config'."
                )

            # Create wrapper that automatically injects self.config
            def config_method(*args, **kwargs):
                if "config" in kwargs:
                    raise ValueError(
                        "Cannote set config parameter externally. Set config property instead."
                    )

                if not self.config:
                    raise ValueError(f"Config not set for {self.task_key}")
                if len(args) >= len(sig.parameters):
                    raise ValueError(
                        f"Too many arguments for {self.task_key} function {name}."
                        f"Expecting {len(sig.parameters) - 1} arguments, received {len(args)}."
                    )

                return partial(function, self.config)(*args, **kwargs)

            setattr(self, name, config_method)
        else:
            setattr(self, name, function)


class TaskRegistry:
    _handlers: Dict[str, TaskHandler] = {}

    @staticmethod
    def get_task_key(job_type: str, task_type: str | None = None) -> str:
        key = job_type
        if task_type:
            key += f"_{task_type}"
        return key

    @classmethod
    def register_function(
        cls, task_key: str, function: Callable, method_name: Optional[str] = None
    ):
        if task_key in cls._handlers:
            handler = cls._handlers[task_key]
        else:
            handler = TaskHandler(task_key)
            cls._handlers[task_key] = handler

        handler.set_method(function, method_name)

    @classmethod
    def register_functions(cls, task_key: str, *functions: Callable):
        if task_key in cls._handlers:
            handler = cls._handlers[task_key]
        else:
            handler = TaskHandler(task_key)
            cls._handlers[task_key] = handler

        for function in functions:
            handler.set_method(function)

    @classmethod
    def register_config(cls, task_key: str, config_type: Any):
        if task_key in cls._handlers:
            handler = cls._handlers[task_key]
        else:
            handler = TaskHandler(task_key)
            cls._handlers[task_key] = handler

        handler._config_type = config_type

    @classmethod
    def get_handler(cls, task_key: str = "") -> TaskHandler:
        if task_key not in cls._handlers:
            raise ValueError(
                f"No handler registered for task type: {task_key}. "
                f"Available types: {list(cls._handlers.keys())}"
            )
        return cls._handlers[task_key]

    @classmethod
    def is_registered(cls, task_key: str) -> bool:
        return task_key in cls._handlers

    @classmethod
    def list_registered_types(cls) -> List[str]:
        return list(cls._handlers.keys())
