from typing import List, Dict, Any, Callable, Optional
import inspect
from functools import partial
from pyfm.domain.conftypes import ConfigBase


class ConfigHandler:
    def __init__(self, handler_key: str):
        self._config_type = ConfigBase
        self._config = None
        self.handler_key = handler_key

    @property
    def key(self):
        if not hasattr(self.config, "key"):
            raise ValueError(
                f"Config for handler `{self.handler_key}` has no key attribute set"
            )
        return self._config.key

    @property
    def config(self):
        if self._config is None:
            raise ValueError(f"Config not set for {self.handler_key}")
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
                    raise ValueError(f"Config not set for {self.handler_key}")
                if len(args) >= len(sig.parameters):
                    raise ValueError(
                        f"Too many arguments for {self.handler_key} function {name}."
                        f"Expecting {len(sig.parameters) - 1} arguments, received {len(args)}."
                    )

                return partial(function, self.config)(*args, **kwargs)

            setattr(self, name, config_method)
        else:
            setattr(self, name, function)


class HandlerRegistry:
    _handlers: Dict[str, ConfigHandler] = {}

    @staticmethod
    def get_handler_key(module_scope: str, config_key: str) -> str:
        handler_key = f"{module_scope}_{config_key}"
        return handler_key

    @classmethod
    def register_function(
        cls, handler_key: str, function: Callable, method_name: Optional[str] = None
    ):
        if handler_key in cls._handlers:
            handler = cls._handlers[handler_key]
        else:
            handler = ConfigHandler(handler_key)
            cls._handlers[handler_key] = handler

        handler.set_method(function, method_name)

    @classmethod
    def register_functions(cls, handler_key: str, *functions: Callable):
        if handler_key in cls._handlers:
            handler = cls._handlers[handler_key]
        else:
            handler = ConfigHandler(handler_key)
            cls._handlers[handler_key] = handler

        for function in functions:
            handler.set_method(function)

    @classmethod
    def register_config(cls, handler_key: str, config_type: Any):
        if handler_key in cls._handlers:
            handler = cls._handlers[handler_key]
        else:
            handler = ConfigHandler(handler_key)
            cls._handlers[handler_key] = handler

        handler._config_type = config_type

    @classmethod
    def get_handler(cls, handler_key: str) -> ConfigHandler:
        if handler_key not in cls._handlers:
            raise ValueError(
                f"No handler registered for handler type: {handler_key}. "
                f"Available types: {list(cls._handlers.keys())}"
            )
        return cls._handlers[handler_key]

    @classmethod
    def is_registered(cls, handler_key: str) -> bool:
        return handler_key in cls._handlers

    @classmethod
    def list_registered_types(cls) -> List[str]:
        return list(cls._handlers.keys())
