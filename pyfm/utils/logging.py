import logging
import sys
from typing import Optional


class PyFMLogger:
    _instance: Optional["PyFMLogger"] = None
    _logger: Optional[logging.Logger] = None

    def __new__(cls) -> "PyFMLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._logger is None:
            self._setup_logging("INFO")

    def _setup_logging(self, level: str) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)-5s - %(message)s",
            style="%",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True,
        )
        self._logger = logging.getLogger()
        self._logger.setLevel(level)

    def set_logging_level(self, level: str) -> logging.Logger:
        if self._logger is None:
            raise RuntimeError("Logger not initialized")
        self._logger.setLevel(level)
        return self._logger

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            raise RuntimeError("Logger not initialized")
        return self._logger


_pyfm_logger = PyFMLogger()


def get_logger() -> logging.Logger:
    return _pyfm_logger.logger


def set_logging_level(level: str) -> logging.Logger:
    return _pyfm_logger.set_logging_level(level)
