import logging
import sys
from typing import Optional


class RankFilter(logging.Filter):
    """Filter that adds MPI rank to log records."""

    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        record.rank = self.rank
        return True


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
        # Try to detect MPI
        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            comm_size = comm.Get_size()
            rank = comm.Get_rank()
            has_mpi = comm_size > 1
        except (ImportError, AttributeError):
            has_mpi = False
            rank = 0

        # Choose format based on MPI detection
        if has_mpi:
            log_format = "%(asctime)s [R%(rank)d] %(levelname)-5s - %(message)s"
        else:
            log_format = "%(asctime)s - %(levelname)-5s - %(message)s"

        logging.basicConfig(
            format=log_format,
            style="%",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True,
        )
        self._logger = logging.getLogger()
        self._logger.setLevel(level)

        # Add rank filter if MPI detected
        if has_mpi:
            rank_filter = RankFilter(rank)
            self._logger.addFilter(rank_filter)

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
