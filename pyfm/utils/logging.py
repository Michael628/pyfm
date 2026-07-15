import logging
import os
import sys
from typing import Optional


def _detect_mpi_from_env() -> tuple[bool, int]:
    """Detect ``(has_mpi, rank)`` from launcher env vars, without importing mpi4py.

    Importing ``mpi4py.MPI`` runs ``MPI_Init``, which aborts the process on hosts
    without an MPI fabric (e.g. HPC login nodes) -- and that abort is not a
    catchable Python exception. MPI launchers export rank/size in the
    environment before the process starts, so read those instead. Falls back to
    single-rank when no launcher is detected.
    """
    for size_var, rank_var in (
        ("PMI_SIZE", "PMI_RANK"),  # MPICH / Cray PALS / Intel MPI
        ("OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_RANK"),  # Open MPI
    ):
        if size_var in os.environ:
            try:
                size = int(os.environ[size_var])
                rank = int(os.environ.get(rank_var, 0))
            except ValueError:
                continue
            return size > 1, rank
    return False, 0


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
        # Detect MPI from launcher env vars (never imports mpi4py; see helper).
        has_mpi, rank = _detect_mpi_from_env()

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
