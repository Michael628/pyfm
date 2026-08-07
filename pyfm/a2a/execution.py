"""Main execution logic for A2A contractions."""

import itertools
import os
import pickle
import typing as t
from time import perf_counter

try:
    import cupy as xp
except ImportError:
    import numpy as xp

from pyfm import utils
from pyfm.a2a.types import DiagramConfig, ContractConfig, ContractType
from pyfm.a2a.contractions import (
    conn_2pt,
    qed_conn_4pt,
    sib_conn_3pt,
)


def execute(
    contraction: t.Tuple[str],
    diagram_config: DiagramConfig,
    contract_config: ContractConfig,
):
    """Execute the appropriate contraction based on diagram configuration."""
    logger = utils.get_logger()
    if hasattr(xp, "cuda"):
        my_device = contract_config.rank % xp.cuda.runtime.getDeviceCount()
        logger.debug(f"Rank {contract_config.rank} is using gpu device {my_device}")
        xp.cuda.Device(my_device).use()

    logger.debug(
        f"Rank {contract_config.rank}/{contract_config.comm_size} "
        f"processing contraction"
    )
    logger.info(f"Processing mode: {', '.join(contraction)}")

    contraction_types = {
        ContractType.TWOPOINT: lambda: conn_2pt(
            contraction, diagram_config, contract_config
        ),
        ContractType.SIB: lambda: sib_conn_3pt(
            contraction, diagram_config, contract_config
        ),
        ContractType.PHOTEX: lambda: qed_conn_4pt(
            contraction, diagram_config, contract_config
        ),
        ContractType.SELFEN: lambda: qed_conn_4pt(
            contraction, diagram_config, contract_config
        ),
    }

    if diagram_config.contraction_type in contraction_types:
        run = contraction_types[diagram_config.contraction_type]
    else:
        raise ValueError(
            f"No contraction implementation for `{diagram_config.contraction_type}`."
        )

    return run()
