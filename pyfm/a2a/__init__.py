#! /usr/bin/env python3
"""
pyfm A2A contraction submodule

This module provides the main interface for general A2A contractions.
"""

from pyfm.a2a.contractions import (
    contract,
    conn_2pt,
    qed_conn_4pt,
    sib_conn_3pt,
)
from pyfm.a2a.execution import execute
from pyfm.a2a.mesonloader import (
    iter_meson_fields,
    load_meson,
    meson_mass_alter,
    get_meson_cache,
    clear_meson_cache,
    get_index_range,
)
from pyfm.a2a.time_operations import convert_to_numpy, time_average
from pyfm.a2a.register import register_a2a, get_a2a_handler
from pyfm.domain import DiagramConfig, ContractConfig

# Maintain backward compatibility by exposing all functions at module level
__all__ = [
    "contract",
    "conn_2pt",
    "convert_to_numpy",
    "execute",
    "iter_meson_fields",
    "load_meson",
    "meson_mass_alter",
    "get_meson_cache",
    "clear_meson_cache",
    "get_index_range",
    "qed_conn_4pt",
    "sib_conn_3pt",
    "time_average",
    "get_a2a_handler",
]

register_a2a(ContractConfig)
register_a2a(DiagramConfig)
