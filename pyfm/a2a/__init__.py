#! /usr/bin/env python3
"""
pyfm A2A contraction submodule

This module provides the main interface for general A2A contractions.
"""

from .contractions import (
    contract,
    conn_2pt,
    make_contraction_key,
    qed_conn_4pt,
    sib_conn_3pt,
)
from .execution import execute, main
from .meson_loader import MesonLoader
from .time_operations import convert_to_numpy, time_average

# Maintain backward compatibility by exposing all functions at module level
__all__ = [
    "contract",
    "conn_2pt",
    "convert_to_numpy",
    "execute",
    "main",
    "make_contraction_key",
    "MesonLoader",
    "qed_conn_4pt",
    "sib_conn_3pt",
    "time_average",
]

if __name__ == "__main__":
    # Preserve the original command-line interface
    import sys

    if len(sys.argv) != 2:
        raise ValueError("Must provide input yaml file.")
    main(sys.argv[1])
