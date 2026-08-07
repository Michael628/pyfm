#!/usr/bin/env python3
"""
Performance analysis utilities for Hadrons output files.

This module analyzes performance data from Hadrons simulation output files,
focusing on three major cost categories:
1. IRL solver (epack modules)
2. CG solves (quark_ modules)
3. Meson field calculations (mf_ modules)
"""

import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

BENCHMARK_SCHEMA_VERSION = 1
HADRONS_LMI_TASK_KEY = "nanny_hadrons_lmi"
GRID_LMI_TASK_KEY = "nanny_grid"
SUPPORTED_LMI_TASK_KEYS = (HADRONS_LMI_TASK_KEY, GRID_LMI_TASK_KEY)
# Backward-compatible alias until backend integration switches to task-aware keys.
LMI_TASK_KEY = HADRONS_LMI_TASK_KEY
BENCHMARK_COMPONENTS = (
    "epack",
    "ranLL",
    "ama",
    "meson_field_local",
    "meson_field_onelink",
)


@dataclass(frozen=True)
class ModuleObservation:
    """Ordered observation of a Hadrons measurement-step module."""

    module_name: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    elapsed_seconds: Optional[float] = None


@dataclass(frozen=True)
class PerformanceAnalysis:
    """Structured performance data parsed from a Hadrons output file."""

    file_path: str
    lattice_grid: Optional[str]
    communicator_sizes: Dict[str, Optional[int]]
    timings: Dict[str, float]
    module_observations: List[ModuleObservation]
    is_incomplete: bool
    mf_custom_timers: Dict[str, Dict[str, float]]
    epack_stats: Dict[str, Any]
    cg_iterations: Dict[str, List[int]]
    total_time: float
    categories: Dict[str, Dict]


def parse_time_value(time_str: str) -> float:
    """
    Convert time string to seconds.

    Args:
        time_str: Time string like "1201 s", "67.87 s", "290 ms", "85 us"

    Returns:
        Time value in seconds
    """
    time_str = time_str.strip()

    # Handle scientific notation
    if "e+" in time_str or "e-" in time_str:
        parts = time_str.split()
        value = float(parts[0])
        if len(parts) > 1:
            unit = parts[1]
        else:
            return value  # assume seconds if no unit
    else:
        parts = time_str.split()
        if len(parts) < 2:
            return 0.0
        value = float(parts[0])
        unit = parts[1]

    # Convert to seconds
    if unit == "s":
        return value
    elif unit == "ms":
        return value / 1000.0
    elif unit == "us":
        return value / 1_000_000.0
    else:
        return value  # assume seconds


def extract_communicator_sizes(file_path: str) -> Dict[str, Optional[int]]:
    """
    Extract World and Node communicator sizes from the output file.

    Args:
        file_path: Path to the Hadrons output file

    Returns:
        Dictionary with 'world_size' and 'node_size' keys
    """
    world_size = None
    node_size = None

    world_pattern = re.compile(r"World communicator of size\s*(\d+)")
    node_pattern = re.compile(r"Node  communicator of size\s*(\d+)")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if world_size is None:
                world_match = world_pattern.search(line)
                if world_match:
                    world_size = int(world_match.group(1))

            if node_size is None:
                node_match = node_pattern.search(line)
                if node_match:
                    node_size = int(node_match.group(1))

            # Early exit if both found
            if world_size is not None and node_size is not None:
                break

    return {"world_size": world_size, "node_size": node_size}


def extract_lattice_grid(file_path: str) -> Optional[str]:
    """
    Extract lattice grid size from the output file (from --grid flag).

    Args:
        file_path: Path to the Hadrons output file

    Returns:
        Grid string (e.g., "144.144.144.288") or None if not found
    """
    grid_pattern = re.compile(r"--grid\s+(\d+\.\d+\.\d+\.\d+)")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            grid_match = grid_pattern.search(line)
            if grid_match:
                return grid_match.group(1)

    return None


def extract_epack_statistics(file_path: str) -> Dict[str, Any]:
    """
    Extract IRL/epack solver statistics (restart iterations, Lanczos steps, eigenvector write time).

    Args:
        file_path: Path to the Hadrons output file

    Returns:
        Dictionary containing epack solver statistics
    """
    epack_stats: Dict[str, Any] = {}

    # Patterns to match IRL solver info
    restart_pattern = re.compile(r"Restart iteration\s*=\s*(\d+)")
    iterations_pattern = re.compile(r"--\s*Iterations\s*=\s*(\d+)")
    nconv_pattern = re.compile(r"--\s*Nconv\s*=\s*(\d+)")

    # Patterns to match eigenvector writing
    writing_start_pattern = re.compile(
        r"Hadrons\s*:\s*Message\s*:\s*([\d.]+)\s+s\s*:\s*Writing eigenvector 0"
    )
    writing_evec_pattern = re.compile(
        r"Hadrons\s*:\s*Message\s*:\s*([\d.]+)\s+s\s*:\s*Writing eigenvector (\d+)"
    )
    write_complete_pattern = re.compile(
        r"writeLatticeObject: unvectorize overhead ([\d.]+) s"
    )

    restart_iterations = []
    total_iterations = None
    nconv = None
    lanczos_step_count = 0

    # Eigenvector write tracking
    eigenvector_write_start = None
    last_eigenvector_num = None
    last_write_complete_time = None

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        prev_line = None
        for line in f:
            # Count Lanczos steps
            if "Lanczos step alpha" in line:
                lanczos_step_count += 1

            # Track restart iterations
            restart_match = restart_pattern.search(line)
            if restart_match:
                restart_iterations.append(int(restart_match.group(1)))

            # Get total iterations from summary
            iter_match = iterations_pattern.search(line)
            if iter_match:
                total_iterations = int(iter_match.group(1))

            # Get number of converged eigenvectors
            nconv_match = nconv_pattern.search(line)
            if nconv_match:
                nconv = int(nconv_match.group(1))

            # Track eigenvector writing
            writing_match = writing_evec_pattern.search(line)
            if writing_match:
                evec_num = int(writing_match.group(2))
                if evec_num == 0:
                    # First eigenvector write
                    eigenvector_write_start = float(writing_match.group(1))
                last_eigenvector_num = evec_num

            # Track last write completion (after the last eigenvector message)
            if last_eigenvector_num is not None:
                complete_match = write_complete_pattern.search(line)
                if complete_match:
                    # Extract timestamp from this line
                    time_match = re.search(r":\s*([\d.]+)\s+s\s*:", line)
                    if time_match:
                        last_write_complete_time = float(time_match.group(1))

            prev_line = line

    if restart_iterations or lanczos_step_count > 0:
        epack_stats["restart_iterations"] = len(restart_iterations)
        epack_stats["lanczos_steps"] = lanczos_step_count
        if total_iterations is not None:
            epack_stats["total_iterations"] = total_iterations
        if nconv is not None:
            epack_stats["nconv"] = nconv

    # Add eigenvector write time if eigenvectors were written
    if eigenvector_write_start is not None and last_write_complete_time is not None:
        eigenvector_write_time = last_write_complete_time - eigenvector_write_start
        epack_stats["eigenvector_write_time"] = eigenvector_write_time
        epack_stats["num_eigenvectors_written"] = (
            last_eigenvector_num + 1
        )  # +1 because numbering starts at 0

    return epack_stats


def extract_mf_custom_timers(file_path: str) -> Dict[str, Dict[str, float]]:
    """
    Extract custom timer information (kernel time, IO time) for mf_ modules.

    Args:
        file_path: Path to the Hadrons output file

    Returns:
        Dictionary mapping mf_ module names to their custom timer data
    """
    mf_custom_timers = {}

    # Pattern to match module step headers
    module_pattern = re.compile(r"Measurement step \d+/\d+ \(module '(mf_[^']+)'\)")

    # Pattern to match custom timer lines
    kernel_pattern = re.compile(r"kernel:\s+([\d.e+-]+\s+(?:s|ms|us))\s+\(")
    io_total_pattern = re.compile(r"IO: total:\s+([\d.e+-]+\s+(?:s|ms|us))\s+\(")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        current_module = None
        in_custom_timers = False

        for line in f:
            # Check for module header
            module_match = module_pattern.search(line)
            if module_match:
                current_module = module_match.group(1)
                in_custom_timers = False
                if current_module not in mf_custom_timers:
                    mf_custom_timers[current_module] = {}

            # Check for CUSTOM TIMERS section
            if current_module and "* CUSTOM TIMERS" in line:
                in_custom_timers = True
                continue

            # Check for end of timers section
            if in_custom_timers and "Memory management" in line:
                in_custom_timers = False
                current_module = None

            # Extract timer values
            if in_custom_timers and current_module:
                kernel_match = kernel_pattern.search(line)
                if kernel_match:
                    time_str = kernel_match.group(1)
                    mf_custom_timers[current_module]["kernel"] = parse_time_value(
                        time_str
                    )

                io_match = io_total_pattern.search(line)
                if io_match:
                    time_str = io_match.group(1)
                    mf_custom_timers[current_module]["io_total"] = parse_time_value(
                        time_str
                    )

    return mf_custom_timers


def extract_cg_iteration_counts(file_path: str) -> Dict[str, List[int]]:
    """
    Extract CG iteration counts for quark_ama modules.

    Args:
        file_path: Path to the Hadrons output file

    Returns:
        Dictionary mapping module names to lists of iteration counts
    """
    cg_iterations = defaultdict(list)

    # Pattern to match module step headers (only quark_ama modules)
    module_pattern = re.compile(
        r"Measurement step \d+/\d+ \(module '(quark_ama_[^']+)'\)"
    )

    # Pattern to match MixedPrecisionConjugateGradient summary lines
    cg_pattern = re.compile(
        r"MixedPrecisionConjugateGradient: Inner CG iterations (\d+)"
    )

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        current_module = None

        for line in f:
            # Check for quark_ama module header
            module_match = module_pattern.search(line)
            if module_match:
                current_module = module_match.group(1)
                continue

            # Reset current_module when we hit a non-ama module
            # This prevents collecting CG iterations from other modules
            if "Measurement step" in line and "quark_ama" not in line:
                current_module = None
                continue

            # Extract CG iteration counts from MixedPrecisionConjugateGradient summary
            if current_module:
                cg_match = cg_pattern.search(line)
                if cg_match:
                    iteration_count = int(cg_match.group(1))
                    cg_iterations[current_module].append(iteration_count)

    return dict(cg_iterations)


def extract_module_step_observations(
    file_path: str,
) -> Tuple[List[ModuleObservation], bool]:
    """Extract ordered module observations from measurement-step headers."""
    module_steps: List[Tuple[float, str]] = []
    step_pattern = re.compile(
        r"Hadrons\s*:\s*Message\s*:\s+([\d.]+)\s+s\s*:.*Measurement step.*\(module '([^']+)'\)"
    )

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = step_pattern.search(line)
            if match:
                timestamp = float(match.group(1))
                module_name = match.group(2)
                module_steps.append((timestamp, module_name))

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        last_lines = f.readlines()[-100:]
        is_incomplete = not any(
            "Module breakdown" in line or "End of measurement" in line
            for line in last_lines
        )

    observations: List[ModuleObservation] = []
    for i, (time_start, module_name) in enumerate(module_steps):
        if i + 1 < len(module_steps):
            time_end = module_steps[i + 1][0]
            elapsed_seconds = time_end - time_start
        else:
            time_end = None
            elapsed_seconds = None
        observations.append(
            ModuleObservation(
                module_name=module_name,
                start_time=time_start,
                end_time=time_end,
                elapsed_seconds=elapsed_seconds,
            )
        )

    return observations, is_incomplete


def extract_module_timings_from_steps(file_path: str) -> Tuple[Dict[str, float], bool]:
    """
    Extract module timing information from measurement step headers.
    Used when Module breakdown section is not available (incomplete logs).
    """
    observations, is_incomplete = extract_module_step_observations(file_path)
    timings = {
        observation.module_name: observation.elapsed_seconds
        for observation in observations
        if observation.elapsed_seconds is not None
    }
    return timings, is_incomplete


def extract_module_timings(file_path: str) -> Tuple[Dict[str, float], bool]:
    """
    Extract module timing information from the output file.

    Args:
        file_path: Path to the Hadrons output file

    Returns:
        Tuple of (timings dictionary, is_incomplete flag)
    """
    timings = {}
    is_incomplete = False

    # Pattern to match timing lines like:
    # Hadrons : Message  : 12289.842517 s :                               epack: 1201 s (9.8%)
    timing_pattern = re.compile(
        r"Hadrons\s*:\s*Message\s*:\s*[\d.]+\s+s\s*:\s+(\S+):\s+([\d.e+-]+\s+(?:s|ms|us))\s+\("
    )

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        in_module_breakdown = False

        for line in f:
            # Look for the module breakdown section
            if "................ Module breakdown" in line:
                in_module_breakdown = True
                continue

            # Stop at module type breakdown
            if "................ Module type breakdown" in line:
                in_module_breakdown = False
                break

            if in_module_breakdown:
                match = timing_pattern.search(line)
                if match:
                    module_name = match.group(1).strip()
                    time_str = match.group(2).strip()
                    time_seconds = parse_time_value(time_str)
                    timings[module_name] = time_seconds

    # If we didn't find module breakdown section, try extracting from measurement steps
    if not timings:
        timings, is_incomplete = extract_module_timings_from_steps(file_path)
    else:
        # Check if file completed normally
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            last_lines = f.readlines()[-50:]
            is_incomplete = not any(
                "End of measurement" in line or "Grid Finalize" in line
                for line in last_lines
            )

    return timings, is_incomplete


def build_module_observations(
    file_path: str,
    timings: Dict[str, float],
) -> List[ModuleObservation]:
    """Return ordered observations, falling back to breakdown timings when needed."""
    observations, _ = extract_module_step_observations(file_path)
    if observations:
        return observations

    return [
        ModuleObservation(module_name=module_name, elapsed_seconds=time_seconds)
        for module_name, time_seconds in timings.items()
    ]


def derive_node_count(communicator_sizes: Dict[str, Optional[int]]) -> Optional[int]:
    """Derive node count from world/node communicator sizes when unambiguous."""
    world_size = communicator_sizes.get("world_size")
    node_size = communicator_sizes.get("node_size")
    if world_size is None or node_size is None or node_size <= 0:
        return None
    if world_size % node_size != 0:
        return None
    return world_size // node_size


def classify_benchmark_component(module_name: str) -> Optional[str]:
    """Map a Hadrons module name onto a v1 benchmark component."""
    if module_name == "epack":
        return "epack"
    if module_name.startswith("quark_ranLL_") or module_name.startswith("quark_lma_"):
        return "ranLL"
    if module_name.startswith("quark_ama"):
        return "ama"
    if module_name.startswith("mf_") and "onelink" in module_name:
        return "meson_field_onelink"
    if module_name.startswith("mf_") and "local" in module_name:
        return "meson_field_local"
    return None


def count_components(module_names: List[str]) -> Dict[str, int]:
    """Count planned/observed modules by benchmark component."""
    counts = {component: 0 for component in BENCHMARK_COMPONENTS}
    for module_name in module_names:
        component = classify_benchmark_component(module_name)
        if component is not None:
            counts[component] += 1
    return counts


def _as_grid_elem_list(section: Any) -> List[Dict[str, Any]]:
    """Return Grid planned-input elem entries from a possibly missing section."""
    if not isinstance(section, dict):
        return []
    elem = section.get("elem", [])
    return elem if isinstance(elem, list) else []


def _grid_gammas_are_onelink(gammas: str) -> bool:
    """Return whether Grid gamma content represents one-link meson fields."""
    return any(gamma.strip().endswith("_G1") for gamma in gammas.split())


def _grid_gammas_are_local(gammas: str) -> bool:
    """Return whether Grid gamma content represents local meson fields."""
    return any(
        gamma.strip() and not gamma.strip().endswith("_G1")
        for gamma in gammas.split()
    )


def count_grid_planned_components(planned_input: Dict[str, Any]) -> Dict[str, int]:
    """Count Grid planned-input sections by benchmark component."""
    counts = {component: 0 for component in BENCHMARK_COMPONENTS}

    if "epack" in planned_input:
        counts["epack"] = 1

    for contraction in _as_grid_elem_list(planned_input.get("corr")):
        for solver_key in ("quarkSolver", "antiquarkSolver"):
            solver = contraction.get(solver_key)
            if solver == "lma":
                counts["ranLL"] += 1
            elif solver == "mpcg":
                counts["ama"] += 1

    for a2a_entry in _as_grid_elem_list(planned_input.get("a2a")):
        spin_taste = a2a_entry.get("spinTaste", {})
        gammas = spin_taste.get("gammas", "") if isinstance(spin_taste, dict) else ""
        if _grid_gammas_are_local(gammas):
            counts["meson_field_local"] += 1
        if _grid_gammas_are_onelink(gammas):
            counts["meson_field_onelink"] += 1

    return counts


def count_planned_components(task_key: str, planned_input: Any) -> Dict[str, int]:
    """Count planned benchmark components for a supported LMI task."""
    if task_key == HADRONS_LMI_TASK_KEY:
        return count_components(planned_input.schedule)
    if task_key == GRID_LMI_TASK_KEY:
        return count_grid_planned_components(planned_input)
    raise ValueError(f"Unsupported LMI benchmark task key: {task_key!r}")


def group_observed_components(
    observations: List[ModuleObservation],
) -> Dict[str, List[ModuleObservation]]:
    """Group ordered observations by benchmark component."""
    grouped: Dict[str, List[ModuleObservation]] = {
        component: [] for component in BENCHMARK_COMPONENTS
    }
    for observation in observations:
        component = classify_benchmark_component(observation.module_name)
        if component is not None:
            grouped[component].append(observation)
    return grouped


GRID_MESSAGE_PATTERN = re.compile(r"Grid\s*:\s*Message\s*:\s*([\d.]+)\s+s\s*:\s*(.*)")
GRID_READING_EIGENVECTOR_PATTERN = re.compile(r"Reading eigenvector\s+(\d+)")
GRID_CONVERGED_EIGENVECTORS_PATTERN = re.compile(r"Converged\s+(\d+)\s+eigenvectors")
GRID_CORRELATOR_PATTERN = re.compile(
    r"Correlator:.*\((lma|mpcg)\).*\((lma|mpcg)\)"
)
GRID_A2A_COMPLETE_PATTERN = re.compile(
    r"All-to-all meson field construction complete\s*\((\d+)\)"
)


def _grid_message(line: str) -> Optional[Tuple[float, str]]:
    """Return a Grid message timestamp and payload when the line is a Grid message."""
    match = GRID_MESSAGE_PATTERN.search(line)
    if not match:
        return None
    return float(match.group(1)), match.group(2).strip()


def _elapsed_seconds(
    start_time: Optional[float], end_time: Optional[float]
) -> Optional[float]:
    """Return non-negative elapsed seconds for selected Grid markers."""
    if start_time is None or end_time is None or end_time < start_time:
        return None
    return end_time - start_time


def _grid_solver_module_name(solver: str) -> Optional[str]:
    """Return a pseudo Hadrons-style module name for a Grid correlator solver."""
    if solver == "lma":
        return "quark_lma_grid_corr"
    if solver == "mpcg":
        return "quark_ama_grid_corr"
    return None


def _grid_a2a_module_names(gammas: List[str]) -> List[str]:
    """Return pseudo Hadrons-style module names for a Grid A2A gamma block."""
    module_names = []
    if any(gamma and not gamma.endswith("_G1") for gamma in gammas):
        module_names.append("mf_grid_local")
    if any(gamma.endswith("_G1") for gamma in gammas):
        module_names.append("mf_grid_onelink")
    return module_names


def extract_grid_benchmark_observations(
    file_path: str,
) -> Tuple[List[ModuleObservation], bool, Dict[str, Any]]:
    """Extract Grid benchmark observations and epack progress metadata."""
    observations: List[ModuleObservation] = []
    epack_stats: Dict[str, Any] = {}

    epack_solve_start = None
    epack_load_start = None
    epack_load_observed = False
    last_eigenvector_read = None

    corr_start = None
    corr_module_name = None

    a2a_start = None
    a2a_gammas: List[str] = []
    collecting_a2a_gammas = False

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line in lines:
        message = _grid_message(line)
        if message is None:
            continue
        timestamp, payload = message

        if "MODULE: MSolver::StagFermionIRL" in payload and epack_solve_start is None:
            epack_solve_start = timestamp
        elif "Running IRL eigensolver" in payload:
            epack_solve_start = timestamp
        elif "Loading eigenpack" in payload and epack_load_start is None:
            epack_load_start = timestamp
            epack_load_observed = True

        read_match = GRID_READING_EIGENVECTOR_PATTERN.search(payload)
        if read_match:
            last_eigenvector_read = max(
                last_eigenvector_read or -1, int(read_match.group(1))
            )

        converged_match = GRID_CONVERGED_EIGENVECTORS_PATTERN.search(payload)
        if converged_match:
            epack_stats["nconv"] = int(converged_match.group(1))
            if epack_solve_start is not None:
                observations.append(
                    ModuleObservation(
                        module_name="epack",
                        start_time=epack_solve_start,
                        end_time=timestamp,
                        elapsed_seconds=_elapsed_seconds(epack_solve_start, timestamp),
                    )
                )
                epack_solve_start = None

        if "Low mode projector setup complete" in payload and epack_load_start is not None:
            observations.append(
                ModuleObservation(
                    module_name="epack",
                    start_time=epack_load_start,
                    end_time=timestamp,
                    elapsed_seconds=_elapsed_seconds(epack_load_start, timestamp),
                )
            )
            epack_load_start = None

        if "Setting up meson contraction" in payload:
            corr_start = timestamp
            corr_module_name = None
            continue

        corr_match = GRID_CORRELATOR_PATTERN.search(payload)
        if corr_match and corr_start is not None:
            quark_solver, antiquark_solver = corr_match.groups()
            corr_module_name = _grid_solver_module_name(quark_solver)
            if quark_solver != antiquark_solver:
                corr_module_name = None
            continue

        if (
            payload.startswith("Saving correlator")
            and corr_start is not None
            and corr_module_name is not None
        ):
            observations.append(
                ModuleObservation(
                    module_name=corr_module_name,
                    start_time=corr_start,
                    end_time=timestamp,
                    elapsed_seconds=_elapsed_seconds(corr_start, timestamp),
                )
            )
            corr_start = None
            corr_module_name = None
            continue

        if "Computing all-to-all meson fields" in payload:
            a2a_start = timestamp
            a2a_gammas = []
            collecting_a2a_gammas = False
            continue

        if payload == "Spin bilinears:":
            collecting_a2a_gammas = True
            continue

        if collecting_a2a_gammas:
            if payload.startswith("Meson field size:"):
                collecting_a2a_gammas = False
            elif re.fullmatch(r"[A-Z0-9]+_[A-Z0-9]+", payload):
                a2a_gammas.append(payload)
            continue

        a2a_complete_match = GRID_A2A_COMPLETE_PATTERN.search(payload)
        if a2a_complete_match and a2a_start is not None:
            for module_name in _grid_a2a_module_names(a2a_gammas):
                observations.append(
                    ModuleObservation(
                        module_name=module_name,
                        start_time=a2a_start,
                        end_time=timestamp,
                        elapsed_seconds=_elapsed_seconds(a2a_start, timestamp),
                    )
                )
            a2a_start = None
            a2a_gammas = []

    if last_eigenvector_read is not None:
        epack_stats["eigenvectors_read"] = last_eigenvector_read + 1
    if epack_load_observed:
        epack_stats["epack_load_observed"] = True
    if epack_load_start is not None:
        observations.append(
            ModuleObservation(
                module_name="epack",
                start_time=epack_load_start,
                end_time=None,
                elapsed_seconds=None,
            )
        )

    is_incomplete = not any("Grid Finalize" in line for line in lines[-100:])
    return observations, is_incomplete, epack_stats


def sum_elapsed_seconds(observations: List[ModuleObservation]) -> Optional[float]:
    """Sum known observation elapsed seconds; return None if none are known."""
    elapsed_values = [
        observation.elapsed_seconds
        for observation in observations
        if observation.elapsed_seconds is not None
    ]
    if not elapsed_values:
        return None
    return sum(elapsed_values)


def get_planned_lanczos_steps(config: Any) -> Optional[int]:
    """Return the configured IRL Lanczos denominator when available."""
    epack_config = getattr(config, "epack_config", None)
    if epack_config is None or getattr(epack_config, "load", True):
        return None
    lanczos = getattr(epack_config, "lanczos", None)
    if lanczos is None:
        return None
    return getattr(lanczos, "nm", None)


def build_component_metadata(
    component: str,
    config: Any,
    analysis: PerformanceAnalysis,
) -> Dict[str, Any]:
    """Build component-specific benchmark metadata."""
    if component != "epack":
        return {}

    metadata: Dict[str, Any] = {}
    planned_lanczos_steps = get_planned_lanczos_steps(config)
    if planned_lanczos_steps is not None:
        metadata["planned_lanczos_steps"] = planned_lanczos_steps
    planned_eigenvectors = getattr(getattr(config, "epack_config", None), "eigs", None)
    if planned_eigenvectors is not None:
        metadata["planned_eigenvectors"] = planned_eigenvectors
    if "lanczos_steps" in analysis.epack_stats:
        metadata["lanczos_steps"] = analysis.epack_stats["lanczos_steps"]
    if "restart_iterations" in analysis.epack_stats:
        metadata["restart_iterations"] = analysis.epack_stats["restart_iterations"]
    if "nconv" in analysis.epack_stats:
        metadata["nconv"] = analysis.epack_stats["nconv"]
    if "eigenvectors_read" in analysis.epack_stats:
        metadata["eigenvectors_read"] = analysis.epack_stats["eigenvectors_read"]
    if "epack_load_observed" in analysis.epack_stats:
        metadata["epack_load_observed"] = analysis.epack_stats[
            "epack_load_observed"
        ]
    return metadata


def epack_progress_override(
    component: str,
    config: Any,
    analysis: PerformanceAnalysis,
) -> Optional[float]:
    """Use Lanczos step progress for IRL epack when possible."""
    if component != "epack":
        return None

    planned_lanczos_steps = get_planned_lanczos_steps(config)
    observed_lanczos_steps = analysis.epack_stats.get("lanczos_steps")
    if planned_lanczos_steps and observed_lanczos_steps is not None:
        return min(observed_lanczos_steps / planned_lanczos_steps, 1.0)

    epack_config = getattr(config, "epack_config", None)
    planned_eigenvectors = (
        getattr(epack_config, "eigs", None) if epack_config is not None else None
    )
    observed_eigenvectors = analysis.epack_stats.get("eigenvectors_read")
    if (
        getattr(epack_config, "load", False)
        and planned_eigenvectors
        and observed_eigenvectors is not None
    ):
        return min(observed_eigenvectors / planned_eigenvectors, 1.0)

    return None


def build_component_score(
    *,
    planned_count: int,
    observed_modules: List[ModuleObservation],
    node_count: Optional[int],
    metadata: Optional[Dict[str, Any]] = None,
    progress_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a JSON-serializable benchmark component score."""
    observed_count = len(observed_modules)
    elapsed_seconds = sum_elapsed_seconds(observed_modules)

    if progress_override is not None:
        progress = progress_override
    elif planned_count > 0:
        progress = min(observed_count / planned_count, 1.0)
    else:
        progress = None

    observed_node_seconds = (
        elapsed_seconds * node_count
        if elapsed_seconds is not None and node_count is not None
        else None
    )
    normalized_node_seconds = (
        observed_node_seconds / progress
        if observed_node_seconds is not None and progress not in (None, 0)
        else None
    )

    return {
        "planned_count": planned_count,
        "observed_count": observed_count,
        "progress": progress,
        "elapsed_seconds": elapsed_seconds,
        "observed_node_seconds": observed_node_seconds,
        "normalized_node_seconds": normalized_node_seconds,
        "metadata": metadata or {},
    }


def is_quark_module_new_format(module_name: str) -> bool:
    """
    Detect whether a quark module uses the new format (with gamma structure) or old format.

    Args:
        module_name: The quark module name to check

    Returns:
        True if new format (contains gamma identifiers), False if old format
    """
    # New format contains gamma structure identifiers
    gamma_identifiers = ["pion_local", "vec_local", "vec_onelink", "scalar_local"]
    return any(gamma in module_name for gamma in gamma_identifiers)


def categorize_modules(
    timings: Dict[str, float],
    mf_custom: Dict[str, Dict[str, float]],
    epack_stats: Dict[str, Any],
    cg_iterations: Dict[str, List[int]],
) -> Dict[str, Dict]:
    """
    Categorize modules into major cost groups.

    Args:
        timings: Dictionary of module names to execution times
        mf_custom: Dictionary of mf_ modules to custom timer data
        epack_stats: Dictionary of epack solver statistics
        cg_iterations: Dictionary of module names to CG iteration count lists

    Returns:
        Dictionary with categorized modules
    """
    categories = {
        "epack": {},
        "epack_stats": epack_stats,
        "cg_solves": defaultdict(lambda: defaultdict(list)),
        "meson_field": {},
        "other": {},
    }

    for module_name, time_seconds in timings.items():
        if module_name == "epack" or "epack" in module_name:
            categories["epack"][module_name] = time_seconds

        elif module_name.startswith("quark_"):
            # Parse quark module name - supports both old and new formats
            # New format: quark_{operation}_{gamma}_{mass}_t{timeslice}
            #   Example: quark_ranLL_pion_local_mass_l_t0
            # Old format: quark_{operation}_{mass}_t{timeslice}
            #   Example: quark_lma_m000569_t0

            parts = module_name.split("_")
            if len(parts) >= 3:
                is_new_format = is_quark_module_new_format(module_name)

                if is_new_format:
                    # New format parsing
                    if len(parts) >= 4:
                        operation = parts[1]  # ama or ranLL
                        gamma = "_".join(
                            parts[2:4]
                        )  # pion_local, vec_local, vec_onelink

                        # Extract mass label
                        # New format: quark_ranLL_pion_local_mass_l_t0 -> mass = "l"
                        mass = None
                        timeslice = None
                        for i, part in enumerate(parts):
                            # New format: "mass" followed by mass label (e.g., mass_l -> extract "l")
                            if part == "mass" and i + 1 < len(parts):
                                mass = parts[i + 1]  # Extract "l", "h", etc.
                                break  # Found mass, stop looking
                            # Old format: part starts with 'm' followed by digits (e.g., m001326, m000514)
                            elif (
                                part.startswith("m")
                                and len(part) > 1
                                and part[1:].isdigit()
                            ):
                                mass = part  # Extract "m001326", "m000514", etc.
                                break  # Found mass, stop looking
                            # Extract timeslice (comes after mass in module name)
                            if part.startswith("t") and i > 0:
                                timeslice = part

                        # Create category key - always include mass to separate by mass
                        key = f"{gamma}_{operation}"
                        if mass:
                            key += f"_{mass}"

                        # Store timing with timeslice info
                        categories["cg_solves"][key]["timings"].append(time_seconds)
                        if timeslice:
                            categories["cg_solves"][key]["timeslices"].append(timeslice)

                        # Store CG iteration counts (only for ama modules)
                        if operation == "ama" and module_name in cg_iterations:
                            # Store all iteration counts for this module instance
                            # Each count represents one MixedPrecisionConjugateGradient solve
                            categories["cg_solves"][key]["iterations"].extend(
                                cg_iterations[module_name]
                            )
                else:
                    # Old format parsing: quark_{operation}_{mass}_t{timeslice}
                    # Example: quark_lma_m000569_t0
                    operation = parts[1]  # lma or ama

                    # Normalize operation name for consistency
                    if operation == "lma":
                        operation = "ranLL"
                    # ama stays as ama

                    # Extract mass and timeslice
                    mass = None
                    timeslice = None
                    for i, part in enumerate(parts):
                        # Old format: part starts with 'm' followed by digits (e.g., m000569)
                        if (
                            part.startswith("m")
                            and len(part) > 1
                            and part[1:].isdigit()
                        ):
                            mass = part  # Extract "m000569", etc.
                        # Extract timeslice (comes after mass, starts with 't')
                        elif part.startswith("t") and i > 0:
                            timeslice = part

                    # Create category key - no gamma structure for old format
                    key = f"{operation}_{mass}" if mass else operation

                    # Store timing with timeslice info
                    categories["cg_solves"][key]["timings"].append(time_seconds)
                    if timeslice:
                        categories["cg_solves"][key]["timeslices"].append(timeslice)

                    # Mark this category as having combined gamma computation
                    categories["cg_solves"][key]["is_combined_gamma"] = True

                    # Store CG iteration counts (only for ama modules)
                    if operation == "ama" and module_name in cg_iterations:
                        categories["cg_solves"][key]["iterations"].extend(
                            cg_iterations[module_name]
                        )

        elif module_name.startswith("mf_"):
            # Store module timing along with custom timer data
            module_data = {"total_time": time_seconds}
            if module_name in mf_custom:
                module_data.update(mf_custom[module_name])
                # Calculate global sum time as the unaccounted portion
                # global_sum_time = total_time - kernel_time - IO_time
                if "kernel" in module_data and "io_total" in module_data:
                    global_sum_time = (
                        time_seconds - module_data["kernel"] - module_data["io_total"]
                    )
                    module_data["global_sum"] = global_sum_time
            categories["meson_field"][module_name] = module_data

        else:
            categories["other"][module_name] = time_seconds

    return categories


def format_time(seconds: float) -> str:
    """
    Format time in human-readable format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string
    """
    if seconds >= 3600:
        hours = seconds / 3600
        return f"{hours:.2f} hours ({seconds:.2f} s)"
    elif seconds >= 60:
        minutes = seconds / 60
        return f"{minutes:.2f} min ({seconds:.2f} s)"
    elif seconds >= 1:
        return f"{seconds:.2f} s"
    elif seconds >= 0.001:
        ms = seconds * 1000
        return f"{ms:.2f} ms"
    else:
        us = seconds * 1_000_000
        return f"{us:.2f} us"


def print_summary(
    categories: Dict[str, Dict],
    total_time: float,
    communicator_sizes: Dict[str, Optional[int]] = None,
    lattice_grid: Optional[str] = None,
    is_incomplete: bool = False,
):
    """
    Print performance summary.

    Args:
        categories: Categorized module timings
        total_time: Total execution time in seconds
        communicator_sizes: Dictionary with 'world_size' and 'node_size' keys
        lattice_grid: Lattice grid size string (e.g., "144.144.144.288")
        is_incomplete: Whether the log file is incomplete (job ran out of time)
    """
    print("=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    if is_incomplete:
        print()
        print("⚠️  WARNING: This log file is INCOMPLETE - the job ran out of time")
        print("              Timings shown are for modules that completed.")
        print()
    print()

    # Display lattice grid and communicator sizes at the beginning
    if lattice_grid:
        print(f"Lattice grid: {lattice_grid}")

    if communicator_sizes:
        if communicator_sizes["world_size"] is not None:
            print(
                f"World communicator size (nodes * gpus per node): {communicator_sizes['world_size']}"
            )
        if communicator_sizes["node_size"] is not None:
            print(f"Node communicator size: {communicator_sizes['node_size']}")
        print()

    # 1. IRL Solver (epack)
    print("1. IRL SOLVER (epack modules)")
    print("-" * 80)
    if categories["epack"]:
        epack_total = sum(categories["epack"].values())
        for module, time_s in sorted(
            categories["epack"].items(), key=lambda x: x[1], reverse=True
        ):
            pct = (time_s / total_time * 100) if total_time > 0 else 0
            print(f"  {module:40s}: {format_time(time_s):>25s}  ({pct:5.2f}%)")

        # Print epack statistics if available
        epack_stats = categories.get("epack_stats", {})
        if epack_stats:
            if "restart_iterations" in epack_stats:
                print(
                    f"    {'Restart iterations':38s}: {epack_stats['restart_iterations']:>25d}"
                )
            if "lanczos_steps" in epack_stats:
                print(
                    f"    {'Total Lanczos steps':38s}: {epack_stats['lanczos_steps']:>25d}"
                )
            if "nconv" in epack_stats:
                print(
                    f"    {'Converged eigenvectors':38s}: {epack_stats['nconv']:>25d}"
                )
            if "num_eigenvectors_written" in epack_stats:
                print(
                    f"    {'Eigenvectors written to disk':38s}: {epack_stats['num_eigenvectors_written']:>25d}"
                )
            if "eigenvector_write_time" in epack_stats:
                write_time = epack_stats["eigenvector_write_time"]
                write_pct = (write_time / epack_total * 100) if epack_total > 0 else 0
                print(
                    f"    {'Eigenvector write time':38s}: {format_time(write_time):>25s}  ({write_pct:5.2f}% of epack)"
                )

        print(
            f"  {'TOTAL':40s}: {format_time(epack_total):>25s}  "
            f"({epack_total/total_time*100:5.2f}%)"
        )
    else:
        print("  No epack modules found")
    print()

    # 2. CG Solves
    print("2. CG SOLVES (quark_ modules, averaged over time slices)")
    print("-" * 80)
    if categories["cg_solves"]:
        cg_results = []

        for key, data in categories["cg_solves"].items():
            timings = data["timings"]
            if timings:
                avg_time = sum(timings) / len(timings)
                total_time_group = sum(timings)
                count = len(timings)
                cg_results.append((key, avg_time, total_time_group, count, data))

        # Sort by total time descending
        cg_results.sort(key=lambda x: x[2], reverse=True)

        cg_grand_total = 0
        for key, avg_time, total_time_group, count, data in cg_results:
            pct = (total_time_group / total_time * 100) if total_time > 0 else 0

            # Format key to show mass information clearly
            # Convert "pion_local_ama_l" -> "pion_local_ama (mass: l)"
            # Convert "pion_local_ama_m001326" -> "pion_local_ama (mass: m001326)"
            # For old format modules, add "(all gammas combined)" label
            parts = key.rsplit("_", 1)
            if len(parts) == 2 and (parts[1].startswith("m") or len(parts[1]) == 1):
                display_key = f"{parts[0]} (mass: {parts[1]})"
            else:
                display_key = key

            # Add label for old format modules that have combined gamma computation
            if data.get("is_combined_gamma", False):
                display_key += " (all gammas combined)"

            print(f"  {display_key:50s}:")
            print(f"    Average:    {format_time(avg_time):>25s}")
            print(
                f"    Total:      {format_time(total_time_group):>25s}  ({pct:5.2f}%)"
            )
            print(f"    Count:      {count:>25d} time slices")

            # Add iteration statistics if available
            if "iterations" in data and data["iterations"]:
                iterations = data["iterations"]
                min_iter = min(iterations)
                max_iter = max(iterations)
                mean_iter = sum(iterations) / len(iterations)
                print(
                    f"    CG iterations (min/max/mean): {min_iter:>8d} / {max_iter:>8d} / {mean_iter:>8.1f}"
                )

            print()
            cg_grand_total += total_time_group

        print(
            f"  {'CG SOLVES TOTAL':50s}: {format_time(cg_grand_total):>25s}  "
            f"({cg_grand_total/total_time*100:5.2f}%)"
        )
    else:
        print("  No CG solve modules found")
    print()

    # 3. Meson Field Calculations
    print("3. MESON FIELD CALCULATIONS (mf_ modules)")
    print("-" * 80)
    if categories["meson_field"]:
        # Separate local and onelink
        local_modules = {}
        onelink_modules = {}
        other_mf = {}

        for module, data in categories["meson_field"].items():
            if "local" in module and "onelink" not in module:
                local_modules[module] = data
            elif "onelink" in module:
                onelink_modules[module] = data
            else:
                other_mf[module] = data

        mf_total = 0

        if local_modules:
            print("  Local modules:")
            local_total = 0
            for module, data in sorted(
                local_modules.items(), key=lambda x: x[1]["total_time"], reverse=True
            ):
                time_s = data["total_time"]
                local_total += time_s
                pct = (time_s / total_time * 100) if total_time > 0 else 0
                print(f"    {module:48s}: {format_time(time_s):>25s}  ({pct:5.2f}%)")

                # Print kernel and IO time if available
                if "kernel" in data:
                    kernel_time = data["kernel"]
                    kernel_pct = (kernel_time / time_s * 100) if time_s > 0 else 0
                    print(
                        f"      {'Kernel time':46s}: {format_time(kernel_time):>25s}  ({kernel_pct:5.2f}% of module)"
                    )
                if "io_total" in data:
                    io_time = data["io_total"]
                    io_pct = (io_time / time_s * 100) if time_s > 0 else 0
                    print(
                        f"      {'IO time':46s}: {format_time(io_time):>25s}  ({io_pct:5.2f}% of module)"
                    )
                if "global_sum" in data:
                    global_sum_time = data["global_sum"]
                    global_sum_pct = (
                        (global_sum_time / time_s * 100) if time_s > 0 else 0
                    )
                    print(
                        f"      {'Global sum time':46s}: {format_time(global_sum_time):>25s}  ({global_sum_pct:5.2f}% of module)"
                    )

            print(
                f"    {'Local Total':48s}: {format_time(local_total):>25s}  "
                f"({local_total/total_time*100:5.2f}%)"
            )
            print()
            mf_total += local_total

        if onelink_modules:
            print("  Onelink modules:")
            onelink_total = 0
            for module, data in sorted(
                onelink_modules.items(), key=lambda x: x[1]["total_time"], reverse=True
            ):
                time_s = data["total_time"]
                onelink_total += time_s
                pct = (time_s / total_time * 100) if total_time > 0 else 0
                print(f"    {module:48s}: {format_time(time_s):>25s}  ({pct:5.2f}%)")

                # Print kernel and IO time if available
                if "kernel" in data:
                    kernel_time = data["kernel"]
                    kernel_pct = (kernel_time / time_s * 100) if time_s > 0 else 0
                    print(
                        f"      {'Kernel time':46s}: {format_time(kernel_time):>25s}  ({kernel_pct:5.2f}% of module)"
                    )
                if "io_total" in data:
                    io_time = data["io_total"]
                    io_pct = (io_time / time_s * 100) if time_s > 0 else 0
                    print(
                        f"      {'IO time':46s}: {format_time(io_time):>25s}  ({io_pct:5.2f}% of module)"
                    )
                if "global_sum" in data:
                    global_sum_time = data["global_sum"]
                    global_sum_pct = (
                        (global_sum_time / time_s * 100) if time_s > 0 else 0
                    )
                    print(
                        f"      {'Global sum time':46s}: {format_time(global_sum_time):>25s}  ({global_sum_pct:5.2f}% of module)"
                    )

            print(
                f"    {'Onelink Total':48s}: {format_time(onelink_total):>25s}  "
                f"({onelink_total/total_time*100:5.2f}%)"
            )
            print()
            mf_total += onelink_total

        if other_mf:
            print("  Other meson field modules:")
            other_total = 0
            for module, data in sorted(
                other_mf.items(), key=lambda x: x[1]["total_time"], reverse=True
            ):
                time_s = data["total_time"]
                other_total += time_s
                pct = (time_s / total_time * 100) if total_time > 0 else 0
                print(f"    {module:48s}: {format_time(time_s):>25s}  ({pct:5.2f}%)")

                # Print kernel and IO time if available
                if "kernel" in data:
                    kernel_time = data["kernel"]
                    kernel_pct = (kernel_time / time_s * 100) if time_s > 0 else 0
                    print(
                        f"      {'Kernel time':46s}: {format_time(kernel_time):>25s}  ({kernel_pct:5.2f}% of module)"
                    )
                if "io_total" in data:
                    io_time = data["io_total"]
                    io_pct = (io_time / time_s * 100) if time_s > 0 else 0
                    print(
                        f"      {'IO time':46s}: {format_time(io_time):>25s}  ({io_pct:5.2f}% of module)"
                    )
                if "global_sum" in data:
                    global_sum_time = data["global_sum"]
                    global_sum_pct = (
                        (global_sum_time / time_s * 100) if time_s > 0 else 0
                    )
                    print(
                        f"      {'Global sum time':46s}: {format_time(global_sum_time):>25s}  ({global_sum_pct:5.2f}% of module)"
                    )

            print(
                f"    {'Other Total':48s}: {format_time(other_total):>25s}  "
                f"({other_total/total_time*100:5.2f}%)"
            )
            print()
            mf_total += other_total

        print(
            f"  {'MESON FIELD TOTAL':50s}: {format_time(mf_total):>25s}  "
            f"({mf_total/total_time*100:5.2f}%)"
        )
    else:
        print("  No meson field modules found")
    print()

    # 4. Other modules (summary only)
    if categories["other"]:
        other_total = sum(categories["other"].values())
        print("4. OTHER MODULES")
        print("-" * 80)
        print(
            f"  {'Total for other modules':50s}: {format_time(other_total):>25s}  "
            f"({other_total/total_time*100:5.2f}%)"
        )
        print()

    # Grand total
    print("=" * 80)
    print(f"TOTAL EXECUTION TIME: {format_time(total_time)}")
    print("=" * 80)


def analyze_performance(file_path: str) -> PerformanceAnalysis:
    """Analyze a Hadrons output file and return structured performance data."""
    lattice_grid = extract_lattice_grid(file_path)
    communicator_sizes = extract_communicator_sizes(file_path)
    timings, is_incomplete = extract_module_timings(file_path)

    if not timings:
        raise ValueError("No timing data found in the file.")

    module_observations = build_module_observations(file_path, timings)
    mf_custom_timers = extract_mf_custom_timers(file_path)
    epack_stats = extract_epack_statistics(file_path)
    cg_iterations = extract_cg_iteration_counts(file_path)
    total_time = sum(timings.values())
    categories = categorize_modules(
        timings, mf_custom_timers, epack_stats, cg_iterations
    )

    return PerformanceAnalysis(
        file_path=file_path,
        lattice_grid=lattice_grid,
        communicator_sizes=communicator_sizes,
        timings=timings,
        module_observations=module_observations,
        is_incomplete=is_incomplete,
        mf_custom_timers=mf_custom_timers,
        epack_stats=epack_stats,
        cg_iterations=cg_iterations,
        total_time=total_time,
        categories=categories,
    )


def analyze_grid_benchmark_performance(file_path: str) -> PerformanceAnalysis:
    """Analyze Grid output for benchmark JSON generation."""
    lattice_grid = extract_lattice_grid(file_path)
    communicator_sizes = extract_communicator_sizes(file_path)
    module_observations, is_incomplete, grid_epack_stats = (
        extract_grid_benchmark_observations(file_path)
    )
    epack_stats = extract_epack_statistics(file_path) | grid_epack_stats
    timings = {
        f"{observation.module_name}_{index}": observation.elapsed_seconds
        for index, observation in enumerate(module_observations)
        if observation.elapsed_seconds is not None
    }
    total_time = sum(timings.values())

    return PerformanceAnalysis(
        file_path=file_path,
        lattice_grid=lattice_grid,
        communicator_sizes=communicator_sizes,
        timings=timings,
        module_observations=module_observations,
        is_incomplete=is_incomplete,
        mf_custom_timers={},
        epack_stats=epack_stats,
        cg_iterations={},
        total_time=total_time,
        categories={},
    )


def analyze_benchmark_performance(task_key: str, log_file: str) -> PerformanceAnalysis:
    """Analyze a benchmark log using the parser for the configured task."""
    if task_key == HADRONS_LMI_TASK_KEY:
        return analyze_performance(log_file)
    if task_key == GRID_LMI_TASK_KEY:
        return analyze_grid_benchmark_performance(log_file)
    raise ValueError(f"Unsupported LMI benchmark task key: {task_key!r}")


def validate_benchmark_task_key(task_key: str) -> None:
    """Raise when the task key is not supported by benchmark JSON generation."""
    if task_key not in SUPPORTED_LMI_TASK_KEYS:
        supported = ", ".join(SUPPORTED_LMI_TASK_KEYS)
        raise ValueError(
            "Performance benchmark only supports Hadrons LMI and Grid LMI jobs in v1 "
            f"({supported}); got {task_key!r}."
        )


def benchmark_lmi_performance(
    job: str,
    log_file: str,
    yaml_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Build v1 component-first benchmark JSON data for a configured LMI job."""
    from pyfm.nanny.taskbuilder import create_task

    task = create_task(job, yaml_params, series="a", cfg="1")
    validate_benchmark_task_key(task.key)

    analysis = analyze_benchmark_performance(task.key, log_file)
    planned_input = task.handler.build_input_params(task.config)
    node_count = derive_node_count(analysis.communicator_sizes)

    planned_by_component = count_planned_components(task.key, planned_input)
    observed_by_component = group_observed_components(analysis.module_observations)

    components = {
        component: build_component_score(
            planned_count=planned_by_component.get(component, 0),
            observed_modules=observed_by_component.get(component, []),
            node_count=node_count,
            metadata=build_component_metadata(component, task.config, analysis),
            progress_override=epack_progress_override(component, task.config, analysis),
        )
        for component in BENCHMARK_COMPONENTS
    }

    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "job": job,
        "task_key": task.key,
        "log": log_file,
        "metadata": {
            "lattice_grid": analysis.lattice_grid,
            "world_size": analysis.communicator_sizes.get("world_size"),
            "node_size": analysis.communicator_sizes.get("node_size"),
            "node_count": node_count,
            "is_incomplete": analysis.is_incomplete,
            "total_time_seconds": analysis.total_time,
        },
        "components": components,
    }


def analyze_file(file_path: str) -> None:
    """Analyze a Hadrons output file and print a performance summary."""
    analysis = analyze_performance(file_path)
    print_summary(
        analysis.categories,
        analysis.total_time,
        analysis.communicator_sizes,
        analysis.lattice_grid,
        analysis.is_incomplete,
    )


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_performance.py <output_file>", file=sys.stderr)
        print("\nExample:", file=sys.stderr)
        print(
            "  python analyze_performance.py lma-0-142-e4000-n1-c.2442", file=sys.stderr
        )
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        analyze_file(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
