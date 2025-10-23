# PyFM

Nanny, postprocessing, and A2A contraction scripts for lattice QCD calculations.

## Installation

### Requirements

- Python >= 3.11

### Install from source

```bash
pip install -e .
```

This will install PyFM in editable mode along with all required dependencies.

## Scripts

The `scripts/` directory contains standalone utilities for managing lattice QCD workflows:

### Input Generation & Task Management

- **`generate_input.py`** - Generate input files for specific job steps and configurations
  - Usage: `python scripts/generate_input.py -j <job> -s <series> -n <config> [-p params.yaml]`
  - Creates properly formatted input files based on parameter specifications

- **`check_task_completion.py`** - Audit task completion status for a given configuration
  - Usage: `python scripts/check_task_completion.py -j <job> -s <series> -n <config> [-p params.yaml] [-v]`
  - Reports missing/complete files; use `-v` flag to show all files

### Data Processing & Aggregation

- **`aggregate_task_data.py`** - Aggregate all output data for a job step
  - Usage: `python scripts/aggregate_task_data.py -j <job> [-p params.yaml] [-f csv]`
  - Collects and consolidates outputs matching the job specification

- **`process_lmi_df.py`** - Process low-mode interpolator dataframes
  - Usage: `python scripts/process_lmi_df.py <file_paths...>`
  - Transforms raw dataframes (averages over tsource, converts to real, reindexes)
  - Expects input from `dataframes/` directory, outputs to `processed_dataframes/`

- **`merge_completed_df.py`** - Merge multiple processed dataframes into a single file
  - Usage: `python scripts/merge_completed_df.py [-i] <outfile> <filestem>`
  - Use `-i` flag to compute only the intersection of configurations across files
  - Useful for combining results from different runs or ensembles

### Analysis Tools

- **`compare_hdf5_matrices.py`** - Compare a2aMatrix datasets between two HDF5 files
  - Usage: `python scripts/compare_hdf5_matrices.py <file1> <file2> [-v]`
  - Validates numerical consistency between files
  - Reports shape mismatches, dtype differences, and element-wise differences

- **`contract_a2a_diagrams.py`** - Execute all-to-all (A2A) contraction calculations
  - Usage: Configured via parameter file with diagram specifications
  - Computes meson correlators from A2A vectors with support for low/high mode mixing

## License

MIT
