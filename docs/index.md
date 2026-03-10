# PyFM

Nanny, postprocessing, and A2A contraction scripts for lattice QCD calculations.

## Installation

### Python package

Requires Python >= 3.12.

```bash
pip install -e .
```

### Building Grid, Hadrons, and HadronsMILC

PyFM drives the [HadronsMILC](https://github.com/Michael628/HadronsMILC) application, which depends on [Grid](https://github.com/milc-qcd/Grid) and [Hadrons](https://github.com/milc-qcd/Hadrons). The `systems/build.sh` script in this repo handles cloning, configuring, and building all three. Run it from the parent workspace directory that will contain all repos side-by-side:

```bash
# Build all components for a generic scalar (CPU) system
pyfm/systems/build.sh --system scalar --all

# Build for a specific HPC system (e.g. Perlmutter GPU)
pyfm/systems/build.sh --system perlmutter --all --threads 8

# Build dependencies first, then the stack
pyfm/systems/build.sh --gmp --mpfr --lime --system scalar --grid --hadrons --app
```

Available systems: `scalar` (CPU, default), `perlmutter`, `deltaai`, `lq`, `lq2`. See [`systems/README.md`](https://github.com/Michael628/pyfm/blob/main/systems/README.md) for full details on customizing builds and adding new systems.

## Scripts

The `scripts/` directory contains standalone utilities for managing lattice QCD workflows:

### Input Generation & Task Management

- **`generate_input.py`** - Generate input files for specific job steps and configurations
  - Usage: `python scripts/generate_input.py -j <job> -s <series> -n <config> [-p params.yaml]`

- **`check_task_completion.py`** - Audit task completion status for a given configuration
  - Usage: `python scripts/check_task_completion.py -j <job> -s <series> -n <config> [-p params.yaml] [-v]`
  - Use `-v` to show all files (not just missing)

### Data Processing & Aggregation

- **`aggregate_task_data.py`** - Aggregate all output data for a job step
  - Usage: `python scripts/aggregate_task_data.py -j <job> [-p params.yaml] [-f csv]`

- **`process_lmi_df.py`** - Process low-mode interpolator dataframes
  - Usage: `python scripts/process_lmi_df.py <file_paths...>`

- **`merge_completed_df.py`** - Merge multiple processed dataframes into a single file
  - Usage: `python scripts/merge_completed_df.py [-i] <outfile> <filestem>`
  - Use `-i` for intersection of configurations across files

### Analysis Tools

- **`compare_hdf5_matrices.py`** - Compare a2aMatrix datasets between two HDF5 files
  - Usage: `python scripts/compare_hdf5_matrices.py <file1> <file2> [-v]`

- **`contract_a2a_diagrams.py`** - Execute all-to-all (A2A) contraction calculations
