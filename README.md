# PyFM

Lattice QCD workflow toolkit: job nanny, input generation, data aggregation, and A2A contraction.

## Installation

Requires Python >= 3.12.

```bash
pip install -e .
```

This installs the `pyfm` CLI entry point.

### Shell completion (optional)

Tab completion for subcommands and options is built in. Generate a script for your shell and source it once:

```bash
pyfm completion --shell bash >> ~/.bashrc      # then: exec bash (or new terminal)
pyfm completion --shell zsh  > ~/.zsh/_pyfm     # ensure ~/.zsh is on your fpath
pyfm completion --shell fish > ~/.config/fish/completions/pyfm.fish
```

The script assumes `pyfm` is on your `PATH`. Use `--prog` if the executable has a different name (e.g. an alias).

## Workspace setup

```bash
# Initialize a new workspace directory
pyfm workspace setup --workspace /path/to/workspace --scheduler slurm --system perlmutter

# Load the system environment into your shell
eval "$(pyfm workspace env --system perlmutter)"
```

## Building Grid, Hadrons, and HadronsMILC

PyFM drives the [HadronsMILC](https://github.com/Michael628/HadronsMILC) application, which depends on [Grid](https://github.com/milc-qcd/Grid) and [Hadrons](https://github.com/milc-qcd/Hadrons). Run from the parent workspace directory:

```bash
# Build all components for a generic scalar (CPU) system
pyfm build run --system scalar --all

# Build for a specific HPC system (e.g. Perlmutter GPU)
pyfm build run --system perlmutter --all --threads 8

# Build dependencies first, then the stack
pyfm build run --gmp --mpfr --lime --system scalar --grid --hadrons
```

Available systems: `scalar` (CPU, default), `perlmutter`, `deltaai`, `lq`, `lq2`. See [`systems/README.md`](systems/README.md) for details on customizing builds and adding new systems.

## CLI reference

All commands read a YAML parameter file (default `params.yaml`). See [`docs/pyfm-params-yaml-reference.md`](docs/pyfm-params-yaml-reference.md) for a full parameter reference.

### Job nanny

```bash
# Add todo entries for series 'a', configs 1000–2000 (step 10), steps hadrons and contract
pyfm nanny add a hadrons contract --cfg-range 1000 2010 10

# Run the nanny loop (submit and monitor jobs)
pyfm nanny run [-j hadrons]

# Submit a single job manually
pyfm nanny submit -i input_list.txt -j hadrons

# Check job status / audit output files
pyfm nanny check
pyfm nanny check -j hadrons -s a -n 1000 -v
```

### Input generation & aggregation

```bash
# Generate input file for a specific job/series/config
pyfm task generate -j hadrons -s a -n 1000

# Aggregate outputs across all configs
pyfm task aggregate -j hadrons [-f hdf5] [--average] [--skip-existing]
```

### A2A contractions

```bash
pyfm contract run params.yaml [--time-average]
```

### Performance analysis

```bash
# Summarize timing from a Hadrons output file
pyfm audit runtime output.log

# Emit JSON benchmark data for an LMI run
pyfm audit benchmark -j hadrons --log output.log

# Compare the outputs of two jobs of the same task type
pyfm audit output -j baseline rerun -s a -n 1000 [--rtol 1e-9] [--atol 1e-12]
```

## Documentation

- [CLI command reference](docs/pyfm-cli-commands-outline.md)
- [params.yaml parameter reference](docs/pyfm-params-yaml-reference.md)
