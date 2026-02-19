# Building Grid, Hadrons, and HadronsMILC

`build.sh` clones, configures, and builds the full lattice QCD software stack:

- **Grid** – `milc-qcd/Grid`, branch `feature/LMI-develop`
- **Hadrons** – `milc-qcd/Hadrons`, branch `feature/LMI-develop`
- **HadronsMILC** – `Michael628/HadronsMILC`, branch `develop`

## Workspace layout

Run `build.sh` from a top-level workspace directory. It creates the following structure relative to where it's run (`TOPDIR`):

```
TOPDIR/
├── pyfm/                  ← this repo
│   └── systems/           ← build.sh lives here
├── Grid/                  (cloned automatically)
│   ├── build-<system>/
│   └── install-<system>/
├── Hadrons/               (cloned automatically)
│   ├── build-<system>/
│   └── install-<system>/
├── HadronsMILC/           (cloned automatically)
│   ├── build-<system>/
│   └── install-<system>/
└── deps/                  (populated when building dependencies)
    └── install-<system>/
```

Example from a workspace root:

```bash
pyfm/systems/build.sh --system perlmutter --all --threads 8
```

## Usage

```
build.sh [OPTIONS]

Dependency Options:
  --gmp              Build GMP
  --mpfr             Build MPFR (requires GMP)
  --lime             Build LIME
  --hdf5             Build HDF5
  --ssl              Build OpenSSL

Component Options:
  --grid             Build Grid
  --hadrons          Build Hadrons
  --app              Build HadronsMILC
  --all              Build all three components

Build Configuration:
  --system <name>    Target system (default: scalar)
  --ext <name>       Additional suffix appended to build/install directory names
  --threads <n>      Parallel make jobs (default: 4)
  --debug            Enable debug build
  --force            Clean and reconfigure build directories
  --skip-make        Configure only, skip make
```

### Common invocations

```bash
# First-time CPU build from scratch (build deps, then stack)
pyfm/systems/build.sh --gmp --mpfr --lime --system scalar --all

# GPU build on Perlmutter
pyfm/systems/build.sh --system perlmutter --all --threads 16

# Force-rebuild only HadronsMILC
pyfm/systems/build.sh --system perlmutter --app --force

# Configure only (inspect Makefile before building)
pyfm/systems/build.sh --system scalar --grid --skip-make
```

## System configuration

Each system is a subdirectory here (e.g. `perlmutter/`, `scalar/`). A system may contain:

| File | Purpose |
|---|---|
| `configure-params.sh` | Defines `grid_configure`, `hadrons_configure`, `hmilc_configure`, `glma_configure`, `dependency_configure` shell functions |
| `env.sh` | Loads modules and sets environment variables before building |

Configuration is layered in this order (later sources override earlier ones):

1. `configure-params_default.sh` – fallback functions (CPU, no MPI, no accelerator)
2. `systems/<system>/configure-params.sh` – system-specific configure flags
3. `configure-params-<system>[<ext>].sh` in `TOPDIR` – optional local overrides

Environment is loaded from `systems/<system>/env.sh` unless a local `env-<system>[<ext>].sh` exists in `TOPDIR`.

## Adding a new system

1. Create `systems/<myhost>/`.
2. Add `configure-params.sh` with at minimum a `grid_configure` function. Use `scalar/configure-params.sh` as a starting point.
3. If the system needs module loads or `LD_LIBRARY_PATH` adjustments, add `env.sh`.
4. Run: `pyfm/systems/build.sh --system myhost --all`

### `configure-params.sh` function signatures

```bash
# All functions receive BUILD_EXT, BUILD_DEBUG, BUILD_MPI_REDUCTION as globals from build.sh

function grid_configure() {
  local INSTALLDIR=$1   # Grid install prefix
  local TOPDIR=$2       # Workspace root

  ${TOPDIR}/Grid/configure \
    --prefix=${INSTALLDIR} \
    # ... system-specific flags (SIMD, comms, accelerator, etc.)
}

function hadrons_configure() {
  local INSTALLDIR=$1
  local TOPDIR=$2

  ${TOPDIR}/Hadrons/configure \
    --prefix=${INSTALLDIR} \
    --with-grid=${TOPDIR}/Grid/install${BUILD_EXT}
}

function glma_configure() {
  local INSTALLDIR=$1
  local TOPDIR=$2

  ${TOPDIR}/grid-lma/configure \
    --prefix=${INSTALLDIR} \
    --with-grid=${TOPDIR}/Grid/install${BUILD_EXT}
}

function hmilc_configure() {
  local INSTALLDIR=$1
  local TOPDIR=$2

  ${TOPDIR}/HadronsMILC/configure \
    --prefix=${INSTALLDIR} \
    --with-grid=${TOPDIR}/Grid/install${BUILD_EXT} \
    --with-hadrons=${TOPDIR}/Hadrons/install${BUILD_EXT}
}

function dependency_configure() {
  local dep_name=$1    # gmp | mpfr | lime | hdf5 | openssl
  local INSTALLDIR=$2
  # ...
}
```

## Available systems

| System | Target | Notes |
|---|---|---|
| `scalar` | Generic CPU | No MPI, generic SIMD — useful for development/testing |
| `perlmutter` | NERSC Perlmutter (A100 GPUs) | CUDA/nvlink, Cray PE modules |
| `deltaai` | NCSA DeltaAI | GPU build |
| `lq` / `lq2` | Local clusters | Includes SLURM `run.slurm` scripts |
