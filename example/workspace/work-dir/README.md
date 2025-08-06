# PyFM Work Directory

This directory contains a complete example workspace for running lattice QCD calculations using the PyFM framework. It demonstrates the typical workflow from gauge field smearing through hadron spectrum calculations and correlator contractions.

## Directory Structure

### Configuration Files

- **`params.yaml`** - Master configuration file containing all parameters for smear, hadrons, and contract jobs including masses, lattice dimensions, solver parameters, and file paths
- **`todo`** - Task queue file listing pending jobs in format: `{series}.{config} {job_type} {job_id} ...`

### Execution Scripts

- **`smear.sh`** - Execution script for gauge field smearing jobs (creates fat and long links with APBC and KS phases)
- **`hadrons.sh`** - Execution script for Hadrons/Grid calculations (eigenvalue calculation, random wall CG solves, and A2A meson fields)
- **`contract.sh`** - Execution script for A2A meson field contraction jobs

### Input/Output Directories

#### `in/`

Contains input files for all job types

- **`test-l4444-hadrons-a.20.xml`** - Hadrons XML configuration example for low mode averaging

#### `out/`

Contains output logs and results from job executions:

- **`test-l4444-hadrons-a.20`** - Output log from hadrons lma calculation

#### `schedules/`

Contains Hadrons execution scheduling files:

- **`test-l4444-hadrons-a.20.sched`** - Schedule file for desired order of hadrons module execution

### Lattice Data

#### `lat/scidac/`

Gauge field configurations in ILDG format:

- **`l4444a.ildg.20`** - Base gauge links (4×4×4×4 lattice, configuration 20)
- **`fat4444a.ildg.20`** - Fat links (smeared gauge fields for staggered fermions)
- **`lng4444a.ildg.20`** - Long links (naik term smeared field)

### Physics Results

#### `e100n1dt1/`

Results directory organized by calculation parameters:

##### `correlators/m002426/`

Two-point correlation functions for mass m=0.002426:

- **`pion_local/`** - Pion correlators with local operators
  - `ama/` - correlators from random wall solves
  - `ranLL/` - correlators from low-mode contribution to random wall solves
- **`vec_local/`** - Vector meson correlators with local operators
- **`vec_onelink/`** - Vector meson correlators with one-link operators

Each subdirectory contains HDF5 files with correlator data:

- Format: `corr_{operator}_{method}_m{mass}_t{tsource}_{series}.h5`

where tsource is the source time slice

##### `mesons/m002426/mf_a.20/`

Meson correlation functions from direct Hadrons calculations:

- **`G5_G5_0_0_0.h5`** - Pseudoscalar (pion) correlators
- **`GX_G1_0_0_0.h5`**, **`GY_G1_0_0_0.h5`**, **`GZ_G1_0_0_0.h5`** - Vector-onelink correlators
- **`GX_GX_0_0_0.h5`**, **`GY_GY_0_0_0.h5`**, **`GZ_GZ_0_0_0.h5`** - Vector-local correlators

### Example Scripts

#### `slurm-example/`

Example SLURM submission scripts (used on Perlmutter):

- **`smear_example.slurm`** - Example for gauge smearing jobs
- **`hadrons_example.slurm`** - Example for hadrons spectrum calculations
- **`contract_example.slurm`** - Example for A2A contraction jobs

## Workflow

1. **Smearing** (`smear.sh`): Generate fat and long links from base gauge configurations
2. **Hadrons** (`hadrons.sh`): Compute eigenvalues/eigenvectors and meson correlators
3. **Contraction** (`contract.sh`): Perform A2A contractions to compute hadron correlators

The `params.yaml` file controls all aspects of this workflow, while the `todo` file manages job dependencies and execution order. The PyFM nanny system monitors the todo file and automatically submits jobs when dependencies are satisfied.

## File Naming Conventions

- **Series**: `a` = MC ensemble series identifier
- **Configuration**: `20` = specific gauge configuration number
- **Mass**: `002426` = quark mass value (m=0.002426)
- **Tsource**: `t0`, `t1` = source time slice
- **Method**: `ama` (random wall CG solves), `ranLL` (low-mode contribution to random wall solves)

