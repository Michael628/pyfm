# PyFM CLI Command Reference

> Outline of every command exposed by the `pyfm` console script (`pyfm.cli:cli`).
> Each section is scoped to one command and separated by `---` so it can be split into slides.
> All commands are invoked as `pyfm <group> <command> [options]`.

The CLI is organized into seven command groups:

| Group | Purpose |
|-------|---------|
| `nanny` | Automated HPC job submission via todo files |
| `export` | Format-based data export (`export corr`) |
| `task` | Deprecated aliases: `nanny generate`, `export corr` |
| `contract` | Run A2A all-to-all contraction calculations |
| `audit` | Audit runtime performance output logs |
| `build` | Compile lattice QCD software components |
| `workspace` | Initialize & configure a PyFM workspace |

Most commands read a YAML parameter file (default `params.yaml`) containing sections such as `nanny`, `job_setup`, and per-task configuration blocks.

---

## `pyfm nanny run`

**1. What it does**
Runs the nanny loop — the main driver that continuously submits and monitors HPC jobs against the scheduler based on the todo file. Sets `umask 022` and configures logging before entering the loop.

**2. Configuration it pulls from**
- `--param-file` (`params.yaml`) → full nanny configuration, including the `nanny` section (todo file, scheduler) and `job_setup` (job step definitions).
- Optionally restricts the loop to a single job step via `--job`.

**3. The command**
```bash
pyfm nanny run [-p params.yaml] [-j JOB_STEP] [--logging-level INFO]
```

---

## `pyfm nanny submit`

**1. What it does**
Submits a single job to the HPC scheduler (one-shot, outside the loop). Builds the nanny config and the job-step config, sets the `INPUTLIST` environment variable, and dispatches the job.

**2. Configuration it pulls from**
- `--param-file` → `get_nanny_config()` (scheduler settings) and `get_job_config(job, ...)` (the named job step).
- `--input` → input file list, exported as the `INPUTLIST` env var.
- `--job` → which job step to submit (required).

**3. The command**
```bash
pyfm nanny submit -i INPUT_FILE -j JOB_STEP [-p params.yaml] [--logging-level INFO]
```

---

## `pyfm nanny add`

**1. What it does**
Adds todo entries for a gauge-field series and one or more job steps, expanding them across a set of configuration numbers. Validates the step names and parses the config selection before writing to the todo file.

**2. Configuration it pulls from**
- `--param-file` → `nanny.todo_file` (where entries are written) and `job_setup` (valid step names, used by `validate_steps`).
- `-s/--series` (required) → gauge field series label.
- `STEPS` (positional, one or more) → job steps to add.
- Config selection (exactly one required, mutually exclusive):
  - `--config` → individual config numbers (repeatable).
  - `--config-range START STOP STEP` → range, exclusive stop.

**3. The command**
```bash
pyfm nanny add -s SERIES STEP [STEP ...] (--config N [--config N ...] | --config-range START STOP STEP) [-p params.yaml]
```

---

## `pyfm nanny check`

**1. What it does**
Two modes. With no filters, prints a summary of all job statuses. When `--job`, `--series`, and `--config` are all supplied, it builds that specific task and audits its output files (optionally verbose).

**2. Configuration it pulls from**
- `--param-file` → job/task definitions used by `check_jobs()` and `create_task()`.
- `--job` / `--series` / `--config` → narrow to one task for output-file auditing.
- `--verbose` → detailed per-file status.

**3. The command**
```bash
# Summary of all jobs
pyfm nanny check [-p params.yaml]
# Audit one task's outputs
pyfm nanny check -j JOB -s SERIES -n CONFIG [-v] [-p params.yaml]
```

---

## `pyfm nanny generate`

**1. What it does**
Generates the input file(s) for a specific job / series / config combination by calling `write_input_file`, then logs the path of the written input.

**2. Configuration it pulls from**
- `--param-file` → the job-step's task configuration used to render the input file.
- `--job` / `--series` / `--config` → all required; identify which input to build.

**3. The command**
```bash
pyfm nanny generate -j JOB -s SERIES -n CONFIG [-p params.yaml]
```

---

## `pyfm export corr`

**1. What it does**
Aggregates a job step's output data across all configurations into a single file, optionally averaging over configs and skipping configs whose output already exists.

**2. Configuration it pulls from**
- `--param-file` → the job step's aggregator parameters.
- `--job` → which step to aggregate (required).
- `--format` → output format (`csv` default, or `hdf5`).
- `--average` → average over configurations.
- `--skip-existing` → skip already-produced outputs.

**3. The command**
```bash
pyfm export corr -j JOB [-f csv|hdf5] [--average] [--skip-existing] [-p params.yaml] [--logging-level INFO]
```

---

## `pyfm task generate`

Deprecated alias for `pyfm nanny generate`; see that command.

---

## `pyfm task aggregate`

Deprecated alias for `pyfm export corr`; see that command.

---

## `pyfm contract run`

**1. What it does**
Executes A2A (all-to-all) contractions for every diagram defined in the parameter file. Builds a `ContractConfig`, enumerates low/high mode permutations and stochastic seed combinations per diagram, runs the contractions (CPU or GPU, MPI-aware), and writes results to HDF5 — skipping existing outputs unless `overwrite` is set.

**2. Configuration it pulls from**
- Parameter file (positional `PARAM_FILE` **or** `-p/--param-file-opt`) → built into `ContractConfig`, which provides: `diagrams` (per-diagram npoint, eig/stoch ranges, seed indices, perms, outfile), `hardware` (cpu/gpu), `comm_size`/`rank` (MPI), `time`, `overwrite`, and `logging_level`.
- `--time-average` → apply time averaging to each contraction before writing (changes the output array layout to `t2..t(n-1), dt`).

**3. The command**
```bash
pyfm contract run PARAM_FILE [--time-average]
# or
pyfm contract run -p PARAM_FILE [--time-average]
```

---

## `pyfm audit runtime`

**1. What it does**
Parses a single Hadrons output file and prints a human-readable runtime performance summary.

**2. Configuration it pulls from**
- `OUTPUT_FILE` (positional, must exist) → the Hadrons output/log to analyze. No YAML config needed.

**3. The command**
```bash
pyfm audit runtime OUTPUT_FILE
```

---

## `pyfm audit benchmark`

**1. What it does**
Computes component-first benchmark data for a configured Hadrons/Grid LMI run and emits it as JSON to stdout (sorted keys).

**2. Configuration it pulls from**
- `--param-file` → the LMI job configuration used to interpret the log.
- `--job` → job step name (required).
- `--log` → the Hadrons/Grid performance log file (required, must exist).

**3. The command**
```bash
pyfm audit benchmark -j JOB --log LOG_FILE [-p params.yaml]
```

---

## `pyfm build run`

**1. What it does**
Wrapper around `systems/build.sh`. Compiles selected lattice QCD software components (dependencies like GMP/MPFR/LIME/HDF5/OpenSSL/QMP/QIO and the main components Grid, Hadrons, HLMA, GLMA) for a target system profile, translating flags into shell arguments.

**2. Configuration it pulls from**
- `--system` → system profile name (default `scalar`); selects the build environment under `systems/`.
- `--ext` → profile extension/variant tag.
- `--threads` → parallel make threads (default 4).
- Component flags: `--grid`, `--hadrons`, `--hlma`, `--glma`, `--all`.
- Dependency flags: `--gmp`, `--mpfr`, `--lime`, `--hdf5`, `--ssl`, `--qmp`, `--qio`.
- Build modifiers: `--debug`, `--mpi-reduction`, `--force`, `--skip-make`.

**3. The command**
```bash
pyfm build run --system NAME [--all | --grid --hadrons ...] [--threads N] [--debug] [--force] [...]
```

---

## `pyfm workspace setup`

**1. What it does**
Wrapper around `systems/setup-workspace.sh`. Creates a new PyFM workspace — directory structure and config files — for a given system. Relative `--workspace`/`--storage` paths are resolved to absolute (and must already exist).

**2. Configuration it pulls from**
- `--workspace` → workspace root directory.
- `--storage` → storage root for job data.
- `--scheduler` → HPC scheduler type (slurm, pbs, lsf).
- `--lattice` → lattice geometry descriptor.
- `--system` → system profile name to configure.

**3. The command**
```bash
pyfm workspace setup [--workspace DIR] [--storage DIR] [--scheduler slurm] [--lattice DESC] [--system NAME]
```

---

## `pyfm workspace env`

**1. What it does**
Sources `systems/source-system-env.sh` for a system profile and prints `export` statements for any environment variables that differ from the current environment. Designed to be eval'd into your shell.

**2. Configuration it pulls from**
- `--system` → system profile to source (required).
- `--ext` → profile extension/variant tag.
- `--runtime-env` → include runtime env vars (default `true`).

**3. The command**
```bash
eval "$(pyfm workspace env --system NAME [--ext TAG] [--runtime-env true])"
```

---

## Suggestions / Notes

- **Slide grouping:** Consider a divider/title slide before each group (`nanny`, `task`, `contract`, `audit`, `build`, `workspace`) so related commands cluster together.
- **Shared convention:** Almost every data command shares the `-p/--param-file` → `params.yaml` and `--logging-level` pattern. A single "common options" slide up front would let you omit repeating it on each command.
- **Mode-dependent commands:** `nanny check` and `contract run` behave differently based on which flags are present — a small "before/after" or two-column slide works well for these.
- **End-to-end flow:** A workflow slide showing the typical lifecycle (`workspace setup` → `build run` → `nanny add` → `nanny run` → `export corr` → `audit runtime`) would tie the commands together narratively.
- **Worth confirming:** I documented behavior from the source; the exact contents of each `params.yaml` section (e.g. the full `ContractConfig`/`job_setup` schema) could be expanded into appendix slides if your audience needs the config detail.
