# scripts/

Standalone utility scripts, some of which predate the `pyfm` CLI.

## Superseded by the CLI

The following scripts have direct CLI equivalents. Prefer `pyfm` for new work — see the [top-level README](../README.md) for usage.

| Script | CLI equivalent |
|--------|---------------|
| `generate_input.py` | `pyfm nanny generate -j JOB -s SERIES -n CONFIG` |
| `check_task_completion.py` | `pyfm nanny check -j JOB -s SERIES -n CONFIG` |
| `aggregate_task_data.py` | `pyfm export corr -j JOB` |
| `contract_a2a_diagrams.py` | `pyfm contract run PARAM_FILE` |
| `run_nanny.py` | `pyfm nanny run` |
| `submit_job.py` | `pyfm nanny submit -i INPUT -j JOB` |
| `source-system-env.sh` | `eval "$(pyfm workspace env --system NAME)"` |

## Not yet in the CLI

| Script | Purpose |
|--------|---------|
| `aggregate_hadrons_contract_data.py` | Aggregate Hadrons contract output data |
| `compare_hdf5_matrices.py` | Compare a2aMatrix datasets between two HDF5 files |
| `locate_agg_files.py` | Locate aggregated output files |
| `xml_diff.py` | Diff two Hadrons XML input files |
