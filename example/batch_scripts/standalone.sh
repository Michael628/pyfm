#! /bin/bash
#
# Environment configuration for example batch scripts.
#
# Source this file from the batch script to populate the environment it
# assumes is already present, e.g.:
#
#     source standalone.sh
#
# Fill in the values below for your run.

export BASENODES=         # number of nodes per input
export BASETASKS=         # total number of tasks (GPUs) per input; PPN = BASETASKS / BASENODES
export LATTICE=           # global lattice geometry, e.g. "48.48.48.96"
export LAYOUT=            # MPI grid layout, e.g. "1.1.1.4"
export INPUTLIST=         # space-separated list of input files to run
export PYFM_EXECUTABLE=        # one of: grid_lma | HadronsMILC | make_links_hisq
export PYFM_BIND_SCRIPT=  # (optional) CPU/GPU binding wrapper script passed to srun
