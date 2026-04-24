#! /bin/bash

print_help() {
  cat <<'EOF'
Usage: ./setup_workspace.sh [OPTIONS]

Options:
  --workspace <dir>    Workspace top directory (env: PYFM_WORKSPACE_TOPDIR)
  --storage   <dir>    Storage top directory   (env: PYFM_STORAGE_TOPDIR)
  --scheduler <name>   Batch scheduler name, e.g. slurm, pbs (env: PYFM_WORKSPACE_SCHEDULER)
  --lattice   <name>   Lattice tag matching example/params_files/params_l<name>.yaml
                       (env: PYFM_WORKSPACE_LATTICE)
  --system    <name>   System name to substitute into batch script (env: PYFM_SYSTEM_NAME)
  --help               Show this help message

Environment variables of the same name take precedence over the flags.
Requires PYFMTOPDIR to be set in the environment.

Example:
  ./setup_workspace.sh --workspace /work/me --storage /scratch/me \
                       --scheduler slurm --lattice 3248 --system perlmutter
EOF
}

_arg_workspace=""
_arg_scheduler=""
_arg_storage=""
_arg_lattice=""
_arg_system=""

while test $# -gt 0; do
  case "$1" in
    --workspace)
      shift
      _arg_workspace="$1"
      shift
      ;;
    --storage)
      shift
      _arg_storage="$1"
      shift
      ;;
    --scheduler)
      shift
      _arg_scheduler="$1"
      shift
      ;;
    --lattice)
      shift
      _arg_lattice="$1"
      shift
      ;;
    --system)
      shift
      _arg_system="$1"
      shift
      ;;
    --help)
      print_help
      exit 0
      ;;
    *)
      echo "Error: Unknown argument: $1" >&2
      echo "" >&2
      print_help >&2
      exit 1
      ;;
  esac
done

PYFM_WORKSPACE_TOPDIR="${PYFM_WORKSPACE_TOPDIR:-${_arg_workspace}}"
PYFM_WORKSPACE_SCHEDULER="${PYFM_WORKSPACE_SCHEDULER:-${_arg_scheduler}}"
PYFM_STORAGE_TOPDIR="${PYFM_STORAGE_TOPDIR:-${_arg_storage}}"
PYFM_WORKSPACE_LATTICE="${PYFM_WORKSPACE_LATTICE:-${_arg_lattice}}"
PYFM_SYSTEM_NAME="${PYFM_SYSTEM_NAME:-${_arg_system}}"

if [ -z "${PYFMTOPDIR}" ] || [ ! -d "${PYFMTOPDIR}" ]; then
  echo "Error: PYFMTOPDIR is not set or not a valid directory (got: '${PYFMTOPDIR}')" >&2
  exit 1
fi

if [ -z "${PYFM_WORKSPACE_TOPDIR}" ] || [ -z "${PYFM_WORKSPACE_SCHEDULER}" ] || \
   [ -z "${PYFM_STORAGE_TOPDIR}" ] || [ -z "${PYFM_WORKSPACE_LATTICE}" ]; then
  echo "Error: --workspace, --scheduler, --storage, and --lattice are all required" >&2
  echo "" >&2
  print_help >&2
  exit 1
fi

PARAMS_DIR="${PYFMTOPDIR}/pyfm/example/params_files"
PARAMS_FILE="${PARAMS_DIR}/params_l${PYFM_WORKSPACE_LATTICE}.yaml"

if [ ! -f "${PARAMS_FILE}" ]; then
  echo "Error: no params file matching lattice '${PYFM_WORKSPACE_LATTICE}' found at ${PARAMS_FILE}" >&2
  exit 1
fi

if [ ! -d "${PYFM_WORKSPACE_TOPDIR}" ]; then
  echo "Error: workspace topdir does not exist: ${PYFM_WORKSPACE_TOPDIR}" >&2
  exit 1
fi

if [ ! -d "${PYFM_STORAGE_TOPDIR}" ]; then
  echo "Error: storage topdir does not exist: ${PYFM_STORAGE_TOPDIR}" >&2
  exit 1
fi

PYFM_ENS=$(python3 -c "
import sys, yaml
with open('${PARAMS_FILE}') as f:
    data = yaml.safe_load(f)
ens = (data.get('shared_params') or {}).get('ens')
if not ens:
    sys.exit('Error: shared_params.ens not found in ${PARAMS_FILE}')
print(ens)
") || { echo "${PYFM_ENS}" >&2; exit 1; }

WORKSPACE_SUBDIR="${PYFM_WORKSPACE_TOPDIR}/l${PYFM_ENS}"
STORAGE_SUBDIR="${PYFM_STORAGE_TOPDIR}/l${PYFM_ENS}"

mkdir -p "${WORKSPACE_SUBDIR}" "${STORAGE_SUBDIR}"

NEW_PARAMS_FILE="${WORKSPACE_SUBDIR}/$(basename "${PARAMS_FILE}")"
cp "${PARAMS_FILE}" "${NEW_PARAMS_FILE}"

sed -i \
  -e "s|PYFM_WORKSPACE_TOPDIR|${PYFM_WORKSPACE_TOPDIR}|g" \
  -e "s|PYFM_WORKSPACE_SCHEDULER|${PYFM_WORKSPACE_SCHEDULER}|g" \
  "${NEW_PARAMS_FILE}"

mkdir -p "${WORKSPACE_SUBDIR}/in" "${WORKSPACE_SUBDIR}/out" "${WORKSPACE_SUBDIR}/schedules"

mkdir -p "${STORAGE_SUBDIR}/eigen" "${STORAGE_SUBDIR}/lat/scidac" "${STORAGE_SUBDIR}/lat/v5"

ln -sfn "${STORAGE_SUBDIR}/eigen" "${WORKSPACE_SUBDIR}/eigen"
ln -sfn "${STORAGE_SUBDIR}/lat" "${WORKSPACE_SUBDIR}/lat"

TODO_FILE="${WORKSPACE_SUBDIR}/todo"
if [ ! -f "${TODO_FILE}" ]; then
  echo "SERIES.CFG JOB_STEP1 0 JOB_STEP2 0 JOB_STEP3 0" > "${TODO_FILE}"
  echo "Created todo file at ${TODO_FILE}"
else
  echo "Note: todo file already exists at ${TODO_FILE} — leaving it untouched"
fi

BATCH_SRC="${PYFMTOPDIR}/pyfm/example/batch_scripts/example.${PYFM_WORKSPACE_SCHEDULER}"
BATCH_COPIED=""
if [ -f "${BATCH_SRC}" ]; then
  BATCH_DST="${WORKSPACE_SUBDIR}/$(basename "${BATCH_SRC}")"
  cp "${BATCH_SRC}" "${BATCH_DST}"
  BATCH_COPIED="${BATCH_DST}"
  echo "Copied batch script to ${BATCH_DST}"
else
  echo "Note: no batch script found for scheduler '${PYFM_WORKSPACE_SCHEDULER}' at ${BATCH_SRC}"
fi

if [ -n "${BATCH_COPIED}" ]; then
  if [ -n "${PYFM_SYSTEM_NAME}" ]; then
    sed -i "s|PYFM_SYSTEM_NAME|${PYFM_SYSTEM_NAME}|g" "${BATCH_COPIED}"
    echo "Substituted PYFM_SYSTEM_NAME=${PYFM_SYSTEM_NAME} in ${BATCH_COPIED}"
  elif [ -t 0 ]; then
    read -p "PYFM_SYSTEM_NAME is not set. Enter a system name to substitute (leave blank to skip): " _sysname
    if [ -n "${_sysname}" ]; then
      sed -i "s|PYFM_SYSTEM_NAME|${_sysname}|g" "${BATCH_COPIED}"
      echo "Substituted PYFM_SYSTEM_NAME=${_sysname} in ${BATCH_COPIED}"
    else
      echo "Warning: PYFM_SYSTEM_NAME was not substituted in ${BATCH_COPIED}. This must be replaced before submitting jobs." >&2
    fi
  else
    echo "Warning: PYFM_SYSTEM_NAME is not set. This must be replaced in ${BATCH_COPIED} before submitting jobs." >&2
  fi
fi

echo "Workspace setup complete: ${WORKSPACE_SUBDIR}"
