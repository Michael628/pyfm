#! /bin/bash

function add_common_install_paths() {
  QUDA_INSTALL=${PYFMTOPDIR}/quda/install${PYFM_SYSTEM_EXT}
  HADRONSMILC_INSTALL=${PYFMTOPDIR}/HadronsMILC/install${PYFM_SYSTEM_EXT}
  GRID_INSTALL=${PYFMTOPDIR}/Grid/install${PYFM_SYSTEM_EXT}
  GRID_BUILD_TEST=${PYFMTOPDIR}/Grid/build${PYFM_SYSTEM_EXT}/GridMilc/tests/
  GRIDLMA_INSTALL=${PYFMTOPDIR}/grid-lma/install${PYFM_SYSTEM_EXT}
  MAKELINKSHISQ_INSTALL=${PYFMTOPDIR}/milc_qcd/install${PYFM_SYSTEM_EXT}
  DEPSINSTALL=${PYFMTOPDIR}/deps/install${PYFM_SYSTEM_EXT}
  for b in $QUDA_INSTALL $HADRONSMILC_INSTALL $GRID_INSTALL $GRIDLMA_INSTALL $MAKELINKSHISQ_INSTALL $DEPSINSTALL
  do 
    if [ -d "${b}/lib" ]; then
      export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${b}/lib
    fi
    if [ -d "${b}/bin" ]; then
      export PATH=${PATH}:${b}/bin
    fi
  done

  if [ -d  $GRID_BUILD_TEST ]; then
      export PATH=${PATH}:${GRID_BUILD_TEST}
  fi

  export PYTHONPATH=${PYTHONPATH}:${PYFMTOPDIR}/pyfm

  return 
}

if [ -z "${PYFMTOPDIR}" ] || [ ! -d "${PYFMTOPDIR}" ]; then
  echo "Error: PYFMTOPDIR is not set or not a valid directory (got: '${PYFMTOPDIR}')"
  return 1
fi

_EXT_SUFFIX=''
PYFM_RUNTIME_ENV=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --system=*)      CONFIG_SYSTEM="${1#*=}"; shift ;;
    --system)        CONFIG_SYSTEM="$2"; shift 2 ;;
    --ext=*)         _EXT_SUFFIX="-${1#*=}"; shift ;;
    --ext)           _EXT_SUFFIX="-$2"; shift 2 ;;
    --runtime-env=*) PYFM_RUNTIME_ENV="${1#*=}"; shift ;;
    --runtime-env)   PYFM_RUNTIME_ENV="$2"; shift 2 ;;
    --help)
      echo "Usage: source systems/source-system-env.sh --system <name> [--ext <ext>] [--runtime-env true|false]"
      return 0 ;;
    *) echo "Unknown argument: $1"; return 1 ;;
  esac
done

if [ -z "${CONFIG_SYSTEM}" ]; then
  echo "Error: --system <name> is required"
  return 1
fi

PYFM_SYSTEM_EXT="-${CONFIG_SYSTEM}${_EXT_SUFFIX}"

add_common_install_paths

# Resolve the system env script via an ordered search. Candidates are listed
# from LOWEST to HIGHEST precedence; each existing file overwrites
# PYFM_ENV_SCRIPT, so once the loop exits it holds the highest-precedence
# match. This reproduces the original if/else precedence (cwd > topdir >
# system dir) and, within any single directory, lets an env-<ext>.sh variant
# supersede a plain env.sh (e.g. systems/<name>/env-milc.sh over env.sh).
# Each entry is "<kind>|<path>", where <kind> is "ext" or "fallback" so the
# chosen file's kind is tracked alongside its path.
_CANDIDATES=(
  "${PYFMTOPDIR}/pyfm/systems/${CONFIG_SYSTEM}"
  "${PYFMTOPDIR}"
  "$(pwd)"
)

PYFM_ENV_SCRIPT=''
PYFM_BIND_SCRIPT=''
for _path in "${_CANDIDATES[@]}"; do
  # Set env script
  if [ -f "${_path}/env.sh" ]; then
    PYFM_ENV_SCRIPT="${_path}/env.sh"
  fi
  if [ -f "${_path}/env${PYFM_SYSTEM_EXT}.sh" ]; then
    PYFM_ENV_SCRIPT="${_path}/env${PYFM_SYSTEM_EXT}.sh"
  fi

  # Set gpu bind script
  if [ -f "${_path}/bind-gpu.sh" ]; then
    PYFM_BIND_SCRIPT="${_path}/bind-gpu.sh"
  fi
  if [ -f "${_path}/bind-gpu${PYFM_SYSTEM_EXT}.sh" ]; then
    PYFM_BIND_SCRIPT="${_path}/bind-gpu${PYFM_SYSTEM_EXT}.sh"
  fi
done
unset _path

if [ -z "${PYFM_ENV_SCRIPT}" ]; then
  echo "ERROR: No env.sh found for system '${PYFM_SYSTEM_EXT}'"
  return 1
fi
echo "Loading: ${PYFM_ENV_SCRIPT}"
source "${PYFM_ENV_SCRIPT}"

if [ -f "${PYFM_BIND_SCRIPT}" ]; then
  echo "Found bind script: ${PYFM_BIND_SCRIPT}"
  export PYFM_BIND_SCRIPT
else
  echo "WARNING: No bind-gpu.sh found for system '${PYFM_SYSTEM_EXT}'"
fi


unset _CANDIDATES
