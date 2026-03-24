#! /bin/bash

function add_common_install_paths() {
  QUDA_INSTALL=${PYFMTOPDIR}/quda/install${PYFM_SYSTEM_EXT}
  HADRONSMILC_INSTALL=${PYFMTOPDIR}/HadronsMILC/install${PYFM_SYSTEM_EXT}
  GRIDLMA_INSTALL=${PYFMTOPDIR}/grid-lma/install${PYFM_SYSTEM_EXT}
  MAKELINKSHISQ_INSTALL=${PYFMTOPDIR}/milc_qcd/install${PYFM_SYSTEM_EXT}
  DEPSINSTALL=${PYFMTOPDIR}/deps/install${PYFM_SYSTEM_EXT}
  for b in $QUDA_INSTALL $HADRONSMILC_INSTALL $GRIDLMA_INSTALL $MAKELINKSHISQ_INSTALL $DEPSINSTALL
  do 
    if [ -d "${b}/lib" ]; then
      export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${b}/lib
    fi
    if [ -d "${b}/bin" ]; then
      export PATH=${PATH}:${b}/bin
    fi
  done

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

if [ -f "$(pwd)/env${PYFM_SYSTEM_EXT}.sh" ]; then
  echo "Loading: $(pwd)/env${PYFM_SYSTEM_EXT}.sh"
  source "$(pwd)/env${PYFM_SYSTEM_EXT}.sh"
elif [ -f "${PYFMTOPDIR}/env${PYFM_SYSTEM_EXT}.sh" ]; then
  echo "Loading: ${PYFMTOPDIR}/env${PYFM_SYSTEM_EXT}.sh"
  source "${PYFMTOPDIR}/env${PYFM_SYSTEM_EXT}.sh"
elif [ -f "${PYFMTOPDIR}/pyfm/systems/${CONFIG_SYSTEM}/env${_EXT_SUFFIX}.sh" ]; then
  echo "Loading: ${PYFMTOPDIR}/pyfm/systems/${CONFIG_SYSTEM}/env${_EXT_SUFFIX}.sh"
  source "${PYFMTOPDIR}/pyfm/systems/${CONFIG_SYSTEM}/env${_EXT_SUFFIX}.sh"
else
  echo "No env.sh found for system '${CONFIG_SYSTEM}'"
  return 1
fi
