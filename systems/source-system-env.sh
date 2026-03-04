#! /bin/bash

if [ -z "${PYFMTOPDIR}" ] || [ ! -d "${PYFMTOPDIR}" ]; then
  echo "Error: PYFMTOPDIR is not set or not a valid directory (got: '${PYFMTOPDIR}')"
  return 1
fi

_EXT_SUFFIX=''

while [[ $# -gt 0 ]]; do
  case "$1" in
    --system) CONFIG_SYSTEM="$2"; shift 2 ;;
    --ext)    _EXT_SUFFIX="-$2"; shift 2 ;;
    --help)
      echo "Usage: source systems/source-system-env.sh --system <name> [--ext <ext>]"
      return 0 ;;
    *) echo "Unknown argument: $1"; return 1 ;;
  esac
done

if [ -z "${CONFIG_SYSTEM}" ]; then
  echo "Error: --system <name> is required"
  return 1
fi

PYFM_SYSTEM_EXT="-${CONFIG_SYSTEM}${_EXT_SUFFIX}"

if [ -f "$(pwd)/env${PYFM_SYSTEM_EXT}.sh" ]; then
  source "$(pwd)/env${PYFM_SYSTEM_EXT}.sh"
elif [ -f "${PYFMTOPDIR}/env${PYFM_SYSTEM_EXT}.sh" ]; then
  source "${PYFMTOPDIR}/env${PYFM_SYSTEM_EXT}.sh"
elif [ -f "${PYFMTOPDIR}/pyfm/systems/${CONFIG_SYSTEM}/env.sh" ]; then
  source "${PYFMTOPDIR}/pyfm/systems/${CONFIG_SYSTEM}/env.sh"
else
  echo "No env.sh found for system '${CONFIG_SYSTEM}'"
  return 1
fi
