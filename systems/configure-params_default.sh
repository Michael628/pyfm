#!/bin/bash
# configure-params.sh - Machine-specific configuration for builds
#
# This script provides functions that set environment variables for configure calls.
# It expects the following BUILD_* variables to be set by build.sh before sourcing:
#   - PYFM_SYSTEM_EXT
#   - BUILD_DEBUG
#   - BUILD_MPI_REDUCTION

function glma_configure() {
  local INSTALLDIR=$1

  # Configure arguments for Hadrons
  ${PYFMTOPDIR}/grid-lma/configure \
    --prefix=${INSTALLDIR} \
    --with-grid=${PYFMTOPDIR}/Grid/install${PYFM_SYSTEM_EXT}
}

function hadrons_configure() {
  local INSTALLDIR=$1

  # Configure arguments for Hadrons
  ${PYFMTOPDIR}/Hadrons/configure \
    --prefix=${INSTALLDIR} \
    --with-grid=${PYFMTOPDIR}/Grid/install${PYFM_SYSTEM_EXT}
}

function hlma_configure() {
  local INSTALLDIR=$1

  # Configure arguments for App
  ${PYFMTOPDIR}/HadronsMILC/configure \
  --prefix=${INSTALLDIR} \
  --with-grid=${PYFMTOPDIR}/Grid/install${PYFM_SYSTEM_EXT} \
  --with-hadrons=${PYFMTOPDIR}/Hadrons/install${PYFM_SYSTEM_EXT}

}

function dependency_configure() {
  local dep_name=$1
  local INSTALLDIR=$2

  CONFIG=${PYFMTOPDIR}/deps/${dep_name}/configure
  pcc=cc
  pcxx=CC
  DEP_CONFIGURE_ARGS=""
  # Dependency-specific additions
  case ${dep_name} in
    mpfr)
      DEP_CONFIGURE_ARGS="--with-gmp=${INSTALLDIR}"
    ;;
    hdf5)
      DEP_CONFIGURE_ARGS="--enable-cxx"
      pcxx=mpicxx
    ;;
    openssl)
      CONFIG=${PYFMTOPDIR}/deps/${dep_name}/config
    ;;
    qmp)
      DEP_CONFIGURE_ARGS="--with-qmp-comms-type=MPI"
    ;;
    qio)
      DEP_CONFIGURE_ARGS="--with-qmp=${INSTALLDIR} --enable-qmp-route"
    ;;
  esac
  $CONFIG \
    --prefix=${INSTALLDIR} \
    ${DEP_CONFIGURE_ARGS} \
    CXXFLAGS=-O3 \
    CFLAGS=-O3 \
    CC=$pcc CXX=$pcxx

}
