#!/bin/bash
# configure-params.sh - Machine-specific configuration for builds
#
# This script provides functions that set environment variables for configure calls.
# It expects the following BUILD_* variables to be set by build.sh before sourcing:
#   - PYFM_SYSTEM_EXT
#   - BUILD_DEBUG
#   - BUILD_MPI_REDUCTION

function grid_configure() {
  local INSTALLDIR=$1
  
  if [ $OLD_RNG = 'true' ]; then
    ext_flags='--enable-old-rng'
  fi

  ${PYFMTOPDIR}/Grid/configure \
    --prefix ${INSTALLDIR}      \
    --enable-comms=mpi       \
    --enable-simd=GPU \
    --enable-shm=nvlink \
    --enable-gen-simd-width=64 \
    --enable-accelerator=cuda \
    --disable-fermion-reps \
    ${ext_flags} \
    --disable-unified \
    --disable-gparity \
    --with-lime=${PYFMTOPDIR}/deps/install${PYFM_SYSTEM_EXT} \
    CXX="nvcc" \
    CXXFLAGS="-ccbin CC -gencode arch=compute_90,code=sm_90 -std=c++17 -I${CUBLAS_PATH}/include -DEIGEN_DONT_VECTORIZE" \
    LIBS="-lcublas" \
    LDFLAGS="-L${CUBLAS_PATH}/lib"
}

function hadrons_configure() {
  local INSTALLDIR=$1
  

  unset CXX #Should be grabbed from grid-config

  # Configure arguments for Hadrons
  ${PYFMTOPDIR}/Hadrons/configure \
    --prefix=${INSTALLDIR} \
    --with-grid=${PYFMTOPDIR}/Grid/install${PYFM_SYSTEM_EXT}
}

function hmilc_configure() {
  local INSTALLDIR=$1
  

  unset CXX #Should be grabbed from grid-config
  
  # Configure arguments for App
  ${PYFMTOPDIR}/HadronsMILC/configure \
  --prefix=${INSTALLDIR} \
  --with-grid=${PYFMTOPDIR}/Grid/install${PYFM_SYSTEM_EXT} \
  --with-hadrons=${PYFMTOPDIR}/Hadrons/install${PYFM_SYSTEM_EXT}

