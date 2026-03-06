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
  
  ${PYFMTOPDIR}/Grid/configure \
   --prefix=${INSTALLDIR} \
       --enable-simd=SKL \
       --enable-comms=mpi \
       --disable-gparity \
       --host=x86_64-unknown-linux-gnu \
       --with-lime=${PYFMTOPDIR}/deps/install${PYFM_SYSTEM_EXT} \
       CXX="mpicxx" CC="mpicc" \
       CXXFLAGS="-std=c++17 -xcore-avx512 -O2 -simd -mkl -qopenmp "
