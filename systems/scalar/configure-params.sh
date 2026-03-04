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
  local PYFMTOPDIR=$2
  ${PYFMTOPDIR}/Grid/configure \
   --enable-debug \
   --enable-simd=GEN \
   --enable-comms=none \
   --enable-unified=no \
   --enable-shm=none \
   --enable-reduction=mpi \
   --with-lime=${PYFMTOPDIR}/deps/install${PYFM_SYSTEM_EXT} \
   CXXFLAGS='-std=c++17 -Wno-psabi'
}
