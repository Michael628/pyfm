#!/bin/bash
# configure-params.sh - Machine-specific configuration for builds
#
# This script provides functions that set environment variables for configure calls.
# It expects the following BUILD_* variables to be set by build.sh before sourcing:
#   - BUILD_EXT
#   - BUILD_DEBUG
#   - BUILD_MPI_REDUCTION

function grid_configure() {
  local INSTALLDIR=$1
  local TOPDIR=$2
  ${TOPDIR}/Grid/configure \
   --enable-debug \
   --enable-simd=GEN \
   --enable-comms=none \
   --enable-unified=no \
   --enable-shm=none \
   --enable-reduction=mpi \
   --with-lime=${TOPDIR}/deps/install${BUILD_EXT} \
   CXXFLAGS='-std=c++17 -Wno-psabi'
}
