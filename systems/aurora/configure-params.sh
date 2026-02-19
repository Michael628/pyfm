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

  if [ $BUILD_DEBUG = 'true' ]; then
    ext_flags='--enable-debug'
  fi
  # ext_flags="$ext_flags --enable-accelerator-aware-mpi=yes  --enable-reduction=mpi"
  ext_flags="$ext_flags --enable-accelerator-aware-mpi=no  --enable-reduction=grid"

  ${TOPDIR}/Grid/configure \
   --prefix=${INSTALLDIR} \
   --enable-comms=mpi-auto       \
   --enable-simd=GPU \
   ${ext_flags} \
   --enable-shm=nvlink \
   --enable-gen-simd-width=64 \
   --enable-accelerator=sycl   \
   --disable-fermion-reps \
   --enable-old-rng \
   --disable-unified \
   --disable-gparity \
   --with-hdf5=${TOPDIR}/deps/install${BUILD_EXT} \
   CXX=icpx MPICXX=mpicxx \
   LDFLAGS="-fiopenmp -fsycl -fsycl-device-code-split=per_kernel -fsycl-targets=spir64 -Xs -device -Xs pvc \
   -fsycl-device-lib=all -lze_loader -L${MKLROOT}/lib -qmkl=parallel -fsycl -lsycl -lnuma \
   -L/opt/aurora/24.180.3/spack/unified/0.8.0/install/linux-sles15-x86_64/oneapi-2024.07.30.002/numactl-2.0.14-7v6edad/lib \
   -fPIC -fsycl-max-parallel-link-jobs=16 -fno-sycl-rdc" \
   CXXFLAGS="-fiopenmp -fsycl-unnamed-lambda -fsycl -Wno-tautological-compare -qmkl=parallel -fsycl -fno-exceptions \
   -I/opt/aurora/24.180.3/spack/unified/0.8.0/install/linux-sles15-x86_64/oneapi-2024.07.30.002/numactl-2.0.14-7v6edad/include -fPIC"
}
