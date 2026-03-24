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
  if [ $BUILD_DEBUG = 'true' ]; then
    ext_flags='--enable-debug'
  fi

  if [ $OLD_RNG = 'true' ]; then
    ext_flags='$ext_flags --enable-old-rng'
  fi
  
  CLIME=`spack find --paths c-lime@2-3-9 | grep c-lime| cut -c 15-`
  ${PYFMTOPDIR}/Grid/configure \
      --enable-comms=mpi-auto \
      --prefix ${INSTALLDIR}      \
      --enable-unified=no \
      --enable-shm=nvlink \
      --enable-tracing=none \
      --enable-accelerator=hip \
      --enable-gen-simd-width=64 \
      --disable-gparity \
      --disable-fermion-reps \
      --enable-simd=GPU \
      --with-gmp=$OLCF_GMP_ROOT \
      --with-hdf5=${HDF5_DIR} \
      --with-lime=${CLIME} \
      --with-mpfr=/opt/cray/pe/gcc/mpfr/3.1.4/ \
      --with-fftw=$FFTW_DIR/.. \
      ${ext_flags} \
      CXX=hipcc MPICXX=mpicxx \
      CXXFLAGS="-fPIC -I${ROCM_PATH}/include/ -I${MPICH_DIR}/include -L/lib64 " \
      LDFLAGS="-L/lib64 -L${ROCM_PATH}/lib -L${MPICH_DIR}/lib -lmpi -L${CRAY_MPICH_ROOTDIR}/gtl/lib -lmpi_gtl_hsa -lhipblas -lrocblas"
}

function hadrons_configure() {
  local INSTALLDIR=$1

  # Configure arguments for Hadrons
  ${PYFMTOPDIR}/Hadrons/configure \
    --prefix=${INSTALLDIR} \
    --with-grid=${PYFMTOPDIR}/Grid/install${PYFM_SYSTEM_EXT} \
    CXXFLAGS="-pthread" \
    LDFLAGS="-lpthread"
}

function quda_configure() {
  local INSTALLDIR=$1

  MPI_CFLAGS="-I${MPICH_DIR}/include -g "
  # Note the flags needed to enable XPMEM support when compiling with hipcc:
  MPI_LDFLAGS="-g -Wl,-rpath=${MPICH_DIR}/lib -L${MPICH_DIR}/lib -lmpi -L${GTL_ROOT} -Wl,-rpath=${GTL_ROOT} -lmpi_gtl_hsa ${CRAY_XPMEM_POST_LINK_OPTS} -lxpmem ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}"

  MY_CFLAGS="$(pat_opts include hipcc gpu) $(pat_opts pre_compile hipcc gpu) ${MPI_CFLAGS} --offload-arch=gfx90a $(pat_opts post_compile hipcc gpu) -g -pg"
  HIPFLAGS="$(pat_opts include hipcc gpu) $(pat_opts pre_compile hipcc gpu) --offload-arch=gfx90a $(pat_opts post_compile hipcc gpu)"
  MY_LDFLAGS="$(pat_opts pre_link hipcc gpu) ${MPI_LDFLAGS} --offload-arch=gfx90a  $(pat_opts post_link hipcc gpu) -g -pg"

  cmake ${PYFMTOPDIR}/quda \
      -G "Ninja" \
      -DQUDA_TARGET_TYPE=HIP \
      -DQUDA_GPU_ARCH=gfx90a \
      -DROCM_PATH=${ROCM_PATH} \
      -DCMAKE_INSTALL_PREFIX=$INSTALLDIR \
      -DCMAKE_BUILD_TYPE=RELEASE \
      -DQUDA_BUILD_SHAREDLIB=ON \
      -DQUDA_DIRAC_DEFAULT_OFF=ON \
      -DQUDA_DIRAC_STAGGERED=ON \
      -DQUDA_QMP=ON \
      -DQUDA_QIO=ON \
      -DQUDA_DOWNLOAD_USQCD=ON \
      -DQUDA_MULTIGRID=OFF \
      -DCMAKE_CXX_COMPILER="hipcc" \
      -DCMAKE_C_COMPILER="hipcc" \
      -DQUDA_BUILD_SHAREDLIB=ON \
      -DQUDA_BUILD_ALL_TESTS=ON \
      -DQUDA_CTEST_DISABLE_BENCHMARKS=ON \
      -DCMAKE_C_STANDARD=99 \
      -DCMAKE_CXX_FLAGS="${MY_CFLAGS}" \
      -DCMAKE_C_FLAGS="${MY_CFLAGS}" \
      -DCMAKE_HIP_FLAGS="${MY_CFLAGS}" \
      -DCMAKE_SHARED_LINKER_FLAGS="${MY_LDFLAGS}" \
      -DCMAKE_EXE_LINKER_FLAGS="${MY_LDFLAGS}"


}

function milc_configure() {
  local INSTALLDIR=$1

  # QUDA install path from the build system
  QUDA_INSTALL=${PYFMTOPDIR}/quda/install${PYFM_SYSTEM_EXT}

  LIBQUDA="-Wl,-rpath ${QUDA_INSTALL}/lib -L${QUDA_INSTALL}/lib -lquda -D__gfx90a --amdgpu-target=gfx90a -Wl,-rpath=${ROCM_PATH}/hiprand/lib -L${ROCM_PATH}/hiprand/lib -Wl,-rpath=${ROCM_PATH}/rocfft/lib -L${ROCM_PATH}/rocfft/lib -lhiprand -lrocfft -Wl,-rpath=${ROCM_PATH}/hipblas/lib -L${ROCM_PATH}/hipblas/lib -lhipblas -Wl,-rpath=${ROCM_PATH}/rocblas/lib -L${ROCM_PATH}/rocblas/lib -lrocblas -Wl,-rpath=${ROCM_PATH}/hip/lib"

  export OFFLOAD=HIP
  export MY_CC=hipcc
  export MY_CXX=hipcc
  export COMPILER="gnu"
  export ARCH=""
  export OPT="-g -ggdb -O3 -Ofast --offload-arch=gfx90a"
  export PATH_TO_NVHPCSDK=""
  export CUDA_HOME=""
  export LDFLAGS=" --verbose -L${MPICH_DIR}/lib -lmpi"
  export QUDA_HOME=${QUDA_INSTALL}
  export WANTQUDA=true
  export WANT_FN_CG_GPU=true
  export WANT_FL_GPU=true
  export WANT_GF_GPU=true
  export WANT_FF_GPU=true
  export WANT_MIXED_PRECISION_GPU=2
  export PRECISION=2
  export WANT_GAUGEFIX_OVR_GPU=true
  export WANT_GSMEAR_GPU=true
  export MPP=true
  export OMP=true
  export WANTQIO=true
  export WANTQMP=true
  export QIOPAR=${QUDA_INSTALL}
  export QMPPAR=${QUDA_INSTALL}
  export LIBQUDA="${LIBQUDA}"
  export CGEOM="-DFIX_NODE_GEOM -DFIX_IONODE_GEOM"
  export KSCGMULTI="-DKS_MULTICG=HYBRID "
  export CTIME="-DNERSC_TIME -DCGTIME -DFFTIME -DFLTIME -DGFTIME -DREMAP -DPRTIME -DIOTIME -DGS_TIME"
}

function dependency_configure() {
  local dep_name=$1
  local INSTALLDIR=$2

  CONFIG=${PYFMTOPDIR}/deps/${dep_name}/configure
  pcc=cc
  pcxx=CC
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
  esac
  $CONFIG \
    --prefix=${INSTALLDIR} \
    ${DEP_CONFIGURE_ARGS} \
    CXXFLAGS=-O3 \
    CFLAGS=-O3 \
    CC=$pcc CXX=$pcxx

}

