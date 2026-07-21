#! /bin/bash

ml hdf5/1.14.6
# module load miniforge3
# module reset 
# module use /opt/aurora/24.347.0/spack/unified/0.9.2/install/modulefiles/Core 
# module use /opt/aurora/24.347.0/spack/unified/0.9.2/install/modulefiles/oneapi/2025.0.5 
# module unload mpich 
# module unload oneapi 
# module use /soft/compilers/oneapi/2025.1.0/modulefiles 
# module load oneapi/public/2025.1.0 
# module use /home/bertoni/mpich_module/ 
# module load aurora_test_2025.1 
#
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
git config --global http.proxy http://proxy.alcf.anl.gov:3128
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file"

if [ $PYFM_RUNTIME_ENV = "true" ]; then
  # QUDA Tuning Directory: Change location as needed
  # export QUDA_RESOURCE_PATH=tunecache
  # mkdir -p tunecache

  export QUDA_ENABLE_GDR=1
  export QUDA_MILC_HISQ_RECONSTRUCT=13
  export QUDA_MILC_HISQ_RECONSTRUCT_SLOPPY=9

  # System-specific runtime environment setup
  # export MPIR_CVAR_ENABLE_GPU=1
  # export MPIR_CVAR_DEBUG_SUMMARY=1
  # export MPICH_DBG_LEVEL=VERBOSE
  # export MPICH_DBG_CLASS=ALL
  # export SYCL_UR_TRACE=2 # shows every kernel launch
  # export ONEAPI_DEVICE_SELECTOR=opencl:gpu # sometimes gives more info
  export OMP_NUM_THREADS=8
  export MPICH_OFI_NIC_POLICY=GPU
  export MPICH_CH4_SHM=XPMEM
fi

# Legacy comments from peter boyle
# source ~paboyle/spack/share/spack/setup-env.sh 
# spack load c-lime
# spack load openssl@3.3.1%gcc@7.5.0
# export CLIME=`spack find --paths c-lime | grep ^c-lime | awk '{print $2}' `
# export MPFR=`spack find --paths mpfr    | grep ^mpfr  | awk '{print $2}' `
# export GMP=`spack find --paths gmp   | grep ^gmp  | awk '{print $2}' `
# export LIME=/home/paboyle/GPT/install

