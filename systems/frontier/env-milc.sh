module reset
module load PrgEnv-amd amd/5.3.0 rocm/5.3.0
module load craype-accel-amd-gfx90a
module load cray-mpich/8.1.28
module load cmake
module load perftools
module load ninja
module list

# Build-time environment variables (needed for both build and runtime)
export PK_BUILD_TYPE="Release"
export MPICH_ROOT=${CRAY_MPICH_ROOTDIR}
export GTL_ROOT=${MPICH_ROOT}/gtl/lib
export MPICH_DIR=${MPICH_ROOT}/ofi/rocm-compiler/5.0
export PATH=${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${PATH}
export LD_LIBRARY_PATH=${ROCM_PATH}/llvm/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${ROCM_PATH}/llvm/lib:${MPICH_DIR}/lib:${GTL_ROOT}:${LD_LIBRARY_PATH}

# Can't seem to find mpi.h otherwise:
export C_INCLUDE_PATH=${C_INCLUDE_PATH:+${C_INCLUDE_PATH}:}${MPICH_DIR}/include

if [ "$PYFM_RUNTIME_ENV" = "true" ]; then
  export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:$LD_LIBRARY_PATH

  export QUDA_ENABLE_GDR=1
  export QUDA_MILC_HISQ_RECONSTRUCT=13
  export QUDA_MILC_HISQ_RECONSTRUCT_SLOPPY=9

  export MPICH_ENV_DISPLAY=1
  export MPICH_GPU_SUPPORT_ENABLED=1

  export OMP_NUM_THREADS=6

  export SLURM_CPU_BIND="cores"
  export OMP_PROC_BIND="spread, spread, spread"
fi
