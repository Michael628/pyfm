#!/bin/bash

# module load PrgEnv-gnu
# module load PrgEnv-cray
module load craype-accel-nvidia90
module unload gcc-native
module load gcc-native/12
module load cray-mpich/8.1.33
module load cudatoolkit
module load cray-hdf5

# export NVCOMPILER_COMM_LIBS_HOME=$NVHPC_COMM_LIBS_HOME/12.9/hpcx/latest
# export PATH=$NVHPC_COMM_LIBS_HOME/mpi/bin:$PATH
export MPICH_GPU_SUPPORT_ENABLED=1
export GPU_SUPPORT_ENABLED=1
# export CUBLAS_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/math_libs/12.3/targets/sbsa-linux
export CUBLAS_PATH=${NVHPC_COMM_LIBS_HOME}/../math_libs/12.9/targets/sbsa-linux

#export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/math_libs/12.3/targets/sbsa-linux/lib

if [ "$PYFM_RUNTIME_ENV" = "true" ]; then

  # QUDA Tuning Directory: Change location as needed
  # export QUDA_RESOURCE_PATH=tunecache
  # mkdir -p tunecache

  export QUDA_ENABLE_GDR=1
  export QUDA_MILC_HISQ_RECONSTRUCT=13
  export QUDA_MILC_HISQ_RECONSTRUCT_SLOPPY=9

  export CRAY_ACCEL_TARGET=nvidia90
  export MPICH_RDMA_ENABLED_CUDA=1
  export MPICH_NEMESIS_ASYNC_PROGRESS=1
  export SLURM_CPU_BIND="cores"
  export OMP_PROC_BIND="spread, spread, spread"

  export OMP_NUM_THREADS=72

  export MPICH_OFI_NIC_POLICY=GPU
  #export PMI_MMAP_SYNC_WAIT_TIME=300


  export MPICH_ENV_DISPLAY=1
fi
