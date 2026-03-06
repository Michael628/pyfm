#! /bin/bash

# module unload cray-hdf5-parallel
# module load cray-hdf5
module swap cpe cpe/23.03
module load cray-hdf5
module load cray-fftw
module load libfabric/1.20.1

if [ "$PYFM_RUNTIME_ENV" = "true" ]; then

  # QUDA Tuning Directory: Change location as needed
  # export QUDA_RESOURCE_PATH=tunecache
  # mkdir -p tunecache

  export QUDA_ENABLE_GDR=1
  export QUDA_MILC_HISQ_RECONSTRUCT=13
  export QUDA_MILC_HISQ_RECONSTRUCT_SLOPPY=9

  export MPICH_RDMA_ENABLED_CUDA=1

  export MPICH_ENV_DISPLAY=1
  export MPICH_GPU_SUPPORT_ENABLED=1

  # export OMP_NUM_THREADS=16
  export OMP_NUM_THREADS=32

  export SLURM_CPU_BIND="cores"
  export OMP_PROC_BIND="spread, spread, spread"
  export CRAY_ACCEL_TARGET=nvidia80

fi
