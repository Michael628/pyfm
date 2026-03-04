#! /bin/bash

# module unload cray-hdf5-parallel
# module load cray-hdf5
module swap cpe cpe/23.03
module load cray-hdf5
module load cray-fftw
module load libfabric/1.20.1

if [ -d "${PYFMTOPDIR}" ]; then
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PYFMTOPDIR}/deps/install-perlmutter/lib
  export PATH=${PATH}:${PYFMTOPDIR}/HadronsMILC/install-perlmutter/bin
  export PATH=${PATH}:${PYFMTOPDIR}/grid-lma/install-perlmutter/bin
  export PATH=${PATH}:${PYFMTOPDIR}/milc_qcd/ks_imp_utilities
  export PYTHONPATH=${PYTHONPATH}:${PYFMTOPDIR}/pyfm

  # QUDA Tuning Directory: Change location as needed
  export QUDA_RESOURCE_PATH=tunecache
  mkdir -p tunecache

  export QUDA_ENABLE_GDR=1
  export QUDA_MILC_HISQ_RECONSTRUCT=13
  export QUDA_MILC_HISQ_RECONSTRUCT_SLOPPY=9

  QUDA_INSTALL=${PYFMTOPDIR}/quda/install-perlmutter
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${QUDA_INSTALL}/lib"

  export MPICH_RDMA_ENABLED_CUDA=1

  export MPICH_ENV_DISPLAY=1
  export MPICH_GPU_SUPPORT_ENABLED=1

  # export OMP_NUM_THREADS=16
  export OMP_NUM_THREADS=32

  export SLURM_CPU_BIND="cores"
  export OMP_PROC_BIND="spread, spread, spread"
  export CRAY_ACCEL_TARGET=nvidia80

fi
