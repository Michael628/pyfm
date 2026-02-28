module purge

module load gompi ucc_cuda fftw hdf5

if [ -d "${PYFMTOPDIR}" ]; then
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PYFMTOPDIR}/deps/install-lq2/lib
  export PATH=${PATH}:${PYFMTOPDIR}/HadronsMILC/install-lq2/bin
  export PATH=${PATH}:${PYFMTOPDIR}/milc_qcd/ks_imp_utilities
  export PYTHONPATH=${PYTHONPATH}:${PYFMTOPDIR}/pyfm/pyfm

  # QUDA Tuning Directory: Change location as needed
  export QUDA_RESOURCE_PATH=tunecache
  mkdir -p tunecache

  export QUDA_ENABLE_GDR=1
  export QUDA_MILC_HISQ_RECONSTRUCT=13
  export QUDA_MILC_HISQ_RECONSTRUCT_SLOPPY=9

  QUDA_INSTALL=${PYFMTOPDIR}/quda/install-lq2
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${QUDA_INSTALL}/lib"

  export UCX_IB_GPU_DIRECT_RDMA=yes
  export UCX_MAX_RNDV_RAILS=1
  export UCX_RNDV_THRESH=1mb

  export MPICH_ENV_DISPLAY=1
  export MPICH_GPU_SUPPORT_ENABLED=1

  export OMP_NUM_THREADS=16

  export SLURM_CPU_BIND="cores"
  export OMP_PROC_BIND="spread, spread, spread"
  export CRAY_ACCEL_TARGET=nvidia80

fi
