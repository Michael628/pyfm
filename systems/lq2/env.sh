module purge

module load gompi ucc_cuda fftw hdf5

if [ -d "${PYFMTOPDIR}" ]; then
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PYFMTOPDIR}/deps/install-lq2/lib
  export PATH=${PATH}:${PYFMTOPDIR}/HadronsMILC/install-lq2/bin
  export PYTHONPATH=${PYTHONPATH}:${PYFMTOPDIR}/pyfm/pyfm

  export QUDA_ENABLE_GDR=1
  export UCX_IB_GPU_DIRECT_RDMA=yes
  export UCX_MAX_RNDV_RAILS=1
  export UCX_RNDV_THRESH=1mb

  export OMP_NUM_THREADS=16

fi
