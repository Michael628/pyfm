echo spack
# . /autofs/nccs-svm1_home1/paboyle/Crusher/Grid/spack/share/spack/setup-env.sh
. /autofs/nccs-svm1_home1/paboyle/spack/share/spack/setup-env.sh

module load rocm/6.3.1
module load cray-fftw
module load craype-accel-amd-gfx90a
module load cray-hdf5/1.12.2.11

if [ "$PYFM_RUNTIME_ENV" = "true" ]; then
  # QUDA Tuning Directory: Change location as needed
  # export QUDA_RESOURCE_PATH=tunecache
  # mkdir -p tunecache
  export LD_LIBRARY_PATH=/opt/gcc/mpfr/3.1.4/lib:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CRAY_LD_LIBRARY_PATH}
  export PYFM_GPUMEM_MONITOR_SCRIPT=${PYFMTOPDIR}/pyfm/systems/${PYFM_SYSTEM_EXT#*-}/gpumem.sh

  export QUDA_ENABLE_GDR=1
  export QUDA_MILC_HISQ_RECONSTRUCT=13
  export QUDA_MILC_HISQ_RECONSTRUCT_SLOPPY=9

  # export MPICH_ENV_DISPLAY=1
  export MPICH_GPU_SUPPORT_ENABLED=1

  export OMP_NUM_THREADS=6


  export SLURM_CPU_BIND="cores"
  export OMP_PROC_BIND="spread, spread, spread"

fi
