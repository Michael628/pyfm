#! /bin/bash

# -A m4193_g
# -A mp13_g
#SBATCH -A m1647_g
#SBATCH -C gpu
#SBATCH -q regular
##SBATCH -q premium
##SBATCH -q preempt
#SBATCH --gpu-bind=none
#SBATCH --ntasks-per-gpu 1
#SBATCH --exclusive

echo "INPUTLIST = ${INPUTLIST}"

module purge
module load PrgEnv-gnu/8.3.3 cpe-cuda/23.03
module load cudatoolkit/11.7 craype-accel-nvidia80 craype-x86-milan 
module load cray-hdf5
module load cray-fftw

for infile in ${INPUTLIST}
do
    input=in/${infile}
    output=out/${infile%.txt}

    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HOME}/deps/install/perlmutter/lib

    export CRAY_ACCEL_TARGET=nvidia80

    export SLURM_CPU_BIND="cores"
    export MPICH_RDMA_ENABLED_CUDA=1
    export MPICH_GPU_SUPPORT_ENABLED=1
    export OMP_NUM_THREADS=16

    MPIPARM="-n 64 -N 16 -c 32"
    APP="../bin/make-links-gpu -qmp-geom 2 2 4 4 ${input} ${output}"
    #APP="../bin/make_links_hisq -qmp-geom 2 2 4 4 ${input} ${output}"
    cmd="srun ${MPIPARM} ${APP}"
    echo ${cmd} >> ${output}
    ${cmd}
done

