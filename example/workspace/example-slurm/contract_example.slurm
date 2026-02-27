#! /bin/bash

####### -N (submit script)
####### -n (submit script)
####### -t (submit script)
####### -J (submit script)
##SBATCH -A m4193_g
##SBATCH -A mp13_g
#SBATCH -A m1647_g
##SBATCH -q regular
##SBATCH -q premium
#SBATCH -q overrun
##SBATCH -q preempt
#SBATCH -C gpu
#SBATCH --gpu-bind=none
#SBATCH --ntasks-per-gpu 1
#SBATCH --exclusive
#SBATCH -V

source ~/.bashrc

module load python

module load cudatoolkit/12.2
module load cray-mpich/8.1.26

conda activate gpu-aware-mpi

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HOME}/deps/install/perlmutter/lib

export CRAY_ACCEL_TARGET=nvidia80

export SLURM_CPU_BIND="cores"
export MPICH_RDMA_ENABLED_CUDA=1
export MPICH_GPU_SUPPORT_ENABLED=1

#export CUDA_VISIBLE_DEVICES=0

echo ${INPUTLIST}

for infile in ${INPUTLIST}
do
    output=out/${infile%.yaml}
    input=in/${infile}

    MPIPARM="-N ${BASENODES} -n ${BASETASKS} -G ${BASETASKS} --cpus-per-task $((128/${PPN})) --gpu-bind=none --ntasks-per-gpu 1"
    #MPIPARM="-N 1 -n 1 -G 1 --cpus-per-task 32 --gpu-bind=none --ntasks-per-gpu 1"
    echo $MPIPARM

    srun -u ${MPIPARM} python ../pyfm/a2a/contract.py $input >> $output &
done

wait
