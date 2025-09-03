#! /bin/bash

####### -N (submit script)
####### -n (submit script)
####### -t (submit script)
####### -J (submit script)
# -A m4193_g
#SBATCH -A m1647_g
##SBATCH -A mp13_g
#SBATCH -q overrun
##SBATCH -q regular
##SBATCH -q premium
##SBATCH -t 2:00:00
##SBATCH -q preempt
#SBATCH -C gpu
#SBATCH --gpu-bind=none
#SBATCH --ntasks-per-gpu 1
#SBATCH --exclusive
#SBATCH -V

echo "START_RUN $(date)"
echo "ENS          = ${ENS}"
echo "EIGS         = ${EIGS}"
echo "NOISE        = ${NOISE}"
echo "DT           = ${DT}"
echo "INPUTLIST = ${INPUTLIST}"
echo "BASENODES    = ${BASENODES}"
echo "BASETASKS    = ${BASETASKS}"

# Copy job files to gpfs under PDIR
module purge
module load PrgEnv-gnu/8.3.3 cpe-cuda/23.03
module load cudatoolkit/11.7 craype-accel-nvidia80 craype-x86-milan
module load cray-hdf5
module load cray-fftw

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HOME}/deps/install/perlmutter/lib

export CRAY_ACCEL_TARGET=nvidia80

#export PMI_MMAP_SYNC_WAIT_TIME=300
export SLURM_CPU_BIND="cores"
export MPICH_RDMA_ENABLED_CUDA=1
export MPICH_GPU_SUPPORT_ENABLED=1

#executable=../bin/HadronsXmlRun
executable=../bin/HadronsMILC

PPN=$((${BASETASKS} / ${BASENODES}))

export OPT="--comms-concurrent --comms-overlap "
runargs=" --grid 128.128.128.96 --mpi ${LAYOUT} --accelerator-threads 8 --shm 2048 --shm-mpi 1 $OPT"

OFFSET=0
for inXML in ${INPUTLIST}; do
	input=in/${inXML}
	output=out/${inXML%.xml}
	echo "Input file ${input}"
	echo "Output file ${output}"
	echo "Input = ${input}"
	echo "START_RUN $(date)" >>${output}
	echo "RUNARGS = ${runargs}" >>${output}

	date >>${output}
	echo "Executable: ${executable}" >>${output}
	argstr="${input} ${runargs}"
	export APP="${executable} ${argstr}"
	echo ${APP} >>${output}
	MPIPARM="-N ${BASENODES} -n ${BASETASKS} -G ${BASETASKS} --cpus-per-task $((128 / ${PPN})) --gpu-bind=none --ntasks-per-gpu 1"
	cmd="srun ${MPIPARM} ../bin/bind${PPN}-perlmutter.sh ${APP}"

	echo ${cmd} >>${output}
	${cmd} >>${output} &
done

wait

popd
exit 0
