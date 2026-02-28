#! /bin/bash

####### -N (submit script)
####### -n (submit script)
####### -t (submit script)
####### -J (submit script)
##SBATCH --qos test
#SBATCH --qos opp
##SBATCH --qos normal
#SBATCH -A stgmugm2
##SBATCH -A stagscale
#SBATCH --partition lq2_gpu
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task 1
#SBATCH --exclusive

echo "START_RUN `date`"
echo "ENS          = ${ENS}"
echo "INPUTLIST = ${INPUTLIST}"
echo "BASENODES    = ${BASENODES}"
echo "BASETASKS    = ${BASETASKS}"

PYFMTOPDIR=${HOME}/stgmugm2 # parent directory for pyfm and other code bases


# Set run environment
source $PYFMTOPDIR/pyfm/systems/lq2/env.sh

module list

executable=HadronsMILC


export OPT="--comms-overlap"
runargs=" --grid $LATTICE --mpi ${LAYOUT} --accelerator-threads 8 --shm 2048 --shm-mpi 0 $OPT"

for input in ${INPUTLIST}
do
  output=out/$(basename "${input%.xml}").${SLURM_JOB_ID}
  echo "Input file ${input}"
  echo "Output file ${output}"
  echo "Input = ${input}" 
  echo "START_RUN `date`" >> ${output}
  echo "RUNARGS = ${runargs}" >> ${output}

  date >> ${output}
  echo "Executable: ${executable}" >> ${output}
  argstr="${input} ${runargs}"
  export APP="${executable} ${argstr}"
  echo ${APP} >> ${output}
  MPIPARM="--mpi=pmix -N ${BASENODES} -G ${BASETASKS} -n ${BASETASKS} --cpus-per-task ${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=none"
  cmd="srun ${MPIPARM} ./select_gpu ${APP}"

  echo ${cmd} >> ${output}
  ${cmd} >> ${output} &
  echo ${cmd}
done

wait
exit 0
