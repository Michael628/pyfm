#! /bin/bash

####### -N (submit script)
####### -n (submit script)
####### -t (submit script)
####### -J (submit script)
#SBATCH -A <account-id>
#SBATCH --partition ghx4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem 0
##SBATCH --mem-per-gpu 96G
#SBATCH --gpu-bind=none
#SBATCH -V

echo "START_RUN `date`"
echo "ENS          = ${ENS}"
echo "INPUTLIST = ${INPUTLIST}"
echo "BASENODES    = ${BASENODES}"
echo "BASETASKS    = ${BASETASKS}"

eval $(pyfm workspace env --system deltaai)

module list

# PPN=$((${BASETASKS}/${BASENODES}))
# export SLURM_NTASKS_PER_NODE=$PPN

executable=HadronsMILC

export OPT="--comms-overlap"
runargs=" --grid $LATTICE --mpi ${LAYOUT} --accelerator-threads 32 --shm 2048 --shm-mpi 1 --device-mem 92000 $OPT"

for input in ${INPUTLIST}
do
  output=out/$(basename "${input%.xml}").${SLURM_JOB_ID}
  echo "Input file ${input}"
  echo "Output file ${output}"
  echo "START_RUN `date`" >> ${output}
  echo "RUNARGS = ${runargs}" >> ${output}

  date >> ${output}
  echo "Executable: ${executable}" >> ${output}
  argstr="${input} ${runargs}"
  export APP="${executable} ${argstr}"
  echo ${APP} >> ${output}
  MPIPARM="-N ${BASENODES} -n ${BASETASKS} -G ${BASETASKS} --cpus-per-task ${OMP_NUM_THREADS} --gpus-per-task=1"
  cmd="srun ${MPIPARM} ${PYFM_BIND_SCRIPT} ${APP}"

  echo ${cmd} >> ${output}
  echo ${cmd}
  ${cmd} >> ${output} &
done

wait
exit 0
