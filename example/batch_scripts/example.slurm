#! /bin/bash

####### -N (submit script)
####### -n (submit script)
####### -t (submit script)
####### -J (submit script)
#SBATCH -A (ALLOCATION)
#SBATCH -q (QUEUE)
#SBATCH -C gpu
#SBATCH --gpu-bind=none
#SBATCH --ntasks-per-gpu 1
#SBATCH --exclusive
#SBATCH -V

# Set run environment
source ${PYFMTOPDIR}/pyfm/scripts/source-system-env.sh --system PYFM_SYSTEM_NAME
if [ $? -ne 0 ]; then
  echo "Error: failed to source system env for 'PYFM_SYSTEM_NAME'" >&2
  exit 1
fi

module list

PPN=$((${BASETASKS}/${BASENODES}))


echo $PATH
# Replace PYFM_EXECUTABLE with one of 
# executable=grid_lma
# executable=HadronsMILC
# executable=make_links_hisq

# Change device-mem to reflect ~80% of total gpu memory
export OPT="${OPT} --device-mem 18000"
# export OPT="${OPT} --log Performance"
export OPT="${OPT} --comms-overlap"
runargs=" --grid $LATTICE --mpi ${LAYOUT} --accelerator-threads 16 --shm 2048 --shm-mpi 1 $OPT"

for input in ${INPUTLIST}
do
  output=out/$(basename "${input%.xml}").${SLURM_JOB_ID}
  echo "Input file ${input}"
  echo "Output file ${output}"
  echo "Input = ${input}" 
  echo "START_RUN `date`" >> ${output}
  echo "RUNARGS = ${runargs}" >> ${output}

  date >> ${output}
  echo "Executable: $(which ${executable})" >> ${output}
  argstr="${input} ${runargs}"
  export APP="${executable} ${argstr}"
  echo ${APP} >> ${output}
  MPIPARM="-N ${BASENODES} -G ${BASETASKS} -n ${BASETASKS} --cpus-per-task 32 --ntasks-per-gpu 1 --gpu-bind=none"
  cmd="srun ${MPIPARM} $PYFM_BIND_SCRIPT ${APP}"

  echo ${cmd} >> ${output}
  ${cmd} >> ${output} &
  # ${cmd}
  echo ${cmd}
done

wait
exit 0
