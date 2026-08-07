#! /bin/bash

####### -N (submit script)
####### -n (submit script)
####### -t (submit script)
####### -J (submit script)
#SBATCH -A <account-id>
#SBATCH -q normal
#SBATCH -C gpu
#SBATCH --gpu-bind=none
#SBATCH --ntasks-per-gpu 1
##SBATCH --exclusive
#SBATCH -V

# Set run environment
eval $(pyfm workspace env --system lq2 )
if [ $? -ne 0 ]; then
  echo "Error: failed to source system env for 'lq2'" >&2
  exit 1
fi

module list

PPN=$((${BASETASKS}/${BASENODES}))


echo $PATH
executable="pyfm contract run"

# Change device-mem to reflect ~80% of total gpu memory
runargs="--time-average"

for input in ${INPUTLIST}
do
  output=out/$(basename "${input%.yaml}").${SLURM_JOB_ID}
  echo "Input file ${input}"
  echo "Output file ${output}"
  echo "Input = ${input}" 
  echo "START_RUN `date`" >> ${output}
  echo "RUNARGS = ${runargs}" >> ${output}

  date >> ${output}
  argstr="${input} ${runargs}"
  export APP="${executable} ${argstr}"
  echo ${APP} >> ${output}
  MPIPARM="-N ${BASENODES} -G ${BASETASKS} -n ${BASETASKS} --cpus-per-task 32 --ntasks-per-gpu 1 --gpu-bind=none"
  cmd="srun ${MPIPARM} ${APP}"

  echo ${cmd} >> ${output}
  ${cmd} >> ${output} &
  # ${cmd}
  echo ${cmd}
done

wait
exit 0
