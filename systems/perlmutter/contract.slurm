#! /bin/bash

####### -N (submit script)
####### -n (submit script)
####### -t (submit script)
####### -J (submit script)
#SBATCH -A <account-id>
#SBATCH -q normal
#SBATCH -q regular
##SBATCH -q premium
##SBATCH -q preempt
#SBATCH -C gpu
#SBATCH --gpu-bind=none
#SBATCH --ntasks-per-gpu 1
##SBATCH --exclusive
#SBATCH -V

# Set run environment
eval $(pyfm workspace env --system perlmutter)
if [ $? -ne 0 ]; then
  echo "Error: failed to source system env for 'lq2'" >&2
  exit 1
fi

module list

PPN=$((${BASETASKS}/${BASENODES}))


echo $PATH
executable="pyfm contract run"

runargs="--time-average"

# export FORGE_MPIRUN=/usr/bin/srun

mkdir -p out/

for input in ${INPUTLIST}
do
  output=out/$(basename "${input%.yaml}").${SLURM_JOB_ID}
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
  MPIPARM="-N ${BASENODES} -G ${BASETASKS} -n ${BASETASKS} --cpus-per-task $((128/${PPN}))  --ntasks-per-gpu 1 --gpu-bind=none"
  cmd="srun ${MPIPARM} ${APP}"

  echo ${cmd} >> ${output}
  ${cmd} >> ${output} &
  # ${cmd}
  echo ${cmd}
done

wait
exit 0
