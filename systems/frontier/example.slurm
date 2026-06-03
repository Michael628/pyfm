#!/bin/bash -l

####### -N (submit script)
####### -n (submit script)
####### -t (submit script)
####### -J (submit script)
#SBATCH -A <account-id>
##SBATCH -q debug
#SBATCH -S 16
#SBATCH --exclusive

# Set run environment
eval $(pyfm workspace env --system PYFM_SYSTEM_NAME)
# source $PYFMTOPDIR/pyfm/scripts/source-system-env.sh --system PYFM_SYSTEM_NAME
if [ $? -ne 0 ]; then
  echo "Error: failed to source system env for 'PYFM_SYSTEM_NAME'" >&2
  exit 1
fi

module list

echo $PATH
# Replace PYFM_EXECUTABLE with one of 
# executable=grid_lma
# executable="pyfm contract run"
# executable=HadronsMILC
# executable=make_links_hisq
executable=PYFM_EXECUTABLE

echo "WORKDIR = $WORKDIR"
cd $WORKDIR

export OPT="--comms-overlap --shm-mpi 0"
runargs=" --grid ${LATTICE} --mpi ${LAYOUT} --accelerator-threads 8 --shm 4096 $OPT"

for input in ${INPUTLIST}
do
  output=out/$(basename "${input%.xml}").${SLURM_JOB_ID}
  echo "Input file ${input}"
  echo "Output file ${output}"
  echo "Input = ${input}" 
  echo "START_RUN `date`" >> ${output}
  echo "RUNARGS = ${runargs}" >> ${output}

  date >> ${output}
  echo "Executable: $(which $executable)" >> ${output}
  argstr="${input} ${runargs}"
  export APP="${executable} ${argstr}"
  echo ${APP} >> ${output}
  cmd="srun -N ${BASENODES} -n ${BASETASKS} --ntasks-per-node=${BASETASKS} --gpus-per-task=1 --cpus-per-task=${OMP_NUM_THREADS} --gpu-bind=closest ${APP}"

  echo ${cmd} >> ${output}
  ${cmd} >> ${output} &
done

wait

popd
exit 0


