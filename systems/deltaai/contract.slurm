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
##SBATCH --exclusive
#SBATCH -V

echo "START_RUN `date`"
echo "ENS          = ${ENS}"
echo "INPUTLIST = ${INPUTLIST}"
echo "BASENODES    = ${BASENODES}"
echo "BASETASKS    = ${BASETASKS}"

eval $(pyfm workspace env --system deltaai)

module list

executable="pyfm"
runargs="contract run --time-average"

for input in ${INPUTLIST}
do
  output=out/$(basename "${input%.yaml}").${SLURM_JOB_ID}
  echo "Input file ${input}"
  echo "Output file ${output}"
  echo "START_RUN `date`" >> ${output}
  MPIPARM="-N ${BASENODES} -n ${BASETASKS} --cpus-per-task $((64/${PPN}))"
  cmd="srun ${MPIPARM} ${PYFM_BIND_SCRIPT} ${APP}"

  echo ${cmd} >> ${output}
  echo ${cmd}
  ${cmd} >> ${output} &
done

wait
exit 0
