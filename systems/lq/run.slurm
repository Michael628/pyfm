#! /bin/bash

####### -N (submit script)
####### -n (submit script)
####### -t (submit script)
####### -J (submit script)
#SBATCH -A stgmugm2
##SBATCH --cpus-per-task 20
#SBATCH -q normal

PPN=$((${BASETASKS}/${BASENODES}))
cpus=$((40/$PPN))

export OMP_NUM_THREADS=$cpus
export OMP_PLACES=threads
export THREADS_PER_CORE=1

cpumask="0x000000000000FFFF,0x00000000FFFF0000,0x0000FFFF00000000,0xFFFF000000000000"

executable=../bin/HadronsXmlRun
#executable=../bin/HadronsMILC-avx512-skx


export OPT="--comms-concurrent --shm-mpi 1"
runargs=" --grid 48.48.48.64 --mpi ${LAYOUT} --shm 2048 $OPT"

OFFSET=0
for inXML in ${INPUTXMLLIST}
do
  input=in/${inXML}
  output=out/${inXML%.xml}
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
#  MPIPARM="--mpi=pmix -N ${BASENODES} -n ${BASETASKS} --cpus-per-task 20"
  MPIPARM="--mpi=pmi2 -N ${BASENODES} -n ${BASETASKS} --cpus-per-task ${cpus}"
  cmd="srun ${MPIPARM} ${APP}"

  echo ${cmd} >> ${output}
  ${cmd} >> ${output} &
done

wait

exit 0
