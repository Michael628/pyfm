#! /bin/bash

echo "START_RUN $(date)"
echo "ENS          = ${ENS}"
echo "EIGS         = ${EIGS}"
echo "NOISE        = ${NOISE}"
echo "DT           = ${DT}"
echo "INPUTLIST = ${INPUTLIST}"

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HOME}/work/deps/install/scalar/lib

executable=../bin/HadronsMILC

# export OPT="--shm 1024 --device-mem 1024"
runargs=" --grid 4.4.4.4 $OPT"

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
	cmd="${APP}"

	${cmd} >>${output} &
done

wait

popd
exit 0
