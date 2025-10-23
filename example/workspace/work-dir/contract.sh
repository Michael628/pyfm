#! /bin/bash

source ~/.bashrc

echo ${INPUTLIST}

for infile in ${INPUTLIST}; do
	output=out/${infile%.yaml}
	input=in/${infile}

	cmd="python ../pyfm_scripts/contract_a2a_diagrams.py -p ${input}"
	echo ${cmd} >>${output}
	${cmd} >>${output} &
done

wait
