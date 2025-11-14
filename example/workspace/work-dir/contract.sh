#! /bin/bash

source ~/.bashrc

echo ${INPUTLIST}

for input in ${INPUTLIST}; do
  output=out/$(basename "${input%.yaml}") 

	cmd="python ../pyfm_scripts/contract_a2a_diagrams.py -p ${input}"
	echo ${cmd} >>${output}
	${cmd} >>${output} &
done

wait
