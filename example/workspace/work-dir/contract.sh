#! /bin/bash

source ~/.bashrc

echo ${INPUTLIST}

for infile in ${INPUTLIST}; do
	output=out/${infile%.yaml}
	input=in/${infile}

	cmd="python ../pyfm/a2a/contract.py ${input}"
	echo ${cmd} >>${output}
	${cmd} >>${output} &
done

wait
