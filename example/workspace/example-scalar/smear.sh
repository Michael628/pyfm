#! /bin/bash

echo "INPUTLIST = ${INPUTLIST}"

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HOME}/work/deps/install/scalar/lib

for infile in ${INPUTLIST}; do
	input=in/${infile}
	output=out/${infile%.txt}

	APP="../bin/make_links_hisq -qmp-geom 1 1 1 1 ${input} ${output}"
	cmd=$APP
	echo ${cmd} >>${output}
	${cmd}
done
