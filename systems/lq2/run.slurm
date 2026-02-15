#! /bin/bash

####### -N (submit script)
####### -n (submit script)
####### -t (submit script)
####### -J (submit script)
##SBATCH --qos test
##SBATCH --qos opp
#SBATCH --qos normal
#SBATCH -A stgmugm2
#SBATCH --partition lq2_gpu
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive

##SBATCH --gpus-per-task 1
##SBATCH --gpu-bind=none
##SBATCH --ntasks-per-gpu 1
##SBATCH -V

SRCDIR=/path/to/HadronsMILC/
DEPSDIR=${SRCDIR}../deps/install/lq2
source ${SRCDIR}/systems/lq2/env.sh

#export CRAY_ACCEL_TARGET=nvidia80
#export PMI_MMAP_SYNC_WAIT_TIME=300
#export QUDA_ENABLE_GDR=1
#export UCX_IB_GPU_DIRECT_RDMA=yes
#export UCX_MAX_RNDV_RAILS=1
#export UCX_RNDV_THRESH=1mb


cat << EOF > select_gpu
#!/bin/bash
export GPU_MAP=(0 1 2 3)
export GPU=\${GPU_MAP[\$SLURM_LOCALID]}
export CUDA_VISIBLE_DEVICES=\$GPU
unset ROCR_VISIBLE_DEVICES
echo RANK \$SLURM_LOCALID using GPU \$GPU
exec \$*
EOF
chmod +x ./select_gpu

export UCX_RNDV_THRESH=16kb
export UCX_IB_GPU_DIRECT_RDMA=no
export UCX_MEMTYPE_CACHE=no
export UCX_MAX_RNDV_RAILS=1

export OMP_NUM_THREADS=16
export OMP_PLACES=cores

ppj=${SLURM_NTASKS_PER_NODE}
if [ $PPJ -lt $PPN ]
then
    ppj=${PPJ}
fi

case ${ppj} in
    1|2)
    ;;
    3) cpumask="0x000000000000FFFF,0x00000000FFFF0000,0x0000FFFF00000000"
       ;;
    4) cpumask="0x000000000000FFFF,0x00000000FFFF0000,0x0000FFFF00000000,0xFFFF000000000000"
       ;;
    *) echo "SLURM_NTASKS_PER_NODE not set to one of {1,2,3,4}"
       ;;
esac

executable=../bin/HadronsMILC

#PPN=$((${BASETASKS}/${BASENODES}))

export OPT="--comms-overlap "
runargs=" --grid 48.48.48.64 --mpi ${LAYOUT} --accelerator-threads 8 --shm 2048 --shm-mpi 0 $OPT"

OFFSET=0
for inXML in ${INPUTXMLLIST}
do
    if [ $ppj -lt 3 ]
    then
	case ${ppj} in
	    1)
		case $(( $OFFSET % $PPN)) in
		    0)
			cpumask="0x000000000000FFFF"
			;;
		    1)
			cpumask="0x00000000FFFF0000"
			;;
		    2)
			cpumask="0x0000FFFF00000000"
			;;
		    3)
			cpumask="0xFFFF000000000000"
		       ;;
		esac
		;;
	    2)
		case $(( $OFFSET % $ppj)) in
		    0)
			cpumask="0x000000000000FFFF,0x00000000FFFF0000"
			;;
		    1)
			cpumask="0x0000FFFF00000000,0xFFFF000000000000"
			;;
		esac
		;;
	esac
    fi
		     
    export srunbind="--gres-flags=enforce-binding --cpu-bind=mask_cpu:${cpumask}"
    #export srunbind="--gpu-bind=map_gpu:$((2*$OFFSET)),$((2*$OFFSET+1)) --cpu-bind=mask_cpu:${cpumask}"

  input=in/${inXML}
  output=out/${inXML%.xml}
  echo "Input file ${input}"
  echo "Output file ${output}"
  echo "Input = ${input}" 
  echo "START_RUN `date`" >> ${output}
  echo "RUNARGS = ${runargs}" >> ${output}

  export OFFSET
  date >> ${output}
  echo "Executable: ${executable}" >> ${output}
  argstr="${input} ${runargs}"
  export APP="${executable} ${argstr}"
  echo ${APP} >> ${output}
  MPIPARM="--mpi=pmix -N ${BASENODES} -n ${BASETASKS} -G ${BASETASKS} --cpus-per-task 16 ${srunbind}"
  #MPIPARM="--mpi=pmix --exclusive -n ${BASETASKS} --gpu-bind=closest"
#  cmd="srun ${MPIPARM} ../bin/bind${PPN}-gpu-cuda.sh ${APP}"
  cmd="srun ${MPIPARM} ./select_gpu ${APP}"
  #cmd="srun ${MPIPARM} ${APP}"

  echo ${cmd} >> ${output}
  ${cmd} >> ${output} &

  ((OFFSET+=1))
done

wait

exit 0
