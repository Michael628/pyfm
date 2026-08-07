#!/bin/bash

APP=$*

# Get the GPU ID (from CUDA_VISIBLE_DEVICES, SLURM, or command line)
# if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    # If CUDA_VISIBLE_DEVICES is set, use the first one
    # echo "CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES"
    # GPU_ID=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f1)
if [ -n "$SLURM_LOCALID" ]; then
  echo "SLURM_LOCALID $SLURM_LOCALID"
    GPU_ID=$SLURM_LOCALID
else
    echo "Error: No GPU specified. Usage: $0 <gpu_id>"
    echo "Or set CUDA_VISIBLE_DEVICES or run via SLURM"
    exit 1
fi

echo "Binding to GPU $GPU_ID"

NUMA_NODE=$(nvidia-smi topo -m | grep "^GPU$GPU_ID" | awk '{print $(NF-1)}')

echo "GPU $GPU_ID is on NUMA node $NUMA_NODE"
echo "Executing $APP"

# if [ "${SLURM_PROCID:-0}" -eq 0 ]; then
#   echo "This is rank 0"
#   CUDA_VISIBLE_DEVICES=$GPU_ID cuda-gdb --args $APP
# else 
#   echo "This is not rank 0"
#   CUDA_VISIBLE_DEVICES=$GPU_ID $APP > /tmp/rank0.log 2>&1
# fi
#
# exit 0

CUDA_VISIBLE_DEVICES=$GPU_ID $APP
exit 0

case ${NUMA_NODE} in
    4)
        CUDA_VISIBLE_DEVICES=$GPU_ID numactl --physcpubind=0-71 --membind=0 $APP
        ;;

    12)
        CUDA_VISIBLE_DEVICES=$GPU_ID numactl --physcpubind=72-143 --membind=1 $APP
        ;;

    20)
        CUDA_VISIBLE_DEVICES=$GPU_ID numactl --physcpubind=144-215 --membind=2 $APP
        ;;

    28)
        CUDA_VISIBLE_DEVICES=$GPU_ID numactl --physcpubind=216-287 --membind=3 $APP
        ;;
esac

