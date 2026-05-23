#!/bin/bash
# Steven Gottlieb, September 2, 2021.  Based on Summit script with this history:
# Evan Weinberg, evansweinberg@gmail.com
# Binding script for 6 GPUs per node. Based on a script given to me by Kate, which I believe was based on something from Steve, which may have been based on something originally by Kate...

dcgmi profile --pause

export MPICH_OFI_NIC_POLICY GPU
APP=$*

CPU_AFFINITY=$(taskset -pc $$ 2>&1 | awk '{print $6}')

echo "CPU Affinity: $CPU_AFFINITY"

NUMA_NODE=$(nvidia-smi topo -m | grep "$CPU_AFFINITY" | awk '{print $5}')
GPU_ID=$(nvidia-smi topo -m | grep "$CPU_AFFINITY" | awk '{print $1}')
GPU_ID=${GPU_ID:3}

echo "GPU $GPU_ID is on NUMA node $NUMA_NODE"
echo "Executing $APP"

CUDA_VISIBLE_DEVICES=$GPU_ID numactl --physcpubind=${CPU_AFFINITY%,*} --membind=$NUMA_NODE $APP

taskset -pc $$
nvidia-smi topo -m
