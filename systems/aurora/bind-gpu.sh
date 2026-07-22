#!/bin/bash

export NUMA_PMAP=(0 0 0 1 1 1 0 0 0 1 1 1 );
export NUMA_HMAP=(2 2 2 3 3 3 3 2 2 2 2 3 3 3 );

# With EnableImplicitScaling=0, Aurora exposes the 12 GPU tiles as flat
# level_zero devices 0-11 (NOT as 6 packages with 2 subdevices). The
# device.subdevice form ZE_AFFINITY_MASK=0.0 matches nothing in this topology
# -> "No platforms found" -> gpu_selector_v throws. Use a flat single index.
export  GPU_MAP=(0 1 2 3 4 5 6 7 8 9 10 11)

export NUMAP=${NUMA_PMAP[$PALS_LOCAL_RANKID]}
export NUMAH=${NUMA_HMAP[$PALS_LOCAL_RANKID]}
export gpu_id=${GPU_MAP[$PALS_LOCAL_RANKID]}

unset EnableWalkerPartition
export EnableImplicitScaling=0

export ZE_AFFINITY_MASK=$gpu_id
# ONEAPI_DEVICE_SELECTOR (set by frameworks/2025.3.1) cannot coexist with
# ZE_AFFINITY_MASK on this Unified-Runtime oneAPI (2025.3.1); clear it and
# restrict the backend via the legacy ONEAPI_DEVICE_FILTER instead.
unset ONEAPI_DEVICE_SELECTOR
export ONEAPI_DEVICE_FILTER=gpu,level_zero

export SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=0
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=0:4
export SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_D2D_COPY=1

echo "rank $PALS_RANKID ; local rank $PALS_LOCAL_RANKID ; ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK ; NUMA $NUMA "

numactl -p $NUMAP -N $NUMAP  "$@"
