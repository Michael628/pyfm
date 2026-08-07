#!/bin/bash
# Steven Gottlieb, September 2, 2021.  Based on Summit script with this history:
# Evan Weinberg, evansweinberg@gmail.com
# Binding script for 6 GPUs per node. Based on a script given to me by Kate, which I believe was based on something from Steve, which may have been based on something originally by Kate...

#dcgmi profile --pause

export lrank=$(($SLURM_LOCALID % 4))
APP=$*

echo "my rank: $lrank"
case ${lrank} in
    [0])
        #numactl --physcpubind=0-71 --membind=0 $APP
        CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 $APP
        # CUDA_VISIBLE_DEVICES=0 $APP
        ;;

    [1])
        #numactl --physcpubind=72-143 --membind=1 $APP
        CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=1 --membind=1 $APP
        # CUDA_VISIBLE_DEVICES=1 $APP
        ;;

    [2])
        #numactl --physcpubind=32-47,96-111 --membind=2 $APP
        CUDA_VISIBLE_DEVICES=2 numactl --cpunodebind=2 --membind=2 $APP
        # CUDA_VISIBLE_DEVICES=2 $APP
        ;;

    [3])
        # numactl --physcpubind=48-63,112-127 --membind=3 $APP
        CUDA_VISIBLE_DEVICES=3 numactl --cpunodebind=3 --membind=3 $APP
        # CUDA_VISIBLE_DEVICES=3 $APP
        ;;
esac
