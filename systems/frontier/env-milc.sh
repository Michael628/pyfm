module load PrgEnv-amd amd/7.1.1 rocm/7.1.1
module load craype-accel-amd-gfx90a
module load cmake
module load ninja
module list

QUDA_INSTALL=${PYFMTOPDIR}/build/usqcd

MY_LDFLAGS="--verbose -g -Wl,-rpath=${MPICH_DIR}/lib -L${MPICH_DIR}/lib -lmpi"
MY_CFLAGS="-I${MPICH_DIR}/include -g -pg -ggdb -O3 -Ofast --offload-arch=gfx90a"

LIBQUDA="-Wl,-rpath ${QUDA_INSTALL}/lib -L${QUDA_INSTALL}/lib -lquda -D__gfx90a --amdgpu-target=gfx90a -Wl,-rpath=${ROCM_PATH}/hiprand/lib -L${ROCM_PATH}/hiprand/lib -Wl,-rpath=${ROCM_PATH}/rocfft/lib -L${ROCM_PATH}/rocfft/lib -lhiprand -lrocfft -Wl,-rpath=${ROCM_PATH}/hipblas/lib -L${ROCM_PATH}/hipblas/lib -lhipblas -Wl,-rpath=${ROCM_PATH}/rocblas/lib -L${ROCM_PATH}/rocblas/lib -lrocblas -Wl,-rpath=${ROCM_PATH}/hip/lib -g -pg"


if [ "$PYFM_RUNTIME_ENV" = "true" ]; then
# Path to QUDA libraries: Change as needed
export LD_LIBRARY_PATH=`pwd`/build/usqcd/lib:$LD_LIBRARY_PATH

# QUDA Tuning Directory: Change location as needed
export QUDA_RESOURCE_PATH=tunecache
mkdir -p tunecache

# WARNING: QUDA P2P is currently broken on Frontier due to IPC bug(s) in ROCm:
#   - rocm/6.x with QUDA P2P --> silent incorrectness in Dslash halo exchanges
#   - rocm/7.x with QUDA P2P --> hard crash with qudaStreamWaitEvent_ error
# Until those bugs are fixed, QUDA should be run with P2P disabled
export QUDA_ENABLE_P2P=0

export QUDA_ENABLE_GDR=1
export QUDA_MILC_HISQ_RECONSTRUCT=13
export QUDA_MILC_HISQ_RECONSTRUCT_SLOPPY=9
export GPUDIRECT=" -gpudirect "

export MPICH_ENV_DISPLAY=1
export MPICH_GPU_SUPPORT_ENABLED=1

export OMP_NUM_THREADS=6
export OMP_PROC_BIND=spread
MASK_0="0x00fe000000000000"
MASK_1="0xfe00000000000000"
MASK_2="0x0000000000fe0000"
MASK_3="0x00000000fe000000"
MASK_4="0x00000000000000fe"
MASK_5="0x000000000000fe00"
MASK_6="0x000000fe00000000"
MASK_7="0x0000fe0000000000"
MEMBIND="--mem-bind=map_mem:3,3,1,1,0,0,2,2"
CPU_MASK="--cpu-bind=mask_cpu:${MASK_0},${MASK_1},${MASK_2},${MASK_3},${MASK_4},${MASK_5},${MASK_6},${MASK_7}"
fi
