#!/bin/sh

PYFMTOPDIR=$(pwd)
SRCDIR=${PYFMTOPDIR}/quda
PYFM_SYSTEM_EXT="${PYFM_SYSTEM_EXT}"
BUILDDIR=${SRCDIR}/build${PYFM_SYSTEM_EXT}
INSTALLDIR=${SRCDIR}/install${PYFM_SYSTEM_EXT}

module load gcc-native/12.3

echo "module list:"
module list

echo "LD_LIBRARY_PATH:"
echo $LD_LIBRARY_PATH


if [ ! -d $SRCDIR ]
then
  git clone https://github.com/lattice/quda -b develop
fi
pushd $BUILDDIR
## now explictly adding this variable
export CRAY_ACCEL_TARGET=nvidia90

CUBLAS=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/math_libs/12.3/lib64/libcublas.so
CUFFT=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/math_libs/12.3/lib64/libcufft.so

cmake $SRCDIR -DCMAKE_BUILD_TYPE=RELEASE \
	-DCMAKE_INSTALL_PREFIX=$INSTALLDIR \
	-DQUDA_GPU_ARCH=sm_90 -DQUDA_DIRAC_DEFAULT_OFF=ON -DQUDA_DIRAC_STAGGERED=ON \
	-DQUDA_QMP=ON -DQUDA_QIO=ON \
	-DQUDA_MULTIGRID=OFF \
	-DQUDA_SMEAR_GAUSS_TWOLINK=ON \
	-DQUDA_DOWNLOAD_USQCD=ON -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC \
	-DCUDA_cublas_LIBRARY=$CUBLAS \
	-DCUDA_cufft_LIBRARY=$CUFFT \

make -j 32 >& make_quda.log
make install

QUDA_INSTALL=$INSTALLDIR
SRCDIR={PYFMTOPDIR}/milc_qcd

if [ ! -d $SRCDIR ]
  then
 git clone https://github.com/milc-qcd/milc_qcd.git -b develop
fi
pushd $SRCDIR
############ Make ks_spectrum_hisq ##################
#cd ks_spectrum
cd ks_imp_utilities
cp ../Makefile .
make clean

MY_CC=mpicc \
MY_CXX=mpiCC \
ARCH="" \
GPU_ARCH="nvidia" \
OFFLOAD="CUDA" \
COMPILER="gnu" \
OPT="-O3 -Ofast -g" \
LDFLAGS="-g -L/opt/cray/pe/fftw/3.3.10.8/aarch64/lib" \
QUDA_HOME=${QUDA_INSTALL} \
WANTQUDA=true \
WANT_FN_CG_GPU=true \
WANT_FL_GPU=true \
WANT_GF_GPU=true \
WANT_FF_GPU=true \
WANT_MIXED_PRECISION_GPU=2 \
PRECISION=2 \
WANT_GAUGEFIX_OVR_GPU=true \
WANT_GSMEAR_GPU=true \
MPP=true \
OMP=true \
WANTQIO=true \
WANTQMP=true \
WANTPRIMME=false \
QIOPAR=${QUDA_INSTALL} \
QMPPAR=${QUDA_INSTALL} \
CGEOM="-DFIX_NODE_GEOM -DFIX_IONODE_GEOM" \
KSCGMULTI="-DKS_MULTICG=HYBRID " \
CTIME="-DNERSC_TIME -DCGTIME -DFFTIME -DFLTIME -DGFTIME -DREMAP -DPRTIME -DIOTIME -DGS_TIME" \
make -j 1 make_links_hisq >& make_links_hisq.log
#make -j 1 ks_spectrum_hisq >& make_ks_spectrum_hisq.log

popd

echo ""
echo ""
echo "Check that compilation was successful by viewing make_ks_baryon.log:"
echo ""
#tail milc_qcd/milc_qcd/ks_spectrum/make_ks_spectrum_hisq.log
tail milc_qcd/ks_imp_utilities/make_links_hisq.log

