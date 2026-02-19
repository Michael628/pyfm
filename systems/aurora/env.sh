#! /bin/bash

module reset 
module use /opt/aurora/24.347.0/spack/unified/0.9.2/install/modulefiles/Core 
module use /opt/aurora/24.347.0/spack/unified/0.9.2/install/modulefiles/oneapi/2025.0.5 
module unload mpich 
module unload oneapi 
module use /soft/compilers/oneapi/2025.1.0/modulefiles 
module load oneapi/public/2025.1.0 
module use /home/bertoni/mpich_module/ 
module load aurora_test_2025.1 

export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
git config --global http.proxy http://proxy.alcf.anl.gov:3128
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file"

if [ -d "${DEPSDIR}" ]; then
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${DEPSDIR}/lib
fi

# source ~paboyle/spack/share/spack/setup-env.sh 
# spack load c-lime
# spack load openssl@3.3.1%gcc@7.5.0
# export CLIME=`spack find --paths c-lime | grep ^c-lime | awk '{print $2}' `
# export MPFR=`spack find --paths mpfr    | grep ^mpfr  | awk '{print $2}' `
# export GMP=`spack find --paths gmp   | grep ^gmp  | awk '{print $2}' `
# export LIME=/home/paboyle/GPT/install

