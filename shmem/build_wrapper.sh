export SCRIPT_DIR=../Triton-distributed/shmem/rocshmem_bind
echo $SCRIPT_DIR
export ROCSHMEM_INSTALL_DIR=${ROCSHMEM_INSTALL_DIR:-${SCRIPT_DIR}/rocshmem_build/install}
export ROCSHMEM_SRC=${ROCSHMEM_SRC:-${SCRIPT_DIR}/../../3rdparty/rocSHMEM}
export ROCM_PATH=/opt/rocm-6.4.0
export OMPI_DIR=${OMPI_DIR:-${SCRIPT_DIR}/ompi_build/install/ompi}
#  --no-gpu-bundle-output --cuda-device-only -O3 \
#  -D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1 -fcolor-diagnostics \
#  -fno-crash-diagnostics -mno-tgsplit \
${ROCM_PATH}/lib/llvm/bin/clang++ -x hip --cuda-device-only -std=c++17  -emit-llvm  --offload-arch=gfx942 \
 -I${ROCSHMEM_INSTALL_DIR}/include \
 -I${OMPI_DIR}/include \
 -c rocshmem_wrapper.cc \
 -o rocshmem_wrapper.bc

# hipcc --genco --offload-arch=gfx942 --std=c++17 -O2 -I${ROCSHMEM_INSTALL_DIR}/include -I${OMPI_DIR}/include  -I${ROCM_PATH}/include -L${ROCM_PATH}/lib -l${ROCSHMEM_INSTALL_DIR}/lib/librocshmem.a -l${OMPI_DIR}/lib/libmpi.so -c rocshmem_wrapper.cc -o rocshmem_wrapper.bc

${ROCM_PATH}/lib/llvm/bin/llvm-dis rocshmem_wrapper.bc -o rocshmem_wrapper.ll

echo "hipcc  -fgpu-rdc --hip-link rocshmem_wrapper.bc ${ROCSHMEM_INSTALL_DIR}/lib/librocshmem.a ${OMPI_DIR}/lib/libmpi.so -L${ROCM_PATH}/lib -lamdhip64 -lhsa-runtime64"