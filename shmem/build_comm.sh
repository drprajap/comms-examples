export SCRIPT_DIR=../Triton-distributed/shmem/rocshmem_bind
echo $SCRIPT_DIR
export ROCSHMEM_INSTALL_DIR=${ROCSHMEM_INSTALL_DIR:-${SCRIPT_DIR}/rocshmem_build/install}
export ROCSHMEM_SRC=${ROCSHMEM_SRC:-${SCRIPT_DIR}/../../3rdparty/rocSHMEM}
export ROCM_PATH=/opt/rocm-6.4.0
export OMPI_DIR=${OMPI_DIR:-${SCRIPT_DIR}/ompi_build/install/ompi}
#  --no-gpu-bundle-output --cuda-device-only -O3 \
#  -D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1 -fcolor-diagnostics \
#  -fno-crash-diagnostics -mno-tgsplit \
# ${ROCM_PATH}/llvm/bin/clang-18 -x hip --cuda-device-only -std=c++17  -emit-llvm  --offload-arch=gfx942 \
#  -I${ROCSHMEM_INSTALL_DIR}/include \
#  -I${OMPI_DIR}/include \
#  -c ${SCRIPT_DIR}/comms.cpp \
#  -o comms_rocshmem.bc
# hipcc  --offload-arch=gfx942 -c -fgpu-rdc -x hip comm_rocshmem.cpp -I${ROCM_PATH}/include  \
#  -I${ROCSHMEM_INSTALL_DIR}/include -I${OMPI_DIR}/include -o comm_rocshmem_compile.o

#  hipcc --offload-arch=gfx942 -fgpu-rdc --hip-link comm_rocshmem_compile.o kernel.hsaco  -o comm_rocshmem\
#  ${ROCSHMEM_INSTALL_DIR}/lib/librocshmem.a  ${OMPI_DIR}/lib/libmpi.so \
#  -L${ROCM_PATH}/lib -lamdhip64 -lhsa-runtime64 -v


hipcc  --offload-arch=gfx942 -c -fgpu-rdc -x hip comm_hsaco.cpp -I${ROCM_PATH}/include  \
 -I${ROCSHMEM_INSTALL_DIR}/include -I${OMPI_DIR}/include -o comm_hsaco_compile.o

 hipcc --offload-arch=gfx942 -fgpu-rdc --hip-link comm_hsaco_compile.o -o comm_rocshmem_hsaco \
 ${ROCSHMEM_INSTALL_DIR}/lib/librocshmem.a  ${OMPI_DIR}/lib/libmpi.so \
 -L${ROCM_PATH}/lib -lamdhip64 -lhsa-runtime64
