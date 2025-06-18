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

${ROCM_PATH}/lib/llvm/bin/clang++ -x hip --cuda-device-only -std=c++17  -emit-llvm  --offload-arch=gfx942 \
 -I${ROCSHMEM_INSTALL_DIR}/include \
 -I${OMPI_DIR}/include \
 -c kernel.cpp \
 -o kernel.bc

llvm-link ${ROCSHMEM_INSTALL_DIR}/lib/rocshmem_gpu.bc ${ROCSHMEM_INSTALL_DIR}/lib/rocshmem_backend_ipc.bc ${ROCSHMEM_INSTALL_DIR}/lib/rocshmem_context_device.bc ${ROCSHMEM_INSTALL_DIR}/lib/rocshmem_context_ipc_device_coll.bc ${ROCSHMEM_INSTALL_DIR}/lib/rocshmem_ipc_policy.bc ${ROCSHMEM_INSTALL_DIR}/lib/rocshmem_team.bc ${ROCSHMEM_INSTALL_DIR}/lib/rocshmem_abql_block_mutex.bc  kernel.bc rocshmem_wrapper.bc -o final_kernel.bc
llvm-dis final_kernel.bc -o final_kernel.ll
