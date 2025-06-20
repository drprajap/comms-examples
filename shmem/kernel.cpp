#include <hip/hip_runtime.h>

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;
extern "C" __device__ int rocshmem_my_pe_wrapper();

extern "C" __device__ int rocshmem_n_pes_wrapper();

extern "C" __device__ void *rocshmem_ptr_wrapper(void *dest, int pe);

extern "C" __device__ void rocshmem_int_p_wrapper(int *dest, int value, int pe);

extern "C" __global__ void __attribute__((used)) rocshmem_my_pe_kernel(
    int *pe) {
  *pe = rocshmem_my_pe() + 3;
}
extern "C" __global__ void __attribute__((used)) rocshmem_n_pes_kernel(
    int *num_pes) {
  *num_pes = rocshmem_n_pes_wrapper();
}

extern "C" __global__ void __attribute__((used)) rocshmem_ptr_kernel(void *src,
                                                                     void *dest,
                                                                     int pe) {
  dest = rocshmem_ptr_wrapper(src, pe);
}

extern "C" __global__ void __attribute__((used)) rocshmem_int_p_kernel(
    int *dest, int value, int pe) {
  rocshmem_int_p_wrapper(dest, value, pe);
}

extern "C" __global__ void __attribute__((used)) rocshmem_get_next_pe_kernel(
    int *dest) {
  int mype = rocshmem_my_pe_wrapper();
  int npes = rocshmem_n_pes_wrapper();
  int peer = (mype + 1) % npes;

  rocshmem_int_p_wrapper(dest, mype, peer);
}

extern "C" __global__ void testing_wrapper(int *sym_buf) {
  // int mype = rocshmem_my_pe_wrapper();
  int tid = threadIdx.x;
  if (tid < 4) {
    sym_buf[tid] = tid * 4 + 100;
    printf("device ptr: %p tid: %d sym_buf[%d]: %d ", sym_buf, tid, tid,
           sym_buf[tid]);
  } else {
    sym_buf[tid] = 501;
    printf("device ptr: %p tid: %d sym_buf[%d]: %d ", sym_buf, tid, tid,
           sym_buf[tid]);
  }

  // rocshmem_my_pe();
}
