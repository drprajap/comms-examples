#include <hip/hip_runtime.h>

extern "C" __device__ int rocshmem_my_pe_wrapper();

extern "C" __global__ void __attribute__((used)) add_rocshmem_my_pe(
    int *destination) {
  // int mype = rocshmem_my_pe();
  int mype = rocshmem_my_pe_wrapper();
}
