#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <stdio.h>

#include <iostream>
#include <rocshmem/rocshmem.hpp>

#include "rocshmem_wrapper.h"

#define CHECK_HIP(condition)                                        \
  {                                                                 \
    hipError_t error = condition;                                   \
    if (error != hipSuccess) {                                      \
      fprintf(stderr, "HIP error: %d line: %d\n", error, __LINE__); \
      MPI_Abort(MPI_COMM_WORLD, error);                             \
    }                                                               \
  }
using namespace rocshmem;

extern "C" __device__ int rocshmem_my_pe_wrapper();

__global__ void add_rocshmem_my_pe(int *destination);

// __global__ void add(int *destination){
//     //int mype = rocshmem_my_pe();
//     int mype = rocshmem_my_pe_wrapper();
//     int npes = rocshmem_n_pes();
//     int peer = (mype + 1) % npes;

//     rocshmem_int_p(destination, mype, peer);
// }
int main(int argc, char **argv) {
  int nelem = 256;
  int msg;
  int rank = rocshmem_my_pe();
  int ndevices, my_device = 0;
  CHECK_HIP(hipGetDeviceCount(&ndevices));
  my_device = rank % ndevices;
  CHECK_HIP(hipSetDevice(my_device));

  rocshmem_init();

  int *dest = (int *)rocshmem_malloc(sizeof(int));
  int threadsPerBlock = 256;
  // add<<<dim3(1), dim3(threadsPerBlock), 0, 0>>>(dest);
  add_rocshmem_my_pe<<<dim3(1), dim3(threadsPerBlock), 0, 0>>>(dest);
  CHECK_HIP(hipDeviceSynchronize());
  CHECK_HIP(
      hipMemcpyAsync(&msg, dest, sizeof(int), hipMemcpyDeviceToHost, NULL));

  std::cout << "\n mype " << rocshmem_my_pe() << " value: " << msg << "\n";
  rocshmem_free(dest);
  rocshmem_finalize();

  return 0;
}