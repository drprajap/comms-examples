#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <stdio.h>

#include <iostream>
#include <rocshmem/rocshmem.hpp>

#define CHECK_HIP(condition)                                                  \
  {                                                                           \
    const hipError_t error_code = condition;                                  \
    if (error_code != hipSuccess) {                                           \
      std::cerr << "HIP Error encountered: " << hipGetErrorString(error_code) \
                << " at " << __FILE__ << ": " << __LINE__ << std::endl;       \
      std::runtime_error(hipGetErrorString(error_code));                      \
      MPI_Abort(MPI_COMM_WORLD, error_code);                                  \
    }                                                                         \
  }
using namespace rocshmem;

__global__ void add(int *destination) {
  int mype = rocshmem_my_pe();
  int npes = rocshmem_n_pes();
  int peer = (mype + 1) % npes;

  rocshmem_int_p(destination, mype, peer);
}
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
  std::cout << "\n malloc done! \ndest: " << dest;
  int threadsPerBlock = 256;
  // add<<<dim3(1), dim3(threadsPerBlock), 0, 0>>>(dest);
  add<<<dim3(1), dim3(threadsPerBlock), 0, 0>>>(dest);
  CHECK_HIP(hipDeviceSynchronize());
  CHECK_HIP(
      hipMemcpyAsync(&msg, dest, sizeof(int), hipMemcpyDeviceToHost, NULL));

  std::cout << "\n mype " << rocshmem_my_pe() << " value: " << msg << "\n";
  rocshmem_free(dest);
  rocshmem_finalize();

  return 0;
}