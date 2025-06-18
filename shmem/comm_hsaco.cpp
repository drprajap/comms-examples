#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <stdio.h>

#include <fstream>
#include <iostream>
#include <rocshmem/rocshmem.hpp>
#include <vector>
using namespace rocshmem;
#define CHECK_HIP(cmd)                                                   \
  {                                                                      \
    hipError_t err = cmd;                                                \
    if (err != hipSuccess) {                                             \
      std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl; \
      exit(1);                                                           \
    }                                                                    \
  }

int main() {
  int rank = rocshmem_my_pe();
  int ndevices, my_device = 0;
  CHECK_HIP(hipGetDeviceCount(&ndevices));
  my_device = rank % ndevices;
  CHECK_HIP(hipSetDevice(my_device));

  rocshmem_init();

  int *dest = (int *)rocshmem_malloc(sizeof(int));
  int threadsPerBlock = 256;

  // Load hsaco binary
  std::ifstream file("kernel.hsaco", std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> binary(size);
  file.read(binary.data(), size);

  // Load module and get function
  hipModule_t module;
  hipFunction_t kernel_func;
  CHECK_HIP(hipModuleLoadData(&module, binary.data()));
  CHECK_HIP(hipModuleGetFunction(&kernel_func, module, "add_rocshmem_my_pe"));

  int pe = 0;
  void *args[] = {dest};

  // Launch the kernel
  CHECK_HIP(hipModuleLaunchKernel(kernel_func, 1, 1, 1,  // grid
                                  1, 1, 1,               // block
                                  0, 0,  // shared memory, stream
                                  args, nullptr));

  CHECK_HIP(hipDeviceSynchronize());
  // std::cout << "\n mype " << rocshmem_my_pe() << " value: " << msg << "\n";
  rocshmem_free(dest);
  rocshmem_finalize();
}
