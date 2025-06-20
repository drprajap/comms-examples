#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <stdio.h>

#include <fstream>
#include <iostream>
#include <rocshmem/rocshmem.hpp>
#include <vector>
// #include "rocshmem_wrapper.h"
using namespace rocshmem;

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
__global__ void get_rocshmem_ctx(int *dest, int val, int p) {
  int mype = rocshmem_my_pe();
  int npes = rocshmem_n_pes();

  int peer = (mype + 1) % npes;
  int64_t addr = reinterpret_cast<int64_t>(get_rshmem_ctx());
  int32_t address[2];

  address[0] = static_cast<int32_t>(addr & 0xFFFFFFFF);
  address[1] = static_cast<int32_t>((addr >> 32) & 0xFFFFFFFF);

  rocshmem_int_p(dest, address[0], mype);
  rocshmem_int_p((dest + 1), address[1], mype);
}

__global__ void int_p_kernel(int *destination) {
  int mype = rocshmem_my_pe();
  int npes = rocshmem_n_pes();
  int peer = (mype + 1) % npes;

  rocshmem_int_p(destination, mype, peer);
}

int main(int argc, char **argv) {
  int threadsPerBlock = 256;

  int *msg;
  int rank = rocshmem_my_pe();
  int npes = rocshmem_n_pes();
  int peer = (rank + 1) % npes;
  int ndevices, my_device = 0;
  int N = 4;

  CHECK_HIP(hipGetDeviceCount(&ndevices));
  my_device = rank % ndevices;
  CHECK_HIP(hipSetDevice(my_device));

  std::cout << "\n mype " << rank << " npes: " << rocshmem_n_pes()
            << " peer: " << peer << " ndevices: " << ndevices << "\n";

  rocshmem_init();
  CHECK_HIP(hipDeviceSynchronize());

  int *d_rocshmem_ctx = (int *)rocshmem_malloc(sizeof(void *));
  int64_t *h_rocshmem_ctx;

  rocshmem_barrier_all();

  std::cout << "\n rocshmem_malloc done! dest: " << d_rocshmem_ctx;

  get_rocshmem_ctx<<<dim3(1), dim3(threadsPerBlock), 0, 0>>>(d_rocshmem_ctx, 0,
                                                             rocshmem_my_pe());
  // int_p_kernel<<<dim3(1), dim3(threadsPerBlock), 0, 0>>>(d_rocshmem_ctx);

  CHECK_HIP(hipDeviceSynchronize());
  CHECK_HIP(hipMemcpyAsync(&h_rocshmem_ctx, d_rocshmem_ctx, sizeof(int) * 2,
                           hipMemcpyDeviceToHost, NULL));

  std::cout << "\n mype " << rocshmem_my_pe() << " h_rocshmem_ctx: " << std::hex
            << h_rocshmem_ctx << "\n";

  // Load hsaco binary
  std::ifstream file("kernel.hsaco", std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> binary(size);
  file.read(binary.data(), size);

  // Load module and get function
  hipModule_t module;
  hipFunction_t my_pe_kernel;
  hipFunction_t testing_kernel;
  hipFunction_t get_next_pe_kernel;
  void *sym_addr;
  size_t sym_size;

  CHECK_HIP(hipModuleLoadData(&module, binary.data()));
  rocshmem_hsaco_init(module, (void *)h_rocshmem_ctx);

  CHECK_HIP(
      hipModuleGetFunction(&my_pe_kernel, module, "rocshmem_my_pe_kernel"));
  CHECK_HIP(hipModuleGetFunction(&get_next_pe_kernel, module,
                                 "rocshmem_get_next_pe_kernel"));
  CHECK_HIP(hipModuleGetFunction(&testing_kernel, module, "testing_wrapper"));

  // CHECK_HIP(hipModuleGetGlobal(&sym_addr, &sym_size, module,
  // "ROCSHMEM_CTX_DEFAULT")); CHECK_HIP(hipMemcpy(sym_addr, &h_rocshmem_ctx,
  // sizeof(void *), hipMemcpyHostToDevice)); std::cout << "\n mype " <<
  // rocshmem_my_pe() << " h_rocshmem_ctx: " << h_rocshmem_ctx << "\n";

  CHECK_HIP(hipDeviceSynchronize());

  ///////////// rocshmem_my_pe_kernel
  int *mypebuf = (int *)rocshmem_malloc(sizeof(int) * N);
  rocshmem_barrier_all();

  void *kernel_args[] = {mypebuf};
  size_t arg_size = sizeof(kernel_args);

  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kernel_args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
                    HIP_LAUNCH_PARAM_END};

  CHECK_HIP(hipModuleLaunchKernel(my_pe_kernel, 1, 1, 1,  // grid
                                  1, 1, 1,                // block
                                  0, 0,  // shared memory, stream
                                  nullptr, config));

  CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(hipMemcpyAsync(&msg, mypebuf, sizeof(int) * N,
                           hipMemcpyDeviceToHost, NULL));
  std::cout << "\n mype " << rocshmem_my_pe() << " value: " << msg << "\n";

  /////////// testing_kernel
  int *testbuf = (int *)rocshmem_malloc(sizeof(int) * N);
  msg = nullptr;
  rocshmem_barrier_all();

  void *testkernel_args[] = {testbuf};
  arg_size = sizeof(testkernel_args);

  void *testconfig[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, testkernel_args,
                        HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
                        HIP_LAUNCH_PARAM_END};

  CHECK_HIP(hipModuleLaunchKernel(testing_kernel, 1, 1, 1,  // grid
                                  1, 1, 1,                  // block
                                  0, 0,  // shared memory, stream
                                  nullptr, testconfig));

  CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(hipMemcpyAsync(&msg, testbuf, sizeof(int) * N,
                           hipMemcpyDeviceToHost, NULL));
  std::cout << "\n mype " << rocshmem_my_pe() << " value: " << msg << "\n";

  ///////// get_next_pe_kernel

  int *getnextbuf = (int *)rocshmem_malloc(sizeof(int) * N);
  msg = nullptr;
  rocshmem_barrier_all();

  void *getnextkernel_args[] = {getnextbuf};
  arg_size = sizeof(testkernel_args);

  void *getnextconfig[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, getnextkernel_args,
                           HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
                           HIP_LAUNCH_PARAM_END};

  CHECK_HIP(hipModuleLaunchKernel(get_next_pe_kernel, 1, 1, 1,  // grid
                                  1, 1, 1,                      // block
                                  0, 0,  // shared memory, stream
                                  nullptr, getnextconfig));

  CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(hipMemcpyAsync(&msg, getnextbuf, sizeof(int) * N,
                           hipMemcpyDeviceToHost, NULL));
  std::cout << "\n mype " << rocshmem_my_pe() << " getnextbuf: " << msg << "\n";

  rocshmem_free(testbuf);
  rocshmem_free(getnextbuf);
  rocshmem_free(mypebuf);
  rocshmem_free(d_rocshmem_ctx);
  rocshmem_finalize();
  return 0;
}