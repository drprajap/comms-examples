// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/*
Compile:
hipcc -c -fgpu-rdc -x hip rocshmem_test.cc -I${ROCM_PATH}/include  \
-I${ROSHMEM_INSTALL_DIR}/include -I${OMPI_DIR}/include

Link:
hipcc -fgpu-rdc --hip-link rocshmem_test.o -o rocshmem_test         \
${ROSHMEM_INSTALL_DIR}/lib/librocshmem.a  ${OMPI_DIR}/lib/libmpi.so \
-L${ROCM_PATH}/lib -lamdhip64 -lhsa-runtime64

Run using mpirun:
${OMPI_DIR}/bin/mpirun -np 8 --allow-run-as-root ./rocshmem_test
*/

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <stdio.h>

#include <iostream>
#include <rocshmem/rocshmem.hpp>

#define CHECK_HIP(condition)                                        \
  {                                                                 \
    hipError_t error = condition;                                   \
    if (error != hipSuccess) {                                      \
      fprintf(stderr, "HIP error: %d line: %d\n", error, __LINE__); \
      MPI_Abort(MPI_COMM_WORLD, error);                             \
    }                                                               \
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
  int threadsPerBlock = 256;
  add<<<dim3(1), dim3(threadsPerBlock), 0, 0>>>(dest);
  CHECK_HIP(hipDeviceSynchronize());
  CHECK_HIP(
      hipMemcpyAsync(&msg, dest, sizeof(int), hipMemcpyDeviceToHost, NULL));

  std::cout << "\n mype " << rocshmem_my_pe() << " value: " << msg << "\n";
  rocshmem_free(dest);
  rocshmem_finalize();

  return 0;
}