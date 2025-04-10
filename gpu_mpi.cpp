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

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <stdio.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#define HEAP_SIZE 1 << 30

#define HIP_CHECK(condition)                                                  \
  {                                                                           \
    const hipError_t error_code = condition;                                  \
    if (error_code != hipSuccess) {                                           \
      std::cerr << "HIP Error encountered: " << hipGetErrorString(error_code) \
                << " at " << __FILE__ << ": " << __LINE__ << std::endl;       \
      std::runtime_error(hipGetErrorString(error_code));                      \
    }                                                                         \
  }

/// \brief Example for IPC memory sharing via symmetric heap
/// when all GPU devices are on same node (intra-node)
/// See collectiveComms.md for more details on intra-node communication via
/// IPC.
int main(int argc, char **argv) {
  int myRank, numRanks;
  int numDevices, currDevice = 0;

  void *symmetricHeap;
  size_t heapSize = 1024;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);    // Current PE ID
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);  // total number of PEs

  HIP_CHECK(hipGetDeviceCount(&numDevices));
  currDevice = myRank % numDevices;
  HIP_CHECK(hipSetDevice(currDevice));

  printf("CommSize: %d HIP Device count: %d , CurrDevice: %d Rank: %d\n",
         numRanks, numDevices, currDevice, myRank);

  // Allocate Symmetric heap on local device memory, each device will allocate
  // one
  HIP_CHECK(hipMalloc(&symmetricHeap, HEAP_SIZE));

  // Gets IPC memory handle for local device's symmetric heap that can be given
  // to hipIpcOpenMemHandle, That will map remote heap allocation to local
  // device's address space.
  hipIpcMemHandle_t ipcMemHandle;
  HIP_CHECK(hipIpcGetMemHandle(&ipcMemHandle, symmetricHeap));

  // gathers IPC handles of heap allocation from all devices
  hipIpcMemHandle_t *ipcMemHandles =
      (hipIpcMemHandle_t *)malloc(numDevices * sizeof(hipIpcMemHandle_t));
  MPI_Allgather(&ipcMemHandle, sizeof(hipIpcMemHandle_t), MPI_CHAR,
                ipcMemHandles, sizeof(hipIpcMemHandle_t), MPI_CHAR,
                MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  void **heapBases = (void **)malloc(numDevices * sizeof(void *));
  heapBases[currDevice] = symmetricHeap;

  printf("heapBases[%d]: %p \n", currDevice, heapBases[currDevice]);

  // hipIpcOpenMemHandle maps remote/peer heap allocation into
  // local device address space, heap allocations can be accessed
  // via remoteBuf device pointer
  for (int i = 0; i < numDevices; i++) {
    if (i != currDevice) {
      void *remoteBuf = NULL;
      HIP_CHECK(hipIpcOpenMemHandle(&remoteBuf, ipcMemHandles[i],
                                    hipIpcMemLazyEnablePeerAccess));
      heapBases[i] = remoteBuf;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  for (int i = 0; i < numDevices; i++) {
    printf("GPU: %d - heap from GPU %d: %p\n", currDevice, i, heapBases[i]);
  }

  void **d_heapBases;
  HIP_CHECK(hipMalloc(&d_heapBases, numDevices * (sizeof(void *))));
  HIP_CHECK(hipMemcpy(d_heapBases, heapBases, numDevices * sizeof(void *),
                      hipMemcpyHostToDevice));

  for (int i = 0; i < numDevices; i++) {
    if (i != currDevice) {
      HIP_CHECK(hipIpcCloseMemHandle(heapBases[i]));
    }
  }

  HIP_CHECK(hipFree(d_heapBases));
  HIP_CHECK(hipFree(symmetricHeap));

  free(ipcMemHandles);
  free(heapBases);

  MPI_Finalize();

  return 0;
}

void peer2peerWithIpc(int myRank) {
  int numDevices, currDevice = 0;
  void *symmetricHeap = NULL;
  void *remoteBuf = NULL;
  size_t heapSize = 1024;

  HIP_CHECK(hipGetDeviceCount(&numDevices));
  currDevice = myRank % numDevices;
  HIP_CHECK(hipSetDevice(currDevice));

  printf("HIP Device count: %d , CurrDevice: %d Rank: %d\n", numDevices,
         currDevice, myRank);

  HIP_CHECK(hipMalloc(&symmetricHeap, HEAP_SIZE));

  void **heapBases = (void **)malloc(numDevices * sizeof(void *));
  heapBases[currDevice] = symmetricHeap;

  printf("heapBases[%d]: %p \n", currDevice, heapBases[currDevice]);

  hipIpcMemHandle_t ipcMemHandle;

  // Gets IPC memory handle for local device's symmetric heap that can be given
  // to hipIpcOpenMemHandle, that will map remote heap allocation to local
  // device's address space.
  HIP_CHECK(hipIpcGetMemHandle(&ipcMemHandle, symmetricHeap));

  hipIpcMemHandle_t *ipcMemHandles =
      (hipIpcMemHandle_t *)malloc(numDevices * sizeof(hipIpcMemHandle_t));
  MPI_Allgather(&ipcMemHandle, sizeof(hipIpcMemHandle_t), MPI_CHAR,
                ipcMemHandles, sizeof(hipIpcMemHandle_t), MPI_CHAR,
                MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  // hipIpcOpenMemHandle maps remote/peer heap allocation into
  // local device address space, heap allocations can be accessed
  // via remoteBuf device pointer
  int peerDevice = (currDevice + 1) % 2;
  if (currDevice == 0) {
    HIP_CHECK(hipIpcOpenMemHandle(&remoteBuf, ipcMemHandles[peerDevice],
                                  hipIpcMemLazyEnablePeerAccess));
    printf("GPU: %d - heap from GPU 1: %p\n", currDevice, remoteBuf);
  } else if (currDevice == 1) {
    HIP_CHECK(hipIpcOpenMemHandle(&remoteBuf, ipcMemHandles[peerDevice],
                                  hipIpcMemLazyEnablePeerAccess));
    printf("GPU: %d - heap from GPU 0: %p\n", currDevice, remoteBuf);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  char *hostRemoteBuf = (char *)malloc(heapSize);
  HIP_CHECK(
      hipMemcpy(hostRemoteBuf, remoteBuf, heapSize, hipMemcpyDeviceToHost));

  char *hostLocalBuf = (char *)malloc(heapSize);
  HIP_CHECK(
      hipMemcpy(hostLocalBuf, symmetricHeap, heapSize, hipMemcpyDeviceToHost));

  printf("GPU: %d - data from local buf : %d %d %d\n", currDevice,
         hostLocalBuf[0], hostLocalBuf[1], hostLocalBuf[2]);

  printf("GPU: %d - data from Remote buf : %d %d %d\n", currDevice,
         hostRemoteBuf[0], hostRemoteBuf[1], hostRemoteBuf[2]);

  HIP_CHECK(hipIpcCloseMemHandle(remoteBuf));

  HIP_CHECK(hipFree(symmetricHeap));
  free(hostRemoteBuf);
  free(hostLocalBuf);
  free(ipcMemHandles);
  free(heapBases);
}