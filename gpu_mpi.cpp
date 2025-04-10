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

  HIP_CHECK(hipMalloc(&symmetricHeap, /*HEAP_SIZE*(sizeof(char))*/ heapSize));
  HIP_CHECK(hipMemset(symmetricHeap, currDevice, heapSize));

  void **heapBases = (void **)malloc(numDevices * sizeof(void *));
  heapBases[currDevice] = symmetricHeap;

  printf("heapBases[%d]: %p \n", currDevice, heapBases[currDevice]);

  hipIpcMemHandle_t ipc_handle;

  // Gets IPC memory handle for local device's symmetric heap that can be given
  // to hipIpcOpenMemHandle, that maps this heap allocation to another device's
  // address space.
  HIP_CHECK(hipIpcGetMemHandle(&ipc_handle, symmetricHeap));

  hipIpcMemHandle_t *ipcMemHandles =
      (hipIpcMemHandle_t *)malloc(numDevices * sizeof(hipIpcMemHandle_t));
  MPI_Allgather(&ipc_handle, sizeof(hipIpcMemHandle_t), MPI_CHAR, ipcMemHandles,
                sizeof(hipIpcMemHandle_t), MPI_CHAR, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  void *remoteBuf = NULL;
  HIP_CHECK(hipMalloc(&remoteBuf, heapSize));

  // Open IpcHandle associated with symmetric heap allocation on remote/peer
  // device and maps allocation memory to current device address space,
  // current device can use/access the remote buffer by referencing device
  // pointer returned by hipIpcOpenMemHandle.
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

  char *hostBuf = (char *)malloc(heapSize);
  HIP_CHECK(hipMemcpy(hostBuf, remoteBuf, heapSize, hipMemcpyDeviceToHost));

  char *hostLocalBuf = (char *)malloc(heapSize);
  HIP_CHECK(
      hipMemcpy(hostLocalBuf, symmetricHeap, heapSize, hipMemcpyDeviceToHost));

  printf("GPU: %d - data from local buf : %d %d %d\n", currDevice,
         hostLocalBuf[0], hostLocalBuf[1], hostLocalBuf[2]);

  printf("GPU: %d - data from Remote buf : %d %d %d\n", currDevice, hostBuf[0],
         hostBuf[1], hostBuf[2]);

  HIP_CHECK(hipIpcCloseMemHandle(remoteBuf));

  // char *baseHeap = (char *) heapBases[myRank];
  // printf("baseHeap: %x\n", baseHeap);
  // HIP_CHECK(hipIpcGetMemHandle(&ipcMemHandles[myRank], baseHeap));

  // MPI_Barrier(MPI_COMM_WORLD);

  // printf("ipcHandle: %p\n", ipcMemHandles[myRank]);
  // MPI_Comm shmcomm;
  // MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
  //                     &shmcomm);
  // MPI_Allgather(MPI_IN_PLACE, sizeof(hipIpcMemHandle_t), MPI_CHAR,
  // ipcMemHandles, sizeof(hipIpcMemHandle_t),MPI_CHAR, shmcomm);

  // char **ipcBase;
  // HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&ipcBase),
  //             numRanks * sizeof(char **)));

  // for (int i = 0; i < numRanks; i++) {
  //   if (i != myRank) {
  //     void **ipc_base_uncast = reinterpret_cast<void **>(&ipcBase[i]);
  //     HIP_CHECK(hipIpcOpenMemHandle(ipc_base_uncast, ipcMemHandles[i],
  //                                   hipIpcMemLazyEnablePeerAccess));
  //   } else {
  //     ipcBase[i] = (char *)heapBases[myRank];
  //   }
  // }
  // MPI_Barrier(MPI_COMM_WORLD);

  HIP_CHECK(hipFree(symmetricHeap));
  free(hostBuf);
  free(hostLocalBuf);
  free(ipcMemHandles);
  free(heapBases);

  MPI_Finalize();

  return 0;
}
