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

/// Maintains internal metadata around symmetric heap
/// Used for tracking memory allocations from heap, keeps
/// track of allocated and free blocks.
typedef struct SymmetricHeap {
  uint8_t *base;
  size_t offset;
  size_t size;
};
class Shmem {
 public:
  hipIpcMemHandle_t ipcMemHandle;
  hipIpcMemHandle_t *ipcMemHandles;
  SymmetricHeap heap;

  int myRank;
  int numDevices, currDevice;
  void **heapBases;
  void **d_heapBases;
  void *symmetricheap;

  Shmem(int device, int rank, int numdevices)
      : currDevice(device), myRank(rank), numDevices(numdevices) {
    // Allocate Symmetric heap on local device memory,
    // each device will allocate their own heap.
    HIP_CHECK(hipMalloc(&symmetricheap, HEAP_SIZE));

    heap.base = (uint8_t *)symmetricheap;
    heap.offset = 0;
    heap.size = HEAP_SIZE;

    // Gets IPC memory handle for local device's symmetric heap that can be
    // given to hipIpcOpenMemHandle, That will map remote heap allocation to
    // local device's address space.
    HIP_CHECK(hipIpcGetMemHandle(&ipcMemHandle, symmetricheap));

    // gathers IPC handles of heap allocation from all devices
    ipcMemHandles =
        (hipIpcMemHandle_t *)malloc(numDevices * sizeof(hipIpcMemHandle_t));
    MPI_Allgather(&ipcMemHandle, sizeof(hipIpcMemHandle_t), MPI_CHAR,
                  ipcMemHandles, sizeof(hipIpcMemHandle_t), MPI_CHAR,
                  MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    heapBases = (void **)malloc(numDevices * sizeof(void *));
    heapBases[currDevice] = symmetricheap;

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

    HIP_CHECK(hipMalloc(&d_heapBases, numDevices * (sizeof(void *))));
    HIP_CHECK(hipMemcpy(d_heapBases, heapBases, numDevices * sizeof(void *),
                        hipMemcpyHostToDevice));
  }

  void *shmem_malloc(size_t size) {
    if (size == 0) return NULL;

    if (heap.offset + size > heap.size) {
      return NULL;
    }

    void *ptr = (char *)heap.base + heap.offset;
    heap.offset += size;
    return ptr;
  }

  ~Shmem() {
    for (int i = 0; i < numDevices; i++) {
      if (i != currDevice) {
        HIP_CHECK(hipIpcCloseMemHandle(heapBases[i]));
      }
    }
    HIP_CHECK(hipFree(d_heapBases));
    HIP_CHECK(hipFree(symmetricheap));

    free(ipcMemHandles);
    free(heapBases);

    MPI_Finalize();
  }

  void shmem_free(void *ptr) {
    // Can maintain free list or advanced technique for properly reclaiming
    // space in heap currently do nothing for basic version, if we run out of
    // space, error out for now
  }
};

__device__ __forceinline__ void memcpy(void *dst, void *src, size_t size) {
  uint8_t *dst_bytes{static_cast<uint8_t *>(dst)};
  uint8_t *src_bytes{static_cast<uint8_t *>(src)};

  for (size_t i = 8; i > 1; i >>= 1) {
    while (size >= i) {
      store_asm(src_bytes, dst_bytes, i);
      src_bytes += i;
      dst_bytes += i;
      size -= i;
    }
  }

  if (size == 1) {
    *dst_bytes = *src_bytes;
  }
}

__global__ void get(void *src, int currDevice, int peerDevice, size_t nelems,
                    void **heapBases) {
  unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;

  char *srcHeapBase = heapBases[currDevice];
  char *dstHeapBase = heapBases[peerDevice];

  char *srcCast = reinterpret_cast<char *> src;
  srsrcCast += threadId;

  uint64_t offset = (char *)srcCast - srcHeapBase;

  char *dst = dstHeapBase + offset;
  memcpy(dst, srsrcCast, nelems);
}

__global__ void put(void *src, int currDevice, int peerDevice,
                    void **heapBases) {
  char *srcHeapBase = heapBases[currDevice];
  char *dstHeapBase = heapBases[peerDevice];

  size_t offset = (size_t)src - srcHeapBase;

  char *dst = dstHeapBase + offset;

  dst = src;
}

/// \brief Example for IPC memory sharing via symmetric heap
/// when all GPU devices are on same node (intra-node)
/// See collectiveComms.md for more details on intra-node communication via
/// IPC.
int main(int argc, char **argv) {
  int myRank, numRanks;
  int numDevices, currDevice = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);    // Current PE ID
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);  // total number of PEs

  HIP_CHECK(hipGetDeviceCount(&numDevices));
  currDevice = myRank % numDevices;
  HIP_CHECK(hipSetDevice(currDevice));

  printf("CommSize: %d HIP Device count: %d , CurrDevice: %d Rank: %d\n",
         numRanks, numDevices, currDevice, myRank);

  Shmem shmem_ctx(currDevice, myRank, numDevices);

  for (int i = 0; i < numDevices; i++) {
    printf("GPU: %d - heap from GPU %d: %p\n", currDevice, i,
           shmem_ctx.heapBases[i]);
  }
  int peerDevice = (currDevice + 1) % 2;

  int threadsPerBlock = 256;

  // number of threads in each kernel block
  const dim3 block_dim(threadsPerBlock);
  // number of blocks in grid
  // const dim3 grid_dim((N + threadsPerBlock - 1) /
  //                     threadsPerBlock);

  // GPU 0, Puts data into buffer Get buffer from heap
  // GPU 1, gets data from remote buffer
  if (currDevice == 0) {
    // malloc dword
    void *buf0 = shmem_ctx.shmem_malloc(4);
    put<<<dim3(1), block_dim, 0, hipStreamDefault>>>(
        buf0, currDevice, peerDevice, shmem_ctx.d_heapBases);
  }
  // else if (currDevice == 1) {
  //   void *buf1;

  //   std::cout << "Launching Kernel.." << std::endl;
  //   get<<<grid_dim, block_dim, 0, hipStreamDefault>>>(
  //       buf1, currDevice, peerDevice, shmem_ctx.d_heapBases);
  // }

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