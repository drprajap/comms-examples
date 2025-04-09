# Implementation for GPU-initiated communication

- Allocate Symmetric heap/memory that is distributed across multiple devices. This memory locations from this heap can be accessed by kernel threads via get(read), put(write) and atomic update. Memory is accessible to local GPU or peer GPU via pointer, which resides in symmetric heap and under the hood, driver maps the allocations to this pointer.
- When CPU is used for communication, that limits the ability to launch single fused kernel as communication needs to interleave between different kernel launches which requires CPU involvement and synchronization, so computation needs to be broken down in multiple kernel launches with additional synchronization. 
- GPU<->GPU communication via kernels, no host involvement. This one-sided communication can be translated to load/store across multiple GPUs Intra-node or RDMA primitives provided by network hardware for inter-node communication. This also enhances the capability launch single fused kernel as computation and computation can now interleave together.


## Setup and Initialization
- MPI Initialization
- Setup symmetric heap (~1GB) - via hipMalloc(), allocated in GPU memory, this is run by each PE.
- Get IPC handle for the allocated heap via hipIpcGetMemHandle(), each PE sends this IPC handle to all PEs via MPI_Allgather() and all PEs receive handle from each PE. Now each PE knows heap base from other PE's address space. 
- Each PE calls hipIpcOpenMemHandle() to get heap base buffer pointer of all PEs(ranks) which points to GPU memory of device specific heap allocation, using this pointer (this is mapping of cross-device heap bases into current device's address space). This is essential to share/access buffers/data allocated from symmetric heap by any device, without this device pointer, GPU kernels cannot reference buffers and GPU HW may not resolve the page table mapping to correct memory.
- Now symmetric heap is setup and each PE/rank has same view of this heap and devices can allocate buffers from this heap.

## Buffer Allocation
 Buffers are allocated from symmetric heap. Buffer sizes from all PEs/ranks need to have same allocation size. heap offset needs to be maintained when allocation requests are made from heap. This is analogus to giving sub allocations from memory pool (symmetric heap). Buffer Address being device pointer, it can be used directly by load/store operation. 

## Kernels - Get/put
 For given source pointer and rank/PE_ID, we can calculate heap bases for source and destination rank. Pointer arithmeric can be performed to get rank specific allocation address. and because buffer sizes across PEs are same, offset obtaines from source pointer into the heap can be used to reference destination pointer for get/put operations. Underlying driver memory mapping should take care of correct translation to device specific GPU memory.
 