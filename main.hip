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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// This is a standalone protoyype to demonstrate intra-node (single-node) communication
// 1. initialization Setup Symmetric heap
// 2. rank query
// 3. allocate buffers from heap, get Ipc Handles to pass across process/pes/devices
// 4. write kernel - get/put
// 5. kernel with collective comm operations (allReduce, allGather, allToall, broadcast)

#include <hip/hip_runtime.h>
#include <mpi.h>

#include <cstdlib>
#include <vector>
#include <cstdio>
#include <stdexcept>
#include <iostream>

#define HIP_CHECK(error)          \
    do {                        \
        hipError_t err = error;   \
        if(err != hipSuccess) { \
            std::cerr << "HIP Error: " << hipGetErrorString(err)    \
                      << " at " << __FILE__ << ": "                 \
                      << __LINE__ << std::endl;                      \
            throw std::runtime_error(hipGetErrorString(err));       \
        }   \
        } while(0)


// Kernel
__global__ void add(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// all-to-all kernel
__device__ void allToall() {
    
}

// kernel for issuing memcpy -
// iris - uses mpi4py setup heap, Ex /rocSHMEM/examples/rocshmem_getmem_test.cc -  translate address as per current device rank, 
// getting base and offset specific to device and 
// calculate src / dst address, and then issue tl.load
int main(int argc, char *argv[]){
    try{
        int N = 1024 * 1024;
        size_t sizeInBytes = N * sizeof(float);
        const constexpr unsigned int iterations = 10;

        std::vector<float> h_A(N);
        std::vector<float> h_B(N);
        std::vector<float> h_C(N);
        std::vector<float> ref_C(N);

        for(int i=0; i< N; i++){
            h_A[i] = static_cast<float>(i);
            h_B[i] = static_cast<float>(i*2);
        }

        float *d_A = nullptr;
        float *d_B = nullptr;
        float *d_C = nullptr;

        HIP_CHECK(hipMalloc(&d_A, sizeInBytes));
        HIP_CHECK(hipMalloc(&d_B, sizeInBytes));
        HIP_CHECK(hipMalloc(&d_C, sizeInBytes));

        HIP_CHECK(hipMemcpy(d_A, h_A.data(), sizeInBytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, h_B.data(), sizeInBytes, hipMemcpyHostToDevice));

        int numDevices = 0;
        HIP_CHECK(hipGetDeviceCount(&numDevices));      

        // Cumulative variable to compute the mean time per iteration of the algorithm.
        double kernel_time = 0;

        // Create events to measure the execution time of the kernels.
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        int threadsPerBlock = 256;
        
        const dim3 block_dim(threadsPerBlock); // number of threads in each kernel block
        const dim3 grid_dim((N + threadsPerBlock - 1) / threadsPerBlock); // number of blocks in grid

        std::cout << "Launching Kernel.." << std::endl;

        for(unsigned int i = 0; i < iterations; ++i)
        {
            float kernel_ms{};

            // Record the start event.
            HIP_CHECK(hipEventRecord(start, hipStreamDefault));

            add<<<grid_dim, block_dim, 0,  hipStreamDefault>>>(d_A, d_B, d_C, N);

            // Check if the kernel launch was successful.
            HIP_CHECK(hipGetLastError());

            // Record the stop event and wait until the kernel execution finishes.
            HIP_CHECK(hipEventRecord(stop, hipStreamDefault));
            HIP_CHECK(hipEventSynchronize(stop));

            // Get the execution time of the kernel and add it to the total count.
            HIP_CHECK(hipEventElapsedTime(&kernel_ms, start, stop));
            kernel_time += kernel_ms;
            std::cout << "Kernel " << i << ": execution time = " << kernel_ms
                      << "ms" << std::endl;

        }

        // Destroy hipEvents.
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        std::cout << "Finished Kernel execution!"<< std::endl;

        HIP_CHECK(hipMemcpy(h_C.data(), d_C, sizeInBytes, hipMemcpyDeviceToHost));

        std::cout << "Copied result from Device to Host!" << std::endl;

        // Print the mean time per iteration (in miliseconds) of the algorithm.
        kernel_time /= iterations;
        std::cout << "The mean time needed for each iteration has been " << kernel_time
              << "ms " << std::endl;

        HIP_CHECK(hipFree(d_A));
        HIP_CHECK(hipFree(d_B));
        HIP_CHECK(hipFree(d_C));

        bool error = false;
        // Calculate result for Reference on Host and compare the result
        for(int i=0; i< N; i++){
            ref_C[i] = h_A[i] + h_B[i];

            double diff = h_C[i] - ref_C[i];
            if(diff != 0){
                std::cout << "result mismatch at index: " << i
                        << "ref_c: " << ref_C[i] 
                        << "h_C: " << h_C[i] << std::endl;
                error = true;
            }
        }

        if(error){
            std::cout << "Verification Failed!" << std::endl;
        } else {
            std::cout << "Verification Passed!" << std::endl;
        }
        
    } catch (const std::exception e) {
        std::cerr << "exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}