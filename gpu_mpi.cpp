#include <stdio.h>
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


// Example: IPC memory sharing when all pes are on same node - only works intra-node
// hipMalloc() to allocate device buffers, 
// hipIpcGetMemHandle/hipIpcOpenMemHandle to get IPC handle for device buffers
int main(int argc, char **argv) {
  int i,rank,size,bufsize;
  int *h_buf;
  int *d_buf;
  MPI_Status status;
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  printf("Rank: %d CommSize: %d \n", rank, size);


  int hipDeviceCount, currDevice = 0;

  HIP_CHECK(hipGetDeviceCount(&hipDeviceCount));
  currDevice = rank % hipDeviceCount;
  HIP_CHECK(hipSetDevice(currDevice));

  printf("HIP Device count: %d , CurrDevice: %d\n", hipDeviceCount, currDevice);

  bufsize=100;

  int hipSize = 1 << 30;

  //allocate buffers
  h_buf=(int*) malloc(sizeof(int)*bufsize);
  hipMalloc(&d_buf, bufsize*sizeof(int));

  //initialize buffers
  if(rank==0) {
    for(i=0;i<bufsize;i++)
      h_buf[i]=i;
  }

  if(rank==1) {
    for(i=0;i<bufsize;i++)
      h_buf[i]=-1;
  }

  hipMemcpy(d_buf, h_buf, bufsize*sizeof(int), hipMemcpyHostToDevice);

  //communication
  if(rank==0)
    MPI_Send(d_buf, bufsize, MPI_INT, 1, 123, MPI_COMM_WORLD);

  if(rank==1)
    MPI_Recv(d_buf, bufsize, MPI_INT, 0, 123, MPI_COMM_WORLD, &status);

  //validate results
  if(rank==1) {
    hipMemcpy(h_buf, d_buf, bufsize*sizeof(int), hipMemcpyDeviceToHost);
    printf("buffer Data from rank 1: %d \n", h_buf[95]);
    for(i=0;i<bufsize;i++) {
      if(h_buf[i] != i)
        printf("Error: buffer[%d]=%d but expected %d\n", i, h_buf[i], i);
      }
    fflush(stdout);
  }

  //free buffers
  free(h_buf);
  hipFree(d_buf);
	
  MPI_Finalize();
}