# comms-examples

These examples are standalone HIP examples that uses ROCm-aware openMPI and UCX and try to implement collective comms operations.

the goal for this examples is to have communication and computation to overlap and have GPU invoked communication without involving host or explicit device buffer transfers.
The way it is going to achieve that goal is to setup symmetric heap and map HIP device buffers through the heap, pass pointers across devices and reference cross-device buffers directly from kernel for get/put operations.
Examples will cover  GPU kernel invoked get/put and collective operations like all-reduce for intra-node GPUs.

## Requirements
* ROCm v6.2.0 onwards
* AMD GPU
    * MI300X (tested)
* ROCm-aware Open MPI and UCX
    * UCX is mainly used for inter-node communication over network like InifiniBand, RoCE, Ethernet, but can be used between GPUs on same node (intra-node) as well through shared memory. Currently examples are going to use shared memory for intra-node communication, inter-node examples will follow later.

 ## Building Dependencies
 Build and configure ROCm-aware Open MPI and UCX.
 ```
export INSTALL_DIR=$HOME/ompi_for_gpu
export ROCM_PATH=<rocm-path>
export UCX_DIR=$INSTALL_DIR/ucx
export OMPI_DIR=$INSTALL_DIR/ompi
```

Build UCX
```
git clone https://github.com/openucx/ucx.git -b v1.15.x
cd ucx
./autogen.sh
mkdir build
cd build
../configure -prefix=$UCX_DIR \
    --with-rocm=$ROCM_PATH
make -j 8
make -j 8 install
```

Build Open MPI
```
cd $BUILD_DIR
git clone --recursive https://github.com/open-mpi/ompi.git \
    -b v5.0.x
cd ompi
./autogen.pl
mkdir build
cd build
../configure --prefix=$OMPI_DIR --with-ucx=$UCX_DIR \
    --with-rocm=$ROCM_PATH
make -j 8
make -j 8 install
```

Update environment variable to use correct version of Open MPI and UCX
```
export LD_LIBRARY_PATH=$OMPI_DIR/lib:$UCX_DIR/lib:$ROCM_PATH/lib:$LD_LIBRARY_PATH
export PATH=$OMPI_DIR/bin:$PATH
```

## Building and Running examples
CMakeLists.txt will be updated later when the examples grow in complexities, currently it is not tested.

Compile example gpu_mpi
```
hipcc -o gpu_mpi gpu_mpi.cpp -I$ROCM_PATH/include -I$OMPI_DIR/include -L$OMPI_DIR/lib -lmpi -L$ROCM_PATH/lib -lamdhip64 
```

Run example
```
HIP_VISIBLE_DEVICES=0,1,2,3 mpirun -np 4 ./gpu_mpi
```

 ## References
 For in-depth details on GPU-aware MPI, please see:
 
https://rocm.docs.amd.com/en/latest/how-to/gpu-enabled-mpi.html

https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-gpu-aware-mpi-readme/


