#!/bin/bash

if [[ ! -d build ]]
then
  mkdir build
fi

hipcc main.hip -o add

#hipcc -o ../comms/gpu_mpi gpu_mpi.cpp -I/opt/rocm-6.2.0/include -I$OMPI_DIR/include -L$OMPI_DIR/lib -lmpi -L/opt/rocm-6.2.0/lib -lamdhip64 