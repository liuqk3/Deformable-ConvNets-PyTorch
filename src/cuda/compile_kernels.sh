#!/usr/bin/env bash

nvcc -c -o deformable_convnet_kernels.cu.o deformable_convnet_kernels.cu -x cu -Xcompiler -fPIC #-std=c++11

