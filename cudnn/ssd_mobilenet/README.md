# SSD-Mobilenet V1

## How to build
1. cd ./data
2. tar -xvzf coco-2017-val.tgz
3. cd ../
4. modify the CUDA path in Makefile (CUDA_INCLUDE, CUDA_LIB)
4. make
5. ./ssd_mobilenet
