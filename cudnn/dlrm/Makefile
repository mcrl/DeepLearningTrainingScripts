NVCC := nvcc
CC := gcc
CXX := mpic++

CC_INCLUDE := 
CFLAGS := 
DEFINES := 

SRC_PATH := ./src
OBJ_PATH := ./obj
INCLUDE_PATH := ./include

CUDA_INCLUDE_PATH := -I/usr/local/cuda/include -I/usr/local/cuda/cudnn-7.4.2/include

CC_CUDA_INCLUDE := -I/usr/include/openmpi-x86_64/ -I/usr/include -I/usr/local/cuda/include -I/usr/local/cuda/cudnn-7.4.2/include
CC_CUDA_LIBRARY := -L/usr/lib/x86_64-linux-gnu/ -L/usr/lib64 -L/usr/local/cuda/lib64/ -L/usr/local/cuda/cudnn-7.4.2/lib64 -L/lib64/openmpi/lib -lcudnn -lcublas -lcudart -lnccl -lgomp -lmpi

OBJS := $(patsubst $(SRC_PATH)/%.c,$(OBJ_PATH)/%.o,$(SRCS))
PURE_OBJS := # 
CU_OBJS := $(patsubst $(SRC_PATH)/%.cu,$(OBJ_PATH)/%.o,$(CU_SRCS))
CPP_OBJS := $(patsubst $(SRC_PATH)/%.cpp,$(OBJ_PATH)/%.o,$(CPP_SRCS))

DEFINES :=

.PHONY : all clean

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.cpp
	$(CXX) -c -o $@ $< $(CC_INCLUDE) $(CC_CUDA_INCLUDE) $(CFLAGS) $(DEFINES) -I$(INCLUDE_PATH) -std=c++11 -O3 -fopenmp

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.cu
	$(NVCC) -c -o $@ $< $(CC_INCLUDE) $(CC_CUDA_INCLUDE) $(CFLAGS) $(DEFINES) -I$(INCLUDE_PATH) -std=c++11 -O3 -Xcompiler -fopenmp

all: dlrm

dlrm: obj/main.o obj/data.o obj/configs.o obj/fc.o obj/interaction.o obj/tensor.o obj/activation.o obj/embedding.o obj/embeddingbag.o obj/utils.o obj/timer.o
	$(CXX) -o $@ $^ $(CC_INCLUDE) $(CC_CUDA_INCLUDE) $(CC_LIBRARY) $(CC_CUDA_LIBRARY) $(CFLAGS) -O3

clean:
	rm -f $(OBJ_PATH)/* dlrm dlrm| true

run: dlrm
	./dlrm