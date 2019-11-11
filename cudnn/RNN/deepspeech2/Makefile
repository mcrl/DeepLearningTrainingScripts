DEEPSPEECH := deepspeech.cudnn

NVCC := nvcc
CC := mpicc
CXX := mpic++

SRC_PATH := ./src
OBJ_PATH := ./obj
INCLUDE_PATH := ./include
WARP_CTC_INCLUDE_PATH := ./warp-ctc/include

CC_INCLUDE := -I$(INCLUDE_PATH) -I$(WARP_CTC_INCLUDE_PATH)
CC_LIBRARY := -lm

CC_CUDA_INCLUDE := -I/usr/local/cuda/include -I/usr/local/cuda/cudnn-7.4.2/include
CC_CUDA_LIBRARY := -L/usr/local/cuda/cudnn-7.4.2/lib64 -L/usr/local/cuda/lib64/ -lcudnn -lcudart -lcublas -lnccl
WARP_CTC_LIBRARY := -L./warp-ctc/build -lwarpctc

SRCS := $(wildcard $(SRC_PATH)/*.c)
CU_SRCS := $(wildcard $(SRC_PATH)/*.cu)
CPP_SRCS := $(wildcard $(SRC_PATH)/*.cpp)

OBJS := $(patsubst $(SRC_PATH)/%.c,$(OBJ_PATH)/%.o,$(SRCS))
CU_OBJS := $(patsubst $(SRC_PATH)/%.cu,$(OBJ_PATH)/%.o,$(CU_SRCS))
CPP_OBJS := $(patsubst $(SRC_PATH)/%.cpp,$(OBJ_PATH)/%.o,$(CPP_SRCS))

override CFLAGS += -std=c99

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.c
	$(CC) -c -o $@ $^ $(CC_INCLUDE) $(CC_CUDA_INCLUDE) $(CFLAGS) -O3

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.cpp
	$(CXX) -c -o $@ $^ $(CC_INCLUDE) -std=c++11 -O3

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.cu
	$(NVCC) -c -o $@ $^ $(CC_INCLUDE) $(CC_CUDA_INCLUDE) -O3 -D_FORCE_INLINES

all : $(OBJS) $(CPP_OBJS) $(CU_OBJS)
	$(CXX) -o $(DEEPSPEECH) $^ $(CC_INCLUDE) $(CC_CUDA_INCLUDE) $(CC_LIBRARY) $(CC_CUDA_LIBRARY) $(WARP_CTC_LIBRARY) $(CFLAGS) -O3

clean :
	rm $(DEEPSPEECH) | true
	rm $(OBJ_PATH)/* | true
